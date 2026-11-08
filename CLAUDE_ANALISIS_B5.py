#!/usr/bin/env python3
"""
analisis_b500.py - ¿Hay ventaja medible en BOOM500?   (archivo independiente, v2)

No modifica reporte_*.py ni deriv_datos.py. Ponlo en la misma carpeta del proyecto.

PRUEBA 1  Deriva neta. Media ~ 0 => martingala => ninguna regla de entrada/salida basada
          en el pasado tiene EV > 0 antes de costos.
PRUEBA 2  Hazard de spikes. ¿La probabilidad de spike depende de:
          H  pasos desde el último spike      Z  caída desde el último spike (la "zona")
          M  momentum de los últimos 30 pasos A  intervalo anterior   S  tamaño vs intervalo
          p-values por bootstrap / permutación; Bonferroni x5.
"Paso" = 1 tick (modo ticks) o 1 minuto = 60 ticks (modo m1).

Uso:
  TICKS (la API solo entrega ~24 h por consulta; para acumular, corre --actualizar a diario):
    python analisis_b500.py --actualizar               # baja lo nuevo, lo suma al CSV y analiza TODO el caché
    python analisis_b500.py --ticks 500000             # analiza lo que haya en caché (extiende hacia atrás si puede)
  VELAS M1 (más historial; resolución de 1 minuto):
    python analisis_b500.py --modo m1 --velas 200000   # descarga propia con pausas anti-RateLimit
    python analisis_b500.py --modo m1 --velas 200000 --refrescar
  Opcionales: --spread 0.05 (en puntos de precio) imprime el hazard requerido para cubrirlo.
  AUTOTEST sin API: --sintetico nulo|hazard|precio   (con --modo m1 también)
Requiere: numpy, pandas, websockets
"""
import argparse
import asyncio
import json
import os
import sys

import numpy as np
import pandas as pd

URL_PUBLICO = "wss://api.derivws.com/trading/v1/options/ws/public"
MAX_POR_PETICION = 1000
P_NOMINAL = {"BOOM500": 1 / 500, "BOOM1000": 1 / 1000, "BOOM300N": 1 / 300}
Z_CORTES = np.array([-0.8416212, -0.2533471, 0.2533471, 0.8416212])  # quintiles N(0,1)
VENTANA_MOM = 30
N_FAMILIA = 5  # H, Z, M, A, S -> Bonferroni


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
async def _descargar(simbolo, cantidad, end, parar_en=None):
    """Pagina hacia atrás con 'end'. Para al llegar a 'parar_en'. Si falla, devuelve lo bajado."""
    import websockets
    ticks = {}
    try:
        async with websockets.connect(URL_PUBLICO, open_timeout=20) as ws:
            req = 0
            while len(ticks) < cantidad:
                req += 1
                pedir = min(MAX_POR_PETICION, cantidad - len(ticks) + 1)
                r = {}
                for intento in range(6):
                    await ws.send(json.dumps({"ticks_history": simbolo, "style": "ticks",
                                              "count": pedir, "end": end, "req_id": req}))
                    r = json.loads(await asyncio.wait_for(ws.recv(), 30))
                    if r.get("error") and "limit" in json.dumps(r["error"]).lower():
                        await asyncio.sleep(5 * (intento + 1))
                        continue
                    break
                if r.get("error"):
                    raise RuntimeError(f"Deriv API: {r['error']}")
                h = r.get("history", {})
                nuevos = [(t, p) for t, p in zip(h.get("times", []), h.get("prices", []))
                          if t not in ticks]
                if not nuevos:
                    break
                for t, p in nuevos:
                    ticks[t] = p
                end = min(ticks) - 1
                if req % 25 == 0:
                    print(f"  {len(ticks):,} ticks descargados...", flush=True)
                if parar_en is not None and min(ticks) <= parar_en:
                    break
                await asyncio.sleep(0.05)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Descarga interrumpida ({e!r}). Se guarda lo obtenido; vuelve a correr para continuar.")
    return ticks


def _df_vacio():
    return pd.DataFrame({"epoch": pd.Series(dtype="int64"), "quote": pd.Series(dtype="float64")})


def _guardar(df, nuevos, archivo):
    if nuevos:
        add = pd.DataFrame({"epoch": list(nuevos.keys()), "quote": list(nuevos.values())})
        df = pd.concat([df, add]).drop_duplicates("epoch").sort_values("epoch").reset_index(drop=True)
        df.to_csv(archivo, index=False)
    return df


def cargar_ticks(simbolo, cantidad, archivo, actualizar=False):
    df = pd.read_csv(archivo) if os.path.exists(archivo) else _df_vacio()
    if actualizar:
        parar = int(df["epoch"].max()) if len(df) else None
        print(f"📡 Actualizando {simbolo}: caché {len(df):,} ticks ({archivo})...", flush=True)
        nuevos = asyncio.run(asyncio.wait_for(
            _descargar(simbolo, 10**9 if parar else cantidad, "latest", parar), timeout=7200))
        if parar is not None:
            nuevos = {t: p for t, p in nuevos.items() if t > parar}
            if nuevos and min(nuevos) - parar > 10:
                h = (min(nuevos) - parar) / 3600
                print(f"⚠️ Hueco de {h:.1f} h entre el caché y los ticks nuevos (la API solo da ~24 h). "
                      f"El análisis usará el segmento continuo más largo. Corre --actualizar al menos una vez al día.")
        df = _guardar(df, nuevos, archivo)
        print(f"   +{len(nuevos):,} ticks nuevos | total {len(df):,}")
        return df
    falta = cantidad - len(df)
    if falta > 0:
        end = int(df["epoch"].min()) - 1 if len(df) else "latest"
        print(f"📡 Descargando hasta {falta:,} ticks de {simbolo} (caché: {archivo})...", flush=True)
        nuevos = asyncio.run(asyncio.wait_for(_descargar(simbolo, falta, end), timeout=7200))
        df = _guardar(df, nuevos, archivo)
        if len(df) < cantidad:
            print(f"   La API no devolvió más historial (límite de profundidad). Disponible: {len(df):,} ticks.")
    return df.tail(cantidad).reset_index(drop=True)


async def _descargar_velas(simbolo, cantidad, end="latest"):
    """Velas M1 paginadas hacia atrás, con espera y reintentos ante RateLimit. Devuelve {epoch: close}."""
    import websockets
    velas = {}
    try:
        async with websockets.connect(URL_PUBLICO, open_timeout=20) as ws:
            req = 0
            while len(velas) < cantidad:
                req += 1
                pedir = min(5000, cantidad - len(velas) + 1)
                r = {}
                for intento in range(8):
                    await ws.send(json.dumps({"ticks_history": simbolo, "style": "candles",
                                              "granularity": 60, "count": pedir,
                                              "end": end, "req_id": req}))
                    r = json.loads(await asyncio.wait_for(ws.recv(), 60))
                    if r.get("error") and "limit" in json.dumps(r["error"]).lower():
                        espera = 20 * (intento + 1)
                        print(f"  RateLimit: espero {espera}s (intento {intento + 1}/8)...", flush=True)
                        await asyncio.sleep(espera)
                        continue
                    break
                if r.get("error"):
                    raise RuntimeError(f"Deriv API: {r['error']}")
                nuevos = {int(c["epoch"]): float(c["close"]) for c in r.get("candles", [])
                          if int(c["epoch"]) not in velas}
                if not nuevos:
                    break
                velas.update(nuevos)
                end = min(velas) - 1
                print(f"  {len(velas):,} velas descargadas...", flush=True)
                await asyncio.sleep(3)  # ticks_history_expensive: ritmo lento
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Descarga interrumpida ({e!r}). Se guarda lo obtenido; vuelve a correr para continuar.")
    return velas


def cargar_m1(simbolo, cantidad, archivo, refrescar=False):
    df = pd.read_csv(archivo) if os.path.exists(archivo) else _df_vacio()
    if len(df) >= cantidad and not refrescar:
        return df.tail(cantidad).reset_index(drop=True)
    print(f"📡 Descargando hasta {cantidad:,} velas M1 de {simbolo} (caché: {len(df):,})...", flush=True)
    nuevas = asyncio.run(asyncio.wait_for(_descargar_velas(simbolo, cantidad), timeout=7200))
    df = _guardar(df, nuevas, archivo)
    if len(df) == 0:
        print("❌ No se obtuvo ninguna vela.")
        sys.exit(1)
    dias = (df["epoch"].iloc[-1] - df["epoch"].iloc[0]) / 86400
    print(f"   Obtenidas {len(df):,} velas ({dias:.1f} días). Pedidas: {cantidad:,}.")
    return df.tail(cantidad).reset_index(drop=True)


def generar_sintetico(n, modo, semilla):
    """Generador tipo Boom: spike Bernoulli(1/500), deriva que compensa (martingala)."""
    rng = np.random.default_rng(semilla)
    p, sig, S = 1 / 500, 1.0, 300.0
    mu_ns = -p * S / (1 - p)
    ruido = rng.normal(mu_ns, sig, n)
    u = rng.random(n)
    tam = rng.gamma(4.0, S / 4.0, n)
    d = np.empty(n)
    k, acc = 0, 0.0
    for i in range(n):
        k += 1
        ph = p
        if modo == "hazard" and 350 <= k <= 450:
            ph = 2.0 * p
        elif modo == "precio" and k >= 2:
            if (acc - mu_ns * (k - 1)) / (sig * np.sqrt(k - 1)) < -1.0:
                ph = 2.0 * p
        if u[i] < ph:
            d[i] = tam[i] + ruido[i]
            k, acc = 0, 0.0
        else:
            d[i] = ruido[i]
            acc += ruido[i]
    q = 8000.0 + np.concatenate(([0.0], np.cumsum(d)))
    epoch = 1_700_000_000 + np.arange(n + 1)
    return epoch, q


# ----------------------------------------------------------------------------
# Utilidades estadísticas
# ----------------------------------------------------------------------------
def segmento_continuo(epoch, quote, max_gap):
    """Devuelve el segmento continuo más largo (corta donde dt > max_gap × dt mediano)."""
    dt = np.diff(epoch)
    cortes = np.flatnonzero(dt > max_gap * np.median(dt))
    if len(cortes) == 0:
        return epoch, quote, 1
    lim = np.concatenate(([0], cortes + 1, [len(epoch)]))
    j = int(np.argmax(np.diff(lim)))
    return epoch[lim[j]:lim[j + 1]], quote[lim[j]:lim[j + 1]], len(cortes) + 1


def eventos(mask, d):
    """Agrupa pasos marcados consecutivos en un solo spike. Devuelve (inicios, tamaños, %multi)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx, np.array([]), 0.0
    nuevo = np.concatenate(([True], np.diff(idx) > 1))
    run = np.cumsum(nuevo) - 1
    tam = np.bincount(run, weights=d[idx])
    largo = np.bincount(run)
    return idx[nuevo], tam, float((largo > 1).mean())


def g_filas(ev, ex):
    """Desvianza de Poisson (G) de hazard constante vs hazard por bin. Filas = simulaciones."""
    ev = np.atleast_2d(ev).astype(float)
    ex = np.asarray(ex, float)
    ex = ex if ex.ndim == 2 else np.broadcast_to(ex, ev.shape)
    p = ev.sum(1, keepdims=True) / ex.sum(1, keepdims=True)
    mu = p * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(ev > 0, ev * np.log(ev / mu), 0.0)
    return 2 * t.sum(1) - 2 * (ev - mu).sum(1)


def p_boot(obs, sim):
    return (1 + np.sum(sim >= obs)) / (1 + len(sim))


def spearman_perm(x, y, rng, nperm=5000):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    r = np.corrcoef(rx, ry)[0, 1]
    sim = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(nperm)])
    return r, (1 + np.sum(np.abs(sim) >= abs(r))) / (1 + nperm)


def estado_z(ev_mask, cs, mu, sig, idx):
    """z = caída acumulada desde el último spike, estandarizada (N(0,1) si no hay memoria)."""
    last = np.maximum.accumulate(np.where(ev_mask, idx, -1))
    s = np.empty_like(last)
    s[0] = -1
    s[1:] = last[:-1]
    k = idx - s
    ok = (s >= 0) & (k >= 2)
    z = np.full(len(idx), np.nan)
    i, ss, kk = idx[ok], s[ok], k[ok]
    z[ok] = (cs[i - 1] - cs[ss] - mu * (kk - 1)) / (sig * np.sqrt(kk - 1))
    return z, ok


def tabla_quintiles(z, ok, ev_mask):
    b = np.digitize(z[ok], Z_CORTES)
    ex = np.bincount(b, minlength=5)
    ev = np.bincount(b, weights=ev_mask[ok].astype(float), minlength=5)
    return ev, ex


# ----------------------------------------------------------------------------
# Análisis
# ----------------------------------------------------------------------------
def analizar(epoch, quote, simbolo="BOOM500", k_sigma=6.0, spread=None,
             sims=2000, sims_z=300, semilla=0, verbose=True, ticks_por_paso=1, max_gap=5):
    rng = np.random.default_rng(semilla)
    P = print if verbose else (lambda *a, **k: None)
    tpp = ticks_por_paso
    unidad = "tick" if tpp == 1 else f"paso de {tpp} ticks"
    epoch, q, nseg = segmento_continuo(np.asarray(epoch), np.asarray(quote, float), max_gap)
    d = np.diff(q)
    n = len(d)
    dt = np.diff(epoch)
    med_dt = float(np.median(dt))
    saltos = int((dt > 1.5 * med_dt).sum())
    p_tick_nom = P_NOMINAL.get(simbolo, 1 / 500)
    p_nom = 1 - (1 - p_tick_nom) ** tpp  # prob. nominal de ≥1 spike por paso

    P("=" * 78)
    P(f"ANÁLISIS {simbolo}: {n:,} pasos ({unidad}) | {pd.to_datetime(epoch[0], unit='s')} -> "
      f"{pd.to_datetime(epoch[-1], unit='s')} | dt mediano {med_dt:g}s")
    if nseg > 1:
        P(f"⚠️ La serie tenía {nseg} segmentos (huecos > {max_gap}×dt). Se analiza solo el más largo.")
    if saltos:
        P(f"⚠️ {saltos} saltos de tiempo > 1.5×dt (datos faltantes). Con >0.1% de saltos, desconfía.")

    # ---- detección de spikes ------------------------------------------------
    med = np.median(d)
    sig_r = 1.4826 * np.median(np.abs(d - med))
    P(f"\nσ robusto del ruido por paso = {sig_r:.5f}  | mediana de d = {med:.5f}")
    P("Sensibilidad del umbral (spike = d > mediana + k·σ):")
    P("   k   spikes   tasa por paso   ÷ nominal   %multi")
    for kk in (4, 5, 6, 8, 10, 15):
        ei, _, mm = eventos(d > med + kk * sig_r, d)
        P(f"  {kk:>2}  {len(ei):>7}   1/{n / max(len(ei), 1):>8.1f}      {len(ei) / n / p_nom:>6.2f}      {mm:>6.1%}")
    mask_tick = d > med + k_sigma * sig_r
    ev_idx, tam, _ = eventos(mask_tick, d)
    m = len(ev_idx)
    P(f"Umbral usado k={k_sigma:g}: {m} spikes. Si la tasa no se estabiliza al subir k, "
      f"el umbral mezcla ruido con spikes.")
    if tpp > 1:
        P(f"(M1: un minuto con 2+ spikes cuenta como 1; el nominal por minuto es {p_nom:.4f}.)")
    if m < 60:
        P("❌ Muy pocos spikes para concluir. Necesitas más datos.")
        return {"error": "pocos_spikes", "n_spikes": m}

    ns = d[~mask_tick]
    mu_ns, sig_ns = ns.mean(), ns.std(ddof=1)
    p_paso = m / n
    S_med = tam.mean()

    # ---- PRUEBA 1: deriva neta ---------------------------------------------
    mu = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    t_mu = mu / se
    f500 = 500 / tpp
    P("\n" + "-" * 78)
    P("PRUEBA 1 - DERIVA NETA")
    P(f"  tasa de spike        : {p_paso:.5f} por paso (1/{1 / p_paso:.1f})   nominal 1/{1 / p_nom:.1f}")
    P(f"  tamaño medio spike   : {S_med:.4f}")
    P(f"  deriva entre spikes  : {mu_ns:.6f} por paso")
    comp = (p_paso * S_med) / (-(1 - p_paso) * mu_ns) if mu_ns < 0 else float("nan")
    P(f"  compensación spike/deriva = {comp:.3f}  (1.000 = diseño martingala exacto)")
    P(f"  media neta por paso  : {mu:.6f}   IC95% [{mu - 1.96 * se:.6f}, {mu + 1.96 * se:.6f}]   t = {t_mu:.2f}")
    P(f"  por 500 ticks        : {mu * f500:.4f}   IC95% [{(mu - 1.96 * se) * f500:.4f}, {(mu + 1.96 * se) * f500:.4f}]")
    bl = np.array_split(d, 10)
    tb = [b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) for b in bl]
    P("  t por bloque (10)    : " + " ".join(f"{x:5.1f}" for x in tb) + "   (poco fiable con <50 spikes por bloque)")
    if abs(t_mu) < 2:
        P("  => Deriva neta indistinguible de 0. Tratar como martingala: sin EV>0 antes de costos.")
    else:
        P(f"  => Deriva neta {'NEGATIVA' if mu < 0 else 'POSITIVA'} significativa. "
          f"Hay sesgo direccional de largo plazo; el spread decide si es explotable.")
    if spread is not None and mu_ns < 0:
        d_t = -mu_ns / tpp                          # deriva por tick entre spikes
        p_t = 1 - (1 - p_paso) ** (1 / tpp)         # hazard base por tick
        p_eq = d_t / (S_med + d_t)                  # hazard de equilibrio sin spread
        P(f"\n  EQUILIBRIO con spread {spread:g} (hazard medio por tick que debe darse durante la tenencia):")
        P(f"    base medido {p_t:.5f} | equilibrio sin spread {p_eq:.5f} ({p_eq / p_t:.2f}× base)")
        P("    tenencia N ticks   hazard requerido   ÷ base")
        for N in (60, 300, 900):
            req = p_eq + spread / (S_med * N)
            P(f"    {N:>10}         {req:.5f}        {req / p_t:6.2f}×")
        P("    Una zona solo es operable si su hazard medio supera ese requerido de forma estable fuera de muestra.")

    # ---- PRUEBA 2 --------------------------------------------------------------
    P("\n" + "-" * 78)
    P("PRUEBA 2 - ¿HAY MEMORIA EN LOS SPIKES?   (H0: hazard constante, independiente del pasado)")
    gaps = np.diff(ev_idx)
    cola = n - 1 - ev_idx[-1]
    mg = len(gaps)
    p_hat = mg / (gaps.sum() + cola)
    P(f"  intervalos: {mg} | media {gaps.mean():.1f} pasos | p̂ = {p_hat:.5f} (1/{1 / p_hat:.1f})")

    # H: hazard por pasos desde el último spike
    nb = int(min(10, max(4, mg // 40)))
    qs = np.linspace(0, 1, nb + 1)[1:-1]
    inner = np.unique(np.ceil(np.log(1 - qs) / np.log(1 - p_hat)).astype(np.int64))
    lows = np.concatenate(([1], inner + 1))
    ups = np.concatenate((inner, [1 << 60]))

    def stats_bins(g2d, cola_):
        ev = np.stack([((g2d >= a) & (g2d <= b)).sum(1) for a, b in zip(lows, ups)], 1)
        ex = np.stack([np.clip(np.minimum(g2d, b) - a + 1, 0, None).sum(1)
                       + max(0, min(cola_, b) - a + 1) for a, b in zip(lows, ups)], 1)
        return ev, ex

    ev_o, ex_o = stats_bins(gaps[None, :], cola)
    G_o = g_filas(ev_o, ex_o)[0]
    sim_g = np.empty(sims)
    for j in range(0, sims, 200):
        w = min(200, sims - j)
        gs = rng.geometric(p_hat, size=(w, mg))
        e_s, x_s = stats_bins(gs, cola)
        sim_g[j:j + w] = g_filas(e_s, x_s)
    pH = p_boot(G_o, sim_g)
    pot = 196 / np.sqrt(mg / len(lows))
    P("\n  H) Hazard por pasos desde el último spike")
    P("     k (pasos)          eventos   exposición   hazard×1000   ÷p̂    IC95% (÷p̂)")
    for i in range(len(lows)):
        e, x = ev_o[0, i], ex_o[0, i]
        h = e / x
        ci = 1.96 / np.sqrt(max(e, 1))
        hi = f"{ups[i]}" if ups[i] < (1 << 59) else "∞"
        P(f"     {lows[i]:>5}-{hi:<8}   {e:>7}   {x:>10}   {h * 1000:>10.3f}   {h / p_hat:>5.2f}   "
          f"[{max(h / p_hat * (1 - ci), 0):.2f}, {h / p_hat * (1 + ci):.2f}]")
    P(f"     G = {G_o:.2f}   p (bootstrap) = {pH:.4f}   (≈{mg / len(lows):.0f} eventos por bin => ±{pot:.0f}% de error relativo)")

    # Z: caída desde el último spike (la "zona")
    dn = d.copy()
    dn[mask_tick] = mu_ns
    cs = np.cumsum(dn)
    idx = np.arange(n)
    ev_mask = np.zeros(n, bool)
    ev_mask[ev_idx] = True
    z, ok = estado_z(ev_mask, cs, mu_ns, sig_ns, idx)
    evz, exz = tabla_quintiles(z, ok, ev_mask)
    Gz = g_filas(evz, exz)[0]
    p_z = evz.sum() / exz.sum()
    sim_z = np.empty(sims_z)
    for j in range(sims_z):
        em = rng.random(n) < p_z
        zz, okk = estado_z(em, cs, mu_ns, sig_ns, idx)
        e_s, x_s = tabla_quintiles(zz, okk, em)
        sim_z[j] = g_filas(e_s, x_s)[0]
    pZ = p_boot(Gz, sim_z)
    P("\n  Z) Hazard según cuánto cayó el precio desde el último spike (Q1 = cayó MÁS de lo esperado)")
    for i in range(5):
        h = evz[i] / exz[i]
        P(f"     Q{i + 1}: eventos {int(evz[i]):>5}  exposición {exz[i]:>9}  hazard×1000 {h * 1000:7.3f}  ÷media {h / p_z:5.2f}")
    P(f"     G = {Gz:.2f}   p (bootstrap) = {pZ:.4f}")

    # M: momentum local (proxy de EMAs)
    w = VENTANA_MOM
    zm = np.full(n, np.nan)
    ii = idx[w + 1:]
    zm[ii] = (cs[ii - 1] - cs[ii - 1 - w] - mu_ns * w) / (sig_ns * np.sqrt(w))
    okm = ~np.isnan(zm)
    evm, exm = tabla_quintiles(zm, okm, ev_mask)
    Gm = g_filas(evm, exm)[0]
    p_m = evm.sum() / exm.sum()
    sim_m = g_filas(rng.binomial(exm, p_m, size=(sims, 5)), exm)
    pM = p_boot(Gm, sim_m)
    P(f"\n  M) Hazard según momentum de los últimos {w} pasos (Q1 = bajista fuerte, Q5 = alcista fuerte)")
    for i in range(5):
        h = evm[i] / exm[i]
        P(f"     Q{i + 1}: eventos {int(evm[i]):>5}  exposición {exm[i]:>9}  hazard×1000 {h * 1000:7.3f}  ÷media {h / p_m:5.2f}")
    P(f"     G = {Gm:.2f}   p (bootstrap) = {pM:.4f}")

    # A: autocorrelación de intervalos / S: tamaño vs intervalo
    rA, pA = spearman_perm(gaps[:-1], gaps[1:], rng)
    rS, pS = spearman_perm(gaps, tam[1:], rng)
    P(f"\n  A) Spearman intervalo[t] vs intervalo[t+1]: rho = {rA:+.3f}   p = {pA:.4f}   (±{1.96 / np.sqrt(mg):.3f} es ruido)")
    P(f"  S) Spearman tamaño del spike vs intervalo previo: rho = {rS:+.3f}   p = {pS:.4f}")

    # ---- Veredicto (depende de la potencia) -----------------------------------
    pv = {"H": pH, "Z": pZ, "M": pM, "A": pA, "S": pS}
    pa = {k: min(1.0, v * N_FAMILIA) for k, v in pv.items()}
    P("\n" + "=" * 78)
    P("RESUMEN (p ajustado = min(1, 5·p), Bonferroni sobre H, Z, M, A, S)")
    nombres = {"H": "pasos desde último spike", "Z": "caída desde último spike",
               "M": f"momentum {w} pasos", "A": "autocorr. intervalos", "S": "tamaño vs intervalo"}
    for k in "HZMAS":
        P(f"  {k}  {nombres[k]:<26} p = {pv[k]:.4f}   p ajustado = {pa[k]:.4f}  {'<-- SEÑAL' if pa[k] < 0.05 else ''}")
    señal = [k for k in "HZMAS" if pa[k] < 0.05]
    nec = len(lows) * 171
    if señal:
        P(f"\nVEREDICTO: hay estructura en {', '.join(señal)} (p ajustado < 0.05). Siguiente paso: medir EV neto de")
        P("spread con validación walk-forward en datos NUEVOS (no esta muestra) antes de construir nada encima.")
    elif abs(t_mu) >= 2:
        P("\nVEREDICTO: sin memoria detectable en los spikes, pero hay deriva neta significativa. Evalúa su EV contra el spread.")
    elif pot <= 15:
        P(f"\nVEREDICTO: SIN MEMORIA DETECTABLE y sin deriva neta, con potencia suficiente ({m} spikes, ±{pot:.0f}% por bin).")
        P("Se descartan efectos mayores a ese error: ningún indicador basado en precio tiene EV>0 antes de costos aquí.")
    else:
        P(f"\nVEREDICTO: INCONCLUSO. No hay evidencia de ventaja, pero tampoco de su ausencia: {m} spikes dan ±{pot:.0f}% por bin.")
        P(f"Solo se descartan efectos grandes. Para ±15% por bin necesitas ≈{nec:,} spikes (tienes {m}).")
    return {"n_spikes": m, "t_mu": t_mu, "mu": mu, "p": pv, "p_adj": pa, "p_hat": p_hat, "pot": pot}


def main():
    ap = argparse.ArgumentParser(description="Prueba de deriva y memoria de spikes en BOOM500")
    ap.add_argument("--simbolo", default="BOOM500")
    ap.add_argument("--modo", choices=["ticks", "m1"], default="ticks")
    ap.add_argument("--ticks", type=int, default=500_000, help="ticks a usar/descargar (modo ticks)")
    ap.add_argument("--velas", type=int, default=200_000, help="velas M1 a usar/descargar (modo m1)")
    ap.add_argument("--actualizar", action="store_true",
                    help="(ticks) baja solo lo nuevo, lo suma al caché y analiza TODO el caché")
    ap.add_argument("--refrescar", action="store_true", help="(m1) ignora el caché y vuelve a descargar")
    ap.add_argument("--archivo", default=None, help="caché CSV (default ticks_<simbolo>.csv / velas_m1_<simbolo>.csv)")
    ap.add_argument("--k", type=float, default=6.0, help="umbral de spike en σ robustas")
    ap.add_argument("--spread", type=float, default=None, help="spread en puntos de precio (opcional)")
    ap.add_argument("--max-gap", type=float, default=5.0, help="corta la serie donde dt > max-gap × dt mediano")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--sims-z", type=int, default=300)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--sintetico", choices=["nulo", "hazard", "precio"], default=None,
                    help="autotest con datos simulados (no usa la API)")
    a = ap.parse_args()

    tpp = 60 if a.modo == "m1" else 1
    if a.sintetico:
        print(f"[AUTOTEST sintético: {a.sintetico} | modo {a.modo}]")
        n_ticks = a.velas * 60 if a.modo == "m1" else a.ticks
        epoch, q = generar_sintetico(n_ticks, a.sintetico, a.semilla)
        if a.modo == "m1":
            epoch, q = epoch[::60], q[::60]
    elif a.modo == "m1":
        archivo = a.archivo or f"velas_m1_{a.simbolo}.csv"
        df = cargar_m1(a.simbolo, a.velas, archivo, a.refrescar)
        epoch, q = df["epoch"].to_numpy(), df["quote"].to_numpy()
    else:
        archivo = a.archivo or f"ticks_{a.simbolo}.csv"
        df = cargar_ticks(a.simbolo, a.ticks, archivo, a.actualizar)
        epoch, q = df["epoch"].to_numpy(), df["quote"].to_numpy()
    if len(q) < (3_000 if a.modo == "m1" else 50_000):
        print(f"❌ Solo {len(q):,} datos disponibles; muy pocos para analizar.")
        sys.exit(1)
    res = analizar(epoch, q, a.simbolo, a.k, a.spread, a.sims, a.sims_z, a.semilla,
                   ticks_por_paso=tpp, max_gap=a.max_gap)
    if "error" in res:
        sys.exit(1)


if __name__ == "__main__":
    main()