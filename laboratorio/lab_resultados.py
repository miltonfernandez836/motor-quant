"""Laboratorio, paso 3: resultados de operar las zonas del replay (un trade por zona tocada) vs un control al azar.

Uso:
    python laboratorio/lab_resultados.py --simbolo BOOM500

Para cada zona tocada con foto previa (pt_* no vacías) simula UN trade en el primer toque:
  - entrada según smc_nucleo.CONFIG["ENTRADA"] ("borde" = borde de la zona por donde entra el precio; "50" = mitad),
    SL en el borde opuesto, TP a 1R, 2R y 3R (una columna de resultado para cada uno);
  - resolución con las velas M15 posteriores. En la vela de entrada solo se comprueba el SL (no se sabe en qué
    orden ocurrieron el máximo y el mínimo antes de entrar, así que no se cuenta un TP ahí). Si una misma vela
    toca SL y TP, cuenta SL primero;
  - r = resultado con la pérdida a precio de SL (-1); r_peor = la pérdida al extremo adverso de la vela que toca
    el SL (en Boom los spikes pueden saltar el SL);
  - máximo MAX_VELAS velas M15 tras la de entrada; si no resuelve, cierra al último precio ("sin_resolver");
  - MFE/MAE en R: recorrido favorable/adverso hasta el SL (o el límite de tiempo), sin depender del TP.
Control: N_CONTROL entradas al azar por cada trade real (semilla fija), en velas M15 del periodo del replay, con la
misma dirección, el mismo riesgo en precio y los mismos TP; entran al cierre de la vela elegida.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import smc_nucleo as smc  # noqa: E402
from lab_datos import DIR_DATOS, cargar  # noqa: E402
from lab_replay import pasos_del_replay, preparar  # noqa: E402

CONFIG_LAB = {
    "TP_R": (1, 2, 3),            # múltiplos de R para el take profit
    "MAX_VELAS": 192,             # tiempo máximo del trade, en velas M15 tras la de entrada
    "N_CONTROL": 20,              # entradas al azar por cada trade real
    "SEMILLA": 20260101,          # semilla fija del control y del bootstrap
    "BOOTSTRAP": 5000,            # remuestreos para el intervalo de confianza
    "BREAKEVEN_2R": 100 / 3,      # % de aciertos de equilibrio con TP a 2R (pérdida = -1R)
}
FOTO = ["r1", "r2", "r3", "r4", "puntaje", "sesgo_h1", "sesgo_h4", "pos_tr", "tamano_atr", "atr", "hora"]


def simular(H, L, C, j, entrada, riesgo, desde_entrada):
    """Simula un trade de COMPRA (para venta se pasan los precios negados). j = posición de la vela de entrada.
    Devuelve {'mfe', 'mae', tp: (estado, r, r_peor)} o None si no hay velas posteriores."""
    N = len(H)
    fin = min(j + CONFIG_LAB["MAX_VELAS"], N - 1)
    ini = j if desde_entrada else j + 1
    if ini > fin:
        return None
    hi, lo = H[ini:fin + 1], L[ini:fin + 1]
    valida = np.arange(ini, fin + 1) > j                  # TP y MFE solo cuentan después de la vela de entrada
    sl = entrada - riesgo
    golpe_sl = lo <= sl
    isl = int(np.argmax(golpe_sl)) if golpe_sl.any() else None
    hasta = isl if isl is not None else len(lo) - 1
    mae = max(0.0, float((entrada - lo[:hasta + 1].min()) / riesgo))
    limite = isl if isl is not None else len(hi)
    favorables = hi[:limite][valida[:limite]]
    mfe = max(0.0, float((favorables.max() - entrada) / riesgo)) if len(favorables) else 0.0
    salida = {"mfe": mfe, "mae": mae}
    for n in CONFIG_LAB["TP_R"]:
        golpe_tp = (hi >= entrada + n * riesgo) & valida
        itp = int(np.argmax(golpe_tp)) if golpe_tp.any() else None
        if isl is not None and (itp is None or isl <= itp):           # SL primero si coinciden
            salida[n] = ("sl", -1.0, float((lo[isl] - entrada) / riesgo))
        elif itp is not None:
            salida[n] = ("tp", float(n), float(n))
        else:
            r = float((C[fin] - entrada) / riesgo)
            salida[n] = ("sin_resolver", r, r)
    return salida


def columnas_resultado(sal):
    f = {"mfe_r": round(sal["mfe"], 3), "mae_r": round(sal["mae"], 3)}
    for n in CONFIG_LAB["TP_R"]:
        estado, r, peor = sal[n]
        f[f"estado_tp{n}"], f[f"r_tp{n}"], f[f"r_peor_tp{n}"] = estado, round(r, 4), round(peor, 4)
    return f


def precios(m15):
    """Arrays de COMPRA y de VENTA (negados: h'=-l, l'=-h, c'=-c) para reutilizar la misma simulación."""
    h, l, c = (m15[x].to_numpy(float) for x in ("high", "low", "close"))
    return {"COMPRA": (h, l, c), "VENTA": (-l, -h, -c)}


def trades_reales(zonas, m15, px):
    filas = []
    modo = smc.CONFIG["ENTRADA"]
    for z in zonas.itertuples(index=False):
        compra = z.direccion == "COMPRA"
        medio = (z.alto + z.bajo) / 2
        entrada = medio if modo == "50" else (z.alto if compra else z.bajo)
        sl = z.bajo if compra else z.alto
        riesgo = abs(entrada - sl)
        j = m15.index.get_loc(pd.Timestamp(z.t_primer_toque, tz="UTC"))
        if riesgo <= 0:
            continue
        H, L, C = px[z.direccion]
        e = entrada if compra else -entrada
        sal = simular(H, L, C, j, e, riesgo, desde_entrada=True)
        if sal is None:
            continue
        fila = {"zona_id": z.zona_id, "tipo": z.tipo, "direccion": z.direccion, "bajo": z.bajo, "alto": z.alto,
                "t_primer_toque": z.t_primer_toque, "estado_final": z.estado_final,
                "entrada": entrada, "sl": sl, "riesgo": round(riesgo, 4)}
        fila.update({f"pt_{c}": getattr(z, f"pt_{c}") for c in FOTO})
        fila.update(columnas_resultado(sal))
        filas.append(fila)
    return pd.DataFrame(filas)


def trades_control(reales, m15, px, desde):
    rng = np.random.default_rng(CONFIG_LAB["SEMILLA"])
    posibles = np.arange(desde, len(m15) - 1 - CONFIG_LAB["MAX_VELAS"])      # con horizonte completo
    filas = []
    for t in reales.itertuples(index=False):
        compra = t.direccion == "COMPRA"
        H, L, C = px[t.direccion]
        for n, j in enumerate(rng.choice(posibles, CONFIG_LAB["N_CONTROL"], replace=False)):
            entrada = float(C[j])                                           # al cierre de la vela elegida
            sal = simular(H, L, C, int(j), entrada, t.riesgo, desde_entrada=False)
            fila = {"zona_id": t.zona_id, "n_control": n, "t_entrada": m15.index[j], "direccion": t.direccion,
                    "entrada": entrada if compra else -entrada, "riesgo": t.riesgo}
            fila.update(columnas_resultado(sal))
            filas.append(fila)
    return pd.DataFrame(filas)


def bootstrap_ic(valores, rng):
    """Media e IC 95% (percentiles) de la media, por bootstrap."""
    v = np.asarray(valores, float)
    idx = rng.integers(0, len(v), size=(CONFIG_LAB["BOOTSTRAP"], len(v)))
    medias = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def tabla(reales, control, grupos):
    """Una tabla con TP a 2R: por grupo, trades reales vs control (el control se agrupa por trade real)."""
    rng = np.random.default_rng(CONFIG_LAB["SEMILLA"])
    ctrl = control.assign(acierto=control["estado_tp2"] == "tp")
    por_trade = ctrl.groupby("zona_id").agg(r=("r_tp2", "mean"), ac=("acierto", "mean"), peor=("r_peor_tp2", "mean"))
    be = CONFIG_LAB["BREAKEVEN_2R"]
    print(f"{'grupo':<14}{'n':>5} {'acierto':>8} {'vs BE':>7}  {'R medio [IC95%]':<24}{'R peor':>8}"
          f" | {'n ctrl':>6} {'acierto':>8}  {'R medio [IC95%]':<24}{'R peor':>8}")
    for nombre, mascara in grupos:
        g = reales[mascara]
        if not len(g):
            continue
        ac = (g["estado_tp2"] == "tp").to_numpy(float)
        m, lo, hi = bootstrap_ic(g["r_tp2"], rng)
        c = por_trade.loc[g["zona_id"]]
        cm, clo, chi = bootstrap_ic(c["r"], rng)
        print(f"{nombre:<14}{len(g):>5} {ac.mean() * 100:>7.1f}% {ac.mean() * 100 - be:>+6.1f}  "
              f"{m:>+6.3f} [{lo:>+6.3f},{hi:>+6.3f}]  {g['r_peor_tp2'].mean():>+8.3f}"
              f" | {len(g) * CONFIG_LAB['N_CONTROL']:>6} {c['ac'].mean() * 100:>7.1f}%  "
              f"{cm:>+6.3f} [{clo:>+6.3f},{chi:>+6.3f}]  {c['peor'].mean():>+8.3f}")


def principal():
    ap = argparse.ArgumentParser(description="Laboratorio: resultados de operar las zonas del replay")
    ap.add_argument("--simbolo", default="BOOM500", choices=list(smc.CONFIG["ACTIVOS"]))
    args = ap.parse_args()
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    simbolo = args.simbolo
    m15 = cargar(simbolo)
    zonas = pd.read_csv(os.path.join(DIR_DATOS, f"zonas_{simbolo}.csv"))
    zonas = zonas[zonas["t_primer_toque"].notna() & zonas["pt_puntaje"].notna()]
    px = precios(m15)
    _, marcos, fines = preparar(simbolo)
    desde = int(pasos_del_replay(m15, marcos, fines)[0])                 # primera vela del periodo de replay
    reales = trades_reales(zonas, m15, px)
    control = trades_control(reales, m15, px, desde)
    reales.to_csv(os.path.join(DIR_DATOS, f"resultados_{simbolo}.csv"), index=False)
    control.to_csv(os.path.join(DIR_DATOS, f"resultados_control_{simbolo}.csv"), index=False)

    p = reales["pt_puntaje"]
    print(f"{simbolo} | TP 2R | break-even {CONFIG_LAB['BREAKEVEN_2R']:.1f}% | trades {len(reales)} | control {len(control)}")
    for grupos in ([("TOTAL", reales["zona_id"].notna())],
                   [("X/4 = 1-2", p <= 2), ("X/4 = 3", p == 3), ("X/4 = 4", p == 4)],
                   [("Extremo", reales["tipo"] == "Extremo"), ("Decisional", reales["tipo"] == "Decisional")],
                   [("COMPRA", reales["direccion"] == "COMPRA"), ("VENTA", reales["direccion"] == "VENTA")]):
        print()
        tabla(reales, control, grupos)

if __name__ == "__main__":
    principal()
