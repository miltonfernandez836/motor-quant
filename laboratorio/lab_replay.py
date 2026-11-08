"""Laboratorio, paso 2: replay del núcleo SMC sobre el historial M15 de BOOM500.

Uso:
    python laboratorio/lab_replay.py --simbolo BOOM500              # replay completo (reanudable)
    python laboratorio/lab_replay.py --simbolo BOOM500 --verificar  # prueba anti look-ahead

Cada cierre de hora (cada 4 velas M15) ejecuta el núcleo con las mismas ventanas y parámetros que el
reporte lite (M15=2000, H1=1000, H4=500, D1=120 velas) usando SOLO velas cerradas hasta ese instante.
Registra las zonas M15 en laboratorio/datos/zonas_{simbolo}.csv (una fila por zona):
  - las zonas que llegan a un estado final (mitigada/invalidada/vencida) se agregan al CSV al cerrarse;
  - las que siguen vivas se agregan al terminar el replay con estado_final = viva.
El avance (último instante procesado y zonas en seguimiento) se guarda en
laboratorio/datos/replay_{simbolo}_estado.json: si la corrida se interrumpe, al relanzarla continúa de ahí.

Los instantes de velas (t_fresca, t_primer_toque) son la hora de APERTURA de la vela M15, en UTC.
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import smc_nucleo as smc  # noqa: E402
from lab_datos import DIR_DATOS, cargar, derivar  # noqa: E402

VENTANAS = {"M15": 2000, "H1": 1000, "H4": 500, "D1": 120}      # las mismas del reporte lite (datos_smc.VENTANAS)
DURACION = {"H1": pd.Timedelta(hours=1), "H4": pd.Timedelta(hours=4), "D1": pd.Timedelta(days=1)}
VELA = pd.Timedelta(minutes=15)
HORA = pd.Timedelta(hours=1)
CADA_CHECKPOINT = 100                                           # pasos entre guardados del avance
CAMPOS_FOTO = ["r1", "r2", "r3", "r4", "puntaje", "sesgo_h1", "sesgo_h4", "pos_tr", "tamano_atr", "atr", "hora"]
COLUMNAS = (["zona_id", "tipo", "direccion", "bajo", "alto", "t_fresca", "t_primer_toque", "estado_final"]
            + [f"ap_{c}" for c in CAMPOS_FOTO] + [f"pt_{c}" for c in CAMPOS_FOTO])
FIN = {"mitigada", "invalidada", "vencida"}


def ruta_csv(simbolo):
    return os.path.join(DIR_DATOS, f"zonas_{simbolo}.csv")


def ruta_estado(simbolo):
    return os.path.join(DIR_DATOS, f"replay_{simbolo}_estado.json")


def preparar(simbolo):
    """M15 de la caché + H1/H4/D1 derivadas (en memoria) y los fines de vela de cada TF."""
    m15 = cargar(simbolo)
    if len(m15) == 0:
        raise SystemExit(f"No hay caché M15 de {simbolo}: ejecuta antes laboratorio/lab_datos.py")
    marcos = {tf: derivar(m15, tf) for tf in ("H1", "H4", "D1")}
    fines = {tf: (marcos[tf].index + DURACION[tf]).tz_localize(None).to_numpy() for tf in marcos}
    return m15, marcos, fines


def datos_en(m15, marcos, fines, k):
    """Datos conocidos al cierre de la vela M15 k (posición): solo velas cerradas hasta ese instante."""
    t_fin = (m15.index[k] + VELA).tz_localize(None).to_datetime64()
    dfs = {"M15": m15.iloc[max(0, k + 1 - VENTANAS["M15"]):k + 1]}
    for tf, df in marcos.items():
        n = int(np.searchsorted(fines[tf], t_fin, side="right"))      # TF cuyo cierre es <= t_fin
        dfs[tf] = df.iloc[max(0, n - VENTANAS[tf]):n]
    return dfs


def paso(m15, marcos, fines, k):
    """Ejecuta el núcleo en el instante de la vela M15 k. Devuelve (resultado de analizar, instante t_fin)."""
    return smc.analizar(datos_en(m15, marcos, fines, k)), m15.index[k] + VELA


def pasos_del_replay(m15, marcos, fines):
    """Posiciones k de los cierres de hora (vela de :45) desde que H4 llena su ventana (warmup mínimo)."""
    cierres_h4 = fines["H4"]
    k_hora = np.nonzero(np.asarray(m15.index.minute == 45))[0]
    t_fin = (m15.index[k_hora] + VELA).tz_localize(None).to_numpy()
    n_h4 = np.searchsorted(cierres_h4, t_fin, side="right")
    n_h1 = np.searchsorted(fines["H1"], t_fin, side="right")
    ok = (n_h4 >= VENTANAS["H4"]) & (n_h1 >= VENTANAS["H1"]) & (k_hora + 1 >= VENTANAS["M15"])
    return k_hora[ok]


def zona_id(simbolo, z, indice):
    return f"{simbolo}-M15-{z['tipo']}-{indice[z['idx_ob']]:%Y%m%dT%H%M}"


def foto(z, res, t_fin):
    """Características de la zona en este paso."""
    atr = float(res["M15"]["atr"][-1])
    pos = z["pos_tr"]
    return {"r1": int(z["r1"]), "r2": int(z["r2"]), "r3": int(z["r3"]), "r4": int(z["estado"] != "mitigada"),
            "puntaje": int(z["puntaje"]), "sesgo_h1": z["sesgo_h1"], "sesgo_h4": z["sesgo_h4"],
            "pos_tr": "sin_tr" if pos is None else ("fuera" if pos < 0 or pos > 100 else round(float(pos), 1)),
            "tamano_atr": round((z["alto"] - z["bajo"]) / atr, 3), "atr": round(atr, 3), "hora": int(t_fin.hour)}


def estado_simple(estado):
    return estado if estado in FIN else "viva"


def registrar(simbolo, seguimiento, cerradas, descartadas, res, t_fin):
    """Actualiza el seguimiento con las zonas M15 de este paso. Devuelve las zonas que acaban de cerrarse.
    Solo se registran las zonas que se vuelven frescas en la última hora antes del paso; las que ya existían
    (warmup) o que aparecen tarde porque al correr la ventana cambió la estructura van a 'descartadas'."""
    indice = res["M15"]["index"]
    nuevas_cerradas = []
    for z in res["M15"]["zonas"]:
        i_fresca = next((i for i, e in z["historial"] if e == "fresca"), None)
        if i_fresca is None:
            continue
        zid = zona_id(simbolo, z, indice)
        if zid in cerradas:
            continue
        r = seguimiento.get(zid)
        if r is None:
            if zid in descartadas:
                continue
            if indice[i_fresca] + VELA <= t_fin - HORA:
                descartadas.add(zid)
                continue
            r = {"zona_id": zid, "tipo": z["tipo"], "direccion": z["dir"], "bajo": z["bajo"], "alto": z["alto"],
                 "t_fresca": f"{indice[i_fresca]:%Y-%m-%d %H:%M}", "t_primer_toque": "",
                 "estado_final": "viva", "ap": foto(z, res, t_fin), "pt": None}
            seguimiento[zid] = r
        if z["primer_toque"] is None:
            r["pt"] = foto(z, res, t_fin)            # última pasada antes del primer toque
        elif not r["t_primer_toque"]:
            r["t_primer_toque"] = f"{indice[z['primer_toque']]:%Y-%m-%d %H:%M}"
        r["estado_final"] = estado_simple(z["estado"])
        if z["estado"] in FIN:
            nuevas_cerradas.append(seguimiento.pop(zid))
            cerradas.add(zid)
    return nuevas_cerradas


def fila(r):
    f = {c: r[c] for c in COLUMNAS[:8]}
    for pref, snap in (("ap", r["ap"]), ("pt", r["pt"])):
        for c in CAMPOS_FOTO:
            f[f"{pref}_{c}"] = "" if snap is None else snap[c]
    return f


def escribir_filas(simbolo, registros):
    """Agrega filas al CSV (crea la cabecera si hace falta) sin repetir zonas ya escritas."""
    ruta = ruta_csv(simbolo)
    existentes = set()
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8", newline="") as f:
            existentes = {r["zona_id"] for r in csv.DictReader(f)}
    nuevas = [r for r in registros if r["zona_id"] not in existentes]
    nuevo_archivo = not os.path.exists(ruta)
    with open(ruta, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo_archivo:
            w.writeheader()
        w.writerows(fila(r) for r in nuevas)


def guardar_estado(simbolo, k, seguimiento, cerradas, descartadas):
    tmp = ruta_estado(simbolo) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"k": int(k), "seguimiento": seguimiento, "cerradas": sorted(cerradas),
                   "descartadas": sorted(descartadas)}, f)
    os.replace(tmp, ruta_estado(simbolo))


def replay(simbolo):
    os.makedirs(DIR_DATOS, exist_ok=True)
    m15, marcos, fines = preparar(simbolo)
    ks = pasos_del_replay(m15, marcos, fines)
    seguimiento, cerradas, descartadas = {}, set(), set()
    if os.path.exists(ruta_estado(simbolo)) and os.path.exists(ruta_csv(simbolo)):
        with open(ruta_estado(simbolo), encoding="utf-8") as f:
            est = json.load(f)
        seguimiento, cerradas, descartadas = est["seguimiento"], set(est["cerradas"]), set(est["descartadas"])
        ks = ks[ks > est["k"]]
        print(f"↻ Reanudando tras la vela {m15.index[est['k']]} ({len(cerradas)} zonas cerradas, "
              f"{len(seguimiento)} en seguimiento); faltan {len(ks)} pasos")
    else:
        for ruta in (ruta_csv(simbolo), ruta_estado(simbolo)):
            if os.path.exists(ruta):
                os.remove(ruta)
        print(f"▶ Replay nuevo: {len(ks)} pasos, del {m15.index[ks[0]] + VELA} al {m15.index[ks[-1]] + VELA}")
    pendientes = []
    t0 = time.time()
    for n, k in enumerate(ks, 1):
        res, t_fin = paso(m15, marcos, fines, k)
        pendientes += registrar(simbolo, seguimiento, cerradas, descartadas, res, t_fin)
        if n % CADA_CHECKPOINT == 0 or n == len(ks):
            escribir_filas(simbolo, pendientes)
            pendientes = []
            guardar_estado(simbolo, k, seguimiento, cerradas, descartadas)
            transcurrido = time.time() - t0
            print(f"  paso {n}/{len(ks)} ({t_fin:%Y-%m-%d %H:%M}) | {transcurrido / 60:.1f} min | "
                  f"ETA {transcurrido / n * (len(ks) - n) / 60:.1f} min | cerradas {len(cerradas)} | "
                  f"en seguimiento {len(seguimiento)} | descartadas {len(descartadas)}", flush=True)
    escribir_filas(simbolo, list(seguimiento.values()))      # las que siguen vivas al final del historial
    print(f"✅ Replay terminado: {len(cerradas) + len(seguimiento)} zonas en {ruta_csv(simbolo)}")


def firma_zonas(res):
    """Zonas M15 en forma comparable (con marcas de tiempo, no con posiciones)."""
    m = res["M15"]
    ix = m["index"]
    return sorted((f"{z['tipo']}-{ix[z['idx_ob']]}", z["dir"], z["bajo"], z["alto"], z["estado"], z["r1"], z["r2"],
                   z["r3"], z["puntaje"], z["sesgo_h1"], z["sesgo_h4"], z["pos_tr"],
                   None if z["primer_toque"] is None else str(ix[z["primer_toque"]]),
                   tuple((str(ix[i]), e) for i, e in z["historial"])) for z in m["zonas"])


def verificar(simbolo, cuantos=20):
    """Repite 'cuantos' pasos al azar con los datos CORTADOS en ese instante (H1/H4/D1 derivadas solo de
    las M15 hasta ahí) y compara las zonas con las del replay completo. Deben ser idénticas."""
    m15, marcos, fines = preparar(simbolo)
    ks = pasos_del_replay(m15, marcos, fines)
    semilla = int(time.time())
    random.seed(semilla)
    elegidos = sorted(random.sample(list(ks), cuantos))
    print(f"Verificando {cuantos} pasos al azar (semilla {semilla}) de {len(ks)}")
    distintos = 0
    for k in elegidos:
        completo, t_fin = paso(m15, marcos, fines, k)
        corte = m15.iloc[:k + 1]
        marcos_c = {tf: derivar(corte, tf) for tf in ("H1", "H4", "D1")}
        fines_c = {tf: (marcos_c[tf].index + DURACION[tf]).tz_localize(None).to_numpy() for tf in marcos_c}
        cortado, _ = paso(corte, marcos_c, fines_c, k)
        igual = firma_zonas(completo) == firma_zonas(cortado)
        distintos += not igual
        print(f"  {t_fin:%Y-%m-%d %H:%M}: {len(completo['M15']['zonas'])} zonas | {'IDÉNTICAS' if igual else 'DIFERENTES'}")
    print("✅ Sin look-ahead: las 20 coinciden." if not distintos else f"❌ {distintos} pasos difieren: hay look-ahead.")
    return distintos


def principal():
    ap = argparse.ArgumentParser(description="Laboratorio: replay del núcleo SMC sobre el historial M15")
    ap.add_argument("--simbolo", default="BOOM500", choices=list(smc.CONFIG["ACTIVOS"]))
    ap.add_argument("--verificar", action="store_true", help="repite 20 pasos al azar con datos cortados y compara")
    args = ap.parse_args()
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if args.verificar:
        sys.exit(1 if verificar(args.simbolo) else 0)
    replay(args.simbolo)


if __name__ == "__main__":
    principal()
