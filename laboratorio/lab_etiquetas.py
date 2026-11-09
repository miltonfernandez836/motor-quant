"""Laboratorio, paso 6A: ¿lo que "operaría" a ojo rinde distinto de lo que "no operaría"?

Uso:
    python laboratorio/lab_etiquetas.py [--simbolo BOOM500]

Lee laboratorio/datos/etiquetas.csv (zona_id, decision si/no, fecha; el que baja la galería) y lo cruza con
resultados_{simbolo}.csv y resultados_control_{simbolo}.csv. Funciona con etiquetas parciales. Con TP a 2R, para
"operaría" y "no operaría": n, % de aciertos, R medio y diferencia contra su control emparejado (las 20 entradas al
azar de cada trade, misma dirección y riesgo), con intervalo 95% por bootstrap sobre trades. Añade "operaría menos
no operaría" y la mitad del ancho de cada intervalo: es el tamaño de diferencia (en R) que este n podría detectar.
Si el intervalo de la diferencia incluye el cero, el resultado es "no concluyente" con ese n.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import smc_nucleo as smc  # noqa: E402
from lab_datos import DIR_DATOS  # noqa: E402
from lab_resultados import CONFIG_LAB  # noqa: E402

ALFA = 0.05


def _remuestras(v, rng):
    v = np.asarray(v, float)
    return v[rng.integers(0, len(v), size=(CONFIG_LAB["BOOTSTRAP"], len(v)))].mean(axis=1)


def _ic(medias):
    lo, hi = np.percentile(medias, [100 * ALFA / 2, 100 * (1 - ALFA / 2)])
    return float(lo), float(hi)


def _fmt(media, lo, hi):
    return f"{media:>+7.3f} [{lo:>+7.3f}, {hi:>+7.3f}] ±{(hi - lo) / 2:.3f}"


def preparar(etiquetas, reales, control):
    """Trades con su etiqueta y su control emparejado. Devuelve (trades, n_leidas, n_sin_trade)."""
    et = etiquetas.copy()
    if "fecha" in et:
        et = et.sort_values("fecha")
    et = et.drop_duplicates("zona_id", keep="last")                  # si una zona se etiquetó dos veces, vale la última
    et = et[et["decision"].isin(["si", "no"])]
    t = reales.merge(et[["zona_id", "decision"]], on="zona_id")
    t["ctrl"] = t["zona_id"].map(control.groupby("zona_id")["r_tp2"].mean())
    t["dif"] = t["r_tp2"] - t["ctrl"]
    t["acierto"] = (t["estado_tp2"] == "tp").astype(float)
    return t, len(et), len(et) - len(t)


def analizar(etiquetas, reales, control):
    """Imprime la comparación 'operaría' vs 'no operaría' con TP a 2R."""
    t, leidas, sin_trade = preparar(etiquetas, reales, control)
    rng = np.random.default_rng(CONFIG_LAB["SEMILLA"])
    grupos = {"operaría": t[t["decision"] == "si"], "no operaría": t[t["decision"] == "no"]}
    print(f"Etiquetas válidas: {leidas} | cruzadas con un trade: {len(t)} | sin trade en resultados: {sin_trade} | "
          f"operaría {len(grupos['operaría'])}, no operaría {len(grupos['no operaría'])} | TP 2R "
          f"(break-even {CONFIG_LAB['BREAKEVEN_2R']:.1f}%) | bootstrap sobre trades")
    print(f"\n{'grupo':<14}{'n':>5} {'acierto':>8}  {'R medio [IC95%] ±mitad':<38}{'dif R vs control [IC95%] ±mitad':<38}")
    remuestras = {}
    for nombre, g in grupos.items():
        if len(g) < 2:
            print(f"{nombre:<14}{len(g):>5}  (n insuficiente para calcular un intervalo)")
            continue
        rm, rd = _remuestras(g["r_tp2"], rng), _remuestras(g["dif"], rng)
        remuestras[nombre] = (rm, rd)
        print(f"{nombre:<14}{len(g):>5} {g['acierto'].mean() * 100:>7.1f}%  "
              f"{_fmt(g['r_tp2'].mean(), *_ic(rm)):<38}{_fmt(g['dif'].mean(), *_ic(rd)):<38}")
    if len(remuestras) < 2:
        print("\nNo se puede comparar: falta alguno de los dos grupos (hay que etiquetar más zonas).")
        return
    a, b = grupos["operaría"], grupos["no operaría"]
    dm = remuestras["operaría"][0] - remuestras["no operaría"][0]
    dd = remuestras["operaría"][1] - remuestras["no operaría"][1]
    print(f"{'operaría - no':<14}{'':>5} {(a['acierto'].mean() - b['acierto'].mean()) * 100:>+6.1f}pp  "
          f"{_fmt(a['r_tp2'].mean() - b['r_tp2'].mean(), *_ic(dm)):<38}"
          f"{_fmt(a['dif'].mean() - b['dif'].mean(), *_ic(dd)):<38}")
    lo, hi = _ic(dd)
    medio = (hi - lo) / 2
    if lo <= 0 <= hi:
        print(f"\nNO CONCLUYENTE por falta de datos: el intervalo de la diferencia incluye el cero. Con {len(a)} + {len(b)} "
              f"zonas etiquetadas solo se detectarían diferencias de más de ±{medio:.3f} R (mitad del ancho del intervalo).")
    else:
        print(f"El intervalo de la diferencia excluye el cero ({'a favor de operar' if lo > 0 else 'en contra de operar'}); "
              f"con este n el tamaño detectable es ±{medio:.3f} R. Es una sola comparación: no es evidencia fuerte por sí sola.")


def principal():
    ap = argparse.ArgumentParser(description="Laboratorio: operaría / no operaría vs resultados y control")
    ap.add_argument("--simbolo", default="BOOM500", choices=list(smc.CONFIG["ACTIVOS"]))
    args = ap.parse_args()
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ruta = os.path.join(DIR_DATOS, "etiquetas.csv")
    if not os.path.exists(ruta):
        raise SystemExit(f"No existe {ruta}: descarga etiquetas.csv desde la galería y cópialo ahí.")
    analizar(pd.read_csv(ruta),
             pd.read_csv(os.path.join(DIR_DATOS, f"resultados_{args.simbolo}.csv")),
             pd.read_csv(os.path.join(DIR_DATOS, f"resultados_control_{args.simbolo}.csv")))


if __name__ == "__main__":
    principal()
