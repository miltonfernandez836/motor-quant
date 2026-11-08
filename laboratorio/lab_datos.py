"""Laboratorio, paso 1: datos históricos M15 (12 meses) con caché incremental.

Uso:
    python laboratorio/lab_datos.py --simbolo BOOM500

Descarga velas M15 (granularidad 900) de los últimos 12 meses con el mismo endpoint público y la misma
lógica de reintentos de datos_smc.py (sin modificarla): 3 s de pausa entre peticiones y espera creciente
ante RateLimit. La caché vive en laboratorio/datos/{simbolo}_M15.csv (epoch, open, high, low, close) y cada
ejecución solo descarga lo que falta. H1, H4 y D1 no se guardan: se derivan de M15 con derivar().
"""
import argparse
import asyncio
import os
import sys
import time

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from datos_smc import _peticion_con_reintentos  # noqa: E402
from smc_nucleo import CONFIG  # noqa: E402

GRAN_M15 = 900
MESES = 12
PAGINA = 5000                      # lo que se pide; Deriv entrega como máximo 1000 por petición
DIR_DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos")
VELAS_POR_TF = {"H1": 4, "H4": 16, "D1": 96}          # velas M15 de un bucket completo
REGLA_TF = {"H1": "1h", "H4": "4h", "D1": "1D"}


def ruta_cache(simbolo):
    return os.path.join(DIR_DATOS, f"{simbolo}_M15.csv")


def cargar(simbolo):
    """Caché como DataFrame UTC (índice 'datetime'); vacío si aún no existe."""
    try:
        df = pd.read_csv(ruta_cache(simbolo))
    except FileNotFoundError:
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz="UTC", name="datetime"))
    df.index = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df.index.name = "datetime"
    return df[["open", "high", "low", "close"]].astype(float)


def guardar(simbolo, df):
    os.makedirs(DIR_DATOS, exist_ok=True)
    salida = df.copy()
    salida.insert(0, "epoch", ((salida.index - pd.Timestamp(0, unit="s", tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64"))
    salida.to_csv(ruta_cache(simbolo), index=False)


async def bajar(codigo, end, parar_en, red):
    """Velas hacia atrás desde 'end' hasta cubrir 'parar_en' (epoch) o hasta que Deriv no entregue más."""
    velas = {}
    primera = True
    while True:
        if not primera:
            await asyncio.sleep(red["PAUSA_S"])
        primera = False
        lote = await _peticion_con_reintentos(codigo, GRAN_M15, PAGINA, end, red)
        nuevas = [c for c in lote if c["epoch"] not in velas]
        if not nuevas:
            break
        for c in nuevas:
            velas[c["epoch"]] = c
        oldest = min(velas)
        print(f"   {len(velas)} velas descargadas, la más antigua: "
              f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(oldest))}", flush=True)
        if oldest <= parar_en:
            break
        end = oldest - 1
    return velas


async def actualizar(simbolo, red=None):
    """Completa la caché con lo que falta (reciente y antiguo hasta 12 meses) y devuelve el DataFrame."""
    red = red or CONFIG["RED_LARGO"]
    codigo = CONFIG["ACTIVOS"][simbolo]["codigo"]
    ahora = int(time.time())
    inicio = int((pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=MESES)).timestamp())
    cache = cargar(simbolo)
    nuevas = {}
    if len(cache):
        ep_min, ep_max = int(cache.index[0].timestamp()), int(cache.index[-1].timestamp())
        print(f"📂 Caché: {len(cache)} velas ({cache.index[0]} → {cache.index[-1]})")
        if ep_max + GRAN_M15 < ahora:
            print("⬇️ Descargando lo reciente que falta...", flush=True)
            nuevas.update(await bajar(codigo, "latest", ep_max, red))
        if ep_min > inicio + GRAN_M15:
            print("⬇️ Descargando lo antiguo que falta...", flush=True)
            await asyncio.sleep(red["PAUSA_S"])
            nuevas.update(await bajar(codigo, ep_min - 1, inicio, red))
    else:
        print("⬇️ Sin caché: descargando 12 meses...", flush=True)
        nuevas.update(await bajar(codigo, "latest", inicio, red))
    # solo velas cerradas, ordenadas y sin duplicados
    cerradas = [c for c in nuevas.values() if c["epoch"] + GRAN_M15 <= ahora]
    if cerradas:
        df_nuevo = pd.DataFrame(sorted(cerradas, key=lambda c: c["epoch"]))
        df_nuevo.index = pd.to_datetime(df_nuevo["epoch"], unit="s", utc=True)
        df_nuevo.index.name = "datetime"
        cache = pd.concat([cache, df_nuevo[["open", "high", "low", "close"]].astype(float)])
    cache = cache[~cache.index.duplicated(keep="last")].sort_index()
    cache = cache[cache.index >= pd.Timestamp(inicio, unit="s", tz="UTC")]
    guardar(simbolo, cache)
    return cache


def derivar(df_m15, tf):
    """H1, H4 o D1 (UTC; D1 desde las 00:00 UTC) a partir de M15. Solo buckets completos:
    se descartan los que tienen velas M15 faltantes y el último si aún no ha cerrado."""
    r = df_m15.resample(REGLA_TF[tf]).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    n = df_m15["close"].resample(REGLA_TF[tf]).count()
    r = r[n == VELAS_POR_TF[tf]]
    r.index.name = "datetime"
    return r


def principal():
    ap = argparse.ArgumentParser(description="Laboratorio: descarga M15 de 12 meses con caché incremental")
    ap.add_argument("--simbolo", default="BOOM500", choices=list(CONFIG["ACTIVOS"]))
    args = ap.parse_args()
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    df = asyncio.run(actualizar(args.simbolo))
    print(f"✅ {args.simbolo}: {len(df)} velas M15 ({df.index[0]} → {df.index[-1]}) en {ruta_cache(args.simbolo)}")


if __name__ == "__main__":
    principal()
