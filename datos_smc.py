"""Descarga de velas OHLC por temporalidad para la versión SMC "lite".

No modifica ni importa deriv_datos.py: esa versión lanza error ante RateLimit sin
reintentar. Aquí se usa el mismo endpoint público (sin token) pero con reintentos y
tiempos configurables en CONFIG["RED"] (producción) o CONFIG["RED_LARGO"] (--modo-largo).

Cada corrida baja una ventana corta. Solo en --dry-run se guarda una caché local
(carpeta ignorada por git) para no volver a descargar durante el desarrollo.
"""
import asyncio
import json
import os
import time

import pandas as pd
import websockets
from websockets.exceptions import WebSocketException

from smc_nucleo import CONFIG

URL_PUBLICO = "wss://api.derivws.com/trading/v1/options/ws/public"
MAX_POR_PETICION = 1000

# Granularidad en segundos y ventana por defecto de cada temporalidad
GRANULARIDAD = {"M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}
VENTANAS = {"M15": 2000, "H1": 1000, "H4": 500, "D1": 120}
TFS = ("M15", "H1", "H4", "D1")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_smc")


class ErrorDeriv(RuntimeError):
    """Error de la API que no vale la pena reintentar (p. ej. símbolo inválido)."""


class ErrorDescarga(RuntimeError):
    """La descarga de una temporalidad falló (tras agotar los reintentos)."""

    def __init__(self, simbolo, tf, causa):
        self.simbolo, self.tf, self.causa = simbolo, tf, causa
        super().__init__(f"descarga {simbolo} {tf} falló: {causa!r}")


def _es_reintentable(codigo):
    return str(codigo).lower() in ("ratelimit", "rate_limit", "toomanyrequests")


async def _peticion(simbolo, granularidad, cuantas, end):
    """Una sola petición ticks_history (conexión nueva). Lanza ErrorDeriv/ConnectionError."""
    async with websockets.connect(URL_PUBLICO, open_timeout=20) as ws:
        await ws.send(json.dumps({
            "ticks_history": simbolo, "style": "candles",
            "granularity": granularidad, "count": cuantas,
            "end": end, "req_id": 1,
        }))
        resp = json.loads(await asyncio.wait_for(ws.recv(), 30))
    err = resp.get("error")
    if err:
        if _es_reintentable(err.get("code")):
            raise ConnectionError(f"RateLimit: {err}")
        raise ErrorDeriv(f"Deriv API: {err}")
    return resp.get("candles", [])


async def _peticion_con_reintentos(simbolo, granularidad, cuantas, end, red):
    """Reintenta ante RateLimit y fallos de red. Cada intento tiene un tope duro de
    red['TIMEOUT_PETICION_S']; antes del reintento k espera red['ESPERAS_S'][k-1]."""
    esperas = tuple(red["ESPERAS_S"])
    intentos = len(esperas) + 1
    ultimo = None
    for intento in range(1, intentos + 1):
        try:
            return await asyncio.wait_for(_peticion(simbolo, granularidad, cuantas, end),
                                          red["TIMEOUT_PETICION_S"])
        except ErrorDeriv:
            raise
        except (ConnectionError, OSError, asyncio.TimeoutError,
                WebSocketException) as e:
            ultimo = e
            if intento <= len(esperas):
                espera = esperas[intento - 1]
                print(f"⚠️ {simbolo} g={granularidad}: {e!r} (intento {intento}/{intentos}); "
                      f"espero {espera}s", flush=True)
                await asyncio.sleep(espera)
    raise RuntimeError(f"sin respuesta tras {intentos} intentos: {ultimo!r}")


async def descargar_velas_smc(simbolo, tf, cantidad=None, red=None):
    """Devuelve un DataFrame open/high/low/close con índice UTC (tz-aware).

    Solo velas CERRADAS: la vela en formación se descarta, porque las reglas
    (cierre de cuerpo, pivotes) no deben usar una vela incompleta.
    """
    red = red or CONFIG["RED"]
    gran = GRANULARIDAD[tf]
    cantidad = cantidad or VENTANAS[tf]
    velas = {}
    end = "latest"
    primera = True
    while len(velas) < cantidad:
        if not primera:
            await asyncio.sleep(red["PAUSA_S"])
        primera = False
        pedir = min(MAX_POR_PETICION, cantidad - len(velas) + (1 if end == "latest" else 0))
        lote = await _peticion_con_reintentos(simbolo, gran, pedir, end, red)
        nuevas = [c for c in lote if c["epoch"] not in velas]
        if not nuevas:
            break
        for c in nuevas:
            velas[c["epoch"]] = c
        end = min(velas) - 1  # seguir hacia atrás

    ahora = int(time.time())
    cerradas = [c for c in velas.values() if c["epoch"] + gran <= ahora]
    if not cerradas:
        raise RuntimeError(f"{simbolo} {tf}: sin velas cerradas")
    df = pd.DataFrame(sorted(cerradas, key=lambda c: c["epoch"])).tail(cantidad)
    df.index = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df.index.name = "datetime"
    return df[["open", "high", "low", "close"]].astype(float)


async def descargar_todo(simbolo, ventanas=None, tfs=TFS, red=None, progreso=None):
    """Descarga secuencial (con pausa entre temporalidades) -> {tf: DataFrame}.

    Si una temporalidad falla lanza ErrorDescarga(tf=...). 'progreso' (dict opcional) se
    actualiza con {'tf_actual': ...} para saber en qué TF estaba si algo externo corta el proceso."""
    red = red or CONFIG["RED"]
    ventanas = {**VENTANAS, **(ventanas or {})}
    progreso = progreso if progreso is not None else {}
    datos = {}
    for i, tf in enumerate(tfs):
        if i:
            await asyncio.sleep(red["PAUSA_S"])
        progreso["tf_actual"] = tf
        print(f"📡 Descargando {simbolo} {tf} ({ventanas[tf]} velas)...", flush=True)
        try:
            datos[tf] = await descargar_velas_smc(simbolo, tf, ventanas[tf], red)
        except Exception as e:
            raise ErrorDescarga(simbolo, tf, e) from e
    progreso["tf_actual"] = None
    return datos


# =====================================================================
# Caché local (SOLO para --dry-run; carpeta ignorada por git)
# =====================================================================
def guardar_cache(simbolo, datos, carpeta=None, ahora=None):
    """Guarda las velas de la última descarga exitosa (un CSV por TF + meta con la hora)."""
    carpeta = carpeta or CACHE_DIR
    os.makedirs(carpeta, exist_ok=True)
    for tf, df in datos.items():
        df.to_csv(os.path.join(carpeta, f"{simbolo}_{tf}.csv"))
    meta = {"guardado_en": ahora if ahora is not None else time.time(), "tfs": list(datos)}
    with open(os.path.join(carpeta, f"{simbolo}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)   # la meta va al final: sin ella, la caché se considera inexistente


def cargar_cache(simbolo, tfs=TFS, max_edad_s=None, carpeta=None, ahora=None):
    """(datos, edad_en_segundos) si la caché existe, está completa y es más joven que max_edad_s;
    en cualquier otro caso None."""
    carpeta = carpeta or CACHE_DIR
    max_edad_s = CONFIG["CACHE_MAX_EDAD_S"] if max_edad_s is None else max_edad_s
    ahora = ahora if ahora is not None else time.time()
    try:
        with open(os.path.join(carpeta, f"{simbolo}_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        edad = ahora - float(meta["guardado_en"])
        if edad < 0 or edad >= max_edad_s or not set(tfs) <= set(meta["tfs"]):
            return None
        datos = {}
        for tf in tfs:
            df = pd.read_csv(os.path.join(carpeta, f"{simbolo}_{tf}.csv"), index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "datetime"
            datos[tf] = df[["open", "high", "low", "close"]].astype(float)
        return datos, edad
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
