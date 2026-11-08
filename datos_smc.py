"""Descarga de velas OHLC por temporalidad para la versión SMC "lite".

No modifica ni importa deriv_datos.py: esa versión lanza error ante RateLimit sin
reintentar. Aquí se usa el mismo endpoint público (sin token) pero con reintentos
(espera de 20 s x intento, máximo 8) y 3 s de pausa entre peticiones.

Cada corrida baja una ventana corta y no guarda nada en disco.
"""
import asyncio
import json
import time

import pandas as pd
import websockets

URL_PUBLICO = "wss://api.derivws.com/trading/v1/options/ws/public"
MAX_POR_PETICION = 1000

# Granularidad en segundos y ventana por defecto de cada temporalidad
GRANULARIDAD = {"M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}
VENTANAS = {"M15": 2000, "H1": 1000, "H4": 500, "D1": 120}

MAX_REINTENTOS = 8
ESPERA_BASE_S = 20      # espera = ESPERA_BASE_S * numero_de_intento
PAUSA_ENTRE_PETICIONES_S = 3
PETICION_TIMEOUT_S = 60     # tope duro por petición completa (conectar + enviar + recibir + cerrar)


class ErrorDeriv(RuntimeError):
    """Error de la API que no vale la pena reintentar (p. ej. símbolo inválido)."""


def _es_reintentable(codigo):
    return str(codigo).lower() in ("ratelimit", "rate_limit", "toomanyrequests")


async def _peticion(simbolo, granularidad, cuantas, end):
    """Una sola petición ticks_history (conexión nueva). Lanza ErrorDeriv/RateLimit."""
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


async def _peticion_con_reintentos(simbolo, granularidad, cuantas, end):
    """Reintenta ante RateLimit y fallos de red: espera 20 s x intento (máx. 8)."""
    ultimo = None
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            return await asyncio.wait_for(_peticion(simbolo, granularidad, cuantas, end),
                                          PETICION_TIMEOUT_S)
        except ErrorDeriv:
            raise
        except (ConnectionError, OSError, asyncio.TimeoutError,
                websockets.exceptions.WebSocketException) as e:
            ultimo = e
            espera = ESPERA_BASE_S * intento
            print(f"⚠️ {simbolo} g={granularidad}: {e!r} (intento {intento}/{MAX_REINTENTOS}); "
                  f"espero {espera}s", flush=True)
            if intento < MAX_REINTENTOS:
                await asyncio.sleep(espera)
    raise RuntimeError(f"Sin respuesta tras {MAX_REINTENTOS} intentos: {ultimo!r}")


async def descargar_velas_smc(simbolo, tf, cantidad=None):
    """Devuelve un DataFrame open/high/low/close con índice UTC (tz-aware).

    Solo velas CERRADAS: la vela en formación se descarta, porque las reglas
    (cierre de cuerpo, pivotes) no deben usar una vela incompleta.
    """
    gran = GRANULARIDAD[tf]
    cantidad = cantidad or VENTANAS[tf]
    velas = {}
    end = "latest"
    primera = True
    while len(velas) < cantidad:
        if not primera:
            await asyncio.sleep(PAUSA_ENTRE_PETICIONES_S)
        primera = False
        pedir = min(MAX_POR_PETICION, cantidad - len(velas) + (1 if end == "latest" else 0))
        lote = await _peticion_con_reintentos(simbolo, gran, pedir, end)
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


async def descargar_todo(simbolo, ventanas=None, tfs=("M15", "H1", "H4", "D1")):
    """Descarga secuencial (con pausa entre temporalidades) -> {tf: DataFrame}."""
    ventanas = {**VENTANAS, **(ventanas or {})}
    datos = {}
    for i, tf in enumerate(tfs):
        if i:
            await asyncio.sleep(PAUSA_ENTRE_PETICIONES_S)
        print(f"📡 Descargando {simbolo} {tf} ({ventanas[tf]} velas)...", flush=True)
        datos[tf] = await descargar_velas_smc(simbolo, tf, ventanas[tf])
    return datos
