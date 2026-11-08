"""Descarga de velas desde la API pública nueva de Deriv (sin token).

El endpoint legacy (ws.derivws.com / ws.binaryws.com, app_id=1089) ya no responde
(HTTP 520). El nuevo devuelve máximo 1000 velas por petición, así que se pagina
hacia atrás con 'end'.
"""
import asyncio
import json
import pandas as pd
import websockets

URL_PUBLICO = "wss://api.derivws.com/trading/v1/options/ws/public"
MAX_POR_PETICION = 1000


async def descargar_velas(simbolo, granularidad, cantidad):
    """Devuelve un DataFrame OHLC indexado por 'datetime' con 'cantidad' velas."""
    velas = {}
    end = "latest"
    async with websockets.connect(URL_PUBLICO, open_timeout=20) as ws:
        req_id = 0
        while len(velas) < cantidad:
            req_id += 1
            pedir = min(MAX_POR_PETICION, cantidad - len(velas) + 1)
            await ws.send(json.dumps({
                "ticks_history": simbolo, "style": "candles",
                "granularity": granularidad, "count": pedir,
                "end": end, "req_id": req_id,
            }))
            resp = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if resp.get("error"):
                raise RuntimeError(f"Deriv API: {resp['error']}")
            lote = resp.get("candles", [])
            nuevas = [c for c in lote if c["epoch"] not in velas]
            if not nuevas:
                break
            for c in nuevas:
                velas[c["epoch"]] = c
            end = min(velas) - 1  # seguir hacia atrás

    df = pd.DataFrame(sorted(velas.values(), key=lambda c: c["epoch"])).tail(cantidad)
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s")
    df = df.set_index("datetime").drop(columns=["epoch"])
    return df
