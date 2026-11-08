# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Quantitative "Smart Money Concepts" (SMC) backtest/report bot for Deriv synthetic indices (V75 = `R_75`, `BOOM500`, `BOOM1000`). Each `reporte_*.py` script downloads 5000 15‑min candles from the Deriv API, detects Order Blocks (OB) after a break of structure, backtests them, renders a Plotly chart, and sends the stats + image to Telegram. Code, comments and output are in Spanish.

## Commands

No build/lint/test setup exists. Python deps (mirrors `.github/workflows/bot_diario.yml`; there is no `requirements.txt`):

```
pip install websockets pandas numpy "plotly<6" requests kaleido==0.2.1
```

Run a report (needs env vars `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`; without them the Telegram `requests.post` fails and the script exits 1):

```
python reporte_v75.py      # also: reporte_boom500.py, reporte_boom1000.py
```

Locally, image export needs `kaleido`; without it the scripts run download + backtest and then fail at `fig.write_image`.

CI: `.github/workflows/bot_diario.yml` is **`workflow_dispatch` only** — an external cron service triggers it, so do not add a `schedule:`. Python 3.10, 10‑minute timeout, runs the three reports (`reporte_v75.py`, `reporte_boom500.py`, `reporte_boom1000.py`) independently and fails the job if any fails. Secrets: `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` (`DER_TOK` is no longer used). Don't use `cache: 'pip'` in `setup-python` (no requirements file → step errors).

## Architecture

- **Data source (`deriv_datos.py`)**: `descargar_velas(simbolo, granularidad, cantidad)` uses Deriv's public, token‑less WebSocket `wss://api.derivws.com/trading/v1/options/ws/public`. The legacy endpoints (`ws.derivws.com` / `ws.binaryws.com`, `app_id=1089`, `python-deriv-api`) return HTTP 520 and hang — don't go back to them. The new endpoint caps at 1000 candles per request, so the function paginates backwards via `end`.
- **The three `reporte_*.py` scripts are near-identical copies** (`reporte_boom500.py` and `reporte_v75.py` differ only in `ACTIVO`, report title and output image name; `reporte_boom1000.py` has its own hard-coded IA thresholds). There is no shared backtest module — a logic fix must be applied to every copy.
- Each script has two parts:
  1. `motor_principal()` — calls `descargar_velas` (`ACTIVO`, `INTERVALO=900`, `CANTIDAD=5000`) and computes sensors: RSI(14), swing `Pico`/`Valle` (rolling window `2n+1`, n=4), `FVG_Alcista`/`FVG_Bajista`. Returns `None` on error.
  2. `main()` — the "Juez": iterates candles; a close beyond the last swing high/low (BOS) plus an FVG within 5 candles of the OB defines a buy/sell OB. Entry = OB edge, SL = opposite edge, TP = 2R. Outcome simulated forward (Win/Loss/Pendiente).
- Error handling: any failure (no data, 120 s `asyncio.wait_for` timeout, Telegram non‑200) ends with `sys.exit(1)` so the Actions job shows red. Don't reintroduce a blanket `sys.exit(0)`.
- Setups are labelled `⭐` (base), `⭐F` (OB in discount/premium half), `⭐R` (RSI divergence), `⭐FR` (both); `👁️` marks the hard-coded "IA" filter. Win/loss counters are kept separately for buys (`_c`) and sells (`_v`) and per label.
- The "IA" filter is a decision tree with thresholds **hard-coded per asset** (e.g. `cuerpo_bos <= 252.63`, `156.10 < tamano_ob <= 173.55` in BOOM1000), derived offline in the notebooks; re-derive per asset, don't copy between scripts.
- Every run overwrites `dataset_sintetico_ml.csv` (features: `Tipo, Velas_Impulso, Tamano_OB, Cuerpo_BOS, Profundidad_Pts, Delta_RSI, Target`) in the cwd — all assets write the same filename, so the last one run wins.
- Chart is written with `fig.write_image` (kaleido) after converting the datetime index to strings, then posted to Telegram via `sendPhoto`.

## Notebooks / data

`motor_v1.ipynb`, `laboratorio_boom500.ipynb`, `laboratorio_boom1000.ipynb` are the research labs where the engine and the ML thresholds are developed before being pasted into the scripts. `datos_v1_limpios.csv` and `auditoria_OB*.csv` are saved datasets/audit outputs from those experiments.
