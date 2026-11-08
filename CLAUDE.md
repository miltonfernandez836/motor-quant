# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Quantitative "Smart Money Concepts" (SMC) backtest/report bot for Deriv synthetic indices (V75 = `R_75`, `BOOM500`, `BOOM1000`). Each `reporte_*.py` script downloads 5000 15‑min candles from the Deriv API, detects Order Blocks (OB) after a break of structure, backtests them, renders a Plotly chart, and sends the stats + image to Telegram. Code, comments and output are in Spanish.

## Commands

No build/lint/test setup exists. Python deps (see `.github/workflows/bot_diario.yml`):

```
pip install python-deriv-api pandas numpy plotly requests kaleido==0.1.0
```

Run a report (needs env vars `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`; without them the Telegram `requests.post` fails):

```
python reporte_v75.py      # also: reporte_boom500.py, reporte_boom1000.py
```

CI: `.github/workflows/bot_diario.yml` is manual-only (`workflow_dispatch`), Python 3.10, 5‑minute timeout, runs `reporte_v75.py` and `reporte_boom500.py` (**not** `reporte_boom1000.py`). Secrets: `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `DER_TOK`. The local `.venv` is Python 3.14 (Windows).

## Architecture

- **The three `reporte_*.py` scripts are near-identical copies** (`reporte_boom500.py` and `reporte_v75.py` differ only in `ACTIVO`, the report title and the output image name). There is no shared module — a logic fix must be applied to every copy. `reporte_boom1000.py` is an older variant (V1.3, ~470 lines vs 485) that may lag behind the other two.
- Each script has two parts:
  1. `motor_principal()` — downloads candles (`ACTIVO`, `INTERVALO=900`, `CANTIDAD=5000`) and computes sensors: RSI(14), swing `Pico`/`Valle` (rolling window of `2n+1`, n=4), `FVG_Alcista`/`FVG_Bajista`.
  2. `main()` — the "Juez": iterates candles; a close beyond the last swing high/low (BOS) plus an FVG within 5 candles of the OB defines a buy/sell OB. Entry = OB edge, SL = opposite edge, TP = 2R. Outcome is simulated forward (Win/Loss/Pendiente).
- Setups are classified by labels `⭐` (base), `⭐F` (OB in discount/premium half), `⭐R` (RSI divergence), `⭐FR` (both); `👁️` marks the hard-coded "IA" filter. Win/loss counters are kept separately for buys (`_c`) and sells (`_v`) and per label.
- The "IA" filter is a decision tree whose thresholds are **hard-coded per asset** (e.g. `cuerpo_bos <= 252.63`, `156.10 < tamano_ob <= 173.55` in BOOM1000). These come from offline analysis in the notebooks and must be re-derived per asset, not copied between scripts.
- Every run overwrites `dataset_sintetico_ml.csv` (features: `Tipo, Velas_Impulso, Tamano_OB, Cuerpo_BOS, Profundidad_Pts, Delta_RSI, Target`) in the cwd — all assets write the same filename, so the last one run wins.
- Chart is written with `fig.write_image` (kaleido) after converting the datetime index to strings ("parche nuclear") to avoid Plotly/kaleido date issues, then posted to Telegram via `sendPhoto`-style `requests.post`.

## Notebooks / data

`motor_v1.ipynb`, `laboratorio_boom500.ipynb`, `laboratorio_boom1000.ipynb` are the research labs where the engine and the ML thresholds are developed before being pasted into the scripts. `datos_v1_limpios.csv` and `auditoria_OB*.csv` are saved datasets/audit outputs from those experiments.
