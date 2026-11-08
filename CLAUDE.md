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

## Versión SMC "lite" (archivos nuevos, independientes de los `reporte_*.py`)

Reporte de zonas vivas (OB Extremo/Decisional) con fichas de verificación para BOOM500. **No toca** `reporte_*.py`, `deriv_datos.py` ni `bot_diario.yml`. Cada corrida baja una ventana corta y recalcula todo desde cero; no guarda datos en el repo.

- `datos_smc.py`: descarga M15=2000, H1=1000, H4=500, D1=120 con el mismo endpoint público pero **con reintentos** ante RateLimit y fallos de red, y 3 s de pausa entre peticiones. Producción (`CONFIG["RED"]`): tope de 30 s por petición y 3 reintentos con esperas de 20/40/60 s; `--modo-largo` (`CONFIG["RED_LARGO"]`): 8 reintentos (20 s × intento) y tope de 60 s. Descarta la vela en formación. Devuelve OHLC con índice UTC. Si una temporalidad falla lanza `ErrorDescarga` con el TF.
- `smc_nucleo.py`: toda la detección, sin red ni gráfico (reutilizable en un "laboratorio"). Todos los parámetros están en el bloque `CONFIG` (incluye el dict `CAPAS` de capas opcionales, todas en `False`). Un solo recorrido vela a vela: **nada en t usa velas > t** y un pivote de ventana n se confirma en `idx + n`. ATR = mediana móvil del rango verdadero (los umbrales de tamaño van en ATR, no en puntos).
  - Flujo: pivotes → HH/HL/LH/LL → LG (mecha fuera, cierra dentro) vs BOS (cierre de cuerpo) → CDC (LG + ruptura del strong point opuesto) → TR → OB → R1–R4 → estados → liquidez (TP1 = IRL a favor, TP final = ERL) → sesgo.
  - **R1** (barrió liquidez) exige un LG del lado correcto (de un pivote confirmado n=4 o de EQH/EQL) dentro de las `CONFIG["R1_VENTANA"]` = **10 velas previas al OB**, contando la vela del OB; se mide contra la vela de cada OB (Extremo y Decisional por separado).
  - Estados de zona: `en_formacion → fresca → tocada → mitigada | invalidada | vencida`. Solo fresca/tocada son candidatas. Un *toque* = entrada tras ≥ 1 vela completa fuera; 2 toques o llegar al 50% = mitigada.
  - Edad máxima por temporalidad en `CONFIG["EDAD_MAX"]` (M15=1000, H1=500, H4=300).
- `reporte_smc.py`: orquesta datos → núcleo → fichas → imagen → Telegram. Args: `--dry-run` (guarda `reporte_smc.png` y `reporte_smc.html`, no envía), `--capas a,b`, `--velas-grafico 400`, `--modo-largo`, `--sin-cache`. Cada capa del gráfico es un `legendgroup` de Plotly (rectángulos como `Scatter` con `fill="toself"`, para que la leyenda los apague). Todo llega en **un único mensaje**: la foto con el texto como caption (máx. 1024 caracteres; si se pasa, se recorta primero la línea de zonas fuera del gráfico y nunca las fichas).
  - **Tiempos de producción** (`CONFIG["RED"]`): timeout global 180 s (un hilo watchdog mata el proceso; el timeout "amable" salta 15 s antes para poder avisar), 30 s por petición, exportar el PNG con tope de `CONFIG["TIMEOUT_PNG_S"]` = 40 s (si se agota falla con "kaleido no respondió"). `bot_smc.yml` usa `timeout-minutes: 5`. `--modo-largo` usa `CONFIG["RED_LARGO"]` (540 s, 8 reintentos).
  - **Fallos**: cualquier fallo (descarga con el TF, timeout, PNG, Telegram rechaza, error inesperado) envía un mensaje corto a Telegram con el motivo (sin el token) y termina con código 1. En `--dry-run` no se avisa a Telegram.
  - **Caché del dry-run**: solo con `--dry-run` se guardan las velas de la última descarga exitosa en `.cache_smc/` (ignorada por git) y se reutilizan si tienen menos de 1 hora (`CONFIG["CACHE_MAX_EDAD_S"]`); `--sin-cache` fuerza una descarga nueva. Producción nunca lee ni escribe la caché.
- **Activos** (`--simbolo`, tabla `CONFIG["ACTIVOS"]`): `BOOM500` (por defecto), `BOOM1000` y `V75` (código Deriv `R_75`). Misma configuración para los tres; cada uno tiene su nombre de título/mensaje, sus archivos de salida (`reporte_smc.png/html` para BOOM500, `reporte_smc_{simbolo}.png/html` para los otros) y su caché separada en `.cache_smc/`.
- `.github/workflows/bot_smc.yml` (BOOM500), `bot_smc_b1000.yml` (`--simbolo BOOM1000`) y `bot_smc_v75.yml` (`--simbolo V75`): `workflow_dispatch` only, mismos secrets, ejecutan solo `reporte_smc.py`.
- Tests: `python -m pytest tests` (velas armadas a mano + test anti-futuro que compara datos hasta t contra datos completos en muchos cortes). Requiere `pytest`.
  - **Mensaje** (`armar_mensaje`): línea 1 `📊 BOOM 500 M15 · {día mes hh:mm UTC de la última vela} · Precio {entero}`; línea 2 `Sesgo: H4 ↑/↓/↔ · H1 ↑/↓/↔`; bloques `🟢 COMPRA` y `🔴 VENTA` (se omite el vacío) con `(a favor del sesgo)` o `(⚠ contra sesgo)`; una línea por zona `N · bajo–alto · Extremo/Decisional · X/4 · falla R# (solo las que fallan) · TP {TP1}` + `premium`/`descuento`/`fuera TR`; leyenda R1–R4; una línea con las 2 zonas H1/H4 fuera del gráfico más cercanas (precio del borde más cercano y distancia %) y `+N más`; última línea `No es señal de entrada.` La numeración N (COMPRA primero, VENTA después, cada una por cercanía al precio) es la misma de la etiqueta del gráfico (`1 · EXT 3/4`). El dry-run imprime el mismo texto.
  - **Fase 1.5**: las zonas llevan sesgo H1/H4, marca de contra sesgo y posición en el TR. Premium/descuento y el 50% se calculan sobre el **TR de trabajo** (último TR establecido); el TR en formación se dibuja aparte, con línea fina y etiqueta «en formación».
  - **Zonas H1/H4 fuera del eje**: el eje Y no se estira; el mensaje añade una línea por zona viva fuera del rango (tipo, rango, distancia % al precio) y el gráfico una flecha en el borde. Zonas M15 nacidas antes de la ventana visible se dibujan desde el borde izquierdo con su fecha de origen en la etiqueta.
- Fases: la 1 y la 1.5 están hechas; capas opcionales de la Fase 2/3 (FVG+IFC, EQH/EQL y IRL/ERL como líneas, albercas D1, inducement, SMT, trendline, wick POI, FVG breaker, modelos trampa) solo tienen la estructura en `CONFIG["CAPAS"]`.
- Local en Windows: el script fuerza UTF-8 en la salida (la consola cp1252 rompe con emojis). El `.venv` local es entorno de desarrollo y usa `plotly>=6` + `kaleido>=1` (kaleido 1.x necesita Chrome); **no alinearlo con el CI** (`plotly<6` + `kaleido==0.2.1`) porque en Windows `kaleido==0.2.1` se cuelga al exportar cualquier figura (visto con Python 3.10 y 3.14). Instalar con `uv pip install "plotly>=6,<7" "kaleido>=1,<2"` (cerrar antes procesos kaleido colgados, que bloquean archivos). El CI en Linux no se ha probado con esta rama.
