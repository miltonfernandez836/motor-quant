"""Reporte SMC "lite" (BOOM500, BOOM1000, V75 con --simbolo): zonas vivas como fichas + gráfico M15 + Telegram.

Cada corrida baja una ventana corta (datos_smc), recalcula todo desde cero (smc_nucleo)
y no guarda datos. Variables de entorno: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, DERIV_TOKEN.

Uso:
    python reporte_smc.py --dry-run                      # no envía; guarda PNG y HTML
    python reporte_smc.py --dry-run --capas fvg_ifc,smt  # capas opcionales (Fase 2/3)
    python reporte_smc.py --velas-grafico 400
"""
import argparse
import asyncio
import copy
import os
import sys
import threading

import pandas as pd
import plotly.graph_objects as go
import requests

import smc_nucleo as smc
from datos_smc import ErrorDescarga, cargar_cache, descargar_todo, guardar_cache

SIMBOLO = "BOOM500"
NOMBRE_ACTIVO = smc.CONFIG["ACTIVOS"][SIMBOLO]["nombre"]   # nombre por defecto de las funciones de presentación
LIMITE_CAPTION = 1024           # máximo de texto de una foto en Telegram
FIN_MENSAJE = "No es señal de entrada."
VELA_S = 900

# Capas opcionales cuya lógica ya está implementada (Fase 1: ninguna)
CAPAS_IMPLEMENTADAS = set()

COLOR = {  # (relleno, borde)
    "COMPRA": ("rgba(38,166,154,0.22)", "rgba(38,166,154,0.95)"),
    "VENTA": ("rgba(239,83,80,0.22)", "rgba(239,83,80,0.95)"),
}


# =====================================================================
# Mensaje (imagen + texto como caption, máx. 1024 caracteres)
# =====================================================================
MESES = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")
FLECHA = {"alcista": "↑", "bajista": "↓"}          # 'rango' (u otro) -> ↔
LEYENDA = "R1 liquidez barrida · R2 rompió estructura · R3 punto interno · R4 sin mitigar"


def encabezado_sesgo(res):
    return f"H4: {res['H4']['sesgo']} | H1: {res['H1']['sesgo']}"


def etiqueta_corta(n, z):
    return f"{n} · {'EXT' if z['tipo'] == 'Extremo' else 'DEC'} {z['puntaje']}/4"


def linea_zona(n, z):
    """N · bajo–alto · tipo · X/4 · falla R# (solo las que fallan) · TP {TP1} [· premium/descuento/fuera TR]."""
    fallan = [r for r, ok in (("R1", z["r1"]), ("R2", z["r2"]), ("R3", z["r3"]),
                              ("R4", z["estado"] != "mitigada")) if not ok]
    tp1 = z.get("tp1")
    partes = [str(n), f"{z['bajo']:.0f}–{z['alto']:.0f}", z["tipo"], f"{z['puntaje']}/4"]
    if fallan:
        partes.append("falla " + " ".join(fallan))
    partes.append(f"TP {tp1:.0f}" if tp1 is not None else "TP s/d")
    pos = z.get("pos_tr")
    if pos is not None:
        partes.append("fuera TR" if pos < 0 or pos > 100 else ("descuento" if pos < 50 else "premium"))
    return " · ".join(partes)


def linea_fuera(fuera):
    """Las 2 zonas H1/H4 fuera del gráfico más cercanas al precio y '+N más'."""
    if not fuera:
        return ""
    partes = [f"{f['tf']} {f['dir'].lower()} {(f['alto'] if not f['arriba'] else f['bajo']):.0f} "
              f"({f['dist_pct']:+.1f}%)" for f in fuera[:2]]
    if len(fuera) > 2:
        partes.append(f"+{len(fuera) - 2} más")
    return "Fuera del gráfico: " + " · ".join(partes)


def armar_mensaje(res, cands, precio, hora, fuera=(), nombre=NOMBRE_ACTIVO):
    """Texto del caption. 'cands' ya viene en el orden de numeración (el mismo del gráfico):
    COMPRA primero y luego VENTA, cada una por cercanía. Si pasa de 1024 caracteres se recorta
    primero la línea de zonas fuera del gráfico; las fichas nunca se recortan."""
    sesgo = lambda tf: FLECHA.get(res[tf]["sesgo"], "↔")
    cab = [f"📊 {nombre} M15 · {hora.day} {MESES[hora.month - 1]} {hora:%H:%M} · Precio {precio:.0f}",
           f"Sesgo: H4 {sesgo('H4')} · H1 {sesgo('H1')}"]
    bloques = []
    for dir_, icono in (("COMPRA", "🟢"), ("VENTA", "🔴")):
        zs = [(n, z) for n, z in enumerate(cands, 1) if z["dir"] == dir_]
        if zs:
            contra = any(z["contra"] for _, z in zs)
            bloques.append([f"{icono} {dir_} " + ("(⚠ contra sesgo)" if contra else "(a favor del sesgo)")]
                           + [linea_zona(n, z) for n, z in zs])
    if not bloques:
        bloques.append(["Sin zonas vivas (frescas o tocadas) por ahora."])

    def armar(con_fuera):
        secciones = ["\n".join(cab)] + ["\n".join(b) for b in bloques] + [LEYENDA]
        if con_fuera and fuera:
            secciones.append(linea_fuera(fuera))
        secciones.append(FIN_MENSAJE)
        return "\n\n".join(secciones)

    texto = armar(True)
    return texto if len(texto) <= LIMITE_CAPTION else armar(False)


# =====================================================================
# Gráfico
# =====================================================================
def _x(index, i):
    return index[i].strftime("%Y-%m-%d %H:%M:%S")


def _rect(x0, x1, y0, y1):
    return [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0]


class Capas:
    """Cada capa = un grupo de leyenda de Plotly (se enciende/apaga con un clic)."""

    def __init__(self, fig):
        self.fig = fig
        self.grupos = {}

    def registrar(self, nombre, color, visible=True):
        """Entrada de leyenda del grupo (siempre existe, aunque no tenga dibujos)."""
        if nombre in self.grupos:
            return
        self.grupos[nombre] = color
        self.fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name=nombre, legendgroup=nombre,
            showlegend=True, line=dict(color=color, width=3),
            visible=True if visible else "legendonly", hoverinfo="skip"))

    def trazo(self, nombre, **kw):
        kw.setdefault("hoverinfo", "skip")
        self.fig.add_trace(go.Scatter(legendgroup=nombre, showlegend=False, **kw))


def rango_eje_y(df15, n_velas, cands=()):
    """Rango del eje Y = velas visibles + todas las zonas M15 dibujadas, con un margen del 1%.
    Las zonas H1/H4 lejanas NO lo estiran (van como flecha en el borde)."""
    vis = df15.iloc[-n_velas:]
    ymin = min([float(vis["low"].min())] + [z["bajo"] for z in cands])
    ymax = max([float(vis["high"].max())] + [z["alto"] for z in cands])
    pad = (ymax - ymin) * 0.01
    return [ymin - pad, ymax + pad]


def construir_figura(df15, res, cands, n_velas, activas, fuera=(), nombre=NOMBRE_ACTIVO):
    m15 = res["M15"]
    index = m15["index"]
    n_tot = len(index)
    i0 = max(0, n_tot - n_velas)
    vis = df15.iloc[i0:]
    xs = [_x(index, i) for i in range(i0, n_tot)]
    x_izq = xs[0]
    x_der = (index[-1] + pd.Timedelta(seconds=VELA_S * 8)).strftime("%Y-%m-%d %H:%M:%S")
    rango_y = rango_eje_y(df15, n_velas, cands)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=xs, open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"],
        name=nombre, showlegend=False,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    capas = Capas(fig)

    # Orden de la leyenda: capas del núcleo siempre activas
    capas.registrar("OB Extremo", "#26a69a")
    capas.registrar("OB Decisional", "#80cbc4")
    capas.registrar("TR de trabajo + 50%", "#ffd54f")
    capas.registrar("TR en formación", "#bcaaa4")
    capas.registrar("Strong/Weak points", "#ba68c8")
    capas.registrar("CDC", "#ff9800")
    capas.registrar("Franjas H1", "#90a4ae")
    capas.registrar("Franjas H4", "#78909c")

    # --- Franjas de fondo H1 / H4 (zonas vivas que tocan el rango visible del eje Y)
    precio = float(df15["close"].iloc[-1])
    rotulos = []   # (y, tf, texto) de las franjas: se separan verticalmente al final
    for tf in ("H4", "H1"):
        r = res[tf]
        grupo = f"Franjas {tf}"
        for z in smc.candidatas(r, precio, 10**6):
            if z["alto"] < rango_y[0] or z["bajo"] > rango_y[1]:
                continue
            xr, yr = _rect(x_izq, x_der, z["bajo"], z["alto"])
            capas.trazo(grupo, x=xr, y=yr, mode="lines", fill="toself",
                        fillcolor="rgba(144,164,174,0.10)" if tf == "H1" else "rgba(120,144,156,0.14)",
                        line=dict(color="rgba(144,164,174,0.35)", width=1, dash="dot"))
            rotulos.append((z["alto"], tf, f"{tf} {'EXT' if z['tipo'] == 'Extremo' else 'DEC'} {z['dir'][0]}"))


    # --- Flechas en el borde del gráfico: zonas H1/H4 vivas fuera del eje (el eje no se estira)
    span = rango_y[1] - rango_y[0]
    ys_flechas = []   # posiciones fijas de las etiquetas de flecha (también cuentan para la separación)
    x_flecha = (index[-1] + pd.Timedelta(seconds=VELA_S * 5)).strftime("%Y-%m-%d %H:%M:%S")
    for arriba in (True, False):
        for k, f in enumerate([f for f in fuera if f["arriba"] == arriba]):
            y = rango_y[1] - span * (0.03 + 0.035 * k) if arriba else rango_y[0] + span * (0.03 + 0.035 * k)
            ys_flechas.append(y)
            capas.trazo(f"Franjas {f['tf']}", x=[x_flecha], y=[y], mode="markers+text",
                        marker=dict(symbol="triangle-up" if arriba else "triangle-down", size=11,
                                    color="#cfd8dc"),
                        text=[f"{f['tf']} {f['bajo']:.0f}-{f['alto']:.0f} ({f['dist_pct']:+.1f}%)"],
                        textposition="middle left", textfont=dict(size=10, color="#cfd8dc"))

    # Etiquetas H1/H4 del borde derecho: separación vertical mínima entre ellas y respecto a las flechas
    sep = span * 0.03
    puestas = list(ys_flechas)
    for y, tf, texto in sorted(rotulos):
        paso = sep if y < (rango_y[0] + rango_y[1]) / 2 else -sep   # se aleja hacia el centro del gráfico
        while any(abs(y - q) < sep for q in puestas):
            y += paso
        puestas.append(y)
        capas.trazo(f"Franjas {tf}", x=[x_der], y=[y], mode="text", textposition="top left",
                    text=[texto], textfont=dict(size=10, color="#b0bec5"), cliponaxis=False)

    # --- TR de trabajo (último ESTABLECIDO): su 50% separa premium de descuento
    tr = m15["tr_trabajo"]
    if tr is not None and tr["alto"] > tr["bajo"]:
        xa = _x(index, max(tr["idx_ini"], i0))
        medio = (tr["alto"] + tr["bajo"]) / 2
        xr, yr = _rect(xa, x_der, tr["bajo"], tr["alto"])
        capas.trazo("TR de trabajo + 50%", x=xr, y=yr, mode="lines",
                    line=dict(color="rgba(255,213,79,0.55)", width=1))
        capas.trazo("TR de trabajo + 50%", x=[xa, x_der], y=[medio, medio], mode="lines",
                    line=dict(color="rgba(255,213,79,0.9)", width=1, dash="dash"))
        capas.trazo("TR de trabajo + 50%", x=[x_der], y=[medio], mode="text", textposition="top left",
                    text=[f"TR de trabajo {'alcista' if tr['dir'] == 1 else 'bajista'} · 50%"],
                    textfont=dict(size=10, color="#ffd54f"))

    # --- TR en formación: aparte, línea fina y etiqueta propia
    trf = m15["tr_formacion"]
    if trf is not None and trf["alto"] > trf["bajo"]:
        xa = _x(index, max(trf["idx_ini"], i0))
        xr, yr = _rect(xa, x_der, trf["bajo"], trf["alto"])
        capas.trazo("TR en formación", x=xr, y=yr, mode="lines",
                    line=dict(color="rgba(188,170,164,0.6)", width=0.6, dash="dot"))
        capas.trazo("TR en formación", x=[x_der], y=[trf["bajo"]], mode="text", textposition="bottom left",
                    text=[f"TR {'alcista' if trf['dir'] == 1 else 'bajista'} en formación"],
                    textfont=dict(size=9, color="#bcaaa4"))

    # --- Strong / weak points (los más recientes visibles)
    visibles = [p for p in m15["puntos"] if p["idx"] >= i0][-6:]
    nombres = {"strong_low": "Strong L", "strong_high": "Strong H",
               "weak_high": "Weak H", "weak_low": "Weak L"}
    if visibles:
        capas.trazo("Strong/Weak points",
                    x=[_x(index, p["idx"]) for p in visibles], y=[p["precio"] for p in visibles],
                    mode="markers+text", text=[nombres[p["tipo"]] for p in visibles],
                    textposition=["bottom center" if p["tipo"].endswith("low") else "top center" for p in visibles],
                    marker=dict(symbol=["diamond" if p["tipo"].startswith("strong") else "diamond-open"
                                        for p in visibles], size=9, color="#ba68c8"),
                    textfont=dict(size=10, color="#ce93d8"))

    # --- CDC
    for e in [e for e in m15["eventos"] if e["tipo"] == "CDC" and e["idx"] >= i0][-4:]:
        capas.trazo("CDC", x=[_x(index, max(e["idx_nivel"], i0)), _x(index, e["idx"])],
                    y=[e["nivel"], e["nivel"]], mode="lines+text", text=["", "CDC"],
                    textposition="top left", line=dict(color="#ff9800", width=1.5, dash="dashdot"),
                    textfont=dict(size=11, color="#ffb74d"))

    # --- Zonas vivas M15 (rectángulos desde la vela del OB hasta el borde derecho)
    for n, z in enumerate(cands, 1):
        grupo = f"OB {z['tipo']}"
        relleno, borde = COLOR[z["dir"]]
        nacio_antes = z["idx_ob"] < i0
        xa = _x(index, max(z["idx_ob"], i0))
        xr, yr = _rect(xa, x_der, z["bajo"], z["alto"])
        capas.trazo(grupo, x=xr, y=yr, mode="lines", fill="toself", fillcolor=relleno,
                    line=dict(color=borde, width=1.5, dash="solid" if z["tipo"] == "Extremo" else "dash"))
        capas.trazo(grupo, x=[xa, x_der], y=[z["medio"], z["medio"]], mode="lines",
                    line=dict(color=borde, width=0.8, dash="dot"))
        texto = etiqueta_corta(n, z)
        if nacio_antes:   # nació antes de la ventana visible: se indica su fecha de origen
            texto += f" · desde {index[z['idx_ob']].strftime('%d/%m %H:%M')}"
        # Si la zona empieza cerca del borde derecho, la etiqueta va hacia la izquierda (no se corta)
        cerca_borde = (z["idx_ob"] - i0) / max(1, n_tot - i0 + 8) > 0.8
        capas.trazo(grupo, x=[xa], y=[z["alto"]], mode="text", text=[texto], cliponaxis=False,
                    textposition="top left" if cerca_borde else "top right",
                    textfont=dict(size=12, color="white"))

    # --- Capas opcionales (Fase 2/3): se agregan aquí cuando tengan lógica
    for nombre in sorted(activas & CAPAS_IMPLEMENTADAS):
        capas.registrar(nombre, "#4fc3f7")

    fig.update_layout(
        title=f"{nombre} M15 | {encabezado_sesgo(res)}",
        template="plotly_dark", width=1400, height=800, xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=30, t=90, b=40),
        legend=dict(orientation="h", y=1.07, x=0, groupclick="togglegroup"),
        xaxis=dict(range=[x_izq, x_der]), yaxis=dict(range=rango_y))
    return fig


# =====================================================================
# Telegram
# =====================================================================
def enviar_telegram(token, chat_id, ruta_png, mensaje):
    """Un único mensaje: la foto con el texto como pie (máx. 1024 caracteres)."""
    with open(ruta_png, "rb") as foto:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", timeout=60,
                          files={"photo": foto}, data={"chat_id": chat_id, "caption": mensaje})
    if r.status_code != 200:
        print(f"❌ Error al enviar la foto: {r.text}")
        return False
    return True


# =====================================================================
# Orquestación
# =====================================================================
class ErrorReporte(RuntimeError):
    """Fallo del reporte con motivo legible: se avisa a Telegram y el proceso termina con código 1."""


_AVISADO = threading.Event()   # evita avisar dos veces (camino amable + watchdog)


def _sin_secretos(texto, *secretos):
    for s in secretos:
        if s:
            texto = texto.replace(s, "***")
    return texto


def notificar_error(motivo, token, chat_id, dry_run=False, timeout_http=10, nombre=NOMBRE_ACTIVO):
    """Mensaje corto a Telegram con el motivo del fallo (mejor esfuerzo). True si se envió."""
    _AVISADO.set()
    texto = f"⚠️ Reporte SMC {nombre} falló: {_sin_secretos(str(motivo), token, chat_id)}"[:400]
    print(texto, flush=True)
    if dry_run:
        print("(--dry-run: no se avisa a Telegram)", flush=True)
        return False
    if not (token and chat_id):
        print("(sin TELEGRAM_TOKEN / TELEGRAM_CHAT_ID: no se puede avisar)", flush=True)
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=timeout_http,
                          data={"chat_id": chat_id, "text": texto})
        return r.status_code == 200
    except Exception as e:
        print(f"(no se pudo avisar a Telegram: {_sin_secretos(repr(e), token, chat_id)})", flush=True)
        return False


def leer_argumentos(argv=None):
    ap = argparse.ArgumentParser(description="Reporte SMC lite")
    ap.add_argument("--dry-run", action="store_true",
                    help="no envía a Telegram; guarda PNG y HTML en local (y usa/crea la caché de velas)")
    ap.add_argument("--capas", default="", help="capas opcionales separadas por coma")
    ap.add_argument("--velas-grafico", type=int, default=400)
    ap.add_argument("--simbolo", default=SIMBOLO, choices=list(smc.CONFIG["ACTIVOS"]),
                    help="activo de CONFIG[\"ACTIVOS\"] (por defecto BOOM500)")
    ap.add_argument("--modo-largo", action="store_true",
                    help="8 reintentos (20 s x intento) y timeout global de 540 s, en vez del perfil de producción")
    ap.add_argument("--sin-cache", action="store_true",
                    help="con --dry-run: ignora la caché y vuelve a descargar")
    return ap.parse_args(argv)


def configurar_capas(texto):
    cfg = copy.deepcopy(smc.CONFIG)
    pedidas = {c.strip() for c in texto.split(",") if c.strip()}
    desconocidas = pedidas - set(cfg["CAPAS"])
    if desconocidas:
        print(f"❌ Capas desconocidas: {sorted(desconocidas)}. Disponibles: {sorted(cfg['CAPAS'])}")
        sys.exit(2)
    for c in pedidas:
        cfg["CAPAS"][c] = True
        if c not in CAPAS_IMPLEMENTADAS:
            print(f"ℹ️ La capa '{c}' existe en CONFIG pero su lógica llega en una fase posterior; se ignora.")
    return cfg, pedidas


def exportar_png(fig, ruta, timeout_s):
    """Exporta el PNG en un hilo daemon con tope: kaleido puede colgarse sin lanzar ningún error."""
    resultado = {}

    def _trabajo():
        try:
            fig.write_image(ruta)
        except BaseException as e:
            resultado["error"] = e

    hilo = threading.Thread(target=_trabajo, daemon=True)
    hilo.start()
    hilo.join(timeout_s)
    if hilo.is_alive():
        raise ErrorReporte(f"kaleido no respondió (más de {timeout_s:.0f} s exportando el PNG)")
    if "error" in resultado:
        raise ErrorReporte(f"no se pudo generar el PNG (¿kaleido?): {resultado['error']!r}") from resultado["error"]


def perfil_de_red(cfg, args):
    return cfg["RED_LARGO"] if args.modo_largo else cfg["RED"]


async def obtener_datos(args, cfg, red, progreso):
    """Velas de las 4 temporalidades. Con --dry-run reutiliza la caché (< 1 h) y guarda la descarga."""
    if args.dry_run and not args.sin_cache:
        c = cargar_cache(cfg["ACTIVOS"][args.simbolo]["cache"], max_edad_s=cfg["CACHE_MAX_EDAD_S"])
        if c:
            print(f"♻️ --dry-run: uso las velas en caché ({c[1] / 60:.0f} min de antigüedad); "
                  f"--sin-cache fuerza una descarga nueva.")
            return c[0]
    activo = cfg["ACTIVOS"][args.simbolo]
    datos = await descargar_todo(activo["codigo"], red=red, progreso=progreso)
    if args.dry_run:
        try:
            guardar_cache(activo["cache"], datos)
        except OSError as e:
            print(f"⚠️ No se pudo guardar la caché: {e!r}")
    return datos


async def principal(args, cfg, activas, red, progreso, token, chat_id):
    if os.getenv("DERIV_TOKEN"):
        print("🕵️ DERIV_TOKEN detectado (el endpoint público no lo necesita).")
    if not args.dry_run and not (token and chat_id):
        raise ErrorReporte("faltan TELEGRAM_TOKEN / TELEGRAM_CHAT_ID (usa --dry-run para probar sin enviar)")

    activo = cfg["ACTIVOS"][args.simbolo]
    activo_nombre = activo["nombre"]
    datos = await obtener_datos(args, cfg, red, progreso)

    print("🧠 Calculando estructura, zonas y liquidez...")
    res = smc.analizar(datos, cfg)
    df15 = datos["M15"]
    precio = float(df15["close"].iloc[-1])
    # Numeración N (mensaje y gráfico): COMPRA primero y VENTA después, cada una por cercanía al precio
    cands = sorted(smc.candidatas(res["M15"], precio, sin_solape=True),
                   key=lambda z: (z["dir"] != "COMPRA", smc._distancia(z, precio)))
    y_min, y_max = rango_eje_y(df15, args.velas_grafico, cands)
    fuera = smc.zonas_fuera_de_rango(res, precio, y_min, y_max, k=cfg["MAX_FUERA_RANGO"])
    mensaje = armar_mensaje(res, cands, precio, df15.index[-1], fuera, activo_nombre)
    print("\n" + mensaje + "\n")
    print(f"(precio actual {precio:.2f} | zonas M15 totales {len(res['M15']['zonas'])} | "
          f"candidatas {len(cands)} | H1/H4 fuera del eje {len(fuera)})")

    fig = construir_figura(df15, res, cands, args.velas_grafico, activas, fuera, activo_nombre)
    ruta_png = activo["png"]
    if args.dry_run:
        ruta_html = activo["html"]
        fig.write_html(ruta_html, include_plotlyjs="cdn")
        print(f"💾 HTML guardado: {os.path.abspath(ruta_html)}")
    exportar_png(fig, ruta_png, cfg["TIMEOUT_PNG_S"])
    print(f"✅ PNG guardado: {os.path.abspath(ruta_png)}")

    if args.dry_run:
        print("🧪 --dry-run: no se envía nada a Telegram.")
        return
    print("📲 Enviando a Telegram...")
    if not enviar_telegram(token, chat_id, ruta_png, mensaje):
        raise ErrorReporte("Telegram rechazó el envío del reporte (revisa token y chat)")
    print("✅ Enviado.")


def correr(args, timeout=None):
    """Ejecuta el reporte. Devuelve el código de salida: 0 = ok; 1 = falló (y se avisó a Telegram)."""
    cfg, activas = configurar_capas(args.capas)
    red = perfil_de_red(cfg, args)
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    progreso = {}
    limite = timeout if timeout is not None else red["TIMEOUT_GLOBAL_S"] - red["MARGEN_TIMEOUT_S"]
    try:
        asyncio.run(asyncio.wait_for(
            principal(args, cfg, activas, red, progreso, token, chat_id), timeout=limite))
        return 0
    except ErrorReporte as e:
        motivo = str(e)
    except ErrorDescarga as e:
        motivo = f"no se pudo descargar {e.tf} de Deriv: {e.causa!r}"
    except asyncio.TimeoutError:
        tf = progreso.get("tf_actual")
        motivo = f"timeout de {limite:.0f}s" + (f" mientras descargaba {tf}" if tf else "")
    except Exception as e:
        motivo = f"error inesperado ({type(e).__name__}): {e}"
    notificar_error(motivo, token, chat_id, args.dry_run, nombre=cfg["ACTIVOS"][args.simbolo]["nombre"])
    return 1


if __name__ == "__main__":
    for flujo in (sys.stdout, sys.stderr):   # consola de Windows (cp1252) no soporta emojis
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    argumentos = leer_argumentos()
    _red = perfil_de_red(smc.CONFIG, argumentos)

    def _matar_por_cuelgue():
        # Último recurso: si el bucle de eventos queda bloqueado (p. ej. un cierre de websocket que no
        # responde a la cancelación), ni wait_for ni asyncio.run terminan. Esto garantiza el fin del proceso.
        if not _AVISADO.is_set():
            notificar_error(f"el proceso no terminó en {_red['TIMEOUT_GLOBAL_S']}s (watchdog)",
                            os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), argumentos.dry_run,
                            nombre=smc.CONFIG["ACTIVOS"][argumentos.simbolo]["nombre"])
        os._exit(1)

    vigilante = threading.Timer(_red["TIMEOUT_GLOBAL_S"], _matar_por_cuelgue)
    vigilante.daemon = True
    vigilante.start()
    sys.exit(correr(argumentos))
