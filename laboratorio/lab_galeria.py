"""Laboratorio, paso 5: galería de zonas para etiquetar a ojo (sin ver el resultado).

Uso:
    python laboratorio/lab_galeria.py --simbolo BOOM500

1) Elige N zonas al azar (semilla fija) de resultados_{simbolo}.csv y guarda la lista en
   laboratorio/datos/galeria_{simbolo}.csv.
2) Para cada zona dibuja un PNG de velas M15 que termina EXACTAMENTE en la última pasada del replay anterior al
   primer toque (cada cierre de hora: la última vela cerrada es la que cierra en ese instante; no hay ninguna vela
   posterior). Muestra la zona como rectángulo y solo el texto COMPRA/VENTA, tipo y sesgo H1/H4: nada de X/4,
   R1-R4 ni resultados. Exporta con el mismo método que el lite (reporte_smc.exportar_png).
3) Genera laboratorio/galeria_{simbolo}.html: una zona a la vez, botones Operaría / No operaría, contador, orden
   aleatorio, avance guardado en el navegador (localStorage) y descarga de etiquetas.csv en cualquier momento.
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import smc_nucleo as smc  # noqa: E402
from lab_datos import DIR_DATOS, cargar  # noqa: E402
from reporte_smc import exportar_png  # noqa: E402

CONFIG_GALERIA = {"N_ZONAS": 150, "SEMILLA": 20260102, "N_VELAS": 150, "VELAS_MARGEN_DERECHO": 8,
                  "ANCHO": 1100, "ALTO": 600}
LAB = os.path.dirname(os.path.abspath(__file__))
DIR_GALERIA = os.path.join(LAB, "galeria")
VELA = pd.Timedelta(minutes=15)
FMT = "%Y-%m-%d %H:%M:%S"


def figura(m15, z, instante):
    """Velas M15 hasta la que cierra en 'instante' (la última pasada antes del primer toque) + la zona."""
    j = m15.index.get_loc(instante - VELA)
    vis = m15.iloc[max(0, j + 1 - CONFIG_GALERIA["N_VELAS"]):j + 1]
    ob = pd.Timestamp(z.zona_id.rsplit("-", 1)[1]).tz_localize("UTC")
    x = [t.strftime(FMT) for t in vis.index]
    x_der = (vis.index[-1] + VELA * CONFIG_GALERIA["VELAS_MARGEN_DERECHO"]).strftime(FMT)
    x_zona = max(ob, vis.index[0]).strftime(FMT)
    y_min, y_max = min(vis["low"].min(), z.bajo), max(vis["high"].max(), z.alto)
    pad = (y_max - y_min) * 0.04
    compra = z.direccion == "COMPRA"
    relleno, borde = ("rgba(38,166,154,0.22)", "rgba(38,166,154,0.95)") if compra else ("rgba(239,83,80,0.22)", "rgba(239,83,80,0.95)")
    fig = go.Figure(go.Candlestick(x=x, open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"],
                                   increasing_line_color="#26a69a", decreasing_line_color="#ef5350", showlegend=False))
    fig.add_trace(go.Scatter(x=[x_zona, x_der, x_der, x_zona, x_zona], y=[z.bajo, z.bajo, z.alto, z.alto, z.bajo],
                             mode="lines", fill="toself", fillcolor=relleno, line=dict(color=borde, width=1.5),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        title=f"{z.direccion} · {z.tipo} · sesgo H1 {z.pt_sesgo_h1} · H4 {z.pt_sesgo_h4}",
        template="plotly_dark", width=CONFIG_GALERIA["ANCHO"], height=CONFIG_GALERIA["ALTO"],
        xaxis_rangeslider_visible=False, margin=dict(l=50, r=30, t=60, b=40),
        xaxis=dict(range=[x[0], x_der]), yaxis=dict(range=[y_min - pad, y_max + pad]))
    return fig, x[-1]


HTML = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Galería __SIMBOLO__</title>
<style>
 body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:16px;text-align:center}
 img{max-width:100%;height:auto;border:1px solid #333}
 button{font-size:1.1rem;padding:.7rem 1.4rem;margin:.4rem;border:0;border-radius:6px;cursor:pointer}
 #si{background:#26a69a;color:#fff} #no{background:#ef5350;color:#fff} #bajar{background:#455a64;color:#fff}
 #cont{margin:.6rem 0;font-size:1.05rem} #aviso{color:#ffb74d;font-size:.9rem}
</style></head><body>
<div id="cont"></div>
<div id="vista"></div>
<div><button id="si">Operaría</button><button id="no">No operaría</button></div>
<div><button id="bajar">Descargar etiquetas.csv</button></div>
<div id="aviso"></div>
<script>
const ZONAS = __ZONAS__;
const CLAVE = "galeria___SIMBOLO___v1";
let sinGuardar = false;
function leer(){ try { return JSON.parse(localStorage.getItem(CLAVE)); } catch (e) { sinGuardar = true; return null; } }
function guardar(){ try { localStorage.setItem(CLAVE, JSON.stringify(estado)); } catch (e) { sinGuardar = true; } }
function barajar(a){ for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
let estado = leer();
if (!estado || !estado.orden || estado.orden.length !== ZONAS.length) {
  estado = {orden: barajar(ZONAS.map((_, i) => i)), etiquetas: {}};
  guardar();
}
function siguiente(){ return estado.orden.find(i => !(ZONAS[i].id in estado.etiquetas)); }
function pintar(){
  const hechas = Object.keys(estado.etiquetas).length, i = siguiente();
  document.getElementById("cont").textContent = i === undefined
    ? "Terminado: " + hechas + " de " + ZONAS.length + " etiquetadas"
    : "Zona " + (hechas + 1) + " de " + ZONAS.length + " · etiquetadas: " + hechas;
  document.getElementById("vista").innerHTML = i === undefined ? "" : '<img src="galeria/' + ZONAS[i].img + '" alt="zona">';
  document.getElementById("si").disabled = document.getElementById("no").disabled = (i === undefined);
  document.getElementById("aviso").textContent = sinGuardar ? "Aviso: este navegador no permite guardar el avance; descarga las etiquetas antes de cerrar." : "";
}
function etiquetar(d){
  const i = siguiente(); if (i === undefined) return;
  estado.etiquetas[ZONAS[i].id] = {decision: d, fecha: new Date().toISOString()};
  guardar(); pintar();
}
document.getElementById("si").onclick = () => etiquetar("si");
document.getElementById("no").onclick = () => etiquetar("no");
document.getElementById("bajar").onclick = () => {
  const filas = ["zona_id,decision,fecha"].concat(Object.entries(estado.etiquetas)
    .sort((a, b) => a[1].fecha.localeCompare(b[1].fecha)).map(([id, e]) => id + "," + e.decision + "," + e.fecha));
  const url = URL.createObjectURL(new Blob([filas.join("\\n") + "\\n"], {type: "text/csv;charset=utf-8"}));
  const a = document.createElement("a"); a.href = url; a.download = "etiquetas.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
pintar();
</script></body></html>
"""


def principal():
    ap = argparse.ArgumentParser(description="Laboratorio: galería de zonas para etiquetar")
    ap.add_argument("--simbolo", default="BOOM500", choices=list(smc.CONFIG["ACTIVOS"]))
    args = ap.parse_args()
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    s = args.simbolo
    t0 = time.time()
    m15 = cargar(s)
    res = pd.read_csv(os.path.join(DIR_DATOS, f"resultados_{s}.csv"))
    sel = res.sample(n=CONFIG_GALERIA["N_ZONAS"], random_state=CONFIG_GALERIA["SEMILLA"]).reset_index(drop=True)
    os.makedirs(DIR_GALERIA, exist_ok=True)
    filas = []
    for n, z in enumerate(sel.itertuples(index=False), 1):
        toque = pd.Timestamp(z.t_primer_toque, tz="UTC")
        instante = toque.floor("h")                       # última pasada horaria anterior (o igual) al primer toque
        fig, x_ultima = figura(m15, z, instante)
        imagen = f"zona_{n:03d}.png"
        for intento in (1, 2):
            try:
                exportar_png(fig, os.path.join(DIR_GALERIA, imagen), smc.CONFIG["TIMEOUT_PNG_S"])
                break
            except Exception as e:
                if intento == 2:
                    raise
                print(f"⚠️ {imagen}: {e!r}; reintento")
        filas.append({"n": n, "zona_id": z.zona_id, "instante_pretoque": instante.strftime(FMT),
                      "x_ultima_dibujada": x_ultima, "imagen": imagen})
        if n % 10 == 0:
            print(f"  {n}/{len(sel)} PNG ({time.time() - t0:.0f} s)", flush=True)
    lista = pd.DataFrame(filas)
    lista.to_csv(os.path.join(DIR_DATOS, f"galeria_{s}.csv"), index=False)
    zonas_js = json.dumps([{"id": f["zona_id"], "img": f["imagen"]} for f in filas])
    ruta_html = os.path.join(LAB, f"galeria_{s}.html")
    with open(ruta_html, "w", encoding="utf-8") as f:
        f.write(HTML.replace("__ZONAS__", zonas_js).replace("__SIMBOLO__", s))

    # verificación: ninguna imagen termina después del instante de la última pasada previa al primer toque
    ultima = pd.to_datetime(lista["x_ultima_dibujada"], utc=True)
    limite = pd.to_datetime(lista["instante_pretoque"], utc=True)
    toque = pd.to_datetime(sel["t_primer_toque"], utc=True)
    posteriores = int(((ultima + VELA) > limite).sum())
    exactas = int(((ultima + VELA) == limite).sum())
    antes_del_toque = int((ultima < toque).all() and (ultima + VELA <= toque).all())
    faltan = [i for i in lista["imagen"] if not os.path.exists(os.path.join(DIR_GALERIA, i))]
    print(f"✅ {len(lista)} zonas, {len(lista) - len(faltan)} PNG en {DIR_GALERIA} | HTML: {ruta_html}")
    print(f"Verificación: velas posteriores al instante pre-toque: {posteriores} | terminan exactamente en él: {exactas}/{len(lista)} "
          f"| todas cierran antes del primer toque: {'sí' if antes_del_toque else 'NO'} | PNG que faltan: {len(faltan)}")
    print(f"Tiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    principal()
