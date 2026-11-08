"""Tests de presentación: fichas, mensaje de Telegram y gráfico (sin red)."""
import os
import sys

import plotly.graph_objects as go
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reporte_smc as rep  # noqa: E402
import smc_nucleo as smc  # noqa: E402
from test_smc_nucleo import serie_aleatoria  # noqa: E402

AG = {"open": "first", "high": "max", "low": "min", "close": "last"}


@pytest.fixture(scope="module")
def mundo():
    df = serie_aleatoria(2000, seed=3)
    dfs = {"M15": df, "H1": df.resample("1h").agg(AG).dropna(), "H4": df.resample("4h").agg(AG).dropna()}
    res = smc.analizar(dfs)
    precio = float(df["close"].iloc[-1])
    cands = smc.ordenar_fichas(smc.candidatas(res["M15"], precio), precio)
    return df, res, precio, cands


def _ordenadas(cands, precio):
    return sorted(cands, key=lambda z: (z["dir"] != "COMPRA", smc._distancia(z, precio)))


def test_linea_de_zona_muestra_solo_lo_que_falla_tp_y_posicion_en_tr(mundo):
    _, _, _, cands = mundo
    z = dict(cands[0])
    z.update(tipo="Extremo", puntaje=2, r1=False, r2=True, r3=False, estado="fresca",
             tp1=5955.8, bajo=5915.5, alto=5921.4, pos_tr=30.0)
    assert rep.linea_zona(3, z) == "3 · 5916–5921 · Extremo · 2/4 · falla R1 R3 · TP 5956 · descuento"
    z.update(r1=True, r3=True, puntaje=4, pos_tr=60.0, tipo="Decisional")
    assert rep.linea_zona(1, z) == "1 · 5916–5921 · Decisional · 4/4 · TP 5956 · premium"
    z.update(pos_tr=120.0)
    assert rep.linea_zona(1, z).endswith("· fuera TR")
    z.update(pos_tr=None)
    assert rep.linea_zona(1, z).endswith("TP 5956")


def test_mensaje_formato_bloques_y_ultima_linea(mundo):
    import pandas as pd
    df, res, precio, cands = mundo
    cands = _ordenadas(cands, precio)
    hora = pd.Timestamp("2026-11-08 16:45", tz="UTC")
    msg = rep.armar_mensaje(res, cands, precio, hora)
    lineas = msg.split("\n")
    assert lineas[0] == f"📊 BOOM 500 M15 · 8 nov 16:45 · Precio {precio:.0f}"
    assert lineas[1].startswith("Sesgo: H4 ") and " · H1 " in lineas[1]
    assert rep.LEYENDA in msg and lineas[-1] == "No es señal de entrada."
    for dir_, icono in (("COMPRA", "🟢"), ("VENTA", "🔴")):
        hay = any(z["dir"] == dir_ for z in cands)
        assert (f"{icono} {dir_}" in msg) == hay                      # bloque vacío se omite
    n_zonas = [l for l in lineas if l[:1].isdigit() and " · " in l[:6]]
    assert [l.split(" · ")[0] for l in n_zonas] == [str(i) for i in range(1, len(cands) + 1)]
    z = dict(cands[0], dir="COMPRA", contra=[])                   # encabezado según sesgo
    assert "🟢 COMPRA (a favor del sesgo)" in rep.armar_mensaje(res, [z], precio, hora)
    assert "🟢 COMPRA (⚠ contra sesgo)" in rep.armar_mensaje(res, [dict(z, contra=["H4"])], precio, hora)
    # la etiqueta del gráfico lleva el mismo número N que el mensaje
    assert rep.etiqueta_corta(1, dict(cands[0], tipo="Extremo", puntaje=3)) == "1 · EXT 3/4"
    fig = rep.construir_figura(df, res, cands, 400, set())
    textos = [t.text[0] for t in fig.data if isinstance(t, go.Scatter) and t.mode == "text" and t.text and t.text[0]]
    for n, zz in enumerate(cands, 1):
        assert any(x.startswith(f"{n} · {'EXT' if zz['tipo'] == 'Extremo' else 'DEC'} {zz['puntaje']}/4") for x in textos)


def test_zonas_fuera_del_grafico_van_en_una_linea_y_se_recortan_primero(mundo):
    import pandas as pd
    _, res, precio, cands = mundo
    hora = pd.Timestamp("2026-11-08 16:45", tz="UTC")
    fuera = [{"tf": "H1", "tipo": "Extremo", "dir": "COMPRA", "bajo": 5764.9, "alto": 5784.6,
              "arriba": False, "dist_pct": -3.5},
             {"tf": "H4", "tipo": "Decisional", "dir": "VENTA", "bajo": 6239.8, "alto": 6279.0,
              "arriba": True, "dist_pct": 4.0},
             {"tf": "H1", "tipo": "Extremo", "dir": "COMPRA", "bajo": 5000.0, "alto": 5100.0,
              "arriba": False, "dist_pct": -15.0}]
    msg = rep.armar_mensaje(res, cands, precio, hora, fuera)
    assert "Fuera del gráfico: H1 compra 5785 (-3.5%) · H4 venta 6240 (+4.0%) · +1 más" in msg
    assert "Fuera del gráfico" not in rep.armar_mensaje(res, cands, precio, hora)
    assert len(msg) <= rep.LIMITE_CAPTION
    # si no cabe, se recorta la línea de zonas fuera del gráfico y las fichas quedan completas
    largo = [dict(fuera[0], tf="H1")] * 2
    msg_sin = rep.armar_mensaje(res, cands, precio, hora)
    rep.LIMITE_CAPTION, original = len(msg_sin), rep.LIMITE_CAPTION
    try:
        assert rep.armar_mensaje(res, cands, precio, hora, largo) == msg_sin
    finally:
        rep.LIMITE_CAPTION = original


def test_el_eje_y_no_se_estira_y_las_flechas_marcan_el_borde(mundo):
    df, res, precio, cands = mundo
    rango = rep.rango_eje_y(df, 400, cands)
    fuera = [{"tf": "H1", "tipo": "Extremo", "dir": "VENTA", "bajo": rango[1] * 2, "alto": rango[1] * 2.1,
              "arriba": True, "dist_pct": 100.0},
             {"tf": "H4", "tipo": "Extremo", "dir": "COMPRA", "bajo": rango[0] / 3, "alto": rango[0] / 2,
              "arriba": False, "dist_pct": -50.0}]
    fig = rep.construir_figura(df, res, cands, 400, set(), fuera)
    assert list(fig.layout.yaxis.range) == pytest.approx(rango)            # no se estiró
    flechas = {t.legendgroup: t for t in fig.data if isinstance(t, go.Scatter) and t.marker.symbol in ("triangle-up", "triangle-down")}
    assert flechas["Franjas H1"].marker.symbol == "triangle-up" and flechas["Franjas H1"].y[0] < rango[1]
    assert flechas["Franjas H4"].marker.symbol == "triangle-down" and flechas["Franjas H4"].y[0] > rango[0]
    assert "+100.0%" in flechas["Franjas H1"].text[0]


def test_tr_de_trabajo_y_tr_en_formacion_son_capas_separadas(mundo):
    df, res, _, cands = mundo
    fig = rep.construir_figura(df, res, cands, 400, set())
    entradas = [t.name for t in fig.data if t.showlegend]
    assert "TR de trabajo + 50%" in entradas and "TR en formación" in entradas
    assert "TR activo + 50%" not in entradas
    m15 = res["M15"]
    if m15["tr_formacion"] is not None:
        fino = [t for t in fig.data if t.legendgroup == "TR en formación" and t.mode == "lines"
                and not t.showlegend]
        assert fino and fino[0].line.width < 1
        textos = [t.text[0] for t in fig.data if t.legendgroup == "TR en formación" and t.mode == "text"]
        assert any("en formación" in x for x in textos)
