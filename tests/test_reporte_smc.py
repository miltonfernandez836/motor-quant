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


def test_ficha_incluye_sesgo_h1_h4_premium_descuento_y_marca_contra(mundo):
    _, res, _, cands = mundo
    z = dict(cands[0])
    z.update(sesgo_h1="bajista", sesgo_h4="bajista", contra=["H1", "H4"], pos_tr=30.0, signo=1, dir="COMPRA")
    t = rep.ficha_texto(z)
    assert "sesgo H1: bajista | H4: bajista" in t
    assert "descuento 30% del TR" in t
    assert t.endswith("⚠ CONTRA SESGO H1/H4")
    z["contra"] = ["H4"]
    assert rep.ficha_texto(z).endswith("⚠ CONTRA SESGO H4")
    z["contra"] = []
    assert "CONTRA SESGO" not in rep.ficha_texto(z)


def test_las_fichas_del_mensaje_salen_a_favor_primero(mundo):
    _, res, _, cands = mundo
    marcas = [bool(z["contra"]) for z in cands]
    assert marcas == sorted(marcas)                  # False (a favor) antes que True (contra)
    msg = rep.armar_mensaje(res, cands)
    lineas = [l for l in msg.split("\n") if l[:2] in {f"{i})" for i in range(1, 6)}]
    assert [("CONTRA SESGO" in l) for l in lineas] == marcas


def test_mensaje_lista_una_linea_por_zona_fuera_del_grafico(mundo):
    _, res, _, cands = mundo
    fuera = [{"tf": "H1", "tipo": "Extremo", "dir": "COMPRA", "bajo": 5764.9, "alto": 5784.6,
              "arriba": False, "dist_pct": -3.9},
             {"tf": "H4", "tipo": "Decisional", "dir": "VENTA", "bajo": 6239.8, "alto": 6279.0,
              "arriba": True, "dist_pct": 4.0}]
    msg = rep.armar_mensaje(res, cands, fuera)
    assert "• H1 Extremo COMPRA | 5764.9 - 5784.6 | -3.9% (abajo)" in msg
    assert "• H4 Decisional VENTA | 6239.8 - 6279.0 | +4.0% (arriba)" in msg
    assert msg.rstrip().endswith(rep.FIN_MENSAJE)
    assert "fuera del gráfico" not in rep.armar_mensaje(res, cands)    # sin zonas fuera, sin sección


def test_el_eje_y_no_se_estira_y_las_flechas_marcan_el_borde(mundo):
    df, res, precio, cands = mundo
    rango = rep.rango_eje_y(df, 400)
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
