"""Tests de la Fase 1.5: R1 por ventana, contra sesgo + orden, TR de trabajo, zonas fuera del eje.

Reutiliza los escenarios armados a mano de test_smc_nucleo.py.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import smc_nucleo as smc  # noqa: E402
from test_smc_nucleo import BASE_IMPULSO, CDC_BASE, CFG, espejo, velas  # noqa: E402

CON_CDC = CDC_BASE + [(48.5, 51, 47, 48), (48, 48.2, 30, 31), (31, 31.5, 18, 19)]


# ---------------------------------------------------------------------------
# R1: LG dentro de las 10 velas previas al OB (incluida la vela del OB)
# idx5 barre con mecha el pivote bajo 30 (LG); luego m velas bajistas claras mueven el OB hacia delante
# ---------------------------------------------------------------------------
def escenario_lg_y_ob_separados(m):
    lista = [(40, 40.5, 39.5, 40), (40, 45, 39.8, 44), (44, 44.2, 41, 41.5), (41.5, 41.6, 30, 30.5),
             (30.5, 36, 30.2, 35), (35, 35.5, 29, 30.5)]                 # idx5: LG bajo
    lista += [(31.5, 31.6, 30.6, 30.7)] * m                              # bajistas claras (OB = la última)
    lista += [(30.7, 40, 30.65, 39.5), (39.5, 45.8, 39.0, 45.5)]         # desplazamiento + BOS
    return lista


@pytest.mark.parametrize("m,r1", [(0, True), (10, True), (11, False)])
def test_r1_exige_el_lg_dentro_de_las_10_velas_previas_al_ob(m, r1):
    r = smc.detectar(velas(escenario_lg_y_ob_separados(m)), "M15", CFG)
    lg = [e for e in r["eventos"] if e["tipo"] == "LG" and e["lado"] == "bajo"]
    assert [e["idx"] for e in lg] == [5]
    z = r["zonas"][0]
    assert z["idx_ob"] == 5 + m                       # distancia LG -> OB = m velas
    assert z["r1"] is r1


def test_r1_ventana_es_parametro_de_config():
    lista = escenario_lg_y_ob_separados(11)
    assert smc.detectar(velas(lista), "M15", CFG)["zonas"][0]["r1"] is False
    assert smc.detectar(velas(lista), "M15", {**CFG, "R1_VENTANA": 11})["zonas"][0]["r1"] is True


def test_r1_ignora_un_lg_del_lado_equivocado():
    # un LG de HIGH previo no cuenta para una zona de compra (aquí ni siquiera hay zona)
    lista = [(10, 10.5, 9.5, 10), (10, 12, 9.8, 11), (11, 11.5, 10, 10.2), (10.2, 11, 9.8, 10.5),
             (10.5, 12.6, 10.4, 11.0)]
    r = smc.detectar(velas(lista), "M15", CFG)
    assert [e["lado"] for e in r["eventos"] if e["tipo"] == "LG"] == ["alto"]
    assert r["zonas"] == []


def test_r1_del_decisional_se_mide_contra_su_propia_vela():
    from test_smc_nucleo import CON_HL_INTERNO
    r = smc.detectar(velas(CON_HL_INTERNO), "M15", CFG)
    assert all(z["r1"] is False for z in r["zonas"])   # no hay LG en este escenario


# ---------------------------------------------------------------------------
# Contra sesgo y orden de fichas
# ---------------------------------------------------------------------------
def zona_falsa(signo, h1="rango", h4="rango", bajo=100.0, alto=101.0, estado="fresca"):
    return {"signo": signo, "sesgo_h1": h1, "sesgo_h4": h4, "bajo": bajo, "alto": alto,
            "estado": estado, "tipo": "Extremo", "dir": "COMPRA" if signo == 1 else "VENTA"}


def test_contra_sesgo_compra_contra_bajista_y_venta_contra_alcista():
    assert smc.contra_sesgo(zona_falsa(1, "bajista", "alcista")) == ["H1"]
    assert smc.contra_sesgo(zona_falsa(1, "alcista", "bajista")) == ["H4"]
    assert smc.contra_sesgo(zona_falsa(-1, "alcista", "alcista")) == ["H1", "H4"]
    assert smc.contra_sesgo(zona_falsa(1, "alcista", "alcista")) == []
    assert smc.contra_sesgo(zona_falsa(-1, "bajista", "bajista")) == []


def test_el_sesgo_en_rango_no_contradice_a_nadie():
    assert smc.contra_sesgo(zona_falsa(1, "rango", "rango")) == []
    assert smc.contra_sesgo(zona_falsa(-1, "rango", "bajista")) == []


def test_orden_de_fichas_a_favor_primero_y_luego_por_cercania():
    contra_cerca = zona_falsa(1, "bajista", bajo=99.0, alto=99.5)
    a_favor_lejos = zona_falsa(1, "alcista", bajo=80.0, alto=81.0)
    a_favor_cerca = zona_falsa(1, "alcista", bajo=95.0, alto=96.0)
    for z in (contra_cerca, a_favor_lejos, a_favor_cerca):
        z["contra"] = smc.contra_sesgo(z)
    orden = smc.ordenar_fichas([contra_cerca, a_favor_lejos, a_favor_cerca], 100.0)
    assert orden == [a_favor_cerca, a_favor_lejos, contra_cerca]


def test_analizar_marca_contra_sesgo_con_los_sesgos_de_h1_y_h4():
    df = velas(BASE_IMPULSO)                         # M15 alcista; H1/H4 = espejo (bajista)
    r = smc.analizar({"M15": df, "H1": espejo(df), "H4": espejo(df)}, CFG)
    assert [r[tf]["sesgo"] for tf in ("M15", "H1", "H4")] == ["alcista", "bajista", "bajista"]
    z = r["M15"]["zonas"][0]
    assert z["dir"] == "COMPRA" and z["contra"] == ["H1", "H4"]


# ---------------------------------------------------------------------------
# TR de trabajo (último establecido) vs TR en formación
# ---------------------------------------------------------------------------
def test_tr_de_trabajo_es_el_ultimo_establecido_y_el_actual_esta_en_formacion():
    r = smc.detectar(velas(BASE_IMPULSO), "M15", CFG)
    assert r["tr_trabajo"] is None                   # aún ningún CDC: no hay TR establecido
    assert r["tr_formacion"]["dir"] == 1 and r["tr_formacion"]["estado"] == "en_formacion"

    r = smc.detectar(velas(CON_CDC), "M15", CFG)
    tw, tf = r["tr_trabajo"], r["tr_formacion"]
    assert (tw["dir"], tw["bajo"], tw["alto"], tw["estado"]) == (1, 22.0, 51.0, "establecido")
    assert tf["dir"] == -1 and tf["estado"] == "en_formacion"


def test_pos_tr_se_calcula_sobre_el_tr_de_trabajo_y_no_sobre_el_que_se_forma():
    df = velas(CON_CDC)
    r = smc.analizar({"M15": df, "H1": df, "H4": df}, CFG)["M15"]
    for z in r["zonas"]:
        assert z["pos_tr"] == pytest.approx(smc.posicion_en_tr(z["medio"], r["tr_trabajo"]))
    compra = next(z for z in r["zonas"] if z["dir"] == "COMPRA")   # medio 26.4 dentro del TR 22-51
    assert compra["pos_tr"] == pytest.approx((26.4 - 22) / (51 - 22) * 100)
    base = velas(BASE_IMPULSO)
    sin_tr = smc.analizar({"M15": base, "H1": base, "H4": base}, CFG)["M15"]["zonas"][0]
    assert sin_tr["pos_tr"] is None


def test_clasificar_premium_descuento():
    assert smc.clasificar_pd(None) == "sin TR de trabajo"
    assert smc.clasificar_pd(30.0) == "descuento 30% del TR"
    assert smc.clasificar_pd(50.0) == "premium 50% del TR"
    assert smc.clasificar_pd(80.0) == "premium 80% del TR"
    assert smc.clasificar_pd(-5.0) == "fuera del TR de trabajo"
    assert smc.clasificar_pd(140.0) == "fuera del TR de trabajo"


# ---------------------------------------------------------------------------
# Zonas H1/H4 vivas fuera del eje visible
# ---------------------------------------------------------------------------
def res_falso(**zonas_por_tf):
    return {tf: {"zonas": zs, "cfg": smc.CONFIG} for tf, zs in zonas_por_tf.items()}


def test_zonas_fuera_de_rango_distancia_pct_y_filtros():
    arriba = zona_falsa(-1, bajo=120.0, alto=125.0)
    abajo = zona_falsa(1, bajo=70.0, alto=80.0, estado="tocada")
    dentro = zona_falsa(1, bajo=95.0, alto=98.0)
    parcial = zona_falsa(1, bajo=108.0, alto=115.0)               # toca el rango: no es "fuera"
    mitigada = zona_falsa(1, bajo=10.0, alto=20.0, estado="mitigada")
    res = res_falso(H1=[arriba, abajo, dentro, parcial, mitigada], H4=[])
    fuera = smc.zonas_fuera_de_rango(res, 100.0, 90.0, 110.0)
    assert len(fuera) == 2
    por_lado = {f["arriba"]: f for f in fuera}
    assert por_lado[True]["dist_pct"] == pytest.approx(20.0)      # borde bajo de la zona: 120 vs 100
    assert por_lado[False]["dist_pct"] == pytest.approx(-20.0)    # borde alto de la zona: 80 vs 100


def test_zonas_fuera_de_rango_respeta_el_tope_y_ordena_por_cercania():
    zs = [zona_falsa(1, bajo=200.0 + i * 10, alto=205.0 + i * 10) for i in range(6)]
    fuera = smc.zonas_fuera_de_rango(res_falso(H1=zs, H4=[]), 100.0, 90.0, 110.0, k=4)
    assert [round(f["dist_pct"]) for f in fuera] == [100, 110, 120, 130]
