"""Tests del núcleo SMC con velas armadas a mano (pytest).

Los escenarios usan parámetros pequeños (pivote n=1, ATR de 3 velas) para que las
secuencias sean cortas y legibles. Cada vela es (open, high, low, close).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import smc_nucleo as smc  # noqa: E402

CFG = {"N_PIVOTE": 1, "N_IRL": 1, "ATR_VENTANA": 3, "DESPLAZ_ATR": 0.5,
       "EDAD_MAX": {"M15": 1000}}


def velas(lista):
    idx = pd.date_range("2026-01-01", periods=len(lista), freq="15min", tz="UTC")
    return pd.DataFrame(lista, columns=["open", "high", "low", "close"], index=idx).astype(float)


def espejo(df, k=100.0):
    """Refleja los precios (compra <-> venta): o'=k-o, h'=k-l, l'=k-h, c'=k-c."""
    return pd.DataFrame({"open": k - df["open"], "high": k - df["low"],
                         "low": k - df["high"], "close": k - df["close"]}, index=df.index)


def tipos(res, tipo):
    return [e for e in res["eventos"] if e["tipo"] == tipo]


# ---------------------------------------------------------------------------
# Escenarios
# ---------------------------------------------------------------------------
# Pivote alto W=45 (idx1). Caída hasta el strong low 22 (idx4, vela con OB bajista),
# impulso sin retrocesos internos y BOS en idx8 (cierre 49.5 > 45).
BASE_IMPULSO = [
    (40, 40.5, 39.5, 40), (40, 45, 39.8, 44), (44, 44.2, 41, 41.5), (41.5, 41.6, 30, 30.5),
    (30.5, 30.8, 22, 22.5), (22.5, 40, 22.4, 39.5), (39.5, 44, 39, 43.5), (43.5, 44.5, 43, 44.4),
    (44.4, 50, 44, 49.5),
]
# Igual pero el impulso tiene un HL interno bien separado del Extremo (vela 7, mín 36.5)
CON_HL_INTERNO = [
    (40, 40.5, 39.5, 40), (40, 45, 39.8, 44), (44, 44.2, 41, 41.5), (41.5, 41.6, 30, 30.5),
    (30.5, 30.8, 22, 22.5), (22.5, 40, 22.4, 39.5), (39.5, 40.5, 38.5, 38.8),
    (38.8, 39, 36.5, 36.9), (36.9, 44, 36.8, 43.5), (43.5, 44.5, 43, 44.0),
    (44, 50, 43.8, 49.5),
]


# ---------------------------------------------------------------------------
# BOS vs LG
# ---------------------------------------------------------------------------
PREVIA_LG = [(10, 10.5, 9.5, 10), (10, 12, 9.8, 11), (11, 11.5, 10, 10.2), (10.2, 11, 9.8, 10.5)]


def test_mecha_que_cierra_dentro_es_lg_no_bos():
    df = velas(PREVIA_LG + [(10.5, 12.6, 10.4, 11.0)])   # mecha 12.6 > 12, cierra 11
    r = smc.detectar(df, "M15", CFG)
    lg = tipos(r, "LG")
    assert len(lg) == 1 and lg[0]["lado"] == "alto" and lg[0]["nivel"] == 12
    assert tipos(r, "BOS") == [] and r["sesgo"] == "rango"


def test_cierre_de_cuerpo_sobre_el_high_es_bos():
    df = velas(PREVIA_LG + [(10.5, 12.8, 10.4, 12.5)])   # cierra 12.5 > 12
    r = smc.detectar(df, "M15", CFG)
    bos = tipos(r, "BOS")
    assert len(bos) == 1 and bos[0]["dir"] == 1 and bos[0]["idx"] == 4
    assert tipos(r, "LG") == [] and r["sesgo"] == "alcista"


def test_bos_bajista_espejo():
    r = smc.detectar(espejo(velas(PREVIA_LG + [(10.5, 12.8, 10.4, 12.5)])), "M15", CFG)
    assert [e["dir"] for e in tipos(r, "BOS")] == [-1] and r["sesgo"] == "bajista"


# ---------------------------------------------------------------------------
# OB Extremo vs Decisional
# ---------------------------------------------------------------------------
def test_un_solo_poi_cuando_no_hay_hl_interno():
    r = smc.detectar(velas(BASE_IMPULSO), "M15", CFG)
    assert [z["tipo"] for z in r["zonas"]] == ["Extremo"]
    z = r["zonas"][0]
    assert (z["bajo"], z["alto"]) == (22.0, 30.8) and z["idx_ob"] == 4
    assert z["medio"] == pytest.approx(26.4)
    assert (z["cuerpo_bajo"], z["cuerpo_alto"]) == (22.5, 30.5)   # cuerpo de la vela bajista
    assert z["entrada"] == 30.8                                     # entrada en el borde
    assert z["dir"] == "COMPRA" and z["r2"] is True


def test_extremo_y_decisional_cuando_hay_hl_interno_separado():
    r = smc.detectar(velas(CON_HL_INTERNO), "M15", CFG)
    por_tipo = {z["tipo"]: z for z in r["zonas"]}
    assert set(por_tipo) == {"Extremo", "Decisional"}
    ext, dec = por_tipo["Extremo"], por_tipo["Decisional"]
    assert (ext["bajo"], ext["alto"]) == (22.0, 30.8)
    assert (dec["bajo"], dec["alto"]) == (36.5, 39.0) and dec["idx_ob"] == 7
    assert dec["bajo"] - ext["alto"] >= 0.5 * dec["atr"]            # separación >= 0.5 ATR


def test_decisional_se_descarta_si_esta_a_menos_de_medio_atr():
    lista = list(CON_HL_INTERNO)
    # HL interno pegado al Extremo (mín 31.0): separación 0.2 < 0.5 ATR
    lista[5] = (22.5, 33, 22.4, 32.5)
    lista[6] = (32.5, 33.5, 31.5, 31.8)
    lista[7] = (31.8, 32.0, 31.0, 31.2)
    lista[8] = (31.2, 44, 31.1, 43.5)
    r = smc.detectar(velas(lista), "M15", CFG)
    assert [z["tipo"] for z in r["zonas"]] == ["Extremo"]


def test_venta_es_espejo_de_compra():
    r = smc.detectar(espejo(velas(CON_HL_INTERNO)), "M15", CFG)
    por_tipo = {z["tipo"]: z for z in r["zonas"]}
    assert set(por_tipo) == {"Extremo", "Decisional"}
    assert por_tipo["Extremo"]["dir"] == "VENTA"
    assert (por_tipo["Extremo"]["bajo"], por_tipo["Extremo"]["alto"]) == pytest.approx((69.2, 78.0))
    assert por_tipo["Extremo"]["entrada"] == pytest.approx(69.2)   # borde inferior en ventas


def test_wick_poi_cuando_no_hay_vela_bajista_clara():
    lista = [(40, 40.5, 39.5, 40), (40, 45, 39.8, 44), (44, 44.2, 41, 41.5),
             (38, 39, 36, 38.5), (35, 36, 30, 35.5),       # velas alcistas: sin OB bajista
             (35.5, 47, 35.4, 46.5)]                        # BOS (46.5 > 45)
    r = smc.detectar(velas(lista), "M15", CFG)
    z = r["zonas"][0]
    assert z["tipo"] == "Extremo" and z["mecha_poi"] is True
    assert (z["bajo"], z["alto"]) == (30.0, 35.0)               # de la mecha al cuerpo


def test_entrada_en_el_50_por_ciento_configurable():
    r = smc.detectar(velas(BASE_IMPULSO), "M15", {**CFG, "ENTRADA": "50"})
    assert r["zonas"][0]["entrada"] == pytest.approx(26.4)


# ---------------------------------------------------------------------------
# Estados de zona (zona Extremo [22, 30.8], medio 26.4, creada en idx8)
# ---------------------------------------------------------------------------
def estados(extra, cfg=None):
    r = smc.detectar(velas(BASE_IMPULSO + extra), "M15", {**CFG, **(cfg or {})})
    z = r["zonas"][0]
    return z, [e for _, e in z["historial"]]


def test_en_formacion_mientras_haga_nuevos_maximos_y_luego_fresca():
    z, hist = estados([(49.5, 52, 49, 51.5)])
    assert hist == ["en_formacion"]                       # nuevo máximo: sigue en formación
    z, hist = estados([(49.5, 52, 49, 51.5), (51.5, 51.8, 50, 50.5)])
    assert hist == ["en_formacion", "fresca"]


FRESCA = [(49.5, 49.8, 48, 48.5), (48.5, 48.6, 40, 41), (41, 41.5, 35, 35.5)]   # idx9..11


def test_fresca_a_tocada_mecha_sin_llegar_al_50():
    z, hist = estados(FRESCA + [(35.5, 36, 29, 31)])      # mín 29 > 26.4
    assert hist == ["en_formacion", "fresca", "tocada"] and z["n_toques"] == 1


def test_tocada_a_mitigada_al_llegar_al_50():
    z, hist = estados(FRESCA + [(35.5, 36, 29, 31), (31, 31.5, 26, 27)])
    assert hist[-1] == "mitigada" and z["n_toques"] == 1


def test_permanecer_dentro_no_cuenta_como_segundo_toque():
    z, hist = estados(FRESCA + [(35.5, 36, 29, 31), (31, 31.2, 29.5, 30.5)])
    assert hist[-1] == "tocada" and z["n_toques"] == 1


def test_dos_toques_tras_salir_una_vela_completa_es_mitigada():
    z, hist = estados(FRESCA + [(35.5, 36, 29, 31),
                                (31, 40, 30.9, 39),            # sale: vela completa fuera
                                (39, 39.5, 29.5, 30)])         # 2º toque, sin llegar al 50%
    assert hist[-1] == "mitigada" and z["n_toques"] == 2


def test_invalidada_por_cierre_de_cuerpo_bajo_el_extremo():
    z, hist = estados(FRESCA[:1] + [(48.5, 48.6, 20, 21)])    # cierra 21 < 22
    assert hist[-1] == "invalidada"


def test_mecha_bajo_el_extremo_que_cierra_dentro_no_invalida():
    z, hist = estados(FRESCA + [(35.5, 36, 21, 25)])          # mecha a 21, cierra 25
    assert hist[-1] == "mitigada"                              # llegó al 50%, pero no invalidada


def test_vencida_por_edad_maxima():
    resto = [(49.5, 49.8, 48, 48.5)] + [(48.5, 49, 48, 48.6)] * 5
    z, hist = estados(resto, {"EDAD_MAX": {"M15": 3}})
    assert hist[-1] == "vencida"


def test_solo_fresca_y_tocada_son_candidatas():
    r = smc.detectar(velas(BASE_IMPULSO + FRESCA + [(35.5, 36, 29, 31)]), "M15", CFG)
    assert [z["estado"] for z in smc.candidatas(r, 31.0)] == ["tocada"]
    r = smc.detectar(velas(BASE_IMPULSO + FRESCA + [(35.5, 36, 29, 31), (31, 31.5, 26, 27)]), "M15", CFG)
    assert smc.candidatas(r, 27.0) == []


# ---------------------------------------------------------------------------
# Reglas R1-R4
# ---------------------------------------------------------------------------
def test_r1_barrido_de_liquidez_previo_marca_la_casilla():
    sin = smc.detectar(velas(BASE_IMPULSO), "M15", CFG)["zonas"][0]
    assert sin["r1"] is False
    # idx3 deja un pivote bajo (30); idx5 lo barre con mecha (29) y cierra dentro (30.5): LG
    lista = [(40, 40.5, 39.5, 40), (40, 45, 39.8, 44), (44, 44.2, 41, 41.5), (41.5, 41.6, 30, 30.5),
             (30.5, 36, 30.2, 35), (35, 35.5, 29, 30.5), (30.5, 40, 30.4, 39.5),
             (39.5, 45.8, 39.0, 45.5)]                          # BOS (45.5 > 45)
    r = smc.detectar(velas(lista), "M15", CFG)
    assert any(e["tipo"] == "LG" and e["lado"] == "bajo" for e in r["eventos"])
    assert r["zonas"] and all(z["r1"] for z in r["zonas"])


# Retroceso con IRL internas: mínimo menor 40 (idx10) y máximo menor 47 (idx11), dentro del TR (22-50)
CON_IRL = [(49.5, 49.8, 48, 48.5), (45, 45.2, 40, 41), (41, 47, 40.5, 46), (46, 46.5, 38, 38.5)]


def test_r3_irl_interna_formada_antes_del_primer_toque():
    z, _ = estados(CON_IRL + [(38.5, 38.6, 29, 30)])      # 1er toque en idx13, IRL conocida en idx11
    assert z["primer_toque"] == 13
    assert z["r3"] is True and z["r3_idx"] == 11


def test_r3_falla_si_el_toque_llega_antes_que_cualquier_irl():
    z, _ = estados([(49.5, 49.8, 25, 26)])                # toque inmediato; el máx 50 es borde del TR
    assert z["r3"] is False


def test_r3_ignora_irl_formadas_despues_del_primer_toque():
    z, hist = estados([(49.5, 49.8, 29, 31), (31, 38, 30.9, 37), (37, 37.5, 33, 33.5),
                       (33.5, 40, 33.2, 39), (39, 39.5, 36, 36.5)])
    assert hist[-1] == "tocada" and z["primer_toque"] == 9
    assert z["r3"] is False                               # el máx 40 (idx12) se formó tras el toque


def test_r4_puntaje_baja_al_mitigarse():
    r = smc.detectar(velas(BASE_IMPULSO + FRESCA + [(35.5, 36, 29, 31)]), "M15", CFG)
    z = r["zonas"][0]
    p_tocada = smc.puntaje(z)
    r = smc.detectar(velas(BASE_IMPULSO + FRESCA + [(35.5, 36, 29, 31), (31, 31.5, 26, 27)]), "M15", CFG)
    assert smc.puntaje(r["zonas"][0]) == p_tocada - 1


# ---------------------------------------------------------------------------
# EQH / EQL
# ---------------------------------------------------------------------------
EQ_BASE = [(10, 10.5, 9.5, 10), (10, 15, 9.8, 14), (14, 14.5, 11, 11.5), (11.5, 12, 10, 10.5),
           (10.5, 15.05, 10.2, 13), (13, 13.5, 11, 11.5)]


def test_eqh_dos_maximos_a_menos_de_0_1_atr():
    r = smc.detectar(velas(EQ_BASE), "M15", CFG)
    assert [(e["tipo"], e["idx1"], e["idx2"]) for e in r["eqs"]] == [("EQH", 1, 4)]
    assert r["eqs"][0]["precio"] == 15.05


def test_maximos_distintos_no_son_eqh():
    lista = list(EQ_BASE)
    lista[4] = (10.5, 16.5, 10.2, 13)
    assert smc.detectar(velas(lista), "M15", CFG)["eqs"] == []


def test_eql_es_el_espejo():
    r = smc.detectar(espejo(velas(EQ_BASE)), "M15", CFG)
    assert [e["tipo"] for e in r["eqs"]] == ["EQL"]


# ---------------------------------------------------------------------------
# CDC, TR, premium/descuento
# ---------------------------------------------------------------------------
CDC_BASE = BASE_IMPULSO + [(49.5, 49.8, 48, 48.5)]      # idx9 confirma el pivote alto 50 (idx8)


def test_cdc_lg_de_un_high_y_cierre_bajo_el_strong_low():
    lista = CDC_BASE + [(48.5, 51, 47, 48), (48, 48.2, 30, 31), (31, 31.5, 18, 19)]
    r = smc.detectar(velas(lista), "M15", CFG)
    cdc = tipos(r, "CDC")
    assert len(cdc) == 1 and cdc[0]["dir"] == -1 and cdc[0]["idx"] == 12
    assert r["sesgo"] == "bajista"
    assert r["trs"][-1]["estado"] == "establecido" and r["tr"]["dir"] == -1


def test_ruptura_del_strong_low_sin_lg_no_es_cdc():
    lista = CDC_BASE + [(48.5, 49.5, 47, 48), (48, 48.2, 30, 31), (31, 31.5, 18, 19)]
    r = smc.detectar(velas(lista), "M15", CFG)
    assert tipos(r, "CDC") == []
    assert [e["dir"] for e in tipos(r, "BOS")] == [1, -1]


def test_tr_alcista_y_posicion_premium_descuento():
    r = smc.detectar(velas(BASE_IMPULSO), "M15", CFG)
    tr = r["tr"]
    assert (tr["dir"], tr["bajo"], tr["alto"], tr["estado"]) == (1, 22.0, 50.0, "en_formacion")
    assert smc.posicion_en_tr(36.0, tr) == pytest.approx(50.0)
    assert smc.posicion_en_tr(25.0, tr) < 50 < smc.posicion_en_tr(45.0, tr)
    r2 = smc.detectar(velas(BASE_IMPULSO + [(49.5, 53, 49, 52)]), "M15", CFG)
    assert r2["tr"]["alto"] == 53.0                         # el TR se extiende con nuevos máximos


ETIQ = [(10, 10.5, 9.5, 10), (10, 15, 9.8, 14), (14, 14.5, 11, 11.5), (11.5, 12, 10, 10.5),
        (10.5, 17, 10.2, 16), (16, 16.5, 13, 13.5), (13.5, 14, 11.5, 12), (12, 15.5, 11.8, 15)]


def test_etiquetas_de_estructura_hh_hl():
    r = smc.detectar(velas(ETIQ), "M15", CFG)
    assert [(p["idx"], p["etiqueta"]) for p in r["pivotes"]] == [(1, ""), (3, ""), (4, "HH"), (6, "HL")]


def test_etiquetas_de_estructura_ll_lh_en_el_espejo():
    r = smc.detectar(espejo(velas(ETIQ)), "M15", CFG)
    assert [(p["idx"], p["etiqueta"]) for p in r["pivotes"]] == [(1, ""), (3, ""), (4, "LL"), (6, "LH")]


# ---------------------------------------------------------------------------
# Liquidez: TP1 / TP final
# ---------------------------------------------------------------------------
def test_tp1_es_la_irl_a_favor_y_tp_final_el_maximo_del_dia_anterior():
    # 4 velas el día UTC anterior (23:00-23:45) con máximo 60 y mínimo 51, luego el escenario
    previo = [(55, 60, 54, 56), (56, 57, 53, 54), (54, 55, 52, 53), (53, 54, 51, 52)]
    lista = previo + BASE_IMPULSO + CON_IRL
    df = velas(lista)
    df.index = pd.date_range("2025-12-31 23:00", periods=len(df), freq="15min", tz="UTC")
    r = smc.detectar(df, "M15", CFG)
    liq = smc.calcular_liquidez(r, df)
    # (el salto a la baja del prefijo genera antes un BOS bajista y su zona de venta: se elige la compra)
    z = next(z for z in r["zonas"] if z["dir"] == "COMPRA")
    assert (z["bajo"], z["alto"]) == (22.0, 30.8)
    assert (liq["pdh"], liq["pdl"]) == (60.0, 51.0)
    assert z["tp1"] == 47.0                  # IRL (máximo menor) a favor, dentro del TR, sin barrer
    assert z["tp_final"] == 60.0             # ERL: máximo del día anterior sin barrer


def test_tp_de_venta_apunta_hacia_abajo():
    previo = [(55, 60, 54, 56), (56, 57, 53, 54), (54, 55, 52, 53), (53, 54, 51, 52)]
    df = espejo(velas(previo + BASE_IMPULSO + CON_IRL))
    df.index = pd.date_range("2025-12-31 23:00", periods=len(df), freq="15min", tz="UTC")
    r = smc.detectar(df, "M15", CFG)
    smc.calcular_liquidez(r, df)
    z = next(z for z in r["zonas"] if z["dir"] == "VENTA" and z["bajo"] == pytest.approx(69.2))
    assert z["tp1"] == pytest.approx(53.0) and z["tp_final"] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# ANTI-FUTURO
# ---------------------------------------------------------------------------
def serie_aleatoria(n=900, seed=7):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1, n)
    r[rng.random(n) < 0.015] += rng.choice([-18, 18], size=(rng.random(n) < 0.015).sum() or 1)[0]
    c = 1000 + np.cumsum(r)
    o = np.concatenate(([1000], c[:-1]))
    h = np.maximum(o, c) + rng.random(n) * 1.5
    l = np.minimum(o, c) - rng.random(n) * 1.5
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def _foto(res, t):
    """Todo lo que debe ser conocido en la vela t, en forma comparable."""
    zonas = []
    for z in res["zonas"]:
        if z["idx_creacion"] > t:
            continue
        hist = [(i, e) for i, e in z["historial"] if i <= t]
        estaticos = {k: z[k] for k in ("id", "tipo", "dir", "idx_ob", "idx_creacion", "bajo", "alto",
                                       "medio", "cuerpo_bajo", "cuerpo_alto", "entrada", "evento",
                                       "r1", "r2", "mecha_poi", "atr", "idx_fuerte")}
        r3 = z["r3_idx"] if (z["r3_idx"] is not None and z["r3_idx"] <= t) else None
        zonas.append((estaticos, hist, r3))
    return {
        "zonas": zonas,
        "eventos": [e for e in res["eventos"] if e["idx"] <= t],
        "pivotes": [p for p in res["pivotes"] if p["conocido_en"] <= t],
        "eqs": [e for e in res["eqs"] if e["conocido_en"] <= t],
        "irls": [i for i in res["irls"] if i["conocido_en"] <= t],
        "puntos": [p for p in res["puntos"] if p["conocido_en"] <= t],
    }


CORTES_ANTI_FUTURO = range(120, 900, 7)      # muchos cortes: una mirada corta al futuro solo
                                             # contamina lo confirmado justo antes de t


AG = {"open": "first", "high": "max", "low": "min", "close": "last"}
CAMPOS_CAPAS = {"fvg": ("dir", "idx", "idx_ifc", "bajo", "alto", "conocido_en"),
                "eqhl": ("tipo", "idx1", "idx2", "bajo", "conocido_en"),
                "inducement": ("tf", "clase", "tipo", "bajo", "t_pivote", "idx_lg", "conocido_en")}


def _dfs_completos(df):
    """M15 + H1/H4/D1 derivadas, descartando la vela superior aún sin cerrar (como en producción)."""
    dfs = {"M15": df}
    for tf, regla in (("H1", "1h"), ("H4", "4h"), ("D1", "1D")):
        r = df.resample(regla).agg(AG).dropna()
        if df.index[-1] + pd.Timedelta(minutes=15) < r.index[-1] + pd.Timedelta(regla):
            r = r.iloc[:-1]
        dfs[tf] = r
    return dfs


def _capas(df):
    dfs = _dfs_completos(df)
    return smc.calcular_capas(smc.analizar(dfs), dfs, set(smc.CONFIG["CAPAS"]) & CAPAS_FASE2)


CAPAS_FASE2 = {"fvg", "eqhl", "irl_erl", "albercas", "inducement", "smt"}


def _foto_capas(capas, t):
    """Elementos estáticos de las capas conocidos en t (el estado fresca/parcial/barrido evoluciona)."""
    return {k: [tuple(e[c] for c in campos) for e in capas[k] if e["conocido_en"] <= t]
            for k, campos in CAMPOS_CAPAS.items()}


def _oraculo_albercas(df):
    """PDH/PDL/PWH/PWL y su barrido calculados a mano solo con velas M15 hasta la última."""
    hoy = df.index[-1].normalize()
    dias = df.index.normalize()
    esperado = {}
    ayer = df[dias == hoy - pd.Timedelta(days=1)]
    hoy_df = df[dias >= hoy]
    if len(ayer):
        esperado["PDH"] = (ayer["high"].max(), bool(hoy_df["high"].max() > ayer["high"].max()))
        esperado["PDL"] = (ayer["low"].min(), bool(hoy_df["low"].min() < ayer["low"].min()))
    lunes = hoy - pd.Timedelta(days=int(hoy.weekday()))
    sem = df[(dias < lunes) & (dias >= lunes - pd.Timedelta(days=7))]
    desde_lunes = df[dias >= lunes]
    if len(sem):
        esperado["PWH"] = (sem["high"].max(), bool(desde_lunes["high"].max() > sem["high"].max()))
        esperado["PWL"] = (sem["low"].min(), bool(desde_lunes["low"].min() < sem["low"].min()))
    return esperado


def test_anti_futuro_datos_hasta_t_igual_a_datos_completos():
    df = serie_aleatoria()
    completo = smc.detectar(df, "M15")
    assert len(completo["zonas"]) > 5                      # la serie sí genera zonas
    capas_completas = _capas(df)
    assert all(len(capas_completas[k]) > 0 for k in ("fvg", "eqhl", "inducement"))   # las capas sí tienen datos
    for t in CORTES_ANTI_FUTURO:
        parcial = smc.detectar(df.iloc[:t + 1], "M15")
        assert _foto(parcial, t) == _foto(completo, t), f"difiere en t={t}"
        # capas de la Fase 2: lo conocido en t coincide con el cálculo sobre datos completos
        capas_parciales = _capas(df.iloc[:t + 1])
        assert _foto_capas(capas_parciales, t) == _foto_capas(capas_completas, t), f"capas difieren en t={t}"
        # y ningún elemento (incluidos irl_erl y smt, que dependen de "hoy") nace después de t
        assert all(e["conocido_en"] <= t for v in capas_parciales.values() for e in v), f"capa del futuro en t={t}"
        # albercas D1/W1: iguales a un cálculo a mano con velas <= t (no pueden mirar días posteriores)
        esperado = _oraculo_albercas(df.iloc[:t + 1])
        obtenido = {e["nombre"]: (e["alto"], e["barrido"]) for e in capas_parciales["albercas"]}
        assert obtenido.keys() == esperado.keys(), f"albercas difieren en t={t}"
        for nombre, (nivel, barrido) in esperado.items():
            assert obtenido[nombre][0] == pytest.approx(nivel) and obtenido[nombre][1] == barrido, f"{nombre} en t={t}"


def test_anti_futuro_alterar_el_futuro_no_cambia_el_pasado():
    df = serie_aleatoria()
    t = 450
    roto = df.copy()
    rng = np.random.default_rng(0)
    roto.iloc[t + 1:] = roto.iloc[t + 1:].to_numpy() * rng.uniform(0.5, 1.5, size=roto.iloc[t + 1:].shape)
    roto["high"] = roto[["open", "high", "low", "close"]].max(axis=1)
    roto["low"] = roto[["open", "high", "low", "close"]].min(axis=1)
    assert _foto(smc.detectar(df, "M15"), t) == _foto(smc.detectar(roto, "M15"), t)


def test_el_pivote_solo_se_conoce_n_velas_despues():
    df = velas(EQ_BASE)
    r = smc.detectar(df, "M15", CFG)
    for p in r["pivotes"]:
        assert p["conocido_en"] == p["idx"] + CFG["N_PIVOTE"]


def test_atr_usa_mediana_y_resiste_spikes():
    base = [(100, 101, 99, 100)] * 10
    df = velas(base + [(100, 160, 99, 150)] + base)       # un spike enorme
    r = smc.detectar(df, "M15", {**CFG, "ATR_VENTANA": 5})
    assert r["atr"][11] == pytest.approx(2.0)               # la media sería ~10; la mediana, 2
