"""Tests de robustez: tiempos de producción, reintentos, aviso de fallos a Telegram y caché de --dry-run.

Sin red: todo se prueba con funciones simuladas (monkeypatch).
"""
import asyncio
import json
import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import datos_smc as D  # noqa: E402
import reporte_smc as rep  # noqa: E402
import smc_nucleo as smc  # noqa: E402
from test_smc_nucleo import serie_aleatoria  # noqa: E402

AG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def dfs_sinteticos():
    df = serie_aleatoria(2000, seed=3)
    dfs = {"M15": df, "H1": df.resample("1h").agg(AG).dropna(), "H4": df.resample("4h").agg(AG).dropna(),
           "D1": df.resample("1D").agg(AG).dropna()}
    for d in dfs.values():
        d.index.name = "datetime"        # igual que las velas reales de datos_smc
    return dfs


# ---------------------------------------------------------------------------
# Tiempos de producción en CONFIG
# ---------------------------------------------------------------------------
def test_tiempos_de_produccion_en_config():
    red = smc.CONFIG["RED"]
    assert red["TIMEOUT_GLOBAL_S"] == 180
    assert red["TIMEOUT_PETICION_S"] == 30
    assert tuple(red["ESPERAS_S"]) == (20, 40, 60)          # 3 reintentos
    assert red["TIMEOUT_GLOBAL_S"] - red["MARGEN_TIMEOUT_S"] > 0


def test_modo_largo_conserva_los_8_reintentos():
    largo = smc.CONFIG["RED_LARGO"]
    assert tuple(largo["ESPERAS_S"]) == (20, 40, 60, 80, 100, 120, 140, 160)
    assert rep.perfil_de_red(smc.CONFIG, rep.leer_argumentos(["--modo-largo"])) is largo
    assert rep.perfil_de_red(smc.CONFIG, rep.leer_argumentos([])) is smc.CONFIG["RED"]


def test_workflow_con_timeouts_y_mismas_librerias_que_bot_diario():
    smc_yml = open(os.path.join(RAIZ, ".github", "workflows", "bot_smc.yml"), encoding="utf-8").read()
    diario = open(os.path.join(RAIZ, ".github", "workflows", "bot_diario.yml"), encoding="utf-8").read()
    assert "timeout-minutes: 15" in smc_yml and smc_yml.count("timeout-minutes: 4") == 3   # job y un paso por activo
    linea = 'pip install websockets pandas numpy "plotly<6" requests kaleido==0.2.1'
    assert linea in smc_yml and linea in diario
    assert "schedule:" not in smc_yml


def test_gitignore_excluye_salidas_y_cache():
    ig = open(os.path.join(RAIZ, ".gitignore"), encoding="utf-8").read().split()
    assert {"reporte_smc.png", "reporte_smc.html", ".cache_smc/"} <= set(ig)


# ---------------------------------------------------------------------------
# Reintentos y tope por petición (datos_smc)
# ---------------------------------------------------------------------------
@pytest.fixture
def esperas(monkeypatch):
    registro = []

    async def dormir(s):
        registro.append(s)
    monkeypatch.setattr(D.asyncio, "sleep", dormir)
    return registro


def test_produccion_reintenta_3_veces_con_esperas_20_40_60(monkeypatch, esperas):
    llamadas = []

    async def falla(*a):
        llamadas.append(1)
        raise ConnectionError("RateLimit")
    monkeypatch.setattr(D, "_peticion", falla)
    with pytest.raises(RuntimeError, match="sin respuesta tras 4 intentos"):
        asyncio.run(D._peticion_con_reintentos("BOOM500", 900, 10, "latest", smc.CONFIG["RED"]))
    assert len(llamadas) == 4 and esperas == [20, 40, 60]


def test_modo_largo_hace_8_reintentos(monkeypatch, esperas):
    llamadas = []

    async def falla(*a):
        llamadas.append(1)
        raise ConnectionError("RateLimit")
    monkeypatch.setattr(D, "_peticion", falla)
    with pytest.raises(RuntimeError, match="sin respuesta tras 9 intentos"):
        asyncio.run(D._peticion_con_reintentos("BOOM500", 900, 10, "latest", smc.CONFIG["RED_LARGO"]))
    assert len(llamadas) == 9 and esperas == [20, 40, 60, 80, 100, 120, 140, 160]


def test_se_recupera_si_un_reintento_funciona(monkeypatch, esperas):
    estado = {"n": 0}

    async def a_veces(*a):
        estado["n"] += 1
        if estado["n"] < 3:
            raise ConnectionError("RateLimit")
        return [{"epoch": 1}]
    monkeypatch.setattr(D, "_peticion", a_veces)
    out = asyncio.run(D._peticion_con_reintentos("BOOM500", 900, 10, "latest", smc.CONFIG["RED"]))
    assert out == [{"epoch": 1}] and esperas == [20, 40]


def test_un_error_de_api_no_reintentable_no_se_reintenta(monkeypatch, esperas):
    llamadas = []

    async def invalido(*a):
        llamadas.append(1)
        raise D.ErrorDeriv("InvalidSymbol")
    monkeypatch.setattr(D, "_peticion", invalido)
    with pytest.raises(D.ErrorDeriv):
        asyncio.run(D._peticion_con_reintentos("XXX", 900, 10, "latest", smc.CONFIG["RED"]))
    assert len(llamadas) == 1 and esperas == []


def test_tope_duro_por_peticion(monkeypatch):
    async def colgada(*a):
        await asyncio.sleep(30)
    monkeypatch.setattr(D, "_peticion", colgada)
    red = {"TIMEOUT_PETICION_S": 0.1, "ESPERAS_S": (), "PAUSA_S": 0}
    t0 = time.time()
    with pytest.raises(RuntimeError, match="sin respuesta tras 1 intentos"):
        asyncio.run(D._peticion_con_reintentos("BOOM500", 900, 10, "latest", red))
    assert time.time() - t0 < 3


def test_descargar_todo_informa_que_tf_fallo(monkeypatch):
    async def velas(simbolo, tf, cantidad, red):
        if tf == "H4":
            raise RuntimeError("sin respuesta")
        return pd.DataFrame()
    monkeypatch.setattr(D, "descargar_velas_smc", velas)
    red = {"PAUSA_S": 0}
    progreso = {}
    with pytest.raises(D.ErrorDescarga) as e:
        asyncio.run(D.descargar_todo("BOOM500", red=red, progreso=progreso))
    assert e.value.tf == "H4" and "H4" in str(e.value) and progreso["tf_actual"] == "H4"


# ---------------------------------------------------------------------------
# Aviso de fallos a Telegram
# ---------------------------------------------------------------------------
@pytest.fixture
def telegram(monkeypatch, tmp_path):
    """Entorno con credenciales, sin red y en una carpeta temporal. Devuelve la lista de POST hechos."""
    monkeypatch.setenv("TELEGRAM_TOKEN", "TOK123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "CHAT9")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(D, "CACHE_DIR", str(tmp_path / ".cache_smc"))
    posts = []

    class Resp:
        status_code = 200
        text = "ok"

    def falso_post(url, **kw):
        posts.append((url.rsplit("/", 1)[-1], kw.get("data", {})))
        return Resp()
    monkeypatch.setattr(rep.requests, "post", falso_post)
    return posts


def args_prod(*extra):
    return rep.leer_argumentos(list(extra))


def test_si_falla_la_descarga_avisa_con_el_tf_y_sale_con_1(monkeypatch, telegram):
    async def falla(*a, **k):
        raise D.ErrorDescarga("BOOM500", "H4", RuntimeError("sin respuesta tras 4 intentos"))
    monkeypatch.setattr(rep, "descargar_todo", falla)
    assert rep.correr(args_prod()) == 1
    assert [m for m, _ in telegram] == ["sendMessage"]
    texto = telegram[0][1]["text"]
    assert telegram[0][1]["chat_id"] == "CHAT9"
    assert "falló" in texto and "H4" in texto and "sin respuesta" in texto and len(texto) <= 400


def test_si_hay_timeout_avisa_en_que_tf_estaba(monkeypatch, telegram):
    async def lenta(simbolo, red=None, progreso=None):
        progreso["tf_actual"] = "D1"
        await asyncio.sleep(10)
    monkeypatch.setattr(rep, "descargar_todo", lenta)
    assert rep.correr(args_prod(), timeout=0.2) == 1
    texto = telegram[0][1]["text"]
    assert "timeout" in texto and "D1" in texto


def test_un_error_inesperado_tambien_avisa_y_oculta_el_token(monkeypatch, telegram):
    async def rota(*a, **k):
        raise ValueError("fallo en https://api.telegram.org/botTOK123/sendPhoto")
    monkeypatch.setattr(rep, "descargar_todo", rota)
    assert rep.correr(args_prod()) == 1
    texto = telegram[0][1]["text"]
    assert "inesperado" in texto and "ValueError" in texto
    assert "TOK123" not in texto and "***" in texto


def test_fallo_al_generar_el_png_avisa(monkeypatch, telegram):
    async def datos(*a, **k):
        return dfs_sinteticos()

    def sin_kaleido(self, *a, **k):
        raise RuntimeError("kaleido no instalado")
    monkeypatch.setattr(rep, "descargar_todo", datos)
    monkeypatch.setattr(go.Figure, "write_image", sin_kaleido)
    assert rep.correr(args_prod()) == 1
    assert [m for m, _ in telegram] == ["sendMessage"] and "PNG" in telegram[0][1]["text"]


def test_si_telegram_rechaza_el_reporte_intenta_avisar_y_sale_con_1(monkeypatch, telegram):
    async def datos(*a, **k):
        return dfs_sinteticos()
    monkeypatch.setattr(rep, "descargar_todo", datos)
    monkeypatch.setattr(go.Figure, "write_image", lambda self, ruta, *a, **k: open(ruta, "wb").write(b"x"))
    monkeypatch.setattr(rep, "enviar_telegram", lambda *a, **k: False)
    assert rep.correr(args_prod()) == 1
    assert "rechazó" in telegram[0][1]["text"]


def test_sin_credenciales_falla_con_1_y_no_intenta_enviar(monkeypatch, telegram):
    monkeypatch.delenv("TELEGRAM_TOKEN")
    assert rep.correr(args_prod()) == 1
    assert telegram == []


def test_dry_run_no_avisa_a_telegram_aunque_falle(monkeypatch, telegram):
    async def falla(*a, **k):
        raise D.ErrorDescarga("BOOM500", "M15", RuntimeError("x"))
    monkeypatch.setattr(rep, "descargar_todo", falla)
    assert rep.correr(args_prod("--dry-run")) == 1
    assert telegram == []


def test_camino_feliz_devuelve_0_y_no_manda_avisos_de_error(monkeypatch, telegram):
    async def datos(*a, **k):
        return dfs_sinteticos()
    monkeypatch.setattr(rep, "descargar_todo", datos)
    monkeypatch.setattr(go.Figure, "write_image", lambda self, ruta, *a, **k: open(ruta, "wb").write(b"x"))
    enviados = []
    monkeypatch.setattr(rep, "enviar_telegram", lambda *a, **k: enviados.append(a[3]) or True)
    assert rep.correr(args_prod()) == 0
    assert telegram == [] and len(enviados) == 1 and rep.FIN_MENSAJE in enviados[0]


# ---------------------------------------------------------------------------
# Caché de --dry-run
# ---------------------------------------------------------------------------
def test_cache_ida_y_vuelta_conserva_los_datos(tmp_path):
    datos = dfs_sinteticos()
    D.guardar_cache("BOOM500", datos, carpeta=str(tmp_path), ahora=1000.0)
    leido, edad = D.cargar_cache("BOOM500", max_edad_s=3600, carpeta=str(tmp_path), ahora=1600.0)
    assert edad == 600.0
    for tf in D.TFS:
        pd.testing.assert_frame_equal(leido[tf], datos[tf], check_freq=False)
        assert str(leido[tf].index.tz) == "UTC"


def test_cache_vencida_incompleta_o_inexistente_no_se_usa(tmp_path):
    carpeta = str(tmp_path)
    assert D.cargar_cache("BOOM500", carpeta=carpeta) is None                       # no existe
    D.guardar_cache("BOOM500", dfs_sinteticos(), carpeta=carpeta, ahora=0.0)
    assert D.cargar_cache("BOOM500", max_edad_s=3600, carpeta=carpeta, ahora=3599.0) is not None
    assert D.cargar_cache("BOOM500", max_edad_s=3600, carpeta=carpeta, ahora=3600.0) is None   # >= 1 h
    os.remove(os.path.join(carpeta, "BOOM500_H4.csv"))
    assert D.cargar_cache("BOOM500", max_edad_s=3600, carpeta=carpeta, ahora=10.0) is None     # incompleta
    with open(os.path.join(carpeta, "BOOM500_meta.json"), "w") as f:
        f.write("{no es json")
    assert D.cargar_cache("BOOM500", max_edad_s=3600, carpeta=carpeta, ahora=10.0) is None     # corrupta


def test_dry_run_guarda_la_descarga_y_la_reutiliza(monkeypatch, telegram, tmp_path):
    descargas = []

    async def datos(*a, **k):
        descargas.append(1)
        return dfs_sinteticos()
    monkeypatch.setattr(rep, "descargar_todo", datos)
    monkeypatch.setattr(go.Figure, "write_image", lambda self, ruta, *a, **k: open(ruta, "wb").write(b"x"))
    assert rep.correr(args_prod("--dry-run")) == 0
    assert len(descargas) == 1 and os.path.exists(tmp_path / ".cache_smc" / "BOOM500_meta.json")
    assert rep.correr(args_prod("--dry-run")) == 0
    assert len(descargas) == 1                                  # 2ª vez: sin descargar
    assert rep.correr(args_prod("--dry-run", "--sin-cache")) == 0
    assert len(descargas) == 2                                  # --sin-cache fuerza la descarga


def test_dry_run_descarta_la_cache_de_mas_de_1_hora(monkeypatch, telegram, tmp_path):
    descargas = []

    async def datos(*a, **k):
        descargas.append(1)
        return dfs_sinteticos()
    monkeypatch.setattr(rep, "descargar_todo", datos)
    monkeypatch.setattr(go.Figure, "write_image", lambda self, ruta, *a, **k: open(ruta, "wb").write(b"x"))
    rep.correr(args_prod("--dry-run"))
    meta = tmp_path / ".cache_smc" / "BOOM500_meta.json"
    m = json.loads(meta.read_text())
    m["guardado_en"] -= 3601
    meta.write_text(json.dumps(m))
    rep.correr(args_prod("--dry-run"))
    assert len(descargas) == 2


def test_produccion_nunca_lee_ni_escribe_la_cache(monkeypatch, telegram, tmp_path):
    D.guardar_cache("BOOM500", dfs_sinteticos())               # caché fresca disponible...
    descargas = []

    async def datos(*a, **k):
        descargas.append(1)
        return dfs_sinteticos()
    monkeypatch.setattr(rep, "descargar_todo", datos)
    monkeypatch.setattr(go.Figure, "write_image", lambda self, ruta, *a, **k: open(ruta, "wb").write(b"x"))
    monkeypatch.setattr(rep, "enviar_telegram", lambda *a, **k: True)
    antes = os.path.getmtime(tmp_path / ".cache_smc" / "BOOM500_meta.json")
    assert rep.correr(args_prod()) == 0
    assert len(descargas) == 1                                  # ...y aun así descarga (no es dry-run)
    assert os.path.getmtime(tmp_path / ".cache_smc" / "BOOM500_meta.json") == antes
