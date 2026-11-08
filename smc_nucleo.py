"""Núcleo SMC: detección de estructura, OB, zonas, liquidez y sesgo.

Independiente de la red y del gráfico (solo numpy/pandas) para poder reutilizarlo
en una versión "laboratorio".

REGLA ANTI-FUTURO: todo se calcula en UN solo recorrido vela a vela (t = 0..N-1).
En la vela t solo se leen posiciones <= t. Un pivote de ventana n se confirma en
t = idx + n (nunca antes). Por eso los resultados sobre datos hasta t coinciden con
los de los datos completos para todo lo conocido en t (ver tests/test_smc_nucleo.py).
"""
import numpy as np
import pandas as pd

# =====================================================================
# CONFIG: todos los parámetros viven aquí
# =====================================================================
CONFIG = {
    # --- Estructura ---
    "N_PIVOTE": 4,            # ventana del pivote mayor (confirmado n velas después)
    "N_IRL": 2,               # ventana de los pivotes menores (IRL)
    "ATR_VENTANA": 14,        # ATR robusto = MEDIANA móvil del rango verdadero
    "EQ_TOL_ATR": 0.1,        # EQH/EQL: diferencia máxima entre pivotes, en ATR
    # --- Order Blocks ---
    "DESPLAZ_ATR": 1.0,       # cuerpo mínimo (en ATR) de la vela de desplazamiento
    "CUERPO_CLARO": 0.3,      # "vela clara": cuerpo >= 30% de su rango
    "MIN_SEP_OB_ATR": 0.5,    # separación mínima Extremo-Decisional, en ATR
    "R1_VENTANA": 10,         # R1: el LG debe estar dentro de las N velas anteriores al OB (incluida la del OB)
    "ENTRADA": "borde",       # "borde" (borde del OB) o "50" (mitad del OB)
    # --- Estados de zona ---
    "EDAD_MAX": {"M15": 1000, "H1": 500, "H4": 300},  # velas hasta "vencida"
    # --- Salida ---
    "MAX_ZONAS_FICHA": 5,     # zonas vivas más cercanas al precio que se muestran
    "MAX_FUERA_RANGO": 10,    # zonas H1/H4 vivas fuera del rango visible que se listan en el mensaje
    "MAX_LIBRES": 60,         # tope de pivotes sin barrer que se vigilan a la vez
    # --- Red y tiempos (los usan datos_smc.py y reporte_smc.py; el núcleo no los lee) ---
    "RED": {                          # perfil de PRODUCCIÓN (GitHub Actions, timeout-minutes: 5)
        "TIMEOUT_GLOBAL_S": 180,      # watchdog: a los 180 s el proceso se mata
        "MARGEN_TIMEOUT_S": 15,       # el timeout "amable" salta antes (180-15) para poder avisar a Telegram
        "TIMEOUT_PETICION_S": 30,     # tope duro de cada petición a Deriv
        "ESPERAS_S": (20, 40, 60),    # 3 reintentos, con estas esperas antes de cada uno
        "PAUSA_S": 3,                 # pausa entre peticiones
    },
    "RED_LARGO": {                    # --modo-largo: 8 reintentos (20 s x intento), para uso manual
        "TIMEOUT_GLOBAL_S": 540,
        "MARGEN_TIMEOUT_S": 15,
        "TIMEOUT_PETICION_S": 60,
        "ESPERAS_S": tuple(20 * i for i in range(1, 9)),
        "PAUSA_S": 3,
    },
    "CACHE_MAX_EDAD_S": 3600,         # solo --dry-run: reutiliza las velas si tienen menos de 1 hora
    # --- Capas opcionales (Fase 2/3; lógica aún no implementada) ---
    "CAPAS": {
        "fvg_ifc": False,             # Fase 2
        "eqh_eql_lineas": False,      # Fase 2
        "irl_erl_lineas": False,      # Fase 2
        "albercas_d1": False,         # Fase 2
        "inducement": False,          # Fase 2
        "smt": False,                 # Fase 2
        "trendline_liquidez": False,  # Fase 3
        "wick_poi": False,            # Fase 3
        "fvg_breaker": False,         # Fase 3
        "modelos_trampa": False,      # Fase 3
    },
}

ESTADOS_VIVOS = ("en_formacion", "fresca", "tocada")
ESTADOS_CANDIDATOS = ("fresca", "tocada")
NOMBRE_SESGO = {1: "alcista", -1: "bajista", 0: "rango"}


# =====================================================================
# Utilidades
# =====================================================================
def atr_mediana(h, l, c, ventana):
    """ATR robusto: mediana móvil (solo pasado) del rango verdadero."""
    pc = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(ventana, min_periods=1).median().to_numpy()


def posicion_en_tr(precio, tr):
    """Posición de un precio dentro del TR en % (0 = mínimo, 100 = máximo, 50 = equilibrio)."""
    if tr is None or tr["alto"] <= tr["bajo"]:
        return None
    return float((precio - tr["bajo"]) / (tr["alto"] - tr["bajo"]) * 100.0)


def _es_pivote(h, l, p, t, n):
    """¿La vela p es pivote de ventana n, conociéndose datos hasta t = p + n?
    Izquierda estricta, derecha no estricta (con empates gana el primero)."""
    alto = h[p] > h[p - n:p].max() and h[p] >= h[p + 1:t + 1].max()
    bajo = l[p] < l[p - n:p].min() and l[p] <= l[p + 1:t + 1].min()
    return alto, bajo


def _ultima_bajista(o, h, l, c, desde, hasta, cuerpo_min):
    """Última vela 'bajista clara' en [desde, hasta] (búsqueda hacia atrás)."""
    for i in range(hasta, desde - 1, -1):
        rango = h[i] - l[i]
        if rango > 0 and c[i] < o[i] and (o[i] - c[i]) >= cuerpo_min * rango:
            return i
    return None


def _primer_desplazamiento(o, c, desde, hasta, minimo):
    """Primera vela alcista con cuerpo >= minimo en [desde, hasta]; si no hay, 'hasta'."""
    for k in range(desde, hasta + 1):
        if c[k] - o[k] >= minimo:
            return k
    return hasta


# =====================================================================
# Creación de OB (se escribe para el caso compra; la venta es el espejo
# usando precios negados: o'=-o, h'=-l, l'=-h, c'=-c)
# =====================================================================
def _crear_obs(signo, o, h, l, c, s, t, atr_t, idx_hl, cfg):
    """Devuelve [Extremo] o [Extremo, Decisional] en coordenadas reales.

    s = índice del strong point (mínimo si compra / máximo si venta)
    t = vela del BOS/CDC;  idx_hl = pivotes (en espacio negado si venta) internos > s
    """
    if signo == 1:
        o_, h_, l_, c_ = o, h, l, c
    else:
        o_, h_, l_, c_ = -o, -l, -h, -c
    if s >= t:
        return []
    cuerpo_min = cfg["CUERPO_CLARO"]
    desplaz = cfg["DESPLAZ_ATR"] * atr_t

    # --- OB EXTREMO: última vela bajista antes del primer desplazamiento desde s
    k = _primer_desplazamiento(o_, c_, s + 1, t, desplaz)
    i_ext = _ultima_bajista(o_, h_, l_, c_, s, k - 1, cuerpo_min)
    if i_ext is None:
        # sin vela bajista clara: mecha del mínimo (wick POI)
        i_ext, poi = s, True
        z_bajo, z_alto = l_[s], min(o_[s], c_[s])
        cu_bajo, cu_alto = min(o_[s], c_[s]), max(o_[s], c_[s])
    else:
        poi = False
        z_bajo, z_alto = l_[i_ext], h_[i_ext]
        cu_bajo, cu_alto = c_[i_ext], o_[i_ext]
    crudas = [("Extremo", i_ext, z_bajo, z_alto, cu_bajo, cu_alto, poi)]

    # --- OB DECISIONAL: en el último HL interno, por encima del Extremo
    hl = [p for p in idx_hl if s < p < t]
    if hl:
        h_hl = hl[-1]
        k2 = _primer_desplazamiento(o_, c_, h_hl + 1, t, desplaz)
        i_dec = _ultima_bajista(o_, h_, l_, c_, h_hl, k2 - 1, cuerpo_min)
        if i_dec is not None and i_dec != i_ext:
            if l_[i_dec] - z_alto >= cfg["MIN_SEP_OB_ATR"] * atr_t:
                crudas.append(("Decisional", i_dec, l_[i_dec], h_[i_dec],
                               c_[i_dec], o_[i_dec], False))

    zonas = []
    for tipo, idx, zb, za, cb, ca, poi in crudas:
        if signo == 1:
            bajo, alto, cu_b, cu_a = zb, za, cb, ca
        else:
            bajo, alto, cu_b, cu_a = -za, -zb, -ca, -cb
        zonas.append({"tipo": tipo, "idx_ob": int(idx), "bajo": float(bajo), "alto": float(alto),
                      "cuerpo_bajo": float(cu_b), "cuerpo_alto": float(cu_a),
                      "mecha_poi": bool(poi)})
    return zonas


# =====================================================================
# Detector principal (una temporalidad)
# =====================================================================
def detectar(df, tf="M15", cfg=None):
    """Recorre las velas y devuelve estructura, zonas, liquidez y sesgo de UNA temporalidad."""
    cfg = {**CONFIG, **(cfg or {})}
    n, n2 = cfg["N_PIVOTE"], cfg["N_IRL"]
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    N = len(df)
    atr = atr_mediana(h, l, c, cfg["ATR_VENTANA"])
    edad_max = cfg["EDAD_MAX"].get(tf, 10**9)
    max_libres = cfg["MAX_LIBRES"]

    pivotes, eqs, irls = [], [], []
    libres = {"alto": [], "bajo": []}       # pivotes aún sin barrer ni romper
    pool_eq = {"alto": [], "bajo": []}      # pivotes no rotos por cierre (incluye barridos por LG): para EQH/EQL
    ult_precio = {"alto": None, "bajo": None}
    eventos, puntos, zonas, trs = [], [], [], []
    bias = 0
    fuerte = None                           # strong point vigente: {'idx','precio'}
    lg_desde = {"alto": False, "bajo": False}  # LG ocurrido desde el último BOS/CDC
    tr = None

    def registrar_ruptura(d, tipo, t, ref):
        """BOS/CDC alcista (d=+1) o bajista (d=-1) en la vela t."""
        nonlocal bias, fuerte, tr
        # tramo del impulso: desde el weak point roto (BOS) o desde el strong roto (flip/CDC)
        inicio = ref["idx"] if tipo == "BOS" and ref is not None else fuerte["idx"]
        if d == 1:
            s = inicio + int(np.argmin(l[inicio:t + 1]))
            precio_s = l[s]
            nombre_fuerte, nombre_debil = "strong_low", "weak_high"
        else:
            s = inicio + int(np.argmax(h[inicio:t + 1]))
            precio_s = h[s]
            nombre_fuerte, nombre_debil = "strong_high", "weak_low"
        nivel = ref["precio"] if ref is not None else fuerte["precio"]
        eventos.append({"tipo": tipo, "dir": d, "idx": t, "nivel": float(nivel),
                        "idx_nivel": int(ref["idx"] if ref is not None else fuerte["idx"])})
        puntos.append({"tipo": nombre_fuerte, "idx": int(s), "precio": float(precio_s), "conocido_en": t})
        if ref is not None and tipo == "BOS":
            puntos.append({"tipo": nombre_debil, "idx": int(ref["idx"]),
                           "precio": float(ref["precio"]), "conocido_en": t})

        # --- OBs (Extremo y, si corresponde, Decisional)
        if d == 1:
            idx_hl = [p["idx"] for p in pivotes if p["tipo"] == "bajo"]
        else:
            idx_hl = [p["idx"] for p in pivotes if p["tipo"] == "alto"]
        lado_lg = "bajo" if d == 1 else "alto"
        lgs_lado = [e["idx"] for e in eventos if e["tipo"] == "LG" and e["lado"] == lado_lg]
        for z in _crear_obs(d, o, h, l, c, s, t, atr[t], idx_hl, cfg):
            # R1: LG de un pivote confirmado (n=N_PIVOTE) o de EQH/EQL dentro de las R1_VENTANA velas
            # previas al OB (incluida su vela). Todo LG ya nace de un pivote confirmado/EQ (ver 'libres').
            r1 = any(z["idx_ob"] - cfg["R1_VENTANA"] <= i <= z["idx_ob"] for i in lgs_lado)
            medio = (z["alto"] + z["bajo"]) / 2.0
            entrada = medio if cfg["ENTRADA"] == "50" else (z["alto"] if d == 1 else z["bajo"])
            z.update({
                "id": f"{tf}-{'C' if d == 1 else 'V'}-{z['tipo'][:3]}-{z['idx_ob']}",
                "tf": tf, "dir": "COMPRA" if d == 1 else "VENTA", "signo": d,
                "idx_creacion": t, "evento": tipo, "atr": float(atr[t]),
                "medio": float(medio), "entrada": float(entrada),
                "idx_fuerte": int(s), "precio_fuerte": float(precio_s),
                "r1": bool(r1), "r2": True,  # R2: la zona nace de un BOS/CDC con cuerpo
                "r3": False, "r3_idx": None,
                "estado": "en_formacion", "n_toques": 0, "primer_toque": None,
                "ext_imp": float(h[t] if d == 1 else l[t]),
                "historial": [(t, "en_formacion")],
            })
            # R3 retroactiva: IRL formada entre la vela del OB y hoy (todo conocido en t)
            for irl in irls:
                if irl["idx"] > z["idx_ob"] and _irl_en_rango(irl["precio"], z):
                    z["r3"], z["r3_idx"] = True, irl["conocido_en"]
                    break
            zonas.append(z)

        # --- TR nuevo (el anterior se archiva; un CDC lo deja 'establecido')
        if tr is not None:
            if tipo == "CDC":
                tr["estado"] = "establecido"
            trs.append(tr)
        tr = {"dir": d, "idx_ini": int(s), "idx_bos": t, "estado": "en_formacion",
              "bajo": float(precio_s if d == 1 else l[t]),
              "alto": float(h[t] if d == 1 else precio_s)}
        bias = d
        fuerte = {"idx": int(s), "precio": float(precio_s)}
        lg_desde["alto"] = lg_desde["bajo"] = False

    def _irl_en_rango(precio, z):
        lo = min(z["precio_fuerte"], z["ext_imp"])
        hi = max(z["precio_fuerte"], z["ext_imp"])
        return lo < precio < hi      # estricto: los bordes del TR no son IRL (son ERL/strong)

    for t in range(N):
        # ---- 1. Pivotes mayores confirmados en t (pivote en p = t - n)
        p = t - n
        if p >= n:
            es_alto, es_bajo = _es_pivote(h, l, p, t, n)
            for tipo, activo, precio in (("alto", es_alto, h[p]),
                                         ("bajo", es_bajo, l[p])):
                if not activo:
                    continue
                prev = ult_precio[tipo]
                if prev is None:
                    etiqueta = ""            # el primer pivote no tiene con qué compararse
                elif tipo == "alto":
                    etiqueta = "HH" if precio > prev else "LH"
                else:
                    etiqueta = "LL" if precio < prev else "HL"
                ult_precio[tipo] = precio
                pivotes.append({"tipo": tipo, "idx": p, "precio": float(precio),
                                "conocido_en": t, "etiqueta": etiqueta})
                nuevo = {"idx": p, "precio": float(precio), "eq": False}
                for otro in pool_eq[tipo]:  # EQH/EQL: un 2º máximo apenas por encima también cuenta
                    if abs(otro["precio"] - precio) <= cfg["EQ_TOL_ATR"] * atr[t]:
                        otro["eq"] = nuevo["eq"] = True
                        eqs.append({"tipo": "EQH" if tipo == "alto" else "EQL",
                                    "idx1": otro["idx"], "idx2": p,
                                    "precio": float(max(precio, otro["precio"]) if tipo == "alto"
                                                    else min(precio, otro["precio"])),
                                    "conocido_en": t})
                libres[tipo].append(nuevo)
                pool_eq[tipo].append(nuevo)
                for lista in (libres[tipo], pool_eq[tipo]):
                    if len(lista) > max_libres:
                        lista.pop(0)

        # ---- 2. Pivotes menores (IRL) confirmados en t
        p2 = t - n2
        if p2 >= n2:
            a2, b2 = _es_pivote(h, l, p2, t, n2)
            for tipo, activo, precio in (("alto", a2, h[p2]), ("bajo", b2, l[p2])):
                if not activo:
                    continue
                irl = {"tipo": tipo, "idx": p2, "precio": float(precio), "conocido_en": t}
                irls.append(irl)
                for z in zonas:  # R3: formada entre el OB y su primer toque
                    if (not z["r3"] and z["estado"] in ESTADOS_VIVOS and p2 > z["idx_ob"]
                            and (z["primer_toque"] is None or p2 < z["primer_toque"])
                            and _irl_en_rango(precio, z)):
                        z["r3"], z["r3_idx"] = True, t

        # ---- 3. Liquidez barrida (LG) y rupturas sobre pivotes sin barrer
        # Weak point a romper = el pivote más extremo aún libre (los pivotes internos menores
        # no son BOS). Con sesgo, solo cuentan los formados después del strong point.
        desde = fuerte["idx"] if fuerte is not None else -1
        c_altos = [p for p in libres["alto"] if p["idx"] > desde]
        c_bajos = [p for p in libres["bajo"] if p["idx"] > desde]
        ref_alto = dict(max(c_altos, key=lambda p: (p["precio"], p["idx"]))) if c_altos else None
        ref_bajo = dict(min(c_bajos, key=lambda p: (p["precio"], -p["idx"]))) if c_bajos else None
        rompe_alto = ref_alto is not None and c[t] > ref_alto["precio"]
        rompe_bajo = ref_bajo is not None and c[t] < ref_bajo["precio"]
        for lado in ("alto", "bajo"):
            for pv in list(libres[lado]):
                if lado == "alto" and h[t] > pv["precio"]:
                    if c[t] <= pv["precio"]:       # mecha por encima, cierra dentro: LG
                        eventos.append({"tipo": "LG", "lado": "alto", "idx": t,
                                        "idx_nivel": pv["idx"], "nivel": pv["precio"], "eq": pv["eq"]})
                        lg_desde["alto"] = True
                    elif pv in pool_eq[lado]:
                        pool_eq[lado].remove(pv)   # roto por cierre: ya no cuenta para EQ
                    libres[lado].remove(pv)        # barrido o roto: ya no es liquidez libre
                elif lado == "bajo" and l[t] < pv["precio"]:
                    if c[t] >= pv["precio"]:
                        eventos.append({"tipo": "LG", "lado": "bajo", "idx": t,
                                        "idx_nivel": pv["idx"], "nivel": pv["precio"], "eq": pv["eq"]})
                        lg_desde["bajo"] = True
                    elif pv in pool_eq[lado]:
                        pool_eq[lado].remove(pv)
                    libres[lado].remove(pv)

        if bias == 1:
            if c[t] < fuerte["precio"]:            # ruptura del strong low
                registrar_ruptura(-1, "CDC" if lg_desde["alto"] else "BOS", t, None)
            elif rompe_alto:
                registrar_ruptura(1, "BOS", t, ref_alto)
        elif bias == -1:
            if c[t] > fuerte["precio"]:
                registrar_ruptura(1, "CDC" if lg_desde["bajo"] else "BOS", t, None)
            elif rompe_bajo:
                registrar_ruptura(-1, "BOS", t, ref_bajo)
        else:
            if rompe_alto:
                registrar_ruptura(1, "BOS", t, ref_alto)
            elif rompe_bajo:
                registrar_ruptura(-1, "BOS", t, ref_bajo)

        # ---- 4. El TR activo se extiende mientras haga nuevos extremos
        if tr is not None and tr["idx_bos"] < t:
            if tr["dir"] == 1 and h[t] > tr["alto"]:
                tr["alto"] = float(h[t])
            elif tr["dir"] == -1 and l[t] < tr["bajo"]:
                tr["bajo"] = float(l[t])

        # ---- 5. Máquina de estados de las zonas
        for z in zonas:
            if z["estado"] not in ESTADOS_VIVOS or t <= z["idx_creacion"]:
                continue
            if t - z["idx_creacion"] > edad_max:
                z["estado"] = "vencida"; z["historial"].append((t, "vencida")); continue
            compra = z["signo"] == 1
            if z["estado"] == "en_formacion":
                if compra and h[t] > z["ext_imp"]:
                    z["ext_imp"] = float(h[t]); continue
                if not compra and l[t] < z["ext_imp"]:
                    z["ext_imp"] = float(l[t]); continue
                z["estado"] = "fresca"; z["historial"].append((t, "fresca"))
            if compra:
                invalida = c[t] < z["bajo"]
                entra = l[t] <= z["alto"]
                previo_fuera = l[t - 1] > z["alto"]
                llega_50 = l[t] <= z["medio"]
            else:
                invalida = c[t] > z["alto"]
                entra = h[t] >= z["bajo"]
                previo_fuera = h[t - 1] < z["bajo"]
                llega_50 = h[t] >= z["medio"]
            if invalida:
                z["estado"] = "invalidada"; z["historial"].append((t, "invalidada")); continue
            if not entra:
                continue
            if previo_fuera:                      # toque = entrada tras >= 1 vela completa fuera
                z["n_toques"] += 1
                if z["primer_toque"] is None:
                    z["primer_toque"] = t
            if llega_50 or z["n_toques"] >= 2:
                z["estado"] = "mitigada"; z["historial"].append((t, "mitigada"))
            elif z["estado"] == "fresca":
                z["estado"] = "tocada"; z["historial"].append((t, "tocada"))

    # TR de trabajo = último TR ESTABLECIDO (cerrado por un CDC); el actual aún está en formación
    establecidos = [x for x in trs if x["estado"] == "establecido"]
    tr_trabajo = establecidos[-1] if establecidos else None
    tr_formacion = tr if tr is not None and tr["estado"] == "en_formacion" else None

    res = {
        "tf": tf, "index": df.index, "n_velas": N, "atr": atr, "cfg": cfg,
        "pivotes": pivotes, "eqs": eqs, "irls": irls, "eventos": eventos,
        "puntos": puntos, "zonas": zonas, "tr": tr, "trs": trs,
        "tr_trabajo": tr_trabajo, "tr_formacion": tr_formacion,
        "sesgo": NOMBRE_SESGO[bias], "sesgo_num": bias,
    }
    for z in zonas:
        z["puntaje"] = puntaje(z)
    return res


def puntaje(z):
    """Casillas R1-R4 cumplidas (R4 = no mitigada)."""
    return int(z["r1"]) + int(z["r2"]) + int(z["r3"]) + int(z["estado"] != "mitigada")


# =====================================================================
# Liquidez: TP1 (IRL a favor) y TP final (ERL), día UTC anterior
# =====================================================================
def _sin_barrer(precio, idx_desde, tipo, h, l):
    """True si ninguna mecha posterior a idx_desde superó el nivel."""
    if idx_desde + 1 >= len(h):
        return True
    return (h[idx_desde + 1:].max() < precio) if tipo == "alto" else (l[idx_desde + 1:].min() > precio)


def extremos_dia_anterior(df):
    """(máximo, mínimo, idx_inicio_del_día_actual) del día UTC anterior, o (None, None, None)."""
    dias = df.index.normalize()
    hoy = dias[-1]
    ayer = hoy - pd.Timedelta(days=1)
    m = np.asarray(dias == ayer)
    if not m.any():
        return None, None, None
    idx_hoy = int(np.argmax(np.asarray(dias == hoy)))
    return float(df["high"].to_numpy()[m].max()), float(df["low"].to_numpy()[m].min()), idx_hoy


def calcular_liquidez(res, df):
    """Rellena 'tp1' y 'tp_final' en cada zona y devuelve el resumen de liquidez."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    tr = res["tr"]
    pdh, pdl, idx_hoy = extremos_dia_anterior(df)

    # ERL: strong/weak points fuera del TR activo (+ máx/mín del día anterior)
    erl = []
    for pt in res["puntos"]:
        tipo = "alto" if pt["tipo"].endswith("high") else "bajo"
        fuera = tr is None or pt["precio"] > tr["alto"] or pt["precio"] < tr["bajo"]
        if fuera and _sin_barrer(pt["precio"], pt["idx"], tipo, h, l):
            erl.append({"nombre": pt["tipo"], "tipo": tipo, "precio": pt["precio"], "idx": pt["idx"]})
    if pdh is not None:
        if _sin_barrer(pdh, max(idx_hoy - 1, 0), "alto", h, l):
            erl.append({"nombre": "PDH", "tipo": "alto", "precio": pdh, "idx": idx_hoy})
        if _sin_barrer(pdl, max(idx_hoy - 1, 0), "bajo", h, l):
            erl.append({"nombre": "PDL", "tipo": "bajo", "precio": pdl, "idx": idx_hoy})

    # IRL a favor: pivotes menores dentro del TR activo, sin barrer
    irl_vivas = []
    if tr is not None:
        for irl in res["irls"]:
            if tr["bajo"] <= irl["precio"] <= tr["alto"] and _sin_barrer(irl["precio"], irl["idx"], irl["tipo"], h, l):
                irl_vivas.append(irl)

    for z in res["zonas"]:
        if z["signo"] == 1:
            ref = z["alto"]
            tp1 = [x["precio"] for x in irl_vivas if x["tipo"] == "alto" and x["precio"] > ref]
            tpf = [x["precio"] for x in erl if x["tipo"] == "alto" and x["precio"] > ref]
            z["tp1"] = min(tp1) if tp1 else None
            z["tp_final"] = min(tpf) if tpf else None
        else:
            ref = z["bajo"]
            tp1 = [x["precio"] for x in irl_vivas if x["tipo"] == "bajo" and x["precio"] < ref]
            tpf = [x["precio"] for x in erl if x["tipo"] == "bajo" and x["precio"] < ref]
            z["tp1"] = max(tp1) if tp1 else None
            z["tp_final"] = max(tpf) if tpf else None
    return {"erl": erl, "irl": irl_vivas, "pdh": pdh, "pdl": pdl}


# =====================================================================
# Multi-temporalidad y selección de candidatas
# =====================================================================
def analizar(dfs, cfg=None):
    """Corre el detector sobre M15/H1/H4. Las zonas se toman de M15; H1/H4 dan sesgo y franjas."""
    res = {tf: detectar(dfs[tf], tf, cfg) for tf in ("M15", "H1", "H4") if tf in dfs}
    m15 = res["M15"]
    m15["liquidez"] = calcular_liquidez(m15, dfs["M15"])
    for z in m15["zonas"]:
        z["sesgo_h1"] = res["H1"]["sesgo"] if "H1" in res else "rango"
        z["sesgo_h4"] = res["H4"]["sesgo"] if "H4" in res else "rango"
        z["contra"] = contra_sesgo(z)
        z["pos_tr"] = posicion_en_tr(z["medio"], m15["tr_trabajo"])
    return res


def contra_sesgo(z):
    """Temporalidades (H1/H4) cuyo sesgo contradice la dirección de la zona ('rango' no contradice)."""
    opuesto = "bajista" if z["signo"] == 1 else "alcista"
    return [tf for tf, clave in (("H1", "sesgo_h1"), ("H4", "sesgo_h4")) if z[clave] == opuesto]


def clasificar_pd(pos):
    """Texto de premium/descuento de una posición % dentro del TR de trabajo."""
    if pos is None:
        return "sin TR de trabajo"
    if pos < 0 or pos > 100:
        return "fuera del TR de trabajo"
    return f"{'descuento' if pos < 50 else 'premium'} {pos:.0f}% del TR"


def candidatas(res_tf, precio, k=None):
    """Zonas fresca/tocada, ordenadas por cercanía al precio actual (máx. k)."""
    k = k or res_tf["cfg"]["MAX_ZONAS_FICHA"]
    vivas = [z for z in res_tf["zonas"] if z["estado"] in ESTADOS_CANDIDATOS]

    def dist(z):
        if z["bajo"] <= precio <= z["alto"]:
            return 0.0
        return min(abs(precio - z["alto"]), abs(precio - z["bajo"]))
    return sorted(vivas, key=dist)[:k]


def _distancia(z, precio):
    if z["bajo"] <= precio <= z["alto"]:
        return 0.0
    return min(abs(precio - z["alto"]), abs(precio - z["bajo"]))


def ordenar_fichas(zonas, precio):
    """A favor del sesgo H1/H4 primero; dentro de cada grupo, las más cercanas al precio."""
    return sorted(zonas, key=lambda z: (bool(z["contra"]), _distancia(z, precio)))


def zonas_fuera_de_rango(res, precio, y_min, y_max, tfs=("H4", "H1"), k=None):
    """Zonas H1/H4 vivas (fresca/tocada) totalmente fuera de [y_min, y_max].

    Devuelve dicts con la distancia % al precio (borde más cercano de la zona; + arriba, - abajo),
    ordenadas por cercanía y limitadas a k (CONFIG["MAX_FUERA_RANGO"])."""
    k = k or CONFIG["MAX_FUERA_RANGO"]
    fuera = []
    for tf in tfs:
        if tf not in res:
            continue
        for z in candidatas(res[tf], precio, k=10**6):
            if z["bajo"] > y_max:
                arriba, borde = True, z["bajo"]
            elif z["alto"] < y_min:
                arriba, borde = False, z["alto"]
            else:
                continue
            fuera.append({"tf": tf, "tipo": z["tipo"], "dir": z["dir"], "bajo": z["bajo"],
                          "alto": z["alto"], "arriba": arriba,
                          "dist_pct": (borde - precio) / precio * 100.0})
    return sorted(fuera, key=lambda f: abs(f["dist_pct"]))[:k]
