"""Laboratorio, paso 4: estadística de las zonas de BOOM500 frente a su control, con validación hacia adelante.

Uso:
    python laboratorio/lab_estadistica.py --simbolo BOOM500

Lee resultados_{simbolo}.csv y resultados_control_{simbolo}.csv (paso 3). Todo con TP a 2R salvo la tabla de
sensibilidad. Para cada trade real, su control emparejado son sus N_CONTROL entradas al azar (misma dirección,
mismo riesgo): diferencia del trade = r del trade - r medio de su control. La diferencia de R medio de un grupo
es el promedio de esas diferencias; su intervalo 95% sale de un bootstrap en bloques semanales (se remuestrean
semanas completas de fecha de entrada, para no tratar como independientes trades solapados en el tiempo).
Los grupos con n < MIN_N no se reportan. Validación hacia adelante: el 70% más antiguo de los trades (por fecha de
entrada) y el 30% más reciente. "Sostenido" = diferencia positiva con intervalo que excluye el cero en AMBOS tramos.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import smc_nucleo as smc  # noqa: E402
from lab_datos import DIR_DATOS  # noqa: E402
from lab_resultados import CONFIG_LAB  # noqa: E402

MIN_N = 30                 # n mínimo por grupo (y por lado en R1-R4)
FRACCION_ANTIGUA = 0.70    # tramo antiguo de la validación hacia adelante
ALFA = 0.05                # nivel de la prueba bilateral (IC 95%)
COLUMNAS = ["tabla", "grupo", "tp", "medida", "n", "acierto_pct", "r_zonas", "r_control", "dif", "ic_lo", "ic_hi",
            "sostenido"]


def preparar(reales, control):
    """Añade a cada trade su control emparejado, la semana de entrada y los grupos fijados de antemano."""
    t = reales.copy()
    t["fecha"] = pd.to_datetime(t["t_primer_toque"], utc=True)
    t["semana"] = (t["fecha"] - pd.to_timedelta(t["fecha"].dt.weekday, unit="D")).dt.normalize().astype(str)
    for n in CONFIG_LAB["TP_R"]:
        for medida in (f"r_tp{n}", f"r_peor_tp{n}"):
            t[f"ctrl_{medida}"] = t["zona_id"].map(control.groupby("zona_id")[medida].mean())
        ctrl_ac = control.assign(ac=(control[f"estado_tp{n}"] == "tp").astype(float)).groupby("zona_id")["ac"].mean()
        t[f"ctrl_ac_tp{n}"] = t["zona_id"].map(ctrl_ac)
        t[f"ac_tp{n}"] = (t[f"estado_tp{n}"] == "tp").astype(float)
    opuesto = np.where(t["direccion"] == "COMPRA", "bajista", "alcista")
    contra = (t["pt_sesgo_h1"] == opuesto) | (t["pt_sesgo_h4"] == opuesto)
    pos = pd.to_numeric(t["pt_pos_tr"], errors="coerce")           # 'fuera' / 'sin_tr' -> NaN
    correcto = ((t["direccion"] == "COMPRA") & (pos < 50)) | ((t["direccion"] == "VENTA") & (pos >= 50))
    incorrecto = ((t["direccion"] == "COMPRA") & (pos >= 50)) | ((t["direccion"] == "VENTA") & (pos < 50))
    p = t["pt_puntaje"]
    grupos = [("total", t["zona_id"].notna()), ("X/4 = 4", p == 4), ("X/4 >= 3", p >= 3),
              ("a favor sesgo H1/H4", ~contra), ("contra sesgo H1/H4", contra),
              ("lado correcto (PD)", correcto), ("lado incorrecto (PD)", incorrecto),
              ("Extremo", t["tipo"] == "Extremo"), ("Decisional", t["tipo"] == "Decisional")]
    pares_r = []
    for r in ("r1", "r2", "r3", "r4"):
        cumple = t[f"pt_{r}"] == 1
        pares_r.append((f"{r.upper()} cumple", cumple, f"{r.upper()} no cumple", ~cumple))
    return t, grupos, pares_r


def ic_bloques(valores, semanas, rng):
    """Media e IC 95% de la media con bootstrap por bloques semanales."""
    df = pd.DataFrame({"v": np.asarray(valores, float), "s": np.asarray(semanas)})
    por_semana = df.groupby("s")["v"].agg(["sum", "count"])
    sumas, cuentas = por_semana["sum"].to_numpy(), por_semana["count"].to_numpy()
    idx = rng.integers(0, len(sumas), size=(CONFIG_LAB["BOOTSTRAP"], len(sumas)))
    medias = sumas[idx].sum(axis=1) / cuentas[idx].sum(axis=1)
    lo, hi = np.percentile(medias, [100 * ALFA / 2, 100 * (1 - ALFA / 2)])
    return float(df["v"].mean()), float(lo), float(hi)


def fila_grupo(tabla, nombre, g, n_tp, medida, rng):
    """Resumen de un grupo de trades: n, % aciertos, R medio zonas/control y diferencia con su IC."""
    col = f"r_tp{n_tp}" if medida == "r" else f"r_peor_tp{n_tp}"
    dif, lo, hi = ic_bloques(g[col] - g[f"ctrl_{col}"], g["semana"], rng)
    return {"tabla": tabla, "grupo": nombre, "tp": n_tp, "medida": medida, "n": len(g),
            "acierto_pct": round(g[f"ac_tp{n_tp}"].mean() * 100, 1), "r_zonas": round(float(g[col].mean()), 4),
            "r_control": round(float(g[f"ctrl_{col}"].mean()), 4), "dif": round(dif, 4),
            "ic_lo": round(lo, 4), "ic_hi": round(hi, 4)}


def tabla_grupos(nombre_tabla, t, grupos, pares_r, rng, omitidos):
    """Filas de todos los grupos fijados de antemano con n >= MIN_N (R1-R4 solo si ambos lados lo cumplen)."""
    filas = []
    todos = list(grupos)
    for a, ma, b, mb in pares_r:
        na, nb = int(ma.sum()), int(mb.sum())
        if na >= MIN_N and nb >= MIN_N:
            todos += [(a, ma), (b, mb)]
        else:
            omitidos.setdefault(a.split()[0], (na, nb))
    for nombre, mascara in todos:
        g = t[mascara]
        if len(g) >= MIN_N:
            filas.append(fila_grupo(nombre_tabla, nombre, g, 2, "r", rng))
    return filas


def imprimir(filas, titulo):
    print(f"\n{titulo}")
    print(f"{'grupo':<24}{'n':>5} {'acierto':>8} {'R zonas':>9} {'R control':>10} {'dif R [IC95% bloques semanales]':>34} {'sostenido':>10}")
    for f in filas:
        marca = {True: "SÍ", False: "no", None: ""}[f.get("sostenido")]
        print(f"{f['grupo']:<24}{f['n']:>5} {f['acierto_pct']:>7.1f}% {f['r_zonas']:>+9.3f} {f['r_control']:>+10.3f} "
              f"{f['dif']:>+10.3f} [{f['ic_lo']:>+7.3f}, {f['ic_hi']:>+7.3f}] {marca:>10}")


def principal():
    ap = argparse.ArgumentParser(description="Laboratorio: estadística de zonas vs control con validación hacia adelante")
    ap.add_argument("--simbolo", default="BOOM500", choices=list(smc.CONFIG["ACTIVOS"]))
    args = ap.parse_args()
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    s = args.simbolo
    reales = pd.read_csv(os.path.join(DIR_DATOS, f"resultados_{s}.csv"))
    control = pd.read_csv(os.path.join(DIR_DATOS, f"resultados_control_{s}.csv"))
    t, grupos, pares_r = preparar(reales, control)
    corte = t["fecha"].quantile(FRACCION_ANTIGUA)
    tramos = [("completo", t), ("antiguo 70%", t[t["fecha"] <= corte]), ("reciente 30%", t[t["fecha"] > corte])]
    print(f"{s} | TP 2R | {len(t)} trades | corte 70/30 por fecha de entrada: {corte:%Y-%m-%d %H:%M} "
          f"(antiguo {int((t['fecha'] <= corte).sum())}, reciente {int((t['fecha'] > corte).sum())}) | n mínimo {MIN_N}")

    rng = np.random.default_rng(CONFIG_LAB["SEMILLA"])
    omitidos = {}
    resultado = {}
    for nombre, sub in tramos:
        om = {}
        pares = [(a, ma.reindex(sub.index), b, mb.reindex(sub.index)) for a, ma, b, mb in pares_r]
        resultado[nombre] = tabla_grupos(nombre, sub, [(n, m.reindex(sub.index)) for n, m in grupos], pares, rng, om)
        omitidos[nombre] = om

    # sostenido: diferencia > 0 con IC que excluye el cero en AMBOS tramos
    ant = {f["grupo"]: f for f in resultado["antiguo 70%"]}
    rec = {f["grupo"]: f for f in resultado["reciente 30%"]}
    comparables = [g for g in ant if g in rec]
    sostenidos = [g for g in comparables if ant[g]["dif"] > 0 and ant[g]["ic_lo"] > 0 and rec[g]["dif"] > 0 and rec[g]["ic_lo"] > 0]
    for nombre in resultado:
        for f in resultado[nombre]:
            f["sostenido"] = (f["grupo"] in sostenidos) if f["grupo"] in comparables else None
        imprimir(resultado[nombre], f"=== {nombre} ({len(dict(tramos)[nombre])} trades)")
    for nombre, om in omitidos.items():
        for r, (na, nb) in om.items():
            print(f"[{nombre}] {r} omitido: cumple n={na}, no cumple n={nb} (se exige n >= {MIN_N} en ambos lados)")

    m = len(comparables)
    esperado = m * (ALFA / 2) ** 2
    print(f"\nGrupos sostenidos: {len(sostenidos)} de {m} comparados en ambos tramos"
          f"{' (' + ', '.join(sostenidos) + ')' if sostenidos else ''}")
    print(f"Esperados por azar: {esperado:.3f} (cada tramo con diferencia positiva y IC 95% sin el cero tiene prob. "
          f"{ALFA / 2:.3f} si no hay efecto; ambos tramos: {(ALFA / 2) ** 2:.6f} x {m} comparaciones). "
          f"Referencia simple al 5%: {m * ALFA:.1f}. Los grupos se solapan, así que no son comparaciones independientes.")

    sens = []
    for n_tp in CONFIG_LAB["TP_R"]:
        for medida in ("r", "r_peor"):
            sens.append(fila_grupo("sensibilidad", "total", t, n_tp, medida, rng))
    print("\n=== Sensibilidad (solo total)")
    print(f"{'TP':>4} {'medida':<8}{'n':>5} {'acierto':>8} {'R zonas':>9} {'R control':>10} {'dif R [IC95% bloques semanales]':>34}")
    for f in sens:
        print(f"{f['tp']:>3}R {f['medida']:<8}{f['n']:>5} {f['acierto_pct']:>7.1f}% {f['r_zonas']:>+9.3f} {f['r_control']:>+10.3f} "
              f"{f['dif']:>+10.3f} [{f['ic_lo']:>+7.3f}, {f['ic_hi']:>+7.3f}]")

    salida = pd.DataFrame([f for nombre in resultado for f in resultado[nombre]] + sens, columns=COLUMNAS)
    salida.to_csv(os.path.join(DIR_DATOS, f"estadistica_{s}.csv"), index=False)
    print(f"\n💾 {os.path.join(DIR_DATOS, f'estadistica_{s}.csv')}")


if __name__ == "__main__":
    principal()
