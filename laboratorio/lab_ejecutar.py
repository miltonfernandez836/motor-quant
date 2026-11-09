"""Laboratorio, paso 6B: ejecuta toda la cadena en orden (para lanzarla sola, p. ej. al iniciar sesión).

Uso:
    python laboratorio/lab_ejecutar.py [--simbolo BOOM500] [--forzar]

Orden: lab_datos -> lab_replay (reanuda donde quedó) -> lab_resultados -> lab_estadistica. Cada paso es un proceso
aparte. Todo queda registrado en laboratorio/datos/lab_log.txt. Si un paso falla, se detiene y lo registra (con las
últimas líneas de su salida). Si la cadena ya terminó bien hoy (fecha local), se salta, salvo con --forzar.

Re-anclaje del replay: lab_replay guarda su avance como POSICIÓN en la tabla de velas M15 y lab_datos descarta cada día
las velas de más de 12 meses, así que las posiciones se desplazan. Antes de lab_datos se anota el INSTANTE de la
última pasada del replay y después se reescribe su posición en el archivo de avance, para que reanude sin saltarse horas.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

LAB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(LAB))
import smc_nucleo as smc  # noqa: E402
from lab_datos import DIR_DATOS, cargar  # noqa: E402
from lab_replay import ruta_estado  # noqa: E402

PASOS = ("lab_datos.py", "lab_replay.py", "lab_resultados.py", "lab_estadistica.py")
RUTA_LOG = os.path.join(DIR_DATOS, "lab_log.txt")
LINEAS_DE_ERROR = 15


def registrar(ruta_log, simbolo, texto):
    linea = f"{datetime.now():%Y-%m-%d %H:%M:%S} [{simbolo}] {texto}"
    os.makedirs(os.path.dirname(ruta_log), exist_ok=True)
    with open(ruta_log, "a", encoding="utf-8") as f:
        f.write(linea + "\n")
    print(linea, flush=True)


def ya_corrio_hoy(ruta_log, simbolo):
    """True si el registro tiene una cadena COMPLETA de este activo con la fecha de hoy."""
    if not os.path.exists(ruta_log):
        return False
    marca = f"{datetime.now():%Y-%m-%d}"
    with open(ruta_log, encoding="utf-8") as f:
        return any(l.startswith(marca) and f"[{simbolo}] COMPLETO" in l for l in f)


def instante_del_replay(simbolo):
    """Instante (apertura de la última vela procesada) guardado por lab_replay, o None si aún no hay avance."""
    if not os.path.exists(ruta_estado(simbolo)):
        return None
    with open(ruta_estado(simbolo), encoding="utf-8") as f:
        k = json.load(f)["k"]
    m15 = cargar(simbolo)
    if not 0 <= k < len(m15):
        raise RuntimeError(f"el avance del replay (posición {k}) no cabe en la caché M15 ({len(m15)} velas)")
    return m15.index[k]


def reanclar_replay(simbolo, instante):
    """Reescribe en el archivo de avance la posición del instante guardado en la caché M15 actualizada."""
    if instante is None:
        return "sin avance previo del replay: no hay nada que reanclar"
    m15 = cargar(simbolo)
    if instante not in m15.index:
        raise RuntimeError(f"el instante {instante} del replay ya no está en la caché M15 (¿hace más de 12 meses?)")
    nuevo = int(m15.index.get_loc(instante))
    with open(ruta_estado(simbolo), encoding="utf-8") as f:
        estado = json.load(f)
    anterior = estado["k"]
    estado["k"] = nuevo
    tmp = ruta_estado(simbolo) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(estado, f)
    os.replace(tmp, ruta_estado(simbolo))
    return f"replay reanclado en {instante}: posición {anterior} -> {nuevo}"


def ejecutar(simbolo, forzar=False, pasos=PASOS, ruta_log=RUTA_LOG):
    """Corre la cadena. Devuelve 0 si terminó bien (o se saltó) y 1 si algún paso falló."""
    if not forzar and ya_corrio_hoy(ruta_log, simbolo):
        registrar(ruta_log, simbolo, "SALTADO: la cadena ya terminó bien hoy (usa --forzar para repetirla)")
        return 0
    t0 = time.time()
    registrar(ruta_log, simbolo, f"INICIO (pasos: {', '.join(pasos)})")
    try:
        instante = instante_del_replay(simbolo)
    except Exception as e:
        registrar(ruta_log, simbolo, f"ABORTADO antes de empezar: {e}")
        return 1
    for paso in pasos:
        t1 = time.time()
        ruta = paso if os.path.isabs(paso) else os.path.join(LAB, paso)
        registrar(ruta_log, simbolo, f"{paso}: inicio")
        r = subprocess.run([sys.executable, ruta, "--simbolo", simbolo], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        salida = [l for l in (r.stdout + "\n" + r.stderr).splitlines() if l.strip()]
        if r.returncode != 0:
            registrar(ruta_log, simbolo, f"{paso}: FALLÓ (código {r.returncode}, {time.time() - t1:.0f} s)")
            for l in salida[-LINEAS_DE_ERROR:]:
                registrar(ruta_log, simbolo, f"    | {l}")
            registrar(ruta_log, simbolo, f"ABORTADO en {paso}: los pasos siguientes no se ejecutaron")
            return 1
        registrar(ruta_log, simbolo, f"{paso}: OK ({time.time() - t1:.0f} s) | {salida[-1] if salida else ''}")
        if os.path.basename(paso) == "lab_datos.py":        # la caché pudo desplazarse: reanclar el avance del replay
            try:
                registrar(ruta_log, simbolo, reanclar_replay(simbolo, instante))
            except Exception as e:
                registrar(ruta_log, simbolo, f"ABORTADO: no se pudo reanclar el replay: {e}")
                return 1
    registrar(ruta_log, simbolo, f"COMPLETO ({time.time() - t0:.0f} s)")
    return 0


def principal():
    ap = argparse.ArgumentParser(description="Laboratorio: ejecuta datos -> replay -> resultados -> estadística")
    ap.add_argument("--simbolo", default="BOOM500", choices=list(smc.CONFIG["ACTIVOS"]))
    ap.add_argument("--forzar", action="store_true", help="repite aunque ya haya terminado bien hoy")
    args = ap.parse_args()
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(ejecutar(args.simbolo, args.forzar))


if __name__ == "__main__":
    principal()
