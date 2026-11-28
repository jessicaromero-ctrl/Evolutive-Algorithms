"""Evalúa las 960 configuraciones del espacio de decisión con entrenamiento y validación cruzada reales.
Escribe resultados/tabla_exhaustiva.csv (base de verdad del espacio completo y del frente de Pareto exacto)."""
import time
from pathlib import Path
import pandas as pd
from moea_hiper import espacio_completo, entrenar_y_evaluar, decodificar, NOMBRES_OBJ

Path("resultados").mkdir(exist_ok=True)
inicio = time.perf_counter(); filas = []
for k, g in enumerate(espacio_completo()):
    f1, f2, f3, acc = entrenar_y_evaluar(g)
    hp = decodificar(g)
    filas.append({"g0": g[0], "g1": g[1], "g2": g[2], "g3": g[3], "g4": g[4], "g5": g[5],
                  "capas": "-".join(map(str, hp["capas"])), "lr": hp["lr"], "alpha": hp["alpha"], "batch": hp["batch"],
                  "exactitud": acc, "error": f1, "mflops": f2, "parametros": int(f3)})
    if (k + 1) % 120 == 0:
        print(k + 1, "de 960 en", round(time.perf_counter() - inicio), "s", flush=True)
pd.DataFrame(filas).to_csv("resultados/tabla_exhaustiva.csv", index=False, float_format="%.10g")
print("tabla escrita:", len(filas), "configuraciones en", round(time.perf_counter() - inicio), "s", flush=True)
