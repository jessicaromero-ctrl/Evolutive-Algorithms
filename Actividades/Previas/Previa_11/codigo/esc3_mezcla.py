#!/usr/bin/env python3
"""
esc3_mezcla.py — Escenario 3: MODE aplicada a la mezcla bi-objetivo de mineral.

Problema real de planta (datos ilustrativos del reporte minero): cinco frentes, dos objetivos
en conflicto y una restricción de calidad.
  f1 = −utilidad (USD/t), a minimizar; la utilidad es no lineal en la mezcla.
  f2 = impureza media (ppm), a minimizar.
  Restricción: ley media de zinc ≥ 6.0 %, tratada con dominancia restringida.

Como el frente verdadero no se conoce, el frente de referencia es la unión de los frentes no
dominados de las 40 ejecuciones (variantes × semillas). Los indicadores se calculan contra esa
unión, por lo que deben leerse como medidas relativas entre métodos, no absolutas.

Uso: python3 esc3_mezcla.py --salida ../resultados --figuras ../figuras
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mode_core import (correr_todo, nd_mascara, hipervolumen2d, igd, VARIANTES,
                       ETIQUETAS, SEMILLAS, friedman_holm, GENS, N_POP, EVALS)
from ga_comun import estilo_grafico, PURPURA, LILA, GRIS, PURPURA_OSC

PROBLEMA = "mezcla"
NOMBRES_K = ["Frente 1", "Frente 2", "Frente 3", "Frente 4", "Frente 5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="../resultados")
    ap.add_argument("--figuras", default="../figuras")
    a = ap.parse_args()
    runs = correr_todo(PROBLEMA, ref_hv=None, R_igd=None, registrar_cada=5)
    todos = np.vstack([np.array(r["frente"]) for v in VARIANTES for r in runs[v] if len(r["frente"])])
    m = nd_mascara(todos, np.zeros(len(todos)))
    R = np.unique(todos[m], axis=0)
    rango = R.max(0) - R.min(0)
    lo_ = R.min(0)
    norm = lambda F: (F - lo_) / rango          # normalización por la referencia
    Rn = norm(R)
    ref_hv = np.array([1.1, 1.1])
    hv = np.array([[hipervolumen2d(norm(np.array(r["frente"])), ref_hv) if len(r["frente"]) else 0.0
                    for r in runs[v]] for v in VARIANTES]).T
    ig = np.array([[igd(norm(np.array(r["frente"])), Rn) if len(r["frente"]) else np.inf
                    for r in runs[v]] for v in VARIANTES]).T
    chi, p, comps = friedman_holm(hv, VARIANTES, "fija")
    chi_i, p_i, comps_i = friedman_holm(-ig, VARIANTES, "fija")

    orden = np.argsort(R[:, 1])
    out = {"problema": PROBLEMA, "n_pop": N_POP, "generaciones": GENS, "evaluaciones": EVALS,
           "semillas": SEMILLAS, "variantes": VARIANTES,
           "frente_referencia": R[orden].tolist(), "ref_hv": [1.1, 1.1], "normalizacion": {"min": lo_.tolist(), "rango": rango.tolist()},
           "hv_final": hv.tolist(), "igd_final": ig.tolist(),
           "friedman_hv": [chi, p], "holm_hv": comps,
           "friedman_igd": [chi_i, p_i], "holm_igd": comps_i,
           "corridas": {v: [{"seed": r["seed"], "n_factibles": r["n_factibles"],
                             "frente": r["frente"]} for r in runs[v]] for v in VARIANTES}}
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    with open(Path(a.salida) / "esc3_mezcla.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)

    estilo_grafico()
    colores = {"fija": GRIS, "dither": LILA, "jde": PURPURA, "shade": PURPURA_OSC}
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.9), layout="constrained",
                           gridspec_kw={"width_ratios": [1.1, 0.9, 1.1]})
    ax[0].plot(R[orden, 1], -R[orden, 0], "-", color="#BDBDBD", lw=1.2, label="Referencia")
    marcas = {"fija": "s", "dither": "^", "jde": "o", "shade": "D"}
    for v in VARIANTES:
        F = np.array(runs[v][0]["frente"])
        ax[0].scatter(F[:, 1], -F[:, 0], s=8, marker=marcas[v], color=colores[v], label=ETIQUETAS[v], zorder=3)
    ax[0].set(xlabel="Impureza media (ppm)", ylabel="Utilidad (USD/t)",
              title="(a) Frentes (semilla 11)")
    ax[0].legend(fontsize=6, loc="upper left")
    datos = [hv[:, j] for j in range(len(VARIANTES))]
    bp = ax[1].boxplot(datos, widths=0.55, patch_artist=True, medianprops={"color": PURPURA_OSC})
    for patch, v in zip(bp["boxes"], VARIANTES):
        patch.set_facecolor(colores[v])
    ax[1].set_xticks(range(1, 5), [ETIQUETAS[v] for v in VARIANTES], fontsize=7, rotation=20)
    ax[1].set(ylabel="Hipervolumen final", title="(b) Hipervolumen")
    # (c) composición de la mezcla en cinco puntos del frente de referencia
    Fall = np.vstack([np.array(r["frente"]) for v in VARIANTES for r in runs[v] if len(r["frente"])])
    Xall = np.vstack([np.array(r["X_frente"]) for v in VARIANTES for r in runs[v] if len(r["frente"])])
    P_all = Xall / np.maximum(Xall.sum(1, keepdims=True), 1e-12)
    idx = np.linspace(0, len(R) - 1, 5).astype(int)
    comp = []
    for i_pt in idx:
        pt = R[orden][i_pt]
        d = np.sqrt((((Fall - pt) / np.maximum(rango, 1e-12)) ** 2).sum(1))
        comp.append(P_all[np.argmin(d)])
    comp = np.array(comp)
    puntos = [R[orden][i_pt] for i_pt in idx]
    bottom = np.zeros(len(puntos))
    cols = [PURPURA, "#A866E8", LILA, "#E3D0F7", GRIS]
    for k in range(5):
        ax[2].bar(range(1, len(puntos) + 1), comp[:, k], bottom=bottom, color=cols[k], width=0.6,
                  label=NOMBRES_K[k])
        bottom += comp[:, k]
    ax[2].set(xlabel="Punto del frente (de menor a mayor impureza)", ylabel="Proporción",
              ylim=(0, 1), xticks=range(1, len(puntos) + 1), title="(c) Mezcla en el frente")
    ax[2].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(Path(a.figuras) / "fig3_mezcla.png")
    plt.close(fig)

    for j, v in enumerate(VARIANTES):
        print(f"{v:6s} HV={hv[:, j].mean():.1f}±{hv[:, j].std(ddof=1):.1f} "
              f"IGD={ig[:, j].mean():.4f}±{ig[:, j].std(ddof=1):.4f} factibles="
              f"{np.mean([r['n_factibles'] for r in runs[v]]):.1f}")
    print("Referencia puntos:", len(R), "Friedman HV p=", p, "Friedman IGD p=", p_i)


if __name__ == "__main__":
    main()
