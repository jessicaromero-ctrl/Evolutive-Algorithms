#!/usr/bin/env python3
"""
esc2_zdt3.py — Escenario 2: control de parámetros en MODE sobre ZDT3 (frente discontinuo).

ZDT3 tiene un frente formado por varios segmentos desconectados. Un método que se estanca en
un segmento obtiene un hipervolumen razonable pero una IGD alta. Por eso este escenario usa la
IGD como indicador principal y muestra los frentes finales frente al verdadero.

Uso: python3 esc2_zdt3.py --salida ../resultados --figuras ../figuras
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mode_core import (correr_todo, zdt_frente_ref, VARIANTES, ETIQUETAS, SEMILLAS, friedman_holm,
                       GENS, N_POP, EVALS)
from ga_comun import estilo_grafico, PURPURA, LILA, GRIS, PURPURA_OSC

PROBLEMA = "zdt3"
REF_HV = np.array([1.1, 1.1])


def figura(runs, R, res, ruta):
    estilo_grafico()
    colores = {"fija": GRIS, "dither": LILA, "jde": PURPURA, "shade": PURPURA_OSC}
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.9), layout="constrained",
                           gridspec_kw={"width_ratios": [1.2, 0.9, 1.1]})
    # (a) frentes finales (semilla 11) frente al verdadero
    ax[0].plot(R[:, 0], R[:, 1], "o", ms=1.2, color="#BDBDBD", label="Frente verdadero")
    marcas = {"fija": "s", "dither": "^", "jde": "o", "shade": "D"}
    for v in VARIANTES:
        F = np.array(runs[v][0]["frente"])
        ax[0].scatter(F[:, 0], F[:, 1], s=7, marker=marcas[v], color=colores[v], alpha=0.85,
                      label=ETIQUETAS[v], zorder=3)
    ax[0].set(xlabel="$f_1$", ylabel="$f_2$", title="(a) Frentes finales (semilla 11)")
    ax[0].legend(fontsize=6, loc="upper right")
    # (b) IGD final
    datos = [res["ig"][:, j] for j in range(len(VARIANTES))]
    bp = ax[1].boxplot(datos, widths=0.55, patch_artist=True, medianprops={"color": PURPURA_OSC})
    for patch, v in zip(bp["boxes"], VARIANTES):
        patch.set_facecolor(colores[v])
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(1, 5), [ETIQUETAS[v] for v in VARIANTES], fontsize=7, rotation=20)
    ax[1].set(ylabel="IGD final (escala log.)", title="(b) IGD final")
    # (c) convergencia de IGD
    for v in VARIANTES:
        I = np.array([r["traza"]["igd"] for r in runs[v]])
        g = np.array(runs[v][0]["traza"]["gen"])
        ax[2].plot(g[1:], I.mean(0)[1:], color=colores[v], lw=1.5, label=ETIQUETAS[v])
    ax[2].set_yscale("log")
    ax[2].set(xlabel="Generación", ylabel="IGD medio (log.)", title="(c) Convergencia de IGD")
    ax[2].legend(fontsize=7, loc="upper right")
    fig.savefig(ruta)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="../resultados")
    ap.add_argument("--figuras", default="../figuras")
    a = ap.parse_args()
    R = zdt_frente_ref(PROBLEMA)
    runs = correr_todo(PROBLEMA, ref_hv=REF_HV, R_igd=R, registrar_cada=5)
    hv = np.array([[r["hv_final"] for r in runs[v]] for v in VARIANTES]).T
    ig = np.array([[r["igd_final"] for r in runs[v]] for v in VARIANTES]).T
    chi, p, comps = friedman_holm(-ig, VARIANTES, "fija")
    chi_h, p_h, comps_h = friedman_holm(hv, VARIANTES, "fija")
    res = {"ig": ig, "hv": hv}
    out = {"problema": PROBLEMA, "n_pop": N_POP, "generaciones": GENS, "evaluaciones": EVALS,
           "semillas": SEMILLAS, "variantes": VARIANTES,
           "igd_final": ig.tolist(), "hv_final": hv.tolist(),
           "friedman_igd": [chi, p], "holm_igd": comps,
           "friedman_hv": [chi_h, p_h], "holm_hv": comps_h,
           "corridas": {v: [{k: r[k] for k in ["seed", "hv_final", "igd_final", "n_factibles", "traza"]}
                            for r in runs[v]] for v in VARIANTES},
           "frente_seed11": {v: runs[v][0]["frente"] for v in VARIANTES},
           "frente_verdadero": R.tolist()}
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    with open(Path(a.salida) / "esc2_zdt3.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    figura(runs, R, res, Path(a.figuras) / "fig2_zdt3.png")
    for j, v in enumerate(VARIANTES):
        print(f"{v:6s} IGD={ig[:, j].mean():.5f}±{ig[:, j].std(ddof=1):.5f} HV={hv[:, j].mean():.4f}")
    print("Friedman IGD", out["friedman_igd"], "Holm", comps)


if __name__ == "__main__":
    main()
