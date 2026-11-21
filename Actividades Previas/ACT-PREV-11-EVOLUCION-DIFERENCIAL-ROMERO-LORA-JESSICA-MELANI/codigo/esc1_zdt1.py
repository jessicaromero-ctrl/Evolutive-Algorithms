#!/usr/bin/env python3
"""
esc1_zdt1.py — Escenario 1: control de parámetros en MODE sobre ZDT1 (frente convexo).

Compara cuatro estrategias de control de F y CR con el mismo presupuesto (15 100 evaluaciones)
y diez semillas. Indicadores: hipervolumen (referencia (1.1, 1.1)) e IGD frente al frente
verdadero. Pruebas: Friedman sobre el hipervolumen final y Wilcoxon con corrección de Holm
frente a la estrategia fija.

Uso: python3 esc1_zdt1.py --salida ../resultados --figuras ../figuras
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from mode_core import (correr_todo, zdt_frente_ref, VARIANTES, ETIQUETAS, SEMILLAS, friedman_holm,
                       GENS, N_POP, EVALS)
from ga_comun import estilo_grafico, PURPURA, LILA, GRIS, PURPURA_OSC

PROBLEMA = "zdt1"
REF_HV = np.array([1.1, 1.1])


def resumen(runs):
    hv = np.array([[r["hv_final"] for r in runs[v]] for v in VARIANTES]).T
    ig = np.array([[r["igd_final"] for r in runs[v]] for v in VARIANTES]).T
    chi, p, comps = friedman_holm(hv, VARIANTES, "fija")
    chi_i, p_i, comps_i = friedman_holm(-ig, VARIANTES, "fija")
    return {"hv": hv, "igd": ig, "friedman_hv": [chi, p], "holm_hv": comps,
            "friedman_igd": [chi_i, p_i], "holm_igd": comps_i}


def figura(runs, res, ruta):
    estilo_grafico()
    colores = {"fija": GRIS, "dither": LILA, "jde": PURPURA, "shade": PURPURA_OSC}
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.9), layout="constrained",
                           gridspec_kw={"width_ratios": [1.1, 0.9, 1.1]})
    # (a) hipervolumen medio por generación
    for v in VARIANTES:
        H = np.array([r["traza"]["hv"] for r in runs[v]])
        g = np.array(runs[v][0]["traza"]["gen"])
        m = H.mean(0)
        s = H.std(0, ddof=1)
        ax[0].plot(g, m, color=colores[v], lw=1.6, label=ETIQUETAS[v])
        ax[0].fill_between(g, m - s, m + s, color=colores[v], alpha=0.12, lw=0)
    ax[0].set(xlabel="Generación", ylabel="Hipervolumen", title="(a) Convergencia")
    ax[0].legend(loc="lower right", fontsize=7)
    # (b) hipervolumen final
    datos = [res["hv"][:, j] for j in range(len(VARIANTES))]
    bp = ax[1].boxplot(datos, widths=0.55, patch_artist=True, medianprops={"color": PURPURA_OSC})
    for patch, v in zip(bp["boxes"], VARIANTES):
        patch.set_facecolor(colores[v])
    ax[1].set_xticks(range(1, 5), [ETIQUETAS[v] for v in VARIANTES], fontsize=7, rotation=20)
    ax[1].set(ylabel="Hipervolumen final", title="(b) Final")
    # (c) trayectorias de parámetros (medias sobre semillas)
    g = np.array(runs["jde"][0]["traza"]["gen"])
    F_j = np.mean([r["traza"]["F_media"] for r in runs["jde"]], axis=0)
    CR_j = np.mean([r["traza"]["CR_media"] for r in runs["jde"]], axis=0)
    MF_s = np.mean([np.array(r["traza"]["MF"]).mean(1) for r in runs["shade"]], axis=0)
    MCR_s = np.mean([np.array(r["traza"]["MCR"]).mean(1) for r in runs["shade"]], axis=0)
    ax[2].plot(g, F_j, color=PURPURA, lw=1.6, label="jDE: $\\bar F$")
    ax[2].plot(g, CR_j, color=PURPURA, lw=1.6, ls="--", label="jDE: $\\bar{CR}$")
    ax[2].plot(g, MF_s, color=GRIS, lw=1.2, label="SHADE: $M_F$")
    ax[2].plot(g, MCR_s, color=GRIS, lw=1.2, ls="--", label="SHADE: $M_{CR}$")
    ax[2].set(xlabel="Generación", ylabel="Valor medio", title="(c) Adaptación de parámetros",
              ylim=(0, 1.25), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[2].legend(fontsize=6, loc="upper center", ncol=2, frameon=False)
    fig.savefig(ruta)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="../resultados")
    ap.add_argument("--figuras", default="../figuras")
    a = ap.parse_args()
    R = zdt_frente_ref(PROBLEMA)
    runs = correr_todo(PROBLEMA, ref_hv=REF_HV, R_igd=R, registrar_cada=5)
    res = resumen(runs)
    out = {"problema": PROBLEMA, "n_pop": N_POP, "generaciones": GENS, "evaluaciones": EVALS,
           "semillas": SEMILLAS, "variantes": VARIANTES,
           "hv_final": res["hv"].tolist(), "igd_final": res["igd"].tolist(),
           "friedman_hv": res["friedman_hv"], "holm_hv": res["holm_hv"],
           "friedman_igd": res["friedman_igd"], "holm_igd": res["holm_igd"],
           "corridas": {v: [{k: r[k] for k in ["seed", "hv_final", "igd_final", "n_factibles", "traza"]}
                            for r in runs[v]] for v in VARIANTES},
           "frente_seed11": {v: runs[v][0]["frente"] for v in VARIANTES}}
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    with open(Path(a.salida) / "esc1_zdt1.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    figura(runs, res, Path(a.figuras) / "fig1_zdt1.png")
    for j, v in enumerate(VARIANTES):
        print(f"{v:6s} HV={res['hv'][:, j].mean():.4f}±{res['hv'][:, j].std(ddof=1):.4f} "
              f"IGD={res['igd'][:, j].mean():.5f}")
    print("Friedman HV", res["friedman_hv"], "Holm", res["holm_hv"])


if __name__ == "__main__":
    main()
