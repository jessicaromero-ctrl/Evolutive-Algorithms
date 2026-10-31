#!/usr/bin/env python3
"""
esc1_mezcla.py — Escenario 1: mezcla óptima de mineral de cinco frentes hacia una planta
de flotación polimetálica (zinc como metal objetivo).

Decisión: proporciones p_k de mineral de cada frente, con p en el símplex.
Modelo lineal (referencia exacta): recuperación constante R0, utilidad lineal en p.
  max  Σ_k p_k (P·z_k/100·R0 − c_k − C_proc)
  s.a. Σ_k p_k = 1,  Σ_k z_k p_k ≥ Z_MIN,  Σ_k a_k p_k ≤ A_MAX,  p ≥ 0
Modelo no lineal: la recuperación depende de la ley y de la impureza de la mezcla,
  r(z̄, ā) = R_MAX · z̄/(z̄ + K_Z) · 1/(1 + ā/A_REF),
  con z̄ = Σ p_k z_k y ā = Σ p_k a_k. La utilidad deja de ser lineal y se resuelve con AG,
  evolución diferencial (DE) y búsqueda aleatoria con igual presupuesto.

Los parámetros son ilustrativos, de orden de magnitud, y no corresponden a una operación real.
Uso: python3 esc1_mezcla.py --salida ../resultados --figuras ../figuras
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, differential_evolution
import matplotlib.pyplot as plt

from ga_comun import ga_real, busqueda_real, estilo_grafico, PURPURA, LILA, GRIS, PURPURA_OSC, EVALS

SEMILLAS = [11, 23, 37, 53, 71]
Z_K = np.array([9.5, 7.2, 5.8, 11.4, 4.1])        # ley de zinc (%)
C_K = np.array([18.0, 15.0, 14.0, 24.0, 12.0])    # costo de extracción y acarreo (USD/t)
A_K = np.array([900.0, 300.0, 150.0, 1200.0, 80.0])  # impureza equivalente (ppm)
NOMBRES_K = ["Frente 1", "Frente 2", "Frente 3", "Frente 4", "Frente 5"]
PRECIO = 2800.0          # USD por tonelada de zinc metálico (referencia ilustrativa)
C_PROC = 6.0             # costo de procesamiento (USD/t de alimentación)
R0 = 0.85                # recuperación constante del modelo lineal
R_MAX, K_Z, A_REF = 0.92, 1.5, 800.0
Z_MIN, A_MAX = 6.0, 400.0


def pesos(U):
    U = np.clip(U, 0.0, None)
    s = U.sum(axis=1, keepdims=True)
    return np.where(s > 1e-12, U / np.maximum(s, 1e-12), 1.0 / U.shape[1])


def utilidad(P, lineal):
    """P: (n, 5) proporciones en el símplex. Devuelve utilidad (USD/t) con penalizaciones."""
    z = P @ Z_K
    a = P @ A_K
    c = P @ C_K
    if lineal:
        bruta = PRECIO * (z / 100.0) * R0
    else:
        r = R_MAX * (z / (z + K_Z)) / (1.0 + a / A_REF)
        bruta = PRECIO * (z / 100.0) * r
    pen = 500.0 * np.maximum(0.0, Z_MIN - z) + 0.5 * np.maximum(0.0, a - A_MAX)
    return bruta - c - C_PROC - pen


def utilidad_real(p, lineal):
    """Utilidad sin penalizar, para reportar (solo si la mezcla es factible)."""
    z = float(p @ Z_K)
    a = float(p @ A_K)
    c = float(p @ C_K)
    r = R0 if lineal else R_MAX * (z / (z + K_Z)) / (1.0 + a / A_REF)
    return PRECIO * (z / 100.0) * r - c - C_PROC, z, a, bool(z >= Z_MIN - 1e-9 and a <= A_MAX + 1e-9)


def lp_exacto():
    """Solución exacta del modelo lineal por programación lineal (HiGHS)."""
    coef = PRECIO * Z_K / 100.0 * R0 - C_K - C_PROC
    res = linprog(-coef,
                  A_ub=np.vstack([-Z_K, A_K]), b_ub=np.array([-Z_MIN, A_MAX]),
                  A_eq=np.ones((1, 5)), b_eq=[1.0],
                  bounds=[(0, 1)] * 5, method="highs")
    return res.x, float(-res.fun)


def ejecutar(seed):
    out = {"semilla": seed}
    # Modelo lineal: AG frente a la solución exacta
    ga_l = ga_real(lambda U: utilidad(pesos(U), True), np.zeros(5), np.ones(5), seed)
    p_lin = pesos(ga_l["x"][None])[0]
    out["lineal"] = {"ga_utilidad": utilidad_real(p_lin, True)[0], "ga_p": p_lin.tolist(),
                     "hist": ga_l["hist"].tolist()}
    # Modelo no lineal: AG, DE y búsqueda aleatoria
    f = lambda U: utilidad(pesos(U), False)
    ga = ga_real(f, np.zeros(5), np.ones(5), seed)
    rnd = busqueda_real(f, np.zeros(5), np.ones(5), seed)
    de = differential_evolution(lambda u: -f(u[None])[0], [(0, 1)] * 5, seed=seed, tol=1e-10,
                                maxiter=400, polish=False)
    p_ga = pesos(ga["x"][None])[0]
    p_rnd = pesos(rnd["x"][None])[0]
    p_de = pesos(de.x[None])[0]
    out["no_lineal"] = {
        "ga": utilidad_real(p_ga, False)[:2] + (p_ga.tolist(),),
        "aleatoria": utilidad_real(p_rnd, False)[:2] + (p_rnd.tolist(),),
        "de": utilidad_real(p_de, False)[:2] + (p_de.tolist(),),
        "de_evals": int(de.nfev),
        "hist": ga["hist"].tolist(),
        "lp_en_no_lineal": utilidad_real(LP_P, False)[:2] + (LP_P.tolist(),),
        "mejor_pura": float(max(utilidad_real(np.eye(5)[k], False)[0] for k in range(5))),
    }
    return out


def figura(runs, ruta):
    estilo_grafico()
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.9), layout="constrained",
                           gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    H = np.array([r["lineal"]["hist"] for r in runs])
    gen = np.arange(H.shape[1])
    for h in H:
        ax[0].plot(gen, h, color=LILA, lw=0.8)
    ax[0].axhline(LP_VAL, color=GRIS, ls="--", lw=1.0, label="Óptimo LP")
    ax[0].plot(gen, H.mean(0), color=PURPURA, lw=1.8, label="Media AG")
    ax[0].set(xlabel="Generación", ylabel="Utilidad (USD/t)", title="(a) Modelo lineal")
    ax[0].legend(loc="lower right")
    nombres = ["AG", "DE", "Aleat", "Pura", "LP"]
    datos = [[r["no_lineal"]["ga"][0] for r in runs], [r["no_lineal"]["de"][0] for r in runs],
             [r["no_lineal"]["aleatoria"][0] for r in runs], [r["no_lineal"]["mejor_pura"] for r in runs],
             [r["no_lineal"]["lp_en_no_lineal"][0] for r in runs]]
    bp = ax[1].boxplot(datos, widths=0.55, patch_artist=True, medianprops={"color": PURPURA_OSC})
    for patch, c in zip(bp["boxes"], [PURPURA, "#A866E8", LILA, LILA, GRIS]):
        patch.set_facecolor(c)
    ax[1].set_xticks(range(1, 6), nombres)
    ax[1].set(ylabel="Utilidad no lineal (USD/t)", title="(b) Modelo no lineal")
    ax[1].tick_params(axis="x", labelsize=7)
    W = np.array([r["no_lineal"]["ga"][2] for r in runs])
    bottom = np.zeros(len(runs))
    cols = [PURPURA, "#A866E8", LILA, "#E3D0F7", GRIS]
    for k in range(5):
        ax[2].bar(range(1, len(runs) + 1), W[:, k], bottom=bottom, color=cols[k], width=0.6,
                  label=NOMBRES_K[k])
        bottom += W[:, k]
    ax[2].set(xlabel="Semilla (índice)", ylabel="Proporción en la mezcla", ylim=(0, 1),
              title="(c) Mezcla del AG")
    ax[2].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(ruta)
    plt.close(fig)


LP_P, LP_VAL = lp_exacto()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="../resultados")
    ap.add_argument("--figuras", default="../figuras")
    a = ap.parse_args()
    runs = [ejecutar(s) for s in SEMILLAS]
    res = {"LP": {"p": LP_P.tolist(), "utilidad": LP_VAL},
           "parametros": {"Z_K": Z_K.tolist(), "C_K": C_K.tolist(), "A_K": A_K.tolist(),
                          "PRECIO": PRECIO, "C_PROC": C_PROC, "R0": R0, "R_MAX": R_MAX,
                          "K_Z": K_Z, "A_REF": A_REF, "Z_MIN": Z_MIN, "A_MAX": A_MAX},
           "corridas": runs}
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    with open(Path(a.salida) / "esc1_mezcla.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    figura(runs, Path(a.figuras) / "fig1_mezcla.png")
    print(f"LP lineal óptimo: {LP_VAL:.4f} USD/t, p={np.round(LP_P, 4)}")
    for r in runs:
        print(r["semilla"], "GA lineal", round(r["lineal"]["ga_utilidad"], 4),
              "| NL GA", round(r["no_lineal"]["ga"][0], 3), "DE", round(r["no_lineal"]["de"][0], 3),
              "rnd", round(r["no_lineal"]["aleatoria"][0], 3), "LP en NL", round(r["no_lineal"]["lp_en_no_lineal"][0], 3))


if __name__ == "__main__":
    main()
