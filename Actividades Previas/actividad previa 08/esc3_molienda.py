#!/usr/bin/env python3
"""
esc3_molienda.py — Escenario 3: ajuste óptimo de un circuito de molienda con superficie de
respuesta sintética y restricciones de especificación y potencia.

Variables de decisión: n (fracción de la velocidad crítica), J (carga de bolas, %) y
Q (alimentación, t/h). Modelo sustituto (ilustrativo):
  P80(n,J,Q) = P_A · (Q/120)^{0.5} · exp(−2.2 (n − 0.70)) · (J/32)^{−0.6}          (µm)
  P(n,J,Q)   = P_B · n^{1.5} · (J/32)^{0.9} · (0.6 + 0.4 Q/120)                      (kW)
  R(P80)     = R_MAX · [1 − exp(−(P_REF/P80)^{1.5})]                                 (–)
  V(n,J,Q)   = Q (V_T · R − C_M) − C_E · P                                           (USD/h)
Restricciones: P80 ≤ P80_MAX y P ≤ P_MAX. Se resuelven con penalización cuadrática
relativa. Referencias: malla densa (exacta en la resolución de la malla) y refinamiento
local; comparación con AG, búsqueda aleatoria y Nelder–Mead desde la configuración por defecto.

Los coeficientes son ilustrativos y no corresponden a un circuito real.
Uso: python3 esc3_molienda.py --salida ../resultados --figuras ../figuras
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from ga_comun import ga_real, busqueda_real, estilo_grafico, PURPURA, LILA, GRIS, PURPURA_OSC, EVALS

SEMILLAS = [11, 23, 37, 53, 71]
LO = np.array([0.65, 25.0, 90.0])
HI = np.array([0.85, 40.0, 150.0])
DEFECTO = np.array([0.78, 30.0, 100.0])
P_A, P_B = 180.0, 1200.0
R_MAX, P_REF = 0.92, 150.0
V_T, C_M, C_E = 85.0, 12.0, 0.12
P80_MAX, P_MAX = 150.0, 1000.0
MU = 1.0e4                       # penalización (USD/h) por unidad relativa de violación


def p80(n, J, Q):
    return P_A * (Q / 120.0) ** 0.5 * np.exp(-2.2 * (n - 0.70)) * (J / 32.0) ** -0.6


def potencia(n, J, Q):
    return P_B * n ** 1.5 * (J / 32.0) ** 0.9 * (0.6 + 0.4 * Q / 120.0)


def recuperacion(P80):
    return R_MAX * (1.0 - np.exp(-(P_REF / P80) ** 1.5))


def valor(x):
    """x: (..., 3) = (n, J, Q). Devuelve (V, P80, P, R) sin penalizar."""
    n, J, Q = x[..., 0], x[..., 1], x[..., 2]
    p8 = p80(n, J, Q)
    pw = potencia(n, J, Q)
    r = recuperacion(p8)
    v = Q * (V_T * r - C_M) - C_E * pw
    return v, p8, pw, r


def objetivo(x):
    """Aptitud penalizada (maximizar). x: (m, 3)."""
    v, p8, pw, _ = valor(x)
    pen = MU * (np.maximum(0.0, p8 / P80_MAX - 1.0) + np.maximum(0.0, pw / P_MAX - 1.0))
    return v - pen


TOL = 0.005                      # tolerancia relativa de 0.5 % en las restricciones de planta


def factible(x, tol=TOL):
    _, p8, pw, _ = valor(x)
    return bool(p8 <= P80_MAX * (1 + tol) and pw <= P_MAX * (1 + tol))


def referencia_malla(k=60):
    """Malla densa sobre la caja y refinamiento local desde el mejor punto factible."""
    n = np.linspace(LO[0], HI[0], k)
    J = np.linspace(LO[1], HI[1], k)
    Q = np.linspace(LO[2], HI[2], k)
    G = np.stack(np.meshgrid(n, J, Q, indexing="ij"), axis=-1).reshape(-1, 3)
    f = objetivo(G)
    i = int(np.argmax(f))
    best = G[i]
    res = minimize(lambda z: -objetivo(z[None])[0], best, method="Nelder-Mead",
                   bounds=list(zip(LO, HI)), options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
    x_ref = np.clip(res.x, LO, HI)
    return x_ref, float(objetivo(x_ref[None])[0]), float(objetivo(best[None])[0])


def ejecutar(seed, f_ref):
    f = lambda P: objetivo(P)
    ga = ga_real(f, LO, HI, seed)
    rnd = busqueda_real(f, LO, HI, seed)
    nm = minimize(lambda z: -objetivo(z[None])[0], DEFECTO, method="Nelder-Mead",
                  bounds=list(zip(LO, HI)), options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    return {"semilla": seed,
            "ga_x": ga["x"].tolist(), "ga_fit": ga["f"], "ga_factible": factible(ga["x"]),
            "ga_v": float(valor(ga["x"])[0]), "ga_p80": float(valor(ga["x"])[1]),
            "ga_pot": float(valor(ga["x"])[2]), "hist": ga["hist"].tolist(),
            "aleatoria_x": rnd["x"].tolist(), "aleatoria_v": float(valor(rnd["x"])[0]),
            "aleatoria_factible": factible(rnd["x"]),
            "nm_x": np.clip(nm.x, LO, HI).tolist(), "nm_v": float(valor(np.clip(nm.x, LO, HI))[0]),
            "nm_factible": factible(np.clip(nm.x, LO, HI)), "nm_evals": int(nm.nfev)}


def figura(runs, x_ref, f_ref, ruta):
    estilo_grafico()
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 3.0), layout="constrained",
                           gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
    # (a) contorno de valor en (n, J) a Q de referencia, con fronteras de restricciones
    n = np.linspace(LO[0], HI[0], 120)
    J = np.linspace(LO[1], HI[1], 120)
    N_, J_ = np.meshgrid(n, J, indexing="ij")
    Qr = 130.0                      # Q representativo, por debajo del límite de potencia
    X = np.stack([N_, J_, np.full_like(N_, Qr)], axis=-1)
    v, p8, pw, _ = valor(X)
    ax[0].contourf(J_, N_, np.where((p8 > P80_MAX) | (pw > P_MAX), 1.0, np.nan), levels=[0.5, 1.5],
                   colors=["#E6E6E6"])
    cs = ax[0].contourf(J_, N_, np.where((p8 <= P80_MAX) & (pw <= P_MAX), v, np.nan), levels=14, cmap="Purples")
    ax[0].contour(J_, N_, p8, levels=[P80_MAX], colors=GRIS, linestyles="--", linewidths=1.0)
    ax[0].contour(J_, N_, pw, levels=[P_MAX], colors=GRIS, linestyles=":", linewidths=1.0)
    ax[0].scatter([x_ref[1]], [x_ref[0]], marker="*", s=90, color="white", edgecolor=PURPURA_OSC, zorder=3,
                  label="Óptimo (Q = 150)")
    ax[0].scatter([DEFECTO[1]], [DEFECTO[0]], marker="o", s=22, color=GRIS, zorder=3, label="Por defecto")
    ax[0].set(xlabel="Carga de bolas $J$ (%)", ylabel="Velocidad relativa $n$",
              title=f"(a) Valor, $Q$ = {Qr:.0f} t/h")
    ax[0].legend(fontsize=6.0, loc="lower left")
    cb = fig.colorbar(cs, ax=ax[0], fraction=0.046, pad=0.03)
    cb.set_label("USD/h", fontsize=7)
    # (b) convergencia
    H = np.array([r["hist"] for r in runs])
    gen = np.arange(H.shape[1])
    for h in H:
        ax[1].plot(gen, h, color=LILA, lw=0.8)
    ax[1].axhline(f_ref, color=GRIS, ls="--", lw=1.0, label="Referencia")
    ax[1].plot(gen, H.mean(0), color=PURPURA, lw=1.8, label="Media AG")
    ax[1].set(xlabel="Generación", ylabel="Valor penalizado (USD/h)", title="(b) Convergencia")
    ax[1].legend(loc="lower right")
    # (c) comparación de valores
    datos = {"Ref.": [f_ref], "AG": [r["ga_v"] for r in runs],
             "Aleat.": [r["aleatoria_v"] for r in runs], "NM": [r["nm_v"] for r in runs]}
    etiquetas = list(datos.keys())
    bp = ax[2].boxplot([datos[k] for k in etiquetas], widths=0.55, patch_artist=True,
                       medianprops={"color": PURPURA_OSC})
    for patch, c in zip(bp["boxes"], [GRIS, PURPURA, LILA, "#A866E8"]):
        patch.set_facecolor(c)
    ax[2].set_xticks(range(1, len(etiquetas) + 1), etiquetas, fontsize=7)
    ax[2].set(ylabel="Valor en factible (USD/h)", title="(c) Por método")
    fig.savefig(ruta)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="../resultados")
    ap.add_argument("--figuras", default="../figuras")
    a = ap.parse_args()
    x_ref, f_ref, f_malla = referencia_malla()
    runs = [ejecutar(s, f_ref) for s in SEMILLAS]
    v_def, p8_def, pw_def, r_def = valor(DEFECTO)
    out = {"referencia": {"x": x_ref.tolist(), "v": float(valor(x_ref)[0]), "p80": float(valor(x_ref)[1]),
                          "pot": float(valor(x_ref)[2]), "R": float(valor(x_ref)[3]), "fit": f_ref,
                          "fit_malla_inicial": f_malla},
           "defecto": {"x": DEFECTO.tolist(), "v": float(v_def), "p80": float(p8_def), "pot": float(pw_def),
                       "R": float(r_def), "factible": factible(DEFECTO)},
           "parametros": {"TOL": TOL, "P_A": P_A, "P_B": P_B, "R_MAX": R_MAX, "P_REF": P_REF, "V_T": V_T,
                          "C_M": C_M, "C_E": C_E, "P80_MAX": P80_MAX, "P_MAX": P_MAX, "MU": MU,
                          "LO": LO.tolist(), "HI": HI.tolist()},
           "corridas": runs}
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    with open(Path(a.salida) / "esc3_molienda.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    figura(runs, x_ref, f_ref, Path(a.figuras) / "fig3_molienda.png")
    print("referencia x=", np.round(x_ref, 4), "V=", round(out["referencia"]["v"], 2),
          "P80=", round(out["referencia"]["p80"], 2), "P=", round(out["referencia"]["pot"], 1))
    print("defecto V=", round(out["defecto"]["v"], 2), "factible", out["defecto"]["factible"])
    for r in runs:
        print(r["semilla"], "GA", np.round(r["ga_x"], 3), round(r["ga_v"], 2), r["ga_factible"],
              "| NM", round(r["nm_v"], 2), r["nm_factible"], "| aleat", round(r["aleatoria_v"], 2))


if __name__ == "__main__":
    main()
