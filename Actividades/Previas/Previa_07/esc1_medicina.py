#!/usr/bin/env python3
"""
esc1_medicina.py — Escenario 1: fusión de puntuaciones de riesgo clínico.

Simula una cohorte con seis covariables clínicas correlacionadas y un riesgo latente z.
Cinco modelos de puntuación observan z con ruido de distinta magnitud. Un AG con
codificación real determina los pesos de la combinación convexa que maximiza la precisión
media (AP) en entrenamiento. El desempeño se informa en prueba y se compara con búsqueda
aleatoria, pesos uniformes, el mejor modelo individual y un oráculo que usa z verdadero.

Datos sintéticos: no corresponden a pacientes reales.
Uso: python3 esc1_medicina.py --salida ../resultados ../figuras
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from ga_comun import ga_real, busqueda_real, estilo_grafico, PURPURA, LILA, GRIS, PURPURA_OSC

SEMILLAS = [11, 23, 37, 53, 71]
N = 4000
BETA = np.array([0.9, 0.7, 0.5, 0.4, 0.2, 0.1])          # efecto de cada covariable sobre z
SIGMAS = np.array([0.5, 0.9, 1.5, 2.2, 0.7])             # ruido de cada modelo
NOMBRES = ["Modelo A", "Modelo B", "Modelo C", "Modelo D", "Modelo E"]


def generar(seed):
    rng = np.random.default_rng(seed)
    cov = 0.3 * np.ones((6, 6)) + 0.7 * np.eye(6)          # correlación moderada entre covariables
    X = rng.normal(size=(N, 6)) @ np.linalg.cholesky(cov).T
    z = X @ BETA
    y = (rng.random(N) < 1 / (1 + np.exp(-2.0 * (z - np.quantile(z, 0.90))))).astype(int)
    S = z[:, None] + SIGMAS * rng.normal(size=(N, len(SIGMAS)))
    return z, S, y


def pesos(U):
    """Proyección al símplex: pesos no negativos que suman 1."""
    U = np.clip(U, 0.0, None)
    s = U.sum(axis=1, keepdims=True)
    return np.where(s > 1e-12, U / np.maximum(s, 1e-12), 1.0 / U.shape[1])


def ejecutar(seed):
    z, S, y = generar(seed)
    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.5, stratify=y, random_state=seed)
    mu, sd = S[idx_tr].mean(0), S[idx_tr].std(0)
    Z = (S - mu) / sd
    Ztr, Zte, ytr, yte = Z[idx_tr], Z[idx_te], y[idx_tr], y[idx_te]

    def f(U):  # aptitud = AP en entrenamiento
        Sc = pesos(U) @ Ztr.T
        return np.array([average_precision_score(ytr, s) for s in Sc])

    ga = ga_real(f, np.zeros(5), np.ones(5), seed)
    rnd = busqueda_real(f, np.zeros(5), np.ones(5), seed)
    w_ga = pesos(ga["x"][None])[0]
    w_rnd = pesos(rnd["x"][None])[0]
    w_uni = np.full(5, 0.2)
    mejor = int(np.argmax([average_precision_score(ytr, Ztr[:, k]) for k in range(5)]))
    w_mejor = np.eye(5)[mejor]

    def test(w):
        s = Zte @ w
        return average_precision_score(yte, s), roc_auc_score(yte, s)

    return {
        "semilla": seed,
        "prevalencia": float(y.mean()),
        "ap_train_ga": ga["f"],
        "hist": ga["hist"].tolist(),
        "pesos_ga": w_ga.tolist(),
        "ga": test(w_ga), "aleatoria": test(w_rnd), "uniforme": test(w_uni),
        "mejor_individual": test(w_mejor), "mejor_indice": mejor,
        "oraculo": (average_precision_score(yte, z[idx_te]), roc_auc_score(yte, z[idx_te])),
        "ap_train_aleatoria": rnd["f"],
    }


def figura(runs, ruta):
    estilo_grafico()
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.9), layout="constrained",
                           gridspec_kw={"width_ratios": [1, 1.05, 1.1]})
    # (a) convergencia
    H = np.array([r["hist"] for r in runs])
    gen = np.arange(H.shape[1])
    for h in H:
        ax[0].plot(gen, h, color=LILA, lw=0.8)
    ax[0].plot(gen, H.mean(0), color=PURPURA, lw=1.8, label="Media (5 semillas)")
    ax[0].set(xlabel="Generación", ylabel="AP en entrenamiento", title="(a) Convergencia")
    ax[0].legend(loc="center right")
    # (b) desempeño en prueba
    claves = [("ga", "AG"), ("aleatoria", "Aleat."), ("uniforme", "Unif."),
              ("mejor_individual", "Mejor\nmod."), ("oraculo", "Oráculo")]
    datos = [[r[k][0] for r in runs] for k, _ in claves]
    bp = ax[1].boxplot(datos, widths=0.55, patch_artist=True, medianprops={"color": PURPURA_OSC})
    for patch, c in zip(bp["boxes"], [PURPURA, LILA, LILA, LILA, GRIS]):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax[1].set_xticks(range(1, len(claves) + 1), [n for _, n in claves])
    ax[1].set(ylabel="AP en prueba", title="(b) Desempeño en prueba")
    # (c) pesos del AG por semilla
    W = np.array([r["pesos_ga"] for r in runs])
    bottom = np.zeros(len(runs))
    colores = [PURPURA, "#A866E8", LILA, "#E3D0F7", GRIS]
    for k in range(5):
        ax[2].bar(range(1, len(runs) + 1), W[:, k], bottom=bottom, color=colores[k],
                  label=NOMBRES[k], width=0.6)
        bottom += W[:, k]
    ax[2].set(xlabel="Semilla (índice)", ylabel="Peso $w_k$", title="(c) Pesos aprendidos", ylim=(0, 1))
    ax[2].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(ruta)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="../resultados")
    ap.add_argument("--figuras", default="../figuras")
    a = ap.parse_args()
    runs = [ejecutar(s) for s in SEMILLAS]
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    Path(a.figuras).mkdir(parents=True, exist_ok=True)
    with open(Path(a.salida) / "esc1_medicina.json", "w", encoding="utf-8") as fh:
        json.dump({"nombres": NOMBRES, "sigmas": SIGMAS.tolist(), "corridas": runs}, fh,
                  ensure_ascii=False, indent=1)
    figura(runs, Path(a.figuras) / "fig1_medicina.png")
    for k in ["ga", "aleatoria", "uniforme", "mejor_individual", "oraculo"]:
        v = np.array([r[k] for r in runs])
        print(f"{k:18s} AP={v[:, 0].mean():.4f}±{v[:, 0].std(ddof=1) if len(v) > 1 else 0:.4f}"
              f"  ROC={v[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()
