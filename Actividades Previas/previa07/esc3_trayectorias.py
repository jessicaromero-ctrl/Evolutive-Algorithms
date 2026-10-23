#!/usr/bin/env python3
"""
esc3_trayectorias.py — Escenario 3: calibración de un alineamiento de trayectorias clínicas.

Una trayectoria es una secuencia de eventos clínicos (A = admisión, B = laboratorio, ...,
H = alta). Una vía guía de referencia R = (A, B, ..., H) describe el recorrido esperado.
La puntuación de conformidad de una trayectoria T es el alineamiento global de
Needleman–Wunsch con parámetros de coincidencia m > 0, de sustitución x < 0 y de hueco g < 0:

    F(i,j) = máx{ F(i-1,j-1) + s(T_i,R_j),  F(i-1,j) + g,  F(i,j-1) + g },
    F(i,0) = i·g,   F(0,j) = j·g,   s(a,b) = m si a = b, x en otro caso.

Un AG con codificación real calibra (m, x, g) para maximizar la precisión media (AP) de
la puntuación −F(T) como detector de trayectorias desviadas en entrenamiento. Se compara con
búsqueda aleatoria, con parámetros por defecto (1, −1, −1) y con una malla de evaluación.

Datos sintéticos: trayectorias generadas con tasas controladas de sustitución y repetición.
Uso: python3 esc3_trayectorias.py --salida ../resultados --figuras ../figuras
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
N = 800
PREVALENCIA = 0.25
L = 8
REF = np.arange(8)                       # vía guía A→H
LO = np.array([0.5, -3.0, -3.0])         # cotas de (m, x, g)
HI = np.array([3.0, -0.1, -0.1])
DEFECTO = np.array([1.0, -1.0, -1.0])


def generar(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(N) < PREVALENCIA).astype(int)
    T = np.zeros((N, L), dtype=int)
    for i in range(N):
        p_sust, p_rep = (0.25, 0.20) if y[i] else (0.06, 0.04)
        seq = []
        for j in range(L):
            e = REF[j]
            if rng.random() < p_sust:
                e = int(rng.integers(0, 8))
            seq.append(e)
            if rng.random() < p_rep:
                seq.append(seq[-1])
        seq = seq[:L] if len(seq) >= L else seq + list(rng.integers(0, 8, size=L - len(seq)))
        T[i] = seq
    return T, y


def alineamiento(T, m, x, g):
    """Puntuación global de Needleman–Wunsch de cada fila de T frente a REF."""
    n = T.shape[0]
    M = len(REF)
    F = np.zeros((n, L + 1, M + 1))
    F[:, :, 0] = g * np.arange(L + 1)
    F[:, 0, :] = g * np.arange(M + 1)
    for i in range(1, L + 1):
        for j in range(1, M + 1):
            s = np.where(T[:, i - 1] == REF[j - 1], m, x)
            F[:, i, j] = np.maximum.reduce([F[:, i - 1, j - 1] + s,
                                            F[:, i - 1, j] + g,
                                            F[:, i, j - 1] + g])
    return F[:, L, M]


def puntuacion(T, p):
    return -alineamiento(T, *p)          # mayor = más desviada


def ejecutar(seed):
    T, y = generar(seed)
    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.5, stratify=y, random_state=seed)
    Ttr, Tte, ytr, yte = T[idx_tr], T[idx_te], y[idx_tr], y[idx_te]

    def f(P):
        return np.array([average_precision_score(ytr, puntuacion(Ttr, p)) for p in P])

    ga = ga_real(f, LO, HI, seed)
    rnd = busqueda_real(f, LO, HI, seed)

    def test(p):
        s = puntuacion(Tte, p)
        return average_precision_score(yte, s), roc_auc_score(yte, s)

    return {"semilla": seed, "p_ga": ga["x"].tolist(), "ap_train_ga": ga["f"],
            "hist": ga["hist"].tolist(), "p_aleatoria": rnd["x"].tolist(),
            "ga": test(ga["x"]), "aleatoria": test(rnd["x"]), "defecto": test(DEFECTO),
            "_s_ga": puntuacion(Tte, ga["x"]).tolist(), "_y": yte.tolist()}


def paisaje(seed, n=30):
    """Malla de AP en entrenamiento sobre (g, x) con m fijo en 1.5 (semilla 11)."""
    T, y = generar(seed)
    idx_tr, _ = train_test_split(np.arange(N), test_size=0.5, stratify=y, random_state=seed)
    Ttr, ytr = T[idx_tr], y[idx_tr]
    xs = np.linspace(LO[1], HI[1], n)
    gs = np.linspace(LO[2], HI[2], n)
    M = np.zeros((n, n))
    for a, x in enumerate(xs):
        for b, g in enumerate(gs):
            M[b, a] = average_precision_score(ytr, puntuacion(Ttr, (1.5, x, g)))
    return xs, gs, M


def figura(runs, ruta, xs, gs, M):
    estilo_grafico()
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.9), layout="constrained",
                           gridspec_kw={"width_ratios": [1.15, 1, 1]})
    # (a) paisaje de aptitud
    im = ax[0].imshow(M, origin="lower", extent=[xs[0], xs[-1], gs[0], gs[-1]], aspect="auto",
                      cmap="Purples")
    p = np.array(runs[0]["p_ga"])
    ax[0].scatter([p[1]], [p[2]], marker="*", s=90, color="white", edgecolor=PURPURA_OSC, lw=0.7,
                  zorder=3, label="Óptimo AG (semilla 11)")
    ax[0].set(xlabel="Sustitución $x$", ylabel="Hueco $g$", title="(a) Paisaje de aptitud ($m=1.5$)")
    ax[0].legend(fontsize=6.5, loc="lower left")
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)
    cb.set_label("AP entrenamiento", fontsize=7)
    # (b) convergencia
    H = np.array([r["hist"] for r in runs])
    gen = np.arange(H.shape[1])
    for h in H:
        ax[1].plot(gen, h, color=LILA, lw=0.8)
    ax[1].plot(gen, H.mean(0), color=PURPURA, lw=1.8, label="Media")
    ax[1].set(xlabel="Generación", ylabel="AP en entrenamiento", title="(b) Convergencia")
    ax[1].legend(loc="lower right")
    # (c) distribución de puntuaciones en prueba (semilla 11)
    s = np.array(runs[0]["_s_ga"])
    yy = np.array(runs[0]["_y"])
    bins = np.linspace(s.min(), s.max(), 30)
    ax[2].hist(s[yy == 0], bins=bins, color=LILA, alpha=0.9, label="Conforme")
    ax[2].hist(s[yy == 1], bins=bins, color=PURPURA, alpha=0.7, label="Desviada")
    ax[2].set(xlabel="Puntuación $-F(T)$", ylabel="Trayectorias",
              title="(c) Prueba")
    ax[2].legend(loc="upper right")
    fig.savefig(ruta)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="../resultados")
    ap.add_argument("--figuras", default="../figuras")
    a = ap.parse_args()
    runs = [ejecutar(s) for s in SEMILLAS]
    xs, gs, M = paisaje(11)
    out_runs = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    with open(Path(a.salida) / "esc3_trayectorias.json", "w", encoding="utf-8") as fh:
        json.dump({"corridas": out_runs, "paisaje": {"x": xs.tolist(), "g": gs.tolist(),
                                                   "AP": M.tolist()}},
                  fh, ensure_ascii=False, indent=1)
    figura(runs, Path(a.figuras) / "fig3_trayectorias.png", xs, gs, M)
    for k in ["ga", "aleatoria", "defecto"]:
        v = np.array([r[k] for r in out_runs])
        print(f"{k:10s} AP={v[:, 0].mean():.4f}±{v[:, 0].std(ddof=1) if len(v) > 1 else 0:.4f} "
              f"ROC={v[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()
