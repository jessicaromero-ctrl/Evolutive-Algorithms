#!/usr/bin/env python3
"""
ga_comun.py — Núcleo compartido por los tres escenarios.

Implementa un algoritmo genético (AG) con dos codificaciones:
  - real:    cruce BLX-α y mutación gaussiana acotada al dominio;
  - binaria: cruce de un punto y mutación bit a bit.
Ambas usan selección por torneo binario y reemplazo elitista. Se incluye la búsqueda
aleatoria de control con exactamente el mismo número de evaluaciones de aptitud.

Las funciones de aptitud se reciben como parámetro y se evalúan por lotes (matriz n x d).
Se maximiza.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_POP = 40
GENS = 50
ELITE = 2
K_TORNEO = 2
P_CRUCE = 0.9
ALPHA_BLX = 0.3
SIGMA = 0.05
EVALS = N_POP + GENS * (N_POP - ELITE)  # 1940 evaluaciones por ejecución

PURPURA = "#8400E0"
PURPURA_OSC = "#3B0A66"
LILA = "#C9A6F0"
GRIS = "#52514E"
GRIS_CLARO = "#D9D6D2"


def estilo_grafico():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRIS, "axes.labelcolor": GRIS, "xtick.color": GRIS,
        "ytick.color": GRIS, "axes.grid": True, "grid.color": GRIS_CLARO, "grid.linewidth": 0.6,
        "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100,
    })


def _torneo(apt, rng, n):
    idx = rng.integers(0, len(apt), size=(n, K_TORNEO))
    return idx[np.arange(n), np.argmax(apt[idx], axis=1)]


def ga_real(f, lo, hi, seed):
    """AG con codificación real. Devuelve el mejor individuo, su aptitud y la historia."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    d = len(lo)
    P = lo + rng.random((N_POP, d)) * (hi - lo)
    apt = f(P)
    hist = [apt.max()]
    n_h = N_POP - ELITE
    for _ in range(GENS):
        orden = np.argsort(-apt)
        A = P[_torneo(apt, rng, n_h)]
        B = P[_torneo(apt, rng, n_h)]
        mn, mx = np.minimum(A, B), np.maximum(A, B)
        I = mx - mn
        H = rng.uniform(mn - ALPHA_BLX * I, mx + ALPHA_BLX * I)          # BLX-α
        H = np.where((rng.random(n_h) < P_CRUCE)[:, None], H, A)
        mut = rng.random(H.shape) < 1.0 / d                               # mutación gaussiana
        H = np.clip(H + mut * rng.normal(0.0, SIGMA, H.shape) * (hi - lo), lo, hi)
        apt_h = f(H)
        P = np.vstack([P[orden[:ELITE]], H])
        apt = np.concatenate([apt[orden[:ELITE]], apt_h])
        hist.append(apt.max())
    i = int(np.argmax(apt))
    return {"x": P[i], "f": float(apt[i]), "hist": np.array(hist), "evals": EVALS}


def ga_binaria(f, nbits, seed):
    """AG con codificación binaria (cruce de un punto y mutación bit a bit)."""
    rng = np.random.default_rng(seed)
    P = rng.integers(0, 2, size=(N_POP, nbits), dtype=np.int8)
    apt = f(P)
    hist = [apt.max()]
    n_h = N_POP - ELITE
    for _ in range(GENS):
        orden = np.argsort(-apt)
        A = P[_torneo(apt, rng, n_h)]
        B = P[_torneo(apt, rng, n_h)]
        punto = rng.integers(1, nbits, size=n_h)
        H = np.where(np.arange(nbits)[None, :] < punto[:, None], A, B)  # un punto
        H = np.where((rng.random(n_h) < P_CRUCE)[:, None], H, A)
        H = (H ^ (rng.random(H.shape) < 1.0 / nbits)).astype(np.int8)     # bit a bit
        apt_h = f(H)
        P = np.vstack([P[orden[:ELITE]], H])
        apt = np.concatenate([apt[orden[:ELITE]], apt_h])
        hist.append(apt.max())
    i = int(np.argmax(apt))
    return {"x": P[i], "f": float(apt[i]), "hist": np.array(hist), "evals": EVALS}


def busqueda_real(f, lo, hi, seed):
    """Control: muestreo uniforme con el mismo presupuesto de evaluaciones."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    P = lo + rng.random((EVALS, len(lo))) * (hi - lo)
    apt = f(P)
    i = int(np.argmax(apt))
    return {"x": P[i], "f": float(apt[i]), "evals": EVALS}


def busqueda_binaria(f, nbits, seed):
    """Control binario con el mismo presupuesto de evaluaciones."""
    rng = np.random.default_rng(seed)
    P = rng.integers(0, 2, size=(EVALS, nbits), dtype=np.int8)
    apt = f(P)
    i = int(np.argmax(apt))
    return {"x": P[i], "f": float(apt[i]), "evals": EVALS}
