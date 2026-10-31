#!/usr/bin/env python3
"""
esc2_secuenciamiento.py — Escenario 2: secuenciamiento de frentes de extracción con capacidad
de planta y restricciones de precedencia.

Decisión: periodo de extracción t ∈ {1..T} o no extracción (t = 0) para cada bloque j.
Formulación entera (referencia exacta por MILP, HiGHS):
  x_jt ∈ {0,1},  w_jt = v_j / (1+r)^t
  max  Σ_j Σ_t w_jt x_jt
  s.a. Σ_t x_jt ≤ 1                              (cada bloque se extrae a lo más una vez)
       Σ_j τ_j x_jt ≤ C                          (capacidad de planta por periodo)
       Σ_{τ<t} x_aτ ≥ x_bt   para (a,b) en P     (precedencia: a antes que b)
donde τ_j es el tonelaje del bloque (kt), v_j su valor neto (MUSD) y r la tasa de descuento.

Codificación del AG: cada bloque tiene un gen real g_j ∈ [0, T+1); el periodo es ⌊g_j⌋
(0 = no extracción). La aptitud es el VPN menos penalizaciones por capacidad y precedencia.

Los datos son ilustrativos, de orden de magnitud, y no corresponden a un yacimiento real.
Uso: python3 esc2_secuenciamiento.py --salida ../resultados --figuras ../figuras
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import matplotlib.pyplot as plt

from ga_comun import ga_real, busqueda_real, estilo_grafico, PURPURA, LILA, GRIS, PURPURA_OSC, EVALS

SEMILLA_DATOS = 2024
SEMILLAS = [11, 23, 37, 53, 71]
J, T = 15, 4
R_DESC = 0.10
CAP = 260.0               # capacidad de planta por periodo (kt)
PRECEDENCIAS = [(0, 3), (2, 7), (5, 11)]  # (a, b): el bloque a debe extraerse antes que b
M_CAP = 0.5               # penalización MUSD por kt de exceso de capacidad
M_PREC = 5.0              # penalización MUSD por violación de precedencia
PRECIO, R0, C_OP, C_PROC = 2800.0, 0.85, 20.0, 6.0   # economía ilustrativa (USD/t)


def generar(seed=SEMILLA_DATOS):
    rng = np.random.default_rng(seed)
    ton = rng.integers(40, 121, size=J).astype(float)          # kt por bloque
    ley = rng.uniform(4.0, 12.0, size=J)                       # Zn (%)
    margen = PRECIO * (ley / 100.0) * R0 - C_OP - C_PROC       # USD/t
    valor = ton * margen / 1000.0                              # MUSD
    return ton, ley, margen, valor


TON, LEY, MARGEN, VALOR = generar()
DESC = (1.0 + R_DESC) ** -np.arange(1, T + 1)                  # factor de descuento por periodo
W = VALOR[:, None] * DESC[None, :]                             # (J, T)


def vpn_y_penalizacion(X):
    """X: (n, J, T) binaria. Devuelve VPN, carga por periodo, exceso de capacidad y violaciones."""
    vpn = (X * W[None]).sum(axis=(1, 2))
    carga = (X * TON[None, :, None]).sum(axis=1)               # (n, T)
    exceso = np.maximum(0.0, carga - CAP).sum(axis=1)
    viol = np.zeros(X.shape[0])
    for a, b in PRECEDENCIAS:
        previo = np.concatenate([np.zeros((X.shape[0], 1)), np.cumsum(X[:, a, :], axis=1)[:, :-1]], axis=1)
        viol += np.maximum(0.0, X[:, b, :] - previo).sum(axis=1)
    return vpn, carga, exceso, viol


def decodificar(G):
    """G: (n, J) reales en [0, T+1). Devuelve X (n, J, T) binaria."""
    k = np.floor(G).astype(int)
    X = np.zeros((G.shape[0], J, T), dtype=float)
    for t in range(1, T + 1):
        X[:, :, t - 1] = (k == t)
    return X


def aptitud(G, mc=M_CAP, mp=M_PREC):
    X = decodificar(G)
    vpn, _, exceso, viol = vpn_y_penalizacion(X)
    return vpn - mc * exceso - mp * viol


def reparar(X):
    """Reparación determinista de una matriz binaria (J, T): elimina, en cada periodo con
    sobrecarga, el bloque de menor valor w_jt; después elimina extracciones que violan
    precedencia. Repite hasta que no haya violaciones. Devuelve una X factible."""
    X = X.copy()
    cambio = True
    while cambio:
        cambio = False
        for t in range(T):
            while (X[:, t] * TON).sum() > CAP + 1e-9:
                activos = np.where(X[:, t] == 1)[0]
                j = activos[np.argmin(W[activos, t])]
                X[j, t] = 0
                cambio = True
        for a, b in PRECEDENCIAS:
            for t in range(T):
                if X[b, t] == 1 and X[a, :t].sum() == 0:
                    X[b, t] = 0
                    cambio = True
    return X


def aptitud_reparada(G):
    """Aptitud del AG con reparación: VPN de la solución reparada (siempre factible)."""
    Xs = decodificar(G)
    Xr = np.stack([reparar(x) for x in Xs]).astype(float)
    vpn, _, _, _ = vpn_y_penalizacion(Xr)
    return vpn


def milp_exacto():
    n = J * T
    idx = lambda j, t: j * T + t
    c = -np.array([W[j, t] for j in range(J) for t in range(T)])
    filas, lb, ub = [], [], []
    for j in range(J):                                         # una extracción por bloque
        fila = np.zeros(n)
        for t in range(T):
            fila[idx(j, t)] = 1.0
        filas.append(fila); lb.append(0.0); ub.append(1.0)
    for t in range(T):                                         # capacidad por periodo
        fila = np.zeros(n)
        for j in range(J):
            fila[idx(j, t)] = TON[j]
        filas.append(fila); lb.append(-np.inf); ub.append(CAP)
    for a, b in PRECEDENCIAS:                                  # precedencia
        for t in range(T):
            fila = np.zeros(n)
            fila[idx(b, t)] = 1.0
            for tau in range(t):
                fila[idx(a, tau)] -= 1.0
            filas.append(fila); lb.append(-np.inf); ub.append(0.0)
    A = np.vstack(filas)
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.ones(n),
               bounds=Bounds(0, 1))
    X = np.round(res.x.reshape(J, T)).astype(int)
    return float(-res.fun), X


def ejecutar(seed, valor_exacto, f, nombre):
    """Ejecuta el AG y la búsqueda aleatoria (mismo presupuesto) con la aptitud f."""
    lo, hi = np.zeros(J), np.full(J, T + 1 - 1e-9)
    ga = ga_real(f, lo, hi, seed)
    rnd = busqueda_real(f, lo, hi, seed)
    Xga = decodificar(ga["x"][None])[0]
    if nombre == "reparacion":
        Xga = reparar(Xga)
    Xrnd = decodificar(rnd["x"][None])[0]
    if nombre == "reparacion":
        Xrnd = reparar(Xrnd)
    vpn_g, carga_g, exc_g, viol_g = vpn_y_penalizacion(Xga[None])
    vpn_r, _, exc_r, viol_r = vpn_y_penalizacion(Xrnd[None])
    fact_g = bool(exc_g[0] < 1e-9 and viol_g[0] < 1e-9)
    fact_r = bool(exc_r[0] < 1e-9 and viol_r[0] < 1e-9)
    return {"semilla": seed, "variante": nombre, "ga_fitness": ga["f"], "ga_vpn": float(vpn_g[0]),
            "ga_factible": fact_g,
            "ga_brecha": (valor_exacto - float(vpn_g[0])) if fact_g else None,
            "ga_X": Xga.astype(int).tolist(), "ga_carga": carga_g[0].tolist(),
            "hist": ga["hist"].tolist(), "aleatoria_fitness": rnd["f"],
            "aleatoria_vpn": float(vpn_r[0]), "aleatoria_factible": fact_r}


def figura(runs, X_opt, valor_exacto, ruta):
    estilo_grafico()
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 3.0), layout="constrained",
                           gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})
    im = ax[0].imshow(X_opt, cmap="Purples", aspect="auto", vmin=0, vmax=1)
    ax[0].set_xticks(range(T), [f"t={t + 1}" for t in range(T)])
    ax[0].set_yticks(range(J), [f"{j + 1}" for j in range(J)], fontsize=6)
    ax[0].set(xlabel="Periodo", ylabel="Bloque", title="(a) Secuencia óptima (MILP)")
    for a, b in PRECEDENCIAS:
        ax[0].annotate("", xy=(-0.6, b), xytext=(-0.6, a), arrowprops=dict(arrowstyle="->", color=GRIS, lw=0.5))
    # (b) carga por periodo frente a capacidad
    carga_opt = (X_opt * TON[:, None]).sum(0)
    x = np.arange(T)
    ax[1].bar(x - 0.2, carga_opt, width=0.38, color=PURPURA, label="MILP")
    ax[1].bar(x + 0.2, runs[0]["ga_carga"], width=0.38, color=LILA, label="AG reparación (semilla 11)")
    ax[1].axhline(CAP, color=GRIS, ls="--", lw=1.0, label="Capacidad")
    ax[1].set_xticks(x, [f"t={t + 1}" for t in range(T)])
    ax[1].set(ylabel="Tonelaje extraído (kt)", title="(b) Carga por periodo", ylim=(0, 300))
    ax[1].legend(fontsize=6.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.14), frameon=False)
    # (c) convergencia
    H = np.array([r["hist"] for r in runs])
    gen = np.arange(H.shape[1])
    for h in H:
        ax[2].plot(gen, h, color=LILA, lw=0.8)
    ax[2].axhline(valor_exacto, color=GRIS, ls="--", lw=1.0, label="Óptimo MILP")
    ax[2].plot(gen, H.mean(0), color=PURPURA, lw=1.8, label="Media AG")
    ax[2].set(xlabel="Generación", ylabel="Aptitud (MUSD)", title="(c) Convergencia")
    ax[2].legend(loc="lower right")
    fig.savefig(ruta)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="../resultados")
    ap.add_argument("--figuras", default="../figuras")
    a = ap.parse_args()
    valor, X_opt = milp_exacto()
    variantes = {
        "reparacion": lambda G: aptitud_reparada(G),
        "penalizacion_debil": lambda G: aptitud(G, M_CAP, M_PREC),
        "penalizacion_fuerte": lambda G: aptitud(G, 5.0, 50.0),
    }
    res = {nombre: [ejecutar(s, valor, f, nombre) for s in SEMILLAS] for nombre, f in variantes.items()}
    datos = {"TON": TON.tolist(), "LEY": LEY.tolist(), "MARGEN": MARGEN.tolist(),
             "VALOR": VALOR.tolist(), "CAP": CAP, "R_DESC": R_DESC, "PRECEDENCIAS": PRECEDENCIAS}
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    with open(Path(a.salida) / "esc2_secuenciamiento.json", "w", encoding="utf-8") as fh:
        json.dump({"datos": datos, "milp_vpn": valor, "milp_X": X_opt.tolist(), "variantes": res},
                  fh, ensure_ascii=False, indent=1)
    figura(res["reparacion"], X_opt, valor, Path(a.figuras) / "fig2_secuenciamiento.png")
    print(f"MILP VPN óptimo: {valor:.4f} MUSD; bloques extraídos: {int(X_opt.sum())}")
    for nombre, runs in res.items():
        vpns = [r["ga_vpn"] for r in runs]
        fact = sum(r["ga_factible"] for r in runs)
        brechas = [r["ga_brecha"] for r in runs if r["ga_brecha"] is not None]
        print(f"{nombre:20s} VPN={np.round(vpns, 2)} factibles={fact}/5 "
              f"brecha_media={np.mean(brechas) if brechas else float('nan'):.3f}")


if __name__ == "__main__":
    main()
