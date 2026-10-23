#!/usr/bin/env python3
"""
esc2_grupos_sociales.py — Escenario 2: asignación de un programa preventivo entre zonas.

Problema: seleccionar exactamente K zonas (de Z) para un programa de salud preventiva.
Cada zona pertenece a un grupo social (A, B, C) con distinta necesidad y distinta
eficiencia de alcance. La función objetivo combina cobertura total y equidad:

    F(s; λ) = Σ_i s_i b_i / Σ_i N_i  −  λ · (máx_g c_g − mín_g c_g)  −  μ |Σ_i s_i − K|

donde s_i ∈ {0,1}, b_i = N_i e_g(i), c_g es la cobertura dentro del grupo g y μ penaliza
violar la restricción de presupuesto. Un AG con codificación binaria (Z bits) se compara
con la solución exacta, obtenida por enumeración de las C(Z, K) combinaciones factibles,
y con la estrategia voraz que elige las K zonas de mayor beneficio.

Datos sintéticos: valores esperados de una población hipotética, no datos observados.
Uso: python3 esc2_grupos_sociales.py --salida ../resultados --figuras ../figuras
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ga_comun import ga_binaria, busqueda_binaria, estilo_grafico, PURPURA, LILA, GRIS, PURPURA_OSC

SEMILLA_DATOS = 2024
SEMILLAS = [11, 23, 37, 53, 71]
Z, K, MU = 20, 6, 1.0
GRUPOS = ["A", "B", "C"]
EFICIENCIA = {"A": 0.80, "B": 0.60, "C": 0.40}
TASA_BASE = {"A": 0.06, "B": 0.09, "C": 0.12}
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0]
LAMBDA_REF = 1.0


def generar(seed=SEMILLA_DATOS):
    rng = np.random.default_rng(seed)
    grupo = rng.choice(GRUPOS, size=Z, p=[0.40, 0.35, 0.25])
    poblacion = rng.integers(500, 3001, size=Z)
    tasa = np.array([TASA_BASE[g] for g in grupo]) * np.exp(0.3 * rng.normal(size=Z))
    N_need = poblacion * tasa
    e = np.array([EFICIENCIA[g] for g in grupo])
    return {"grupo": grupo, "N": N_need, "b": N_need * e, "e": e}


DAT = generar()


def componentes(S):
    """S: matriz (n, Z) binaria. Devuelve cobertura total, coberturas por grupo, brecha y tamaño."""
    g = DAT["grupo"]
    total = (S * DAT["b"]).sum(1) / DAT["N"].sum()
    cob = np.stack([(S[:, g == k] * DAT["b"][g == k]).sum(1) / DAT["N"][g == k].sum()
                    for k in GRUPOS], axis=1)
    brecha = cob.max(1) - cob.min(1)
    return total, cob, brecha, S.sum(1)


def objetivo(S, lam):
    total, _, brecha, tam = componentes(S)
    return total - lam * brecha - MU * np.abs(tam - K)


def solucion_exacta(lam):
    """Enumeración de todas las combinaciones de K zonas (C(20,6) = 38 760)."""
    combis = np.array(list(itertools.combinations(range(Z), K)))
    S = np.zeros((len(combis), Z), dtype=np.int8)
    np.put_along_axis(S, combis, 1, axis=1)
    val = objetivo(S, lam)
    i = int(np.argmax(val))
    total, cob, brecha, _ = componentes(S[i:i + 1])
    return {"s": S[i].tolist(), "F": float(val[i]), "total": float(total[0]),
            "cob": cob[0].tolist(), "brecha": float(brecha[0])}


def voraz():
    s = np.zeros(Z, dtype=np.int8)
    s[np.argsort(-DAT["b"])[:K]] = 1
    total, cob, brecha, _ = componentes(s[None])
    return {"s": s.tolist(), "F": float(objetivo(s[None], 0.0)[0]), "total": float(total[0]),
            "cob": cob[0].tolist(), "brecha": float(brecha[0])}


def ejecutar_ag(lam):
    corridas = []
    for sd in SEMILLAS:
        ga = ga_binaria(lambda P: objetivo(P, lam), Z, sd)
        rnd = busqueda_binaria(lambda P: objetivo(P, lam), Z, sd)
        corridas.append({"semilla": sd, "F_ga": ga["f"], "hist": ga["hist"].tolist(),
                         "s_ga": ga["x"].astype(int).tolist(), "tam_ga": int(ga["x"].sum()),
                         "F_aleatoria": rnd["f"]})
    return corridas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="../resultados")
    ap.add_argument("--figuras", default="../figuras")
    a = ap.parse_args()
    exactas = {str(lam): solucion_exacta(lam) for lam in LAMBDAS}
    ags = {str(lam): ejecutar_ag(lam) for lam in LAMBDAS}
    vor = voraz()
    out = {"grupo": DAT["grupo"].tolist(), "N": DAT["N"].tolist(), "b": DAT["b"].tolist(),
           "eficiencia": EFICIENCIA, "K": K, "lambdas": LAMBDAS, "exacta": exactas,
           "ag": ags, "voraz": vor}
    Path(a.salida).mkdir(parents=True, exist_ok=True)
    with open(Path(a.salida) / "esc2_grupos.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    figura(out, Path(a.figuras) / "fig2_grupos.png")
    for lam in LAMBDAS:
        k = str(lam)
        mejoras = [c["F_ga"] - exactas[k]["F"] for c in ags[k]]
        print(f"λ={lam:4}: exacta F={exactas[k]['F']:.4f} total={exactas[k]['total']:.4f} "
              f"brecha={exactas[k]['brecha']:.4f} | AG max-gap={max(mejoras):.2e} "
              f"exactos={sum(m > -1e-9 for m in mejoras)}/5")
    print("voraz:", vor["total"], vor["brecha"], vor["F"])


def figura(out, ruta):
    estilo_grafico()
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.9), layout="constrained",
                           gridspec_kw={"width_ratios": [1.1, 1]})
    # (a) frontera de compromiso exacta: cobertura total frente a brecha entre grupos
    # Los puntos con igual (brecha, cobertura) se agrupan; λ=0 coincide con la voraz.
    puntos = {}
    for l in LAMBDAS:
        d = out["exacta"][str(l)]
        puntos.setdefault((round(d["brecha"], 6), round(d["total"], 6)), []).append(l)
    xs = sorted(puntos, key=lambda p: p[0])
    ax[0].plot([p[0] for p in xs], [p[1] for p in xs], color=LILA, lw=1.2, zorder=1)
    ax[0].scatter([p[0] for p in xs], [p[1] for p in xs], s=52, color=PURPURA,
                  edgecolor=PURPURA_OSC, lw=0.6, zorder=2)
    for p, ls in puntos.items():
        etiqueta = f"λ={ls[0]}" if len(ls) == 1 else f"λ={ls[0]}–{ls[-1]}"
        if p[1] > 0.28:
            etiqueta += " = voraz"
        off = (6, -10) if p[0] > 0.1 else (6, 4)
        ax[0].annotate(etiqueta, p, textcoords="offset points", xytext=off, fontsize=7, color=GRIS)
    ax[0].set(xlabel="Brecha de cobertura entre grupos", ylabel="Cobertura total",
              title="(a) Compromiso eficiencia–equidad", xlim=(-0.02, 0.42), ylim=(0.25, 0.30))
    # (b) cobertura por grupo
    grupos = ["A", "B", "C"]
    lam_b = "1.0"
    series = [("Voraz", out["voraz"]["cob"], GRIS),
              ("Exacta λ=0", out["exacta"]["0.0"]["cob"], LILA),
              (f"Exacta λ={lam_b}", out["exacta"][lam_b]["cob"], PURPURA)]
    x = np.arange(3)
    w = 0.26
    for i, (nombre, cob, c) in enumerate(series):
        ax[1].bar(x + (i - 1) * w, cob, width=w, color=c, label=nombre)
    ax[1].set_xticks(x, [f"Grupo {g}" for g in grupos])
    ax[1].set(ylabel="Cobertura dentro del grupo", title="(b) Cobertura por grupo")
    ax[1].legend(fontsize=7, loc="upper right")
    fig.savefig(ruta)
    plt.close(fig)


if __name__ == "__main__":
    main()
