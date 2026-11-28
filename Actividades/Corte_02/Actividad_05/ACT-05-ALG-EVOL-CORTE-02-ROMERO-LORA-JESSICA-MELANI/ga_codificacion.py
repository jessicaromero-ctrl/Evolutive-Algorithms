#!/usr/bin/env python3
"""
ga_codificacion.py — Algoritmo genético con tres codificaciones de cromosoma para la
fusión lineal de las señales del detector CAS (versión 2).

OBJETIVO DEL MÓDULO
  Aprender el vector de pesos w en el símplex para la puntuación fusionada
      s(w) = Σ_k w_k z_k,   k = 1..6,
  donde z_k son las seis señales out-of-fold producidas por cas_detector_v2.py.
  El fenotipo es w. Se comparan tres codificaciones del cromosoma:
      G1 binaria (código de Gray, 8 bits por gen),
      G2 real (un número real en [0, 1] por gen),
      G3 entera (un entero en {0, ..., 20} por gen).

GARANTÍAS METODOLÓGICAS
  1. Validación externa por proveedor: para el pliegue k, el AG sólo usa etiquetas de los
     pliegues distintos de k (aptitud = PR-AUC sobre esos pliegues). El pliegue k se usa
     únicamente para evaluar la solución final y las líneas base.
  2. Mismo presupuesto de evaluaciones de aptitud en todas las codificaciones y en la búsqueda
     aleatoria. Mismas poblaciones iniciales (M03) para una semilla dada.
  3. Las métricas de presupuesto (top 5 %) siguen el mismo criterio que budget_metrics de
     cas_detector_v2.py.

LIMITACIÓN DECLARADA
  Las señales de los pliegues distintos de k fueron producidas por modelos que sí vieron los
  datos del pliegue k (apilamiento estándar de dos niveles). La fuga resultante es de segundo
  orden y se discute en el informe.

USO
  python ga_codificacion.py --resultados ../results/difficulty_1.00 --salida ../results_ga/difficulty_1.00
"""

import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon, norm
from sklearn.metrics import roc_auc_score, average_precision_score


# ── Constantes declaradas a priori ──────────────────────────────

SIGNALS = ["s_struct", "s_sem", "s_temp", "s_xgb_cas", "s_if", "s_xgb_static"]
K = len(SIGNALS)
N_POP = 60              # tamaño de población
K_TORN = 2              # torneo binario (M12)
E_ELITE = 2             # elitismo (M22)
P_CRUCE = 0.9           # probabilidad de cruce por pareja
E_MAX = 5000            # presupuesto de evaluaciones de aptitud (M26)
N_HIJOS = N_POP - E_ELITE
GENERACIONES = (E_MAX - N_POP) // N_HIJOS
EVALS_FINAL = N_POP + GENERACIONES * N_HIJOS      # 4 990 con los valores anteriores
SEEDS = [11, 23, 37, 53, 71]
BUDGET = 0.05           # presupuesto de revisión (5 % de claims por pliegue)
BITS = 8                # bits por gen en G1
M_ENT = 20              # máximo del entero en G3
ALPHA_BLX = 0.3         # BLX-α en G2
SIGMA = 0.05            # desviación de la mutación gaussiana en G2
BATCH_ALEATORIA = 100   # tamaño de lote en búsqueda aleatoria


# ── Datos y métricas ────────────────────────────────────────────

DATOS = {}


def cargar(ruta_oof):
    df = pd.read_csv(ruta_oof)
    if df[SIGNALS].isna().any().any():
        raise ValueError("Hay señales sin valor; no se puede ajustar la fusión.")
    DATOS["Z"] = df[SIGNALS].to_numpy(float)
    DATOS["y"] = df["is_fraud"].to_numpy(int)
    DATOS["fold"] = df["fold"].to_numpy(int)
    DATOS["amt"] = df["amount_billed"].to_numpy(float)
    return df


def metricas_presupuesto(y, s, amt, rng):
    """Precisión, recall y recall monetario en el top BUDGET; desempate idéntico al detector."""
    s = s + rng.uniform(0, 1e-9, len(s))
    k = max(1, int(math.ceil(len(s) * BUDGET)))
    idx = np.argsort(-s)[:k]
    tp = int(y[idx].sum())
    fraud_amt = amt[y == 1].sum()
    det_amt = amt[idx][y[idx] == 1].sum()
    return tp / k, tp / max(int(y.sum()), 1), det_amt / max(fraud_amt, 1e-9)


def metricas_test(y, s, amt, seed):
    rng = np.random.default_rng(seed)
    prec, rec, ar = metricas_presupuesto(y, s, amt, rng)
    return {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
            "precision_5": prec, "recall_5": rec, "amount_recall_5": ar}


def aptitud(W, Z, y):
    """PR-AUC de cada vector de pesos (filas de W) sobre Z con etiquetas y. Vectorizada.
    Equivale a average_precision_score cuando no hay empates (prueba 4)."""
    S = W @ Z.T                                   # (n_pop, n)
    orden = np.argsort(-S, axis=1, kind="stable")
    ys = y[orden]
    tp = np.cumsum(ys, axis=1)
    rangos = np.arange(1, Z.shape[0] + 1)
    prec = tp / rangos
    return (prec * ys).sum(axis=1) / max(int(y.sum()), 1)


def pesos(U):
    """Fenotipo: normaliza un vector no negativo a la suma 1. Si la suma es 0, uniforme."""
    s = U.sum(axis=1, keepdims=True)
    return np.where(s > 1e-12, U / np.maximum(s, 1e-12), 1.0 / K)


# ── Muestreo inicial común (M03) ────────────────────────────────

def lhs_unit(n, k, rng):
    """Hipercubo latino en [0, 1]^k: cada columna ocupa cada estrato exactamente una vez."""
    estratos = rng.permuted(np.tile(np.arange(n), (k, 1)), axis=1).T
    return (estratos + rng.random((n, k))) / n


# ── G1 · binaria con código de Gray ─────────────────────────────

def bin_desde_u(U):
    enteros = np.rint(np.clip(U, 0, 1) * (2 ** BITS - 1)).astype(np.int64)
    gris = enteros ^ (enteros >> 1)
    desplaz = np.arange(BITS - 1, -1, -1)
    bits = (gris[..., None] >> desplaz) & 1
    return bits.reshape(len(U), K * BITS).astype(np.uint8)


def bin_a_u(G):
    bits = G.reshape(len(G), K, BITS).astype(np.int64)
    binario = np.bitwise_xor.accumulate(bits, axis=2)     # Gray → binario, MSB primero
    potencias = 2 ** np.arange(BITS - 1, -1, -1)
    return (binario * potencias).sum(axis=2) / (2 ** BITS - 1)


def bin_cruzar(A, B, rng):
    """Cruce de un punto por pareja (un punto por cromosoma)."""
    n, L = A.shape
    c = rng.integers(1, L, size=(n, 1))
    m = np.arange(L)[None, :] < c
    return np.where(m, A, B), np.where(m, B, A)


def bin_mutar(H, rng):
    """Volteo de bit con probabilidad 1/L por bit (tasa estándar)."""
    m = rng.random(H.shape) < 1.0 / H.shape[1]
    return H ^ m.astype(H.dtype)


# ── G2 · real ───────────────────────────────────────────────────

def real_desde_u(U):
    return np.clip(U, 0, 1)


def real_a_u(G):
    return G


def real_cruzar(A, B, rng):
    """BLX-α con recorte a [0, 1]."""
    lo, hi = np.minimum(A, B), np.maximum(A, B)
    I = hi - lo
    H1 = np.clip(rng.uniform(lo - ALPHA_BLX * I, hi + ALPHA_BLX * I), 0, 1)
    H2 = np.clip(rng.uniform(lo - ALPHA_BLX * I, hi + ALPHA_BLX * I), 0, 1)
    return H1, H2


def real_mutar(H, rng):
    """Gaussiana acotada, con probabilidad 1/K por gen."""
    m = rng.random(H.shape) < 1.0 / K
    Y = np.clip(H + SIGMA * rng.standard_normal(H.shape), 0, 1)
    return np.where(m, Y, H)


# ── G3 · entera ─────────────────────────────────────────────────

def ent_desde_u(U):
    return np.rint(np.clip(U, 0, 1) * M_ENT).astype(np.int64)


def ent_a_u(G):
    return G / M_ENT


def ent_cruzar(A, B, rng):
    """Cruce uniforme por gen (máscara Bernoulli 0.5)."""
    m = rng.random(A.shape) < 0.5
    return np.where(m, A, B), np.where(m, B, A)


def ent_mutar(H, rng):
    """Paso ±1 con probabilidad 1/K por gen, con recorte a {0, ..., M}."""
    m = rng.random(H.shape) < 1.0 / K
    paso = rng.choice(np.array([-1, 1]), size=H.shape)
    return np.where(m, np.clip(H + paso, 0, M_ENT), H)


CODIFICACIONES = {
    "G1-binaria": dict(desde_u=bin_desde_u, a_u=bin_a_u, cruzar=bin_cruzar, mutar=bin_mutar,
                       dtype=np.uint8, longitud=K * BITS),
    "G2-real": dict(desde_u=real_desde_u, a_u=real_a_u, cruzar=real_cruzar, mutar=real_mutar,
                    dtype=float, longitud=K),
    "G3-entera": dict(desde_u=ent_desde_u, a_u=ent_a_u, cruzar=ent_cruzar, mutar=ent_mutar,
                      dtype=np.int64, longitud=K),
}


# ── Algoritmo genético (siete etapas) ───────────────────────────

def ejecutar_ag(nombre, seed):
    enc = CODIFICACIONES[nombre]
    Z, y = DATOS["Z_tr"], DATOS["y_tr"]
    rng = np.random.default_rng(seed)
    inicio = time.perf_counter()

    U0 = lhs_unit(N_POP, K, rng)                              # ETAPA 1 · M03
    G = enc["desde_u"](U0).astype(enc["dtype"])
    fit = aptitud(pesos(enc["a_u"](G)), Z, y)                 # ETAPA 2 · aptitud directa
    evals, gen = N_POP, 0
    curva = [(evals, float(fit.max()))]

    for _ in range(GENERACIONES):
        torneos = rng.integers(0, N_POP, size=(N_HIJOS, K_TORN))          # ETAPA 3 · M12
        ganadores = torneos[np.arange(N_HIJOS), np.argmax(fit[torneos], axis=1)]
        A, B = G[ganadores[0::2]], G[ganadores[1::2]]
        H1, H2 = enc["cruzar"](A, B, rng)                                 # ETAPA 4
        cruza = rng.random(len(A)) < P_CRUCE
        H1 = np.where(cruza[:, None], H1, A)
        H2 = np.where(cruza[:, None], H2, B)
        H = np.empty((N_HIJOS, G.shape[1]), dtype=enc["dtype"])
        H[0::2], H[1::2] = H1, H2
        H = enc["mutar"](H, rng).astype(enc["dtype"])                     # ETAPA 5
        fH = aptitud(pesos(enc["a_u"](H)), Z, y)                          # repetir ETAPA 2
        evals += N_HIJOS
        elite = np.argsort(-fit, kind="stable")[:E_ELITE]                 # ETAPA 6 · M22
        G = np.vstack([G[elite], H])
        fit = np.concatenate([fit[elite], fH])
        gen += 1
        curva.append((evals, float(fit.max())))                           # ETAPA 7 · M26

    assert evals == EVALS_FINAL and evals <= E_MAX
    mejor = int(np.argmax(fit))
    w = pesos(enc["a_u"](G[mejor:mejor + 1]))[0]
    return {"w": w, "ap_train": float(fit[mejor]), "evaluaciones": evals, "generaciones": gen,
            "unicos": float(len(np.unique(G, axis=0)) / N_POP),
            "segundos": time.perf_counter() - inicio, "curva": curva}


def busqueda_aleatoria(seed):
    """Control: muestreo uniforme en el símplex con el mismo presupuesto; elige por PR-AUC de train."""
    Z, y = DATOS["Z_tr"], DATOS["y_tr"]
    rng = np.random.default_rng(seed + 10_000)
    mejor_ap, mejor_w, usadas = -1.0, None, 0
    while usadas < EVALS_FINAL:
        n = min(BATCH_ALEATORIA, EVALS_FINAL - usadas)
        W = rng.dirichlet(np.ones(K), size=n)
        ap = aptitud(W, Z, y)
        i = int(np.argmax(ap))
        if ap[i] > mejor_ap:
            mejor_ap, mejor_w = float(ap[i]), W[i]
        usadas += n
    return {"w": mejor_w, "ap_train": mejor_ap, "evaluaciones": usadas, "generaciones": 0,
            "unicos": np.nan, "segundos": np.nan, "curva": []}


def uniforme():
    w = np.full(K, 1.0 / K)
    return {"w": w, "ap_train": float(aptitud(w[None, :], DATOS["Z_tr"], DATOS["y_tr"])[0]),
            "evaluaciones": 1, "generaciones": 0, "unicos": np.nan, "segundos": np.nan, "curva": []}


def mejor_senal():
    """Control: la señal individual con mayor PR-AUC en entrenamiento."""
    Z, y = DATOS["Z_tr"], DATOS["y_tr"]
    W = np.eye(K)
    ap = aptitud(W, Z, y)
    i = int(np.argmax(ap))
    return {"w": W[i], "ap_train": float(ap[i]), "evaluaciones": K, "generaciones": 0,
            "unicos": np.nan, "segundos": np.nan, "curva": []}


# ── Tareas paralelas ────────────────────────────────────────────

def preparar_fold(k):
    tr = DATOS["fold"] != k
    te = DATOS["fold"] == k
    DATOS["Z_tr"], DATOS["y_tr"] = DATOS["Z"][tr], DATOS["y"][tr]
    DATOS["Z_te"], DATOS["y_te"], DATOS["amt_te"] = DATOS["Z"][te], DATOS["y"][te], DATOS["amt"][te]


def tarea(t):
    """t = (método, pliegue, semilla). Entrena sólo con pliegues ≠ k y evalúa en el pliegue k."""
    metodo, k, seed = t
    preparar_fold(k)
    if metodo in CODIFICACIONES:
        r = ejecutar_ag(metodo, seed)
    elif metodo == "ALEATORIA":
        r = busqueda_aleatoria(seed)
    elif metodo == "UNIFORME":
        r = uniforme()
    elif metodo == "MEJOR_SENAL":
        r = mejor_senal()
    else:
        raise ValueError(metodo)
    s_te = DATOS["Z_te"] @ r["w"]
    m = metricas_test(DATOS["y_te"], s_te, DATOS["amt_te"], seed + k)
    fila = {"metodo": metodo, "pliegue": k, "semilla": seed, "ap_train": r["ap_train"],
            "evaluaciones": r["evaluaciones"], "generaciones": r["generaciones"],
            "unicos": r["unicos"], "segundos": r["segundos"],
            **{f"w_{s}": float(v) for s, v in zip(SIGNALS, r["w"])}, **m}
    fila["_curva"] = r["curva"]
    return fila


# ── Estadística ─────────────────────────────────────────────────

def holm(p):
    p = np.asarray(p, float)
    m, ajust, acum = len(p), np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        acum = max(acum, min(1.0, (m - i) * p[j]))
        ajust[j] = acum
    return ajust


def wilcoxon_emp(x, y):
    return 1.0 if np.all(x == y) else float(wilcoxon(x, y).pvalue)


def comparar_metodos(M, etiquetas):
    """Friedman sobre bloques (filas) y Wilcoxon-Holm contra el método de menor rango medio."""
    chi2, p = friedmanchisquare(*M.T)
    # rango 1 = mejor valor (mayor PR-AUC o precisión); rankdata ordena de menor a mayor, por eso -M
    rangos = rankdata(-M, axis=1).mean(axis=0)
    mejor = int(np.argmin(rangos))
    otros = [j for j in range(M.shape[1]) if j != mejor]
    ph = dict(zip(otros, holm([wilcoxon_emp(M[:, j], M[:, mejor]) for j in otros])))
    return {"chi2": float(chi2), "p": float(p), "rangos": dict(zip(etiquetas, rangos)),
            "mejor": etiquetas[mejor], "p_holm": {etiquetas[j]: float(ph[j]) for j in otros}}


# ── Pruebas automatizadas (antes de cualquier ejecución) ────────

def pruebas():
    rng = np.random.default_rng(0)
    # G1: el decodificador de Gray invierte al codificador en los 256 valores posibles
    z = np.arange(256)
    gris = z ^ (z >> 1)
    bits = (gris[:, None] >> np.arange(BITS - 1, -1, -1)) & 1
    recuperado = (np.bitwise_xor.accumulate(bits, axis=1) * 2 ** np.arange(BITS - 1, -1, -1)).sum(axis=1)
    assert np.array_equal(recuperado, z), "decodificación de Gray incorrecta"
    # G1: el código de Gray cambia exactamente un bit entre enteros consecutivos
    cambios = np.array([bin(int(a) ^ int(b)).count("1") for a, b in zip(gris[:-1], gris[1:])])
    assert np.all(cambios == 1), "el código no es de Gray"
    U = rng.random((500, K))
    assert np.allclose(bin_a_u(bin_desde_u(U)), np.rint(U * (2 ** BITS - 1)) / (2 ** BITS - 1)), \
        "la codificación binaria no es invertible"
    W = pesos(rng.random((50, K)))
    assert np.allclose(W.sum(axis=1), 1.0), "los pesos no suman 1"
    Z = rng.random((400, K))
    y = (Z @ np.ones(K) + rng.normal(0, 0.8, 400) > K / 2).astype(int)
    w = np.full(K, 1.0 / K)
    assert abs(aptitud(w[None], Z, y)[0] - average_precision_score(y, Z @ w)) < 1e-9, \
        "la aptitud no coincide con average_precision_score"
    for nombre, enc in CODIFICACIONES.items():
        Ua = lhs_unit(20, K, rng)
        Ga = enc["desde_u"](Ua).astype(enc["dtype"])
        A, B = Ga[:10], Ga[10:]
        H1, H2 = enc["cruzar"](A, B, rng)
        assert H1.shape == A.shape and H2.shape == A.shape, f"{nombre}: forma del cruce"
        Hm = enc["mutar"](Ga.copy(), rng)
        assert Hm.shape == Ga.shape, f"{nombre}: forma de la mutación"
        U2 = enc["a_u"](Hm)
        assert np.all((U2 >= -1e-12) & (U2 <= 1 + 1e-12)), f"{nombre}: fenotipo fuera de [0,1]"
    assert EVALS_FINAL == 4990 and EVALS_FINAL <= E_MAX
    return 6


# ── Figuras ─────────────────────────────────────────────────────

def figuras(res, curvas, carpeta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colores = {"G1-binaria": "#2a78d6", "G2-real": "#eb6834", "G3-entera": "#1baf7a",
               "ALEATORIA": "#898781", "UNIFORME": "#52514e", "MEJOR_SENAL": "#c3c2b7"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

    # 1 · convergencia
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for enc in CODIFICACIONES:
        c = curvas[curvas["codificacion"] == enc]
        g = c.groupby("evaluaciones")["ap"].median()
        ax.plot(g.index, g.values, color=colores[enc], lw=2, label=enc)
    ax.set_xlabel("evaluaciones de aptitud")
    ax.set_ylabel("PR-AUC mediana del mejor (entrenamiento)")
    ax.legend(frameon=False)
    ax.grid(color="#e1e0d9", lw=0.6)
    fig.savefig(os.path.join(carpeta, "ga_01_convergencia.png"))
    plt.close(fig)

    # 2 · distribución de PR-AUC en prueba
    orden = ["G1-binaria", "G2-real", "G3-entera", "ALEATORIA", "UNIFORME", "MEJOR_SENAL"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    datos = [res[res["metodo"] == m]["pr_auc"].to_numpy() for m in orden]
    bp = ax.boxplot(datos, tick_labels=orden, patch_artist=True, widths=0.55)
    for patch, m in zip(bp["boxes"], orden):
        patch.set_facecolor(colores[m])
        patch.set_alpha(0.35)
    ax.set_ylabel("PR-AUC en el pliegue de prueba")
    ax.tick_params(axis="x", labelsize=8.5)
    fig.savefig(os.path.join(carpeta, "ga_02_prauc_prueba.png"))
    plt.close(fig)

    # 3 · pesos aprendidos por codificación
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ancho = 0.26
    x = np.arange(K)
    for i, enc in enumerate(CODIFICACIONES):
        sub = res[res["metodo"] == enc]
        medias = [sub[f"w_{s}"].mean() for s in SIGNALS]
        sds = [sub[f"w_{s}"].std(ddof=1) for s in SIGNALS]
        ax.bar(x + (i - 1) * ancho, medias, ancho, yerr=sds, color=colores[enc], label=enc,
               error_kw={"lw": 0.8, "ecolor": "#52514e"})
    ax.set_xticks(x)
    ax.set_xticklabels(SIGNALS, fontsize=8.5)
    ax.set_ylabel("peso medio (símplex)")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(carpeta, "ga_03_pesos.png"))
    plt.close(fig)
    return 3


# ── Principal ───────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="AG con tres codificaciones para la fusión CAS.")
    ap.add_argument("--resultados", required=True, help="carpeta con oof_predictions.csv")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--solo-analisis", action="store_true",
                    help="no reejecuta las corridas; relee corridas.csv y convergencia.csv")
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)
    inicio = time.perf_counter()

    print("0. Pruebas automatizadas:", pruebas(), "de 6 pasaron")
    metodos = list(CODIFICACIONES) + ["ALEATORIA"]
    if args.solo_analisis:
        res = pd.read_csv(os.path.join(args.salida, "corridas.csv"))
        curvas = pd.read_csv(os.path.join(args.salida, "convergencia.csv"))
        print(f"1. Análisis sobre corridas existentes: {len(res)} filas")
    else:
        df = cargar(os.path.join(args.resultados, "oof_predictions.csv"))
        print(f"   {len(df):,} claims · {int(df.is_fraud.sum()):,} fraudes · {df.fold.nunique()} pliegues")
        print(f"   G = {GENERACIONES} generaciones · evaluaciones por corrida = {EVALS_FINAL}")
        tareas = [(m, k, s) for k in range(5) for s in SEEDS for m in metodos]
        tareas += [("UNIFORME", k, s) for k in range(5) for s in SEEDS]
        tareas += [("MEJOR_SENAL", k, s) for k in range(5) for s in SEEDS]
        print(f"1. Ejecutando {len(tareas)} tareas con {args.workers} procesos...")
        with Pool(args.workers) as pool:
            filas = pool.map(tarea, tareas, chunksize=1)
        curvas_rows = []
        for f in filas:
            for ev, apv in f.pop("_curva"):
                curvas_rows.append({"codificacion": f["metodo"], "pliegue": f["pliegue"],
                                    "semilla": f["semilla"], "evaluaciones": ev, "ap": apv})
        res = pd.DataFrame(filas)
        curvas = pd.DataFrame(curvas_rows)
        res.to_csv(os.path.join(args.salida, "corridas.csv"), index=False)
        curvas.to_csv(os.path.join(args.salida, "convergencia.csv"), index=False)

    print("2. Resumen en prueba (media ± DE sobre 25 bloques pliegue × semilla)")
    metricas = ["roc_auc", "pr_auc", "precision_5", "recall_5", "amount_recall_5"]
    resumen = res.groupby("metodo")[metricas].agg(["mean", "std"])
    resumen.columns = [f"{m}_{s}" for m, s in resumen.columns]
    resumen = resumen.loc[[m for m in metodos + ["UNIFORME", "MEJOR_SENAL"] if m in resumen.index]]
    resumen.to_csv(os.path.join(args.salida, "resumen.csv"))
    print(resumen[["pr_auc_mean", "pr_auc_std", "precision_5_mean", "recall_5_mean",
                   "amount_recall_5_mean"]].round(4).to_string())

    print("3. Pruebas estadísticas sobre bloques pliegue × semilla")
    estad = []
    orden = ["G1-binaria", "G2-real", "G3-entera", "ALEATORIA", "UNIFORME", "MEJOR_SENAL"]
    for m in ["pr_auc", "precision_5"]:
        piv = res.pivot_table(index=["pliegue", "semilla"], columns="metodo", values=m)[orden]
        c = comparar_metodos(piv.to_numpy(), orden)
        estad.append({"metrica": m, "chi2": c["chi2"], "p": c["p"], "mejor": c["mejor"],
                      **{f"p_holm_{k}": v for k, v in c["p_holm"].items()},
                      **{f"rango_{k}": v for k, v in c["rangos"].items()}})
    estad = pd.DataFrame(estad)
    estad.to_csv(os.path.join(args.salida, "pruebas_estadisticas.csv"), index=False)
    print(estad[["metrica", "chi2", "p", "mejor"]].to_string(index=False))

    print("4. Figuras")
    n_figs = figuras(res, curvas, args.salida)

    total = time.perf_counter() - inicio
    manifiesto = {"senales": SIGNALS, "n_pop": N_POP, "torneo_k": K_TORN, "elite": E_ELITE,
                  "p_cruce": P_CRUCE, "e_max": E_MAX, "evaluaciones_finales": EVALS_FINAL,
                  "generaciones": GENERACIONES, "semillas": SEEDS, "bits": BITS, "m_entera": M_ENT,
                  "alpha_blx": ALPHA_BLX, "sigma": SIGMA, "figuras": n_figs,
                  "segundos_totales": round(total, 1)}
    with open(os.path.join(args.salida, "manifiesto.json"), "w", encoding="utf-8") as fh:
        json.dump(manifiesto, fh, indent=2, ensure_ascii=False)
    print(f"Listo en {total:.0f} s → {args.salida}")


if __name__ == "__main__":
    main()
