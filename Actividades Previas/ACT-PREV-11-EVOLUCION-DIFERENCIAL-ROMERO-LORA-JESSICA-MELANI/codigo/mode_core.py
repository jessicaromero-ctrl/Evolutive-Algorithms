#!/usr/bin/env python3
"""
mode_core.py — Núcleo de la evolución diferencial multiobjetivo (MODE) con cuatro estrategias
de control de parámetros: fija, dither, autoadaptación jDE y adaptación por historial de éxito
(variante multiobjetivo de SHADE).

Componentes:
  - Mutación DE/rand/1 y cruce binomial (Storn y Price, 1997; Price et al., 2005).
  - Selección ambiental elitista por no dominancia y distancia de multitud (Deb et al., 2002),
    con dominancia restringida (Deb et al., 2002) para problemas con restricciones.
  - Indicadores: hipervolumen bidimensional e IGD respecto de un frente de referencia.

Todos los problemas se minimizan. Para maximizar una función, se usa su negativo.
"""
import numpy as np

VARIANTES = ["fija", "dither", "jde", "shade"]
ETIQUETAS = {"fija": "Fija", "dither": "Dither", "jde": "jDE", "shade": "SHADE-MO"}
SEMILLAS = [11, 23, 37, 53, 71, 89, 101, 113, 127, 139]
N_POP = 100
GENS = 150
F_FIJA, CR_FIJA = 0.5, 0.9
TAU1, TAU2 = 0.1, 0.1           # probabilidades de actualización de jDE
H_MEM = 6                       # tamaño del historial de SHADE-MO
EVALS = N_POP * (GENS + 1)      # evaluaciones por ejecución (incluye la población inicial)


# ── Problemas ─────────────────────────────────────────────────────────────

def zdt_evaluar(nombre, X):
    x1 = X[:, 0]
    g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
    r = x1 / g
    if nombre == "zdt1":
        h = 1.0 - np.sqrt(r)
    elif nombre == "zdt2":
        h = 1.0 - r ** 2
    elif nombre == "zdt3":
        h = 1.0 - np.sqrt(r) - r * np.sin(10.0 * np.pi * x1)
    else:
        raise ValueError(nombre)
    F = np.column_stack([x1, g * h])
    return F, np.zeros(len(X))


def zdt_frente_ref(nombre, n_pts=1000):
    """Frente verdadero muestreado (g = 1)."""
    f1 = np.linspace(0.0, 1.0, 4000)
    if nombre == "zdt1":
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])[:: max(1, 4000 // n_pts)]
    if nombre == "zdt2":
        f2 = 1.0 - f1 ** 2
        return np.column_stack([f1, f2])[:: max(1, 4000 // n_pts)]
    # ZDT3: frente discontinuo; se conservan solo puntos no dominados
    f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
    P = np.column_stack([f1, f2])
    return P[nd_mascara(P, np.zeros(len(P)))]


# Problema aplicado: mezcla bi-objetivo (utilidad frente a impureza), datos del reporte minero
Z_K = np.array([9.5, 7.2, 5.8, 11.4, 4.1])
C_K = np.array([18.0, 15.0, 14.0, 24.0, 12.0])
A_K = np.array([900.0, 300.0, 150.0, 1200.0, 80.0])
PRECIO, C_PROC, R_MAX, K_Z, A_REF, Z_MIN = 2800.0, 6.0, 0.92, 1.5, 800.0, 6.0


def mezcla_evaluar(U):
    """U: (n, 5) en [0,1]. Objetivos: f1 = −utilidad (USD/t), f2 = impureza media (ppm)."""
    s = U.sum(axis=1, keepdims=True)
    P = np.where(s > 1e-12, U / np.maximum(s, 1e-12), 1.0 / U.shape[1])
    z = P @ Z_K
    a = P @ A_K
    c = P @ C_K
    r = R_MAX * (z / (z + K_Z)) / (1.0 + a / A_REF)
    utilidad = PRECIO * (z / 100.0) * r - c - C_PROC
    F = np.column_stack([-utilidad, a])
    cv = np.maximum(0.0, Z_MIN - z)          # restricción: ley media mínima de 6 %
    return F, cv


PROBLEMAS = {
    "zdt1": {"n": 30, "lo": np.zeros(30), "hi": np.ones(30), "ref_pt": np.array([1.1, 1.1]),
             "eval": lambda X: zdt_evaluar("zdt1", X)},
    "zdt3": {"n": 30, "lo": np.zeros(30), "hi": np.ones(30), "ref_pt": np.array([1.1, 1.1]),
             "eval": lambda X: zdt_evaluar("zdt3", X)},
    "mezcla": {"n": 5, "lo": np.zeros(5), "hi": np.ones(5), "ref_pt": None,
               "eval": mezcla_evaluar},
}


# ── Dominancia, ordenación y selección ────────────────────────────────────

def matriz_dominancia(F, cv):
    """D[i, j] = True si i domina a j (dominancia restringida)."""
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    pareto = le & lt
    fi = (cv[:, None] == 0)
    fj = (cv[None, :] == 0)
    D = np.where(fi & fj, pareto,
                 np.where(fi & ~fj, True,
                          np.where(~fi & fj, False, cv[:, None] < cv[None, :])))
    np.fill_diagonal(D, False)
    return D


def nd_mascara(F, cv):
    """Máscara de soluciones no dominadas (y factibles, si se pasan restricciones)."""
    D = matriz_dominancia(F, cv)
    return ~D.any(axis=0)


def ordenar_frentes(F, cv):
    D = matriz_dominancia(F, cv)
    cnt = D.sum(axis=0).astype(int)
    restante = np.ones(len(F), dtype=bool)
    frentes = []
    while restante.any():
        frente = np.where(restante & (cnt == 0))[0]
        if len(frente) == 0:          # no debería ocurrir; evita bucles infinitos
            frente = np.where(restante)[0]
        frentes.append(frente)
        restante[frente] = False
        cnt = cnt - D[frente].sum(axis=0)
    return frentes


def distancia_multitud(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        d[:] = np.inf
        return d
    for k in range(m):
        o = np.argsort(F[:, k])
        d[o[0]] = d[o[-1]] = np.inf
        rango = F[o[-1], k] - F[o[0], k]
        if rango <= 1e-300:
            continue
        d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / rango
    return d


def seleccion(F, cv, n):
    """Selección elitista NSGA-II sobre la población combinada. Devuelve índices elegidos."""
    elegidos = []
    for frente in ordenar_frentes(F, cv):
        if len(elegidos) + len(frente) <= n:
            elegidos.extend(frente.tolist())
        else:
            falta = n - len(elegidos)
            d = distancia_multitud(F[frente])
            orden = frente[np.argsort(-d, kind="stable")]
            elegidos.extend(orden[:falta].tolist())
            break
        if len(elegidos) == n:
            break
    return np.array(elegidos, dtype=int)


# ── Indicadores ───────────────────────────────────────────────────────────

def hipervolumen2d(F, ref):
    """Hipervolumen bidimensional (minimización) de F respecto al punto de referencia ref."""
    P = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    keep = []
    mejor = np.inf
    for p in P:
        if p[1] < mejor:
            keep.append(p)
            mejor = p[1]
    P = np.array(keep)
    x_sig = np.append(P[1:, 0], ref[0])
    return float(np.sum((x_sig - P[:, 0]) * (ref[1] - P[:, 1])))


def igd(F, R):
    """Distancia generacional invertida: promedio de la distancia de cada punto de R a F."""
    if len(F) == 0:
        return np.inf
    d = np.sqrt(((R[:, None, :] - F[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return float(d.mean())


# ── Algoritmo ─────────────────────────────────────────────────────────────

def _indices_distintos(rng, n, excluir):
    R = rng.integers(0, n, size=(n, 3))
    for _ in range(100):
        malo = (R[:, 0] == excluir) | (R[:, 1] == excluir) | (R[:, 2] == excluir) \
            | (R[:, 0] == R[:, 1]) | (R[:, 0] == R[:, 2]) | (R[:, 1] == R[:, 2])
        if not malo.any():
            break
        R[malo] = rng.integers(0, n, size=(malo.sum(), 3))
    return R


def _muestra_F_shade(rng, mu, n):
    F = mu + 0.1 * rng.standard_cauchy(n)
    for _ in range(50):
        malo = F <= 0
        if not malo.any():
            break
        F[malo] = mu[malo] + 0.1 * rng.standard_cauchy(malo.sum())
    F = np.where(F <= 0, 0.5, F)
    return np.minimum(F, 1.0)


def mode(problema, variante, seed, n=N_POP, gens=GENS, ref_hv=None, R_igd=None, registrar_cada=5):
    """Ejecuta MODE con la estrategia de control indicada. Devuelve resultados y trazas."""
    rng = np.random.default_rng(seed)
    P = PROBLEMAS[problema]
    lo, hi, d = P["lo"], P["hi"], P["n"]
    X = lo + rng.random((n, d)) * (hi - lo)
    F, cv = P["eval"](X)
    Fi = np.full(n, F_FIJA)
    CRi = np.full(n, CR_FIJA)
    MF = np.full(H_MEM, 0.5)
    MCR = np.full(H_MEM, 0.5)
    k_mem = 0
    traza = {"gen": [], "hv": [], "igd": [], "F_media": [], "CR_media": [], "MF": [], "MCR": []}

    def registrar(g):
        mask = nd_mascara(F, cv) & (cv == 0)
        Fnd = F[mask]
        traza["gen"].append(g)
        if ref_hv is not None:
            traza["hv"].append(hipervolumen2d(Fnd, ref_hv))
        if R_igd is not None:
            traza["igd"].append(igd(Fnd, R_igd) if len(Fnd) else np.inf)
        traza["F_media"].append(float(Fi.mean()))
        traza["CR_media"].append(float(CRi.mean()))
        traza["MF"].append(MF.tolist())
        traza["MCR"].append(MCR.tolist())

    registrar(0)
    for g in range(1, gens + 1):
        if variante == "fija":
            Fc = np.full(n, F_FIJA)
            CRc = np.full(n, CR_FIJA)
        elif variante == "dither":
            Fc = rng.uniform(0.5, 1.0, n)
            CRc = np.full(n, CR_FIJA)
        elif variante == "jde":
            Fc = np.where(rng.random(n) < TAU1, 0.1 + 0.9 * rng.random(n), Fi)
            CRc = np.where(rng.random(n) < TAU2, rng.random(n), CRi)
        elif variante == "shade":
            r = rng.integers(0, H_MEM, n)
            CRc = np.clip(rng.normal(MCR[r], 0.1), 0.0, 1.0)
            Fc = _muestra_F_shade(rng, MF[r], n)
        else:
            raise ValueError(variante)

        R = _indices_distintos(rng, n, np.arange(n))
        V = X[R[:, 0]] + Fc[:, None] * (X[R[:, 1]] - X[R[:, 2]])
        cruza = rng.random((n, d)) < CRc[:, None]
        cruza[np.arange(n), rng.integers(0, d, n)] = True
        U = np.clip(np.where(cruza, V, X), lo, hi)
        Fu, cvu = P["eval"](U)

        Xc = np.vstack([X, U])
        Fc_all = np.vstack([F, Fu])
        cvc = np.concatenate([cv, cvu])
        Fi_all = np.concatenate([Fi, Fc])
        CRi_all = np.concatenate([CRi, CRc])
        sel = seleccion(Fc_all, cvc, n)

        exito = sel[sel >= n] - n           # índices de hijos que sobreviven
        X, F, cv = Xc[sel], Fc_all[sel], cvc[sel]
        Fi, CRi = Fi_all[sel], CRi_all[sel]

        if variante == "shade" and len(exito) > 0:
            Fs, CRs = Fc[exito], CRc[exito]
            MF[k_mem] = float(np.sum(Fs ** 2) / max(np.sum(Fs), 1e-12))   # media de Lehmer
            MCR[k_mem] = float(np.mean(CRs))
            k_mem = (k_mem + 1) % H_MEM

        if g % registrar_cada == 0 or g == gens:
            registrar(g)

    mask = nd_mascara(F, cv) & (cv == 0)
    return {"variante": variante, "seed": seed, "frente": F[mask].tolist(),
            "X_frente": X[mask].tolist(),
            "n_factibles": int(mask.sum()), "traza": traza,
            "hv_final": hipervolumen2d(F[mask], ref_hv) if ref_hv is not None else None,
            "igd_final": igd(F[mask], R_igd) if (R_igd is not None and mask.any()) else None,
            "evals": EVALS}


def friedman_holm(M, nombres, ref="fija"):
    """M: (bloques, tratamientos). Devuelve chi2, p de Friedman y comparaciones de Wilcoxon con Holm."""
    from scipy.stats import friedmanchisquare, wilcoxon
    chi2, p = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    i_ref = nombres.index(ref)
    comps = []
    for j, nombre in enumerate(nombres):
        if j == i_ref:
            continue
        d = M[:, j] - M[:, i_ref]
        pv = 1.0 if np.allclose(d, 0) else float(wilcoxon(M[:, j], M[:, i_ref]).pvalue)
        comps.append([nombre, float(np.median(d)), pv])
    comps.sort(key=lambda c: c[2])
    m = len(comps)
    ajust = []
    acum = 0.0
    for i, c in enumerate(comps):
        val = min(1.0, (m - i) * c[2])
        acum = max(acum, val)
        ajust.append(min(1.0, acum))
    return float(chi2), float(p), [c + [a] for c, a in zip(comps, ajust)]


def correr_todo(problema, ref_hv=None, R_igd=None, registrar_cada=5, semillas=None):
    """Ejecuta todas las variantes con todas las semillas. Devuelve dict[variante] -> lista de corridas."""
    semillas = SEMILLAS if semillas is None else semillas
    return {v: [mode(problema, v, s, ref_hv=ref_hv, R_igd=R_igd, registrar_cada=registrar_cada)
                for s in semillas] for v in VARIANTES}
