"""NSGA-II para permutaciones (flow shop) con dos operadores de cruce, y búsqueda aleatoria como línea base.

NSGA-II (Deb et al., 2002): ordenamiento por no dominancia, distancia de aglomeración y selección por torneo
binario con criterio (rango, −distancia). Reemplazo μ + λ.
Operadores de permutación:
  OX  (Davis, 1985): copia un segmento del padre 1 y completa con el orden relativo del padre 2.
  PMX (Goldberg y Lingle, 1985): copia un segmento del padre 2 y repara conflictos por correspondencia.
  Mutación por intercambio de dos posiciones.
"""
from __future__ import annotations

import numpy as np

from flowshop_modelo import Instancia, evaluar, frente_no_dominado


class Registro:
    """Cuenta evaluaciones y guarda cada permutación evaluada (una evaluación por individuo)."""

    def __init__(self, inst: Instancia, presupuesto: int):
        self.inst, self.presupuesto = inst, presupuesto
        self.perms, self.objs = [], []

    @property
    def agotado(self) -> bool:
        return len(self.perms) >= self.presupuesto

    def lote(self, X: np.ndarray) -> np.ndarray:
        k = self.presupuesto - len(self.perms)
        if k <= 0:
            return np.empty((0, 3))
        X = X[:k]
        F = evaluar(self.inst, X)
        self.perms.extend(map(tuple, X)); self.objs.append(F)
        return F


def nds(F: np.ndarray):
    """Frentes no dominados sucesivos (listas de índices), minimización. Matriz de dominancia vectorizada."""
    F = np.asarray(F, float)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)     # le[p, q]: p es ≤ q en todos los objetivos
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)      # lt[p, q]: p es < q en algún objetivo
    domina = le & lt                                        # domina[p, q]: p domina a q
    cuenta = domina.sum(axis=0)                             # número de soluciones que dominan a q
    restantes = np.ones(len(F), bool)
    frentes = []
    while restantes.any():
        actual = np.flatnonzero(restantes & (cuenta == 0))
        if len(actual) == 0:
            break
        frentes.append(list(actual))
        cuenta = cuenta - domina[actual].sum(axis=0)
        restantes[actual] = False
        cuenta[actual] = 10**9
    return frentes


def aglomeracion(F: np.ndarray, idx):
    idx = list(idx); k = len(idx); d = np.zeros(k)
    if k <= 2:
        return np.full(k, np.inf)
    sub = F[idx]
    for m in range(F.shape[1]):
        o = np.argsort(sub[:, m]); rango = sub[o[-1], m] - sub[o[0], m]
        d[o[0]] = d[o[-1]] = np.inf
        if rango <= 0:
            continue
        d[o[1:-1]] += (sub[o[2:], m] - sub[o[:-2], m]) / rango
    return d


def ox(p1, p2, rng):
    n = len(p1); a, b = sorted(rng.choice(n, 2, replace=False))
    hijo = np.full(n, -1); hijo[a:b + 1] = p1[a:b + 1]
    usados = set(int(g) for g in hijo[a:b + 1])
    resto = [int(g) for g in np.roll(p2, -(b + 1)) if int(g) not in usados]
    posiciones = [(b + 1 + k) % n for k in range(n) if hijo[(b + 1 + k) % n] == -1]
    for q, g in zip(posiciones, resto):
        hijo[q] = g
    return hijo


def pmx(p1, p2, rng):
    """Copia el segmento de p2; los conflictos fuera del segmento se reparan por correspondencia."""
    n = len(p1); a, b = sorted(rng.choice(n, 2, replace=False))
    hijo = np.full(n, -1)
    hijo[a:b + 1] = p2[a:b + 1]
    segmento = set(int(g) for g in p2[a:b + 1])
    correspondencia = {int(p2[k]): int(p1[k]) for k in range(a, b + 1)}
    for k in range(n):
        if a <= k <= b:
            continue
        g = int(p1[k])
        while g in segmento:
            g = correspondencia[g]
        hijo[k] = g
    return hijo


def mutar(x, rng, pm):
    x = x.copy()
    if rng.random() < pm:
        i, j = rng.choice(len(x), 2, replace=False)
        x[i], x[j] = x[j], x[i]
    return x


def nsga2(inst, presupuesto, semilla, cruce="OX", N=60, pc=0.9, pm=0.3):
    rng = np.random.default_rng(semilla)
    R = Registro(inst, presupuesto)
    operador = {"OX": ox, "PMX": pmx}[cruce]
    P = np.array([rng.permutation(inst.n) for _ in range(N)])
    F = R.lote(P)
    while not R.agotado:
        frentes = nds(F); rango = np.zeros(len(P)); crowd = np.zeros(len(P))
        for r, fr in enumerate(frentes):
            rango[fr] = r; crowd[fr] = aglomeracion(F, fr)

        def torneo():
            i, j = rng.integers(len(P), size=2)
            return i if (rango[i], -crowd[i]) < (rango[j], -crowd[j]) else j

        H = []
        while len(H) < N:
            a, b = P[torneo()], P[torneo()]
            if rng.random() < pc:
                h1, h2 = operador(a, b, rng), operador(b, a, rng)
            else:
                h1, h2 = a.copy(), b.copy()
            H += [mutar(h1, rng, pm), mutar(h2, rng, pm)]
        H = np.array(H[:N])
        FH = R.lote(H)
        H = H[:len(FH)]
        Q = np.vstack([P, H]); FQ = np.vstack([F, FH])
        nuevo = []
        for fr in nds(FQ):
            if len(nuevo) + len(fr) <= N:
                nuevo += fr
            else:
                dist = aglomeracion(FQ, fr)
                nuevo += [fr[k] for k in np.argsort(-dist)[: N - len(nuevo)]]
                break
        P, F = Q[nuevo], FQ[nuevo]
    return R


def aleatoria(inst, presupuesto, semilla):
    rng = np.random.default_rng(semilla)
    R = Registro(inst, presupuesto)
    while not R.agotado:
        k = min(200, presupuesto - len(R.perms))
        R.lote(np.array([rng.permutation(inst.n) for _ in range(k)]))
    return R


ALGORITMOS = {
    "NSGA-II (OX)": lambda inst, B, s: nsga2(inst, B, s, cruce="OX"),
    "NSGA-II (PMX)": lambda inst, B, s: nsga2(inst, B, s, cruce="PMX"),
    "Aleatoria": aleatoria,
}


def objetivos_de(R: Registro) -> np.ndarray:
    return np.vstack(R.objs) if R.objs else np.empty((0, 3))


def hipervolumen(P: np.ndarray, ref: np.ndarray) -> float:
    """Hipervolumen 3D exacto por rebanadas (minimización)."""
    P = np.asarray(P, float); ref = np.asarray(ref, float)
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 2])]
    total = 0.0
    for i in range(len(P)):
        z_sig = P[i + 1, 2] if i + 1 < len(P) else ref[2]
        if z_sig > P[i, 2]:
            total += _hv2d(P[: i + 1, :2], ref[:2]) * (z_sig - P[i, 2])
    return float(total)


def _hv2d(P, ref):
    P = P[np.argsort(P[:, 0])]
    area, mejor = 0.0, ref[1]
    for x, y in P:
        if y < mejor:
            area += (ref[0] - x) * (mejor - y); mejor = y
    return float(area)


def igd(A: np.ndarray, R: np.ndarray) -> float:
    d = np.sqrt(((R[:, None, :] - A[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return float(d.mean())
