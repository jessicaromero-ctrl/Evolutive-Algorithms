"""Evolución diferencial con manejo de restricciones y complementos.

Algoritmos:
  de_clasico  DE/rand/1/bin con F y CR fijos (Storn y Price, 1997).
  jade        JADE: DE/current-to-pbest/1 con archivo externo y adaptación de F y CR (Zhang y Sanderson, 2009).

Complementos:
  Manejo de restricciones por reglas de factibilidad de Deb (2000), sin coeficientes de penalización.
  Corrección de límites por punto medio entre el límite y el padre (Zhang y Sanderson, 2009).
  Archivo externo de padres reemplazados, que conserva diversidad en JADE.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from modelo_mecanizado import costo, violacion, cotas, PAR, Parametros

TOL = 1e-12


@dataclass
class Resultado:
    algoritmo: str
    semilla: int
    mejor_x: np.ndarray
    mejor_costo: float
    mejor_violacion: float
    evaluaciones: int
    historial: list = field(default_factory=list)   # (evaluaciones, mejor costo factible)
    segundos: float = 0.0
    parametros: dict = field(default_factory=dict)


def _factible(v):
    return v <= TOL


def regla_factibilidad(Va, Ca, Vb, Cb) -> bool:
    """True si a es preferible a b (Deb, 2000)."""
    fa, fb = _factible(Va), _factible(Vb)
    if fa and fb:
        return Ca < Cb
    if fa != fb:
        return fa
    return Va < Vb


def _mejor_indice(V, C):
    ok = V <= TOL
    if ok.any():
        return int(np.argmin(np.where(ok, C, np.inf)))
    return int(np.argmin(V))


def _evalua(x, p: Parametros):
    return float(costo(x, p)), float(violacion(x, p))


def _corrige(u, padre, lo, hi):
    """Corrección de límites por punto medio entre el límite violado y el padre."""
    u = np.where(u < lo, (lo + padre) / 2.0, u)
    u = np.where(u > hi, (hi + padre) / 2.0, u)
    return u


def _inicial(NP, lo, hi, rng, p):
    X = lo + rng.random((NP, len(lo))) * (hi - lo)
    C = np.empty(NP); V = np.empty(NP)
    for i in range(NP):
        C[i], V[i] = _evalua(X[i], p)
    return X, C, V


def de_clasico(semilla, presupuesto, NP=30, F=0.5, CR=0.9, p: Parametros = PAR) -> Resultado:
    rng = np.random.default_rng(semilla)
    lo, hi = cotas(p); d = len(lo)
    inicio = time.perf_counter()
    X, C, V = _inicial(NP, lo, hi, rng, p)
    n = NP; hist = []
    while n < presupuesto:
        for i in range(NP):
            if n >= presupuesto:
                break
            r1, r2, r3 = rng.choice([j for j in range(NP) if j != i], 3, replace=False)
            mutante = X[r1] + F * (X[r2] - X[r3])
            cruza = rng.random(d) < CR
            cruza[rng.integers(d)] = True
            u = _corrige(np.where(cruza, mutante, X[i]), X[i], lo, hi)
            cu, vu = _evalua(u, p); n += 1
            if regla_factibilidad(vu, cu, V[i], C[i]):
                X[i], C[i], V[i] = u, cu, vu
        hist.append((n, _mejor_factible(C, V)))
    k = _mejor_indice(V, C)
    return Resultado("DE clásico", semilla, X[k].copy(), float(C[k]), float(V[k]), n, hist,
                     time.perf_counter() - inicio, {"NP": NP, "F": F, "CR": CR, "presupuesto": presupuesto})


def jade(semilla, presupuesto, NP=30, p_top=0.05, c=0.1, p: Parametros = PAR) -> Resultado:
    rng = np.random.default_rng(semilla)
    lo, hi = cotas(p); d = len(lo)
    inicio = time.perf_counter()
    X, C, V = _inicial(NP, lo, hi, rng, p)
    n = NP; hist = []
    mu_F, mu_CR = 0.5, 0.5
    archivo = []
    while n < presupuesto:
        SF, SCR = [], []
        for i in range(NP):
            if n >= presupuesto:
                break
            # parámetros de control adaptados
            F = min(1.0, rng.standard_cauchy() * 0.1 + mu_F)
            while F <= 0:
                F = min(1.0, rng.standard_cauchy() * 0.1 + mu_F)
            CR = float(np.clip(rng.normal(mu_CR, 0.1), 0.0, 1.0))
            # pbest entre el p_top % mejores
            orden = np.lexsort((C, ~_factible_vec(V)))            # factibles primero, luego por costo
            top = orden[: max(1, int(round(p_top * NP)))]
            pb = X[rng.choice(top)]
            # r1 de la población, r2 de población ∪ archivo
            r1 = rng.choice([j for j in range(NP) if j != i])
            pool = np.vstack([X, np.array(archivo)]) if archivo else X
            while True:
                r2 = rng.integers(len(pool))
                if r2 != i and not np.array_equal(pool[r2], X[i]):
                    break
            mutante = X[i] + F * (pb - X[i]) + F * (X[r1] - pool[r2])
            cruza = rng.random(d) < CR
            cruza[rng.integers(d)] = True
            u = _corrige(np.where(cruza, mutante, X[i]), X[i], lo, hi)
            cu, vu = _evalua(u, p); n += 1
            if regla_factibilidad(vu, cu, V[i], C[i]):
                if len(archivo) < NP:
                    archivo.append(X[i].copy())
                else:
                    archivo[rng.integers(NP)] = X[i].copy()
                X[i], C[i], V[i] = u, cu, vu
                SF.append(F); SCR.append(CR)
        if SF:
            SF_ = np.array(SF); SCR_ = np.array(SCR)
            mu_F = (1 - c) * mu_F + c * float(np.sum(SF_ ** 2) / np.sum(SF_))     # media de Lehmer
            mu_CR = (1 - c) * mu_CR + c * float(np.mean(SCR_))
        hist.append((n, _mejor_factible(C, V)))
    k = _mejor_indice(V, C)
    return Resultado("JADE", semilla, X[k].copy(), float(C[k]), float(V[k]), n, hist,
                     time.perf_counter() - inicio, {"NP": NP, "p": p_top, "c": c, "presupuesto": presupuesto})


def _factible_vec(V):
    return V <= TOL


def _mejor_factible(C, V):
    ok = V <= TOL
    return float(C[ok].min()) if ok.any() else np.nan


ALGORITMOS = {"DE clásico": de_clasico, "JADE": jade}
