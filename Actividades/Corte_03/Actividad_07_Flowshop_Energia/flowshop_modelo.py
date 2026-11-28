"""Flow shop de permutación con energía: modelo, generación de instancias y evaluación por lotes.

Problema: n trabajos (lotes) procesados en m máquinas en serie, con el mismo orden de máquinas
para todos los trabajos (flow shop de permutación). La decisión es la secuencia π de los n trabajos.

Objetivos (todos de minimización):
  f1(π) = C_max          tiempo total de terminación (min)
  f2(π) = Σ_j w_j·T_j    tardanza ponderada (min·prioridad)
  f3(π) = E(π)           energía consumida (kWh), con política de apagado óptimo en huecos ociosos

Energía: en cada máquina i, el trabajo ocupado consume P_act_i (kW). Entre dos trabajos consecutivos,
el hueco g consume min(P_idle_i·g, E_arr_i): se mantiene encendida o se apaga y reinicia (costo E_arr_i).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

MAQUINAS = ("Mezclado", "Llenado", "Etiquetado", "Empaque")
NOMBRES_OBJ = ("C_max (min)", "Tardanza ponderada", "Energía (kWh)")


@dataclass(frozen=True)
class Instancia:
    nombre: str
    p: np.ndarray        # (n, m) tiempo de proceso del trabajo j en la máquina i (min)
    d: np.ndarray        # (n,) fecha de entrega (min)
    w: np.ndarray        # (n,) peso de prioridad
    P_act: np.ndarray    # (m,) potencia en operación (kW)
    P_idle: np.ndarray   # (m,) potencia en espera encendida (kW)
    E_arr: np.ndarray    # (m,) energía de arranque tras apagado (kWh)

    @property
    def n(self) -> int:
        return self.p.shape[0]

    @property
    def m(self) -> int:
        return self.p.shape[1]


PARAMETROS_MAQUINA = {
    "P_act": np.array([45.0, 60.0, 20.0, 25.0]),   # kW en operación
    "P_idle": np.array([16.0, 22.0, 8.0, 9.0]),    # kW en espera encendida (≈ 35–40 % de P_act)
    "E_arr": np.array([1.2, 2.0, 0.6, 0.8]),       # kWh por arranque tras apagado
}
RANGO_PROCESO = np.array([[10, 25], [8, 20], [5, 12], [6, 15]])   # min por lote, por máquina


def generar(nombre: str, n: int, semilla: int) -> Instancia:
    """Instancia sintética de una línea de envasado de bebidas, con supuestos documentados."""
    rng = np.random.default_rng(semilla)
    m = len(MAQUINAS)
    p = np.column_stack([rng.integers(RANGO_PROCESO[i, 0], RANGO_PROCESO[i, 1] + 1, size=n)
                         for i in range(m)]).astype(float)
    carga = p.sum(axis=0).max()                       # carga de la máquina más cargada
    lb = carga + p.sum(axis=1).min()                  # cota inferior simple del makespan
    d = np.round(rng.uniform(0.35, 0.95, size=n) * lb)
    w = rng.integers(1, 4, size=n).astype(float)      # prioridad 1 (baja) a 3 (alta)
    return Instancia(nombre, p, d, w, PARAMETROS_MAQUINA["P_act"].copy(),
                     PARAMETROS_MAQUINA["P_idle"].copy(), PARAMETROS_MAQUINA["E_arr"].copy())


def evaluar(inst: Instancia, perms: np.ndarray) -> np.ndarray:
    """Evalúa un lote de permutaciones (B, n). Devuelve (B, 3): (C_max, tardanza ponderada, energía)."""
    perms = np.asarray(perms, dtype=np.int64)
    if perms.ndim == 1:
        perms = perms[None, :]
    B, n = perms.shape
    m = inst.m
    P = inst.p[perms]                                             # (B, n, m)
    C = np.zeros((B, n, m))
    for j in range(n):
        for i in range(m):
            antes_trabajo = C[:, j - 1, i] if j > 0 else 0.0
            antes_maquina = C[:, j, i - 1] if i > 0 else 0.0
            C[:, j, i] = np.maximum(antes_trabajo, antes_maquina) + P[:, j, i]
    S = C - P                                                     # inicios
    cmax = C[:, -1, -1]
    T = np.maximum(0.0, C[:, :, -1] - inst.d[perms])
    tard = (inst.w[perms] * T).sum(axis=1)
    ocupado = (P.sum(axis=1) * inst.P_act / 60.0).sum(axis=1)     # kWh en operación
    huecos = S[:, 1:, :] - C[:, :-1, :]                             # (B, n-1, m), ≥ 0
    ocioso = np.minimum(inst.P_idle * huecos / 60.0, inst.E_arr).sum(axis=(1, 2))
    energia = ocupado + ocioso + inst.E_arr.sum()
    return np.column_stack([cmax, tard, energia])


def todas_las_permutaciones(n: int, lote: int = 40_000):
    """Genera todas las permutaciones de 0..n-1 en lotes (enumeración exacta)."""
    it = itertools.permutations(range(n))
    while True:
        bloque = list(itertools.islice(it, lote))
        if not bloque:
            return
        yield np.array(bloque, dtype=np.int64)


def frente_no_dominado(F: np.ndarray) -> np.ndarray:
    """Índices del frente de Pareto (minimización), sin vectores duplicados.
    Orden lexicográfico: ningún punto posterior puede dominar a uno anterior."""
    F = np.asarray(F, float)
    orden = np.lexsort((F[:, 2], F[:, 1], F[:, 0]))
    kept: list[int] = []
    for k in orden:
        f = F[k]
        if kept:
            K = F[kept]
            if np.any(np.all(K <= f, axis=1) & np.any(K < f, axis=1)):
                continue
            if np.any(np.all(K == f, axis=1)):
                continue
        kept.append(int(k))
    return np.array(kept, dtype=int)


def frente_exacto(inst: Instancia):
    """Frente de Pareto exacto por enumeración de todas las permutaciones (instancias pequeñas)."""
    vec, per = [], []
    for lote in todas_las_permutaciones(inst.n):
        vec.append(evaluar(inst, lote)); per.append(lote)
    F = np.vstack(vec); X = np.vstack(per)
    idx = frente_no_dominado(F)
    return F[idx], X[idx], F
