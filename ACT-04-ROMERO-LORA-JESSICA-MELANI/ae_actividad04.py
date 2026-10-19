#!/usr/bin/env python3
"""
Actividad 04 · Algoritmo genético para fijar precios de temporada.

Versión refactorizada del cuadernillo original. Mantiene las reglas del modelo,
las semillas, el orden de los sorteos aleatorios y las cifras del experimento,
pero organiza el código en funciones independientes, quita código muerto,
ejecuta las pruebas como funciones explícitas y comprueba automáticamente las
cifras publicadas en el README original.

Uso:
    python ae_actividad04.py                      # salida en ./resultados_actividad04
    python ae_actividad04.py --salida carpeta     # salida en otra carpeta
    python ae_actividad04.py --sin-figuras        # sólo tablas y pruebas

Dependencias: numpy, pandas, scipy, matplotlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.optimize import minimize
from scipy.stats import friedmanchisquare, norm, rankdata, wilcoxon

# ============================================================================
# 1. Constantes del estudio
# ============================================================================
CRUCES = ("M14", "M15", "M16")
MUTACIONES = ("M18", "M19", "M20")
CONFIGURACIONES = [(c, m) for c in CRUCES for m in MUTACIONES]
ORDEN_CONFIG = [f"{c}+{m}" for c, m in CONFIGURACIONES]
NOMBRE = {
    "M14": "M14 uniforme", "M15": "M15 aritmética", "M16": "M16 BLX-α",
    "M18": "M18 reinicio", "M19": "M19 gaussiana", "M20": "M20 polinomial",
}
NIVELES = ("marca propia", "marca estándar", "marca premium")

# Parámetros del modelo y del generador de instancias
MARGEN_ACTUAL = 0.45        # regla vigente: precio = costo × 1.45
MARGEN_MINIMO = 0.10        # piso: costo × 1.10
ALZA_MAXIMA = 0.20          # techo: precio actual × 1.20
BRECHA_ESCALERA = 0.08
HOLGURA_INDICE = 0.03
MULTIPLICADOR_COSTO = np.array([0.70, 1.00, 1.40])
CUOTA_NIVEL = np.array([0.35, 0.45, 0.20])
ELASTICIDAD = ((3.0, 5.5), (2.0, 3.8), (1.3, 2.4))
SUSTITUCION = {"D": (0.10, 0.05), "F": (0.30, 0.12)}   # (κ intra-familia, suma inter-familia)
EXISTENCIAS = (1.05, 1.50)

# Parámetros comunes de los algoritmos genéticos
N_POBLACION = 100
K_TORNEO = 2
E_ELITE = 2
P_CRUCE = 0.9
E_MAX = 20_000
TOL_FACTIBLE = 1e-12        # tolerancia relativa para la factibilidad (sólo redondeo de punto flotante)
EVALUACIONES_AG = N_POBLACION + ((E_MAX - N_POBLACION) // (N_POBLACION - E_ELITE)) * (N_POBLACION - E_ELITE)
GEN_25, GEN_50 = 50, 101    # generaciones que corresponden a 5 000 y 9 998 evaluaciones

REJILLA = {"alfa": (0.1, 0.3, 0.5), "sigma": (0.02, 0.05, 0.10), "eta": (5, 20, 50)}
SEMILLAS_CALIBRACION = list(range(20261200, 20261210))
SEMILLAS_EVALUACION = list(range(20261017, 20261047))
FAMILIAS_PRINCIPAL = ("Agua natural 1.5 L", "Refresco de cola 600 mL",
                      "Jugo de naranja 1 L", "Bebida isotónica 600 mL")
DEFINICION = {
    "C24-F": dict(n_familias=8, intensidad="F", semilla=20261090, uso="calibración"),
    "E12-D": dict(n_familias=4, intensidad="D", semilla=20261091, uso="evaluación"),
    "E12-F": dict(n_familias=4, intensidad="F", semilla=20261092, uso="evaluación y caso principal",
                  familias=FAMILIAS_PRINCIPAL, costos_base=(11.0, 13.0, 22.0, 16.0),
                  volumenes=(3000, 3500, 1000, 1500)),
    "E36-D": dict(n_familias=12, intensidad="D", semilla=20261093, uso="evaluación"),
    "E36-F": dict(n_familias=12, intensidad="F", semilla=20261094, uso="evaluación"),
}
INSTANCIAS_EVALUACION = ["E12-D", "E12-F", "E36-D", "E36-F"]

# Colores y trazos: el color codifica la recombinación, el trazo y la marca la mutación
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, TENUE = "#0b0b0b", "#52514e", "#898781"
REJILLA_COLOR, EJE = "#e1e0d9", "#c3c2b7"
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COLOR_CRUCE = {"M14": AZUL, "M15": NARANJA, "M16": AQUA}
TRAZO_MUTACION = {"M18": ":", "M19": "--", "M20": "-"}
MARCA_MUTACION = {"M18": "o", "M19": "s", "M20": "^"}

# Directorio de salida (se fija en main)
RESULTADOS = Path("resultados")
FIGURAS = Path("figuras")


# ============================================================================
# 2. Modelo del problema
# ============================================================================
@dataclass(frozen=True)
class Instancia:
    """Caso de fijación de precios. Precios en pesos; demanda en unidades por temporada."""
    nombre: str
    familias: tuple
    familia: np.ndarray         # familia de cada producto
    nivel: np.ndarray           # 0 propia, 1 estándar, 2 premium
    costo: np.ndarray           # c_j
    precio_actual: np.ndarray   # P_j
    demanda_actual: np.ndarray  # q0_j = q_j(P)
    B: np.ndarray               # pendientes: simétrica y definida positiva
    a: np.ndarray               # intercepto: q(p) = a - B p
    existencias: np.ndarray     # S_j
    piso: np.ndarray            # l_j
    techo: np.ndarray           # u_j
    G: np.ndarray               # reglas comerciales: G p <= h
    h: np.ndarray
    tipo: tuple                 # (regla, producto) de cada fila de G
    brecha_escalera: float
    holgura_indice: float
    pesos_indice: np.ndarray

    @property
    def d(self) -> int:
        return len(self.costo)

    def producto(self, j: int) -> str:
        return f"{self.familias[self.familia[j]]} · {NIVELES[self.nivel[j]]}"

    def demanda(self, X):
        return self.a - np.asarray(X) @ self.B


def ensamblar_instancia(nombre, familias, familia, nivel, costo, precio_actual, demanda_actual, B,
                        existencias, piso, techo, brecha_escalera, holgura_indice) -> Instancia:
    """Calcula el intercepto de la demanda y escribe las reglas comerciales como G p <= h."""
    familia, nivel = np.asarray(familia, int), np.asarray(nivel, int)
    B, P = np.asarray(B, float), np.asarray(precio_actual, float)
    q0, S = np.asarray(demanda_actual, float), np.asarray(existencias, float)
    a = q0 + B @ P
    d = len(P)
    filas, lados, tipo = [], [], []
    for j in range(d):                          # existencias: a_j - B_j p <= S_j
        filas.append(-B[j] / S[j]); lados.append((S[j] - a[j]) / S[j]); tipo.append(("existencias", j))
    for j in range(d):                          # demanda no negativa: B_j p - a_j <= 0
        filas.append(B[j] / q0[j]); lados.append(a[j] / q0[j]); tipo.append(("demanda", j))
    for f in np.unique(familia):                # escalera: (1 + g) p_inferior - p_superior <= 0
        miembros = np.flatnonzero(familia == f)
        miembros = miembros[np.argsort(nivel[miembros], kind="stable")]
        for i, s in zip(miembros[:-1], miembros[1:]):
            fila = np.zeros(d)
            fila[i] = 1.0 + brecha_escalera
            fila[s] = -1.0
            filas.append(fila / P[s]); lados.append(0.0); tipo.append(("escalera", int(s)))
    w = q0 * P / np.sum(q0 * P)                 # índice de precios ponderado por el ingreso actual
    filas.append(w / P); lados.append(1.0 + holgura_indice); tipo.append(("índice", -1))
    return Instancia(nombre, tuple(familias), familia, nivel, np.asarray(costo, float), P, q0, B, a, S,
                     np.asarray(piso, float), np.asarray(techo, float), np.array(filas), np.array(lados),
                     tuple(tipo), float(brecha_escalera), float(holgura_indice), w)


def validar_instancia(inst: Instancia) -> None:
    """Rechaza datos incoherentes; no corrige nada en silencio."""
    d = inst.d
    if inst.B.shape != (d, d) or not np.allclose(inst.B, inst.B.T, rtol=0, atol=1e-12):
        raise ValueError(f"{inst.nombre}: B debe ser simétrica de {d}×{d}.")
    if np.linalg.eigvalsh(inst.B).min() <= 0:
        raise ValueError(f"{inst.nombre}: B no es definida positiva; la utilidad no sería estrictamente cóncava.")
    if not (np.all(inst.costo < inst.piso) and np.all(inst.piso < inst.precio_actual)
            and np.all(inst.precio_actual <= inst.techo)):
        raise ValueError(f"{inst.nombre}: se requiere costo < piso < precio actual <= techo.")
    if not np.allclose(inst.demanda(inst.precio_actual), inst.demanda_actual, rtol=1e-12, atol=1e-9):
        raise ValueError(f"{inst.nombre}: la demanda al precio actual no coincide con q0.")
    if np.any(inst.G @ inst.precio_actual > inst.h + 1e-12):
        raise ValueError(f"{inst.nombre}: los precios actuales deben cumplir las reglas comerciales.")


def generar_instancia(nombre: str, n_familias: int, intensidad: str, semilla: int,
                      familias=None, costos_base=None, volumenes=None) -> Instancia:
    """Genera una categoría de n_familias × 3 productos con la intensidad de sustitución dada."""
    if n_familias < 2 or intensidad not in SUSTITUCION:
        raise ValueError("Se requieren al menos dos familias y una intensidad 'D' o 'F'.")
    rng = np.random.default_rng(semilla)
    F, d = n_familias, 3 * n_familias
    familia, nivel = np.repeat(np.arange(F), 3), np.tile(np.arange(3), F)
    familias = tuple(familias) if familias is not None else tuple(f"Familia {f + 1:02d}" for f in range(F))
    base = np.asarray(costos_base, float) if costos_base is not None else rng.uniform(8, 30, F)
    volumen = np.asarray(volumenes, float) if volumenes is not None else rng.uniform(800, 3500, F)
    costo = np.round(base[familia] * MULTIPLICADOR_COSTO[nivel] * rng.uniform(0.95, 1.05, d), 2)
    precio = np.round(costo * (1 + MARGEN_ACTUAL), 1)
    q0 = np.round(volumen[familia] * CUOTA_NIVEL[nivel] * rng.uniform(0.85, 1.15, d))
    elasticidad = np.array([rng.uniform(*ELASTICIDAD[t]) for t in nivel])
    b = elasticidad * q0 / precio
    kappa, entre = SUSTITUCION[intensidad]
    K = np.where(familia[:, None] == familia[None, :], kappa, entre / (d - 3))
    np.fill_diagonal(K, 0.0)
    raiz = np.sqrt(b)
    B = np.diag(b) - K * np.outer(raiz, raiz)
    existencias = np.round(q0 * rng.uniform(*EXISTENCIAS, d))
    piso = np.round(costo * (1 + MARGEN_MINIMO), 2)
    techo = np.round(precio * (1 + ALZA_MAXIMA), 2)
    inst = ensamblar_instancia(nombre, familias, familia, nivel, costo, precio, q0, B, existencias,
                               piso, techo, BRECHA_ESCALERA, HOLGURA_INDICE)
    validar_instancia(inst)
    return inst


def utilidad(inst: Instancia, X) -> np.ndarray:
    """Utilidad de la temporada de cada fila de X (pesos)."""
    X = np.atleast_2d(X)
    return np.sum((X - inst.costo) * (inst.a - X @ inst.B), axis=1)


def violacion(inst: Instancia, X) -> np.ndarray:
    """Violación agregada V de cada fila de X; vale 0 exactamente en el conjunto factible."""
    X = np.atleast_2d(X)
    return np.maximum(0.0, X @ inst.G.T - inst.h).sum(axis=1)


def evaluar_escalar(inst: Instancia, p) -> dict:
    """Recalcula utilidad y reglas con bucles, a partir de su significado y sin usar G ni h."""
    d = inst.d
    p = [float(v) for v in p]
    q = [inst.a[j] - sum(inst.B[j, k] * p[k] for k in range(d)) for j in range(d)]
    valor = sum((p[j] - inst.costo[j]) * q[j] for j in range(d))
    v = 0.0
    for j in range(d):
        v += max(0.0, (q[j] - inst.existencias[j]) / inst.existencias[j])
        v += max(0.0, -q[j] / inst.demanda_actual[j])
    for f in range(len(inst.familias)):
        miembros = sorted((j for j in range(d) if inst.familia[j] == f), key=lambda j: inst.nivel[j])
        for i, s in zip(miembros[:-1], miembros[1:]):
            v += max(0.0, ((1 + inst.brecha_escalera) * p[i] - p[s]) / inst.precio_actual[s])
    indice = sum(inst.pesos_indice[j] * p[j] / inst.precio_actual[j] for j in range(d))
    v += max(0.0, indice - (1 + inst.holgura_indice))
    en_caja = all(inst.piso[j] <= p[j] <= inst.techo[j] for j in range(d))
    return {"utilidad": valor, "violacion": v, "demanda": q, "indice": indice, "en_caja": en_caja}


# ============================================================================
# 3. Óptimo exacto (certificado KKT) y líneas base
# ============================================================================
def gradiente_utilidad(inst: Instancia, p: np.ndarray) -> np.ndarray:
    return inst.a + inst.B @ inst.costo - 2.0 * inst.B @ p


def matriz_completa(inst: Instancia):
    """Todas las restricciones como A p <= b: reglas comerciales, techos y pisos."""
    d = inst.d
    A = np.vstack([inst.G, np.eye(d), -np.eye(d)])
    b = np.concatenate([inst.h, inst.techo, -inst.piso])
    nombres = list(inst.tipo) + [("techo", j) for j in range(d)] + [("piso", j) for j in range(d)]
    return A, b, nombres


def certificar_kkt(inst: Instancia, x: np.ndarray, tolerancia: float = 1e-9, max_iter: int = 100) -> dict:
    """Resuelve el sistema KKT sobre un conjunto activo y lo corrige hasta cumplir todas las condiciones."""
    d = inst.d
    A, b, nombres = matriz_completa(inst)
    H, g = 2.0 * inst.B, inst.a + inst.B @ inst.costo
    activos = set(np.flatnonzero(b - A @ x < 1e-7).tolist())
    for iteracion in range(1, max_iter + 1):
        act = np.array(sorted(activos), dtype=int)
        m = len(act)
        K = np.block([[H, A[act].T], [A[act], np.zeros((m, m))]])
        solucion = np.linalg.lstsq(K, np.concatenate([g, b[act]]), rcond=None)[0]
        x, lam = solucion[:d], solucion[d:]
        holgura = b - A @ x
        escala = max(1.0, float(np.abs(lam).max())) if m else 1.0
        if m and lam.min() < -tolerancia * escala:
            activos.discard(int(act[np.argmin(lam)]))
            continue
        if holgura.min() < -tolerancia:
            activos.add(int(np.argmin(holgura)))
            continue
        residuo = (float(np.max(np.abs(g - H @ x - A[act].T @ lam))) if m
                   else float(np.max(np.abs(g - H @ x))))
        return {"precios": x, "utilidad": float(utilidad(inst, x)[0]),
                "activas": [nombres[i] for i in act], "multiplicadores": lam, "residuo": residuo,
                "max_violacion": float(max(0.0, -holgura.min())), "iteraciones": iteracion}
    raise RuntimeError(f"{inst.nombre}: el conjunto activo no convergió; no hay certificado.")


def resolver_exacto(inst: Instancia) -> dict:
    """Óptimo global de la utilidad: SLSQP desde cuatro arranques y certificado KKT."""
    objetivo = lambda p: -utilidad(inst, p)[0]
    gradiente = lambda p: -gradiente_utilidad(inst, p)
    reglas = [{"type": "ineq", "fun": lambda p: inst.h - inst.G @ p, "jac": lambda p: -inst.G}]
    cotas = list(zip(inst.piso, inst.techo))
    arranques = {"precio actual": inst.precio_actual, "piso": inst.piso, "techo": inst.techo,
                 "punto medio": (inst.piso + inst.techo) / 2}
    inicio = time.perf_counter()
    puntos = {}
    for nombre, x0 in arranques.items():
        r = minimize(objetivo, x0, jac=gradiente, method="SLSQP", bounds=cotas, constraints=reglas,
                     options={"ftol": 1e-15, "maxiter": 5000})
        puntos[nombre] = np.clip(r.x, inst.piso, inst.techo)
    mejor = max(puntos.values(), key=lambda x: utilidad(inst, x)[0] - 1e9 * violacion(inst, x)[0])
    cert = certificar_kkt(inst, mejor)
    cert["segundos"] = time.perf_counter() - inicio
    cert["slsqp"] = {k: (float(utilidad(inst, x)[0]), float(violacion(inst, x)[0])) for k, x in puntos.items()}
    if cert["max_violacion"] > 1e-9 or cert["residuo"] > 1e-6 * max(1.0, np.abs(inst.a).max()):
        raise RuntimeError(f"{inst.nombre}: el certificado no alcanzó la tolerancia.")
    return cert


def ajuste_parejo(inst: Instancia, paso: float = 0.0005) -> dict:
    """Mejor cambio porcentual común a todos los precios, recortado a la caja de cada producto."""
    t = np.round(np.arange(-0.25, ALZA_MAXIMA + paso / 2, paso), 4)
    X = np.clip(inst.precio_actual[None, :] * (1 + t[:, None]), inst.piso, inst.techo)
    U = np.where(violacion(inst, X) <= TOL_FACTIBLE, utilidad(inst, X), -np.inf)
    i = int(np.argmax(U))
    return {"t": float(t[i]), "precios": X[i], "utilidad": float(U[i])}


# ============================================================================
# 4. Las siete etapas del algoritmo genético
# ============================================================================
# Etapa 1 · M03 hipercubo latino (fija)
def etapa_1_inicializar_M03(N: int, inst: Instancia, rng: np.random.Generator) -> np.ndarray:
    d = inst.d
    estratos = rng.permuted(np.tile(np.arange(N), (d, 1)), axis=1).T
    U = (estratos + rng.random((N, d))) / N
    return inst.piso + U * (inst.techo - inst.piso)


# Etapa 2 · M08 prioridad de factibilidad (fija)
def etapa_2_evaluar_M08(inst: Instancia, X: np.ndarray) -> dict:
    V, U = violacion(inst, X), utilidad(inst, X)
    V = np.where(V <= TOL_FACTIBLE, 0.0, V)
    return {"V": V, "utilidad": U, "factible": V == 0.0}


def orden_M08(ev: dict) -> np.ndarray:
    """Índices del mejor al peor: menor violación y, a igual violación, mayor utilidad."""
    return np.lexsort((-ev["utilidad"], ev["V"]))


def rangos_densos_M08(ev: dict) -> np.ndarray:
    """Rango 0 = mejor; claves idénticas comparten rango."""
    o = orden_M08(ev)
    V, U = ev["V"][o], ev["utilidad"][o]
    nuevo = np.ones(len(o), dtype=bool)
    nuevo[1:] = (V[1:] != V[:-1]) | (U[1:] != U[:-1])
    rangos = np.empty(len(o), dtype=int)
    rangos[o] = np.cumsum(nuevo) - 1
    return rangos


# Etapa 3 · M12 torneo con reemplazo (fija)
def torneo_desde_participantes(rangos: np.ndarray, participantes: np.ndarray) -> np.ndarray:
    return participantes[np.arange(len(participantes)), np.argmin(rangos[participantes], axis=1)]


def etapa_3_seleccionar_M12(rangos: np.ndarray, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return torneo_desde_participantes(rangos, rng.integers(0, len(rangos), size=(n, k)))


# Etapa 4 · Recombinación: M14, M15 y M16 (comparadas)
def cruce_con_mascara(A: np.ndarray, B: np.ndarray, mascara: np.ndarray):
    m = mascara.astype(bool)
    return np.where(m, A, B), np.where(m, B, A)


def etapa_4_recombinar_M14(A, B, rng: np.random.Generator, q: float = 0.5):
    return cruce_con_mascara(A, B, rng.random(A.shape) < q)


def combinacion_aritmetica(A, B, alfa):
    return alfa * A + (1 - alfa) * B, (1 - alfa) * A + alfa * B


def etapa_4_recombinar_M15(A, B, rng: np.random.Generator):
    return combinacion_aritmetica(A, B, rng.random((A.shape[0], 1)))


def etapa_4_recombinar_M16(A, B, rng: np.random.Generator, alfa: float, piso, techo):
    bajo, alto = np.minimum(A, B), np.maximum(A, B)
    I = alto - bajo
    H1 = rng.uniform(bajo - alfa * I, alto + alfa * I)
    H2 = rng.uniform(bajo - alfa * I, alto + alfa * I)
    return np.clip(H1, piso, techo), np.clip(H2, piso, techo)


RECOMBINAR = {
    "M14": lambda A, B, rng, inst, par: etapa_4_recombinar_M14(A, B, rng),
    "M15": lambda A, B, rng, inst, par: etapa_4_recombinar_M15(A, B, rng),
    "M16": lambda A, B, rng, inst, par: etapa_4_recombinar_M16(A, B, rng, par["alfa"], inst.piso, inst.techo),
}


# Etapa 5 · Mutación: M18, M19 y M20 (comparadas)
def etapa_5_mutar_M18(H, pm, rng: np.random.Generator, piso, techo):
    m = rng.random(H.shape) < pm
    return np.where(m, piso + rng.random(H.shape) * (techo - piso), H)


def etapa_5_mutar_M19(H, pm, rng: np.random.Generator, piso, techo, sigma: float):
    m = rng.random(H.shape) < pm
    Y = np.clip(H + sigma * (techo - piso) * rng.standard_normal(H.shape), piso, techo)
    return np.where(m, Y, H)


def etapa_5_mutar_M20(H, pm, rng: np.random.Generator, piso, techo, eta: float):
    m = rng.random(H.shape) < pm
    R = techo - piso
    d1, d2 = (H - piso) / R, (techo - H) / R
    r = rng.random(H.shape)
    e1 = eta + 1.0
    izquierda = (2 * r + (1 - 2 * r) * (1 - d1) ** e1) ** (1 / e1) - 1
    derecha = 1 - (2 * (1 - r) + 2 * (r - 0.5) * (1 - d2) ** e1) ** (1 / e1)
    Y = H + np.where(r <= 0.5, izquierda, derecha) * R
    Y = np.minimum(np.maximum(Y, piso), techo)   # sólo absorbe error de punto flotante
    return np.where(m, Y, H)


MUTAR = {
    "M18": lambda H, pm, rng, inst, par: etapa_5_mutar_M18(H, pm, rng, inst.piso, inst.techo),
    "M19": lambda H, pm, rng, inst, par: etapa_5_mutar_M19(H, pm, rng, inst.piso, inst.techo, par["sigma"]),
    "M20": lambda H, pm, rng, inst, par: etapa_5_mutar_M20(H, pm, rng, inst.piso, inst.techo, par["eta"]),
}
PARAMETRO_DE = {"M14": None, "M15": None, "M16": "alfa", "M18": None, "M19": "sigma", "M20": "eta"}


# Etapa 6 · Reemplazo: M22 generacional con elitismo (fija)
def etapa_6_reemplazar_M22(P, ev, H, evH, e: int):
    elite = orden_M08(ev)[:e]
    P_nueva = np.vstack([P[elite], H])
    ev_nueva = {k: np.concatenate([ev[k][elite], evH[k]]) for k in ev}
    return P_nueva, ev_nueva


# Etapa 7 · Paro: M26 presupuesto de evaluaciones (fija)
def etapa_7_parar_M26(evaluaciones: int, por_generacion: int, E_max: int) -> bool:
    return evaluaciones + por_generacion > E_max


# ============================================================================
# 5. Integración del algoritmo y líneas base
# ============================================================================
def dispersion(P: np.ndarray, inst: Instancia) -> float:
    return float(np.mean(P.std(axis=0) / (inst.techo - inst.piso)))


def algoritmo_genetico(inst: Instancia, cruce: str, mutacion: str, semilla: int, parametros: dict | None = None,
                       N: int = N_POBLACION, k: int = K_TORNEO, e: int = E_ELITE, pc: float = P_CRUCE,
                       pm: float | None = None, E_max: int = E_MAX, historial: bool = True) -> dict:
    """Siete etapas: M03, M08, M12, recombinación y mutación elegidas, M22 y M26."""
    par = dict(parametros or {})
    d = inst.d
    pm = 1.0 / d if pm is None else pm
    rng = np.random.default_rng(semilla)               # un solo generador por corrida
    inicio = time.perf_counter()
    P = etapa_1_inicializar_M03(N, inst, rng)          # ETAPA 1
    ev = etapa_2_evaluar_M08(inst, P)                  # ETAPA 2
    evaluaciones, generacion = N, 0
    hijos, parejas = N - e, (N - e + 1) // 2
    mejor = {"utilidad": -np.inf, "precios": None, "evaluaciones": None}
    registro, clave_anterior = [], (np.inf, np.inf)

    def registrar(frac_hijos: float):
        nonlocal clave_anterior
        i = int(orden_M08(ev)[0])
        clave = (float(ev["V"][i]), float(-ev["utilidad"][i]))
        if clave > clave_anterior:
            raise RuntimeError("La mejor clave empeoró: el elitismo no se cumplió.")
        clave_anterior = clave
        if ev["factible"][i] and ev["utilidad"][i] > mejor["utilidad"]:
            mejor.update(utilidad=float(ev["utilidad"][i]), precios=P[i].copy(), evaluaciones=evaluaciones)
        if historial:
            en_cota = np.mean((P == inst.piso) | (P == inst.techo))
            registro.append({"generacion": generacion, "evaluaciones": evaluaciones,
                             "mejor_factible": mejor["utilidad"] if mejor["precios"] is not None else np.nan,
                             "fraccion_factible": float(ev["factible"].mean()),
                             "fraccion_hijos_factibles": frac_hijos,
                             "dispersion": dispersion(P, inst), "en_cota": float(en_cota)})

    registrar(np.nan)
    while not etapa_7_parar_M26(evaluaciones, hijos, E_max):                # ETAPA 7
        rangos = rangos_densos_M08(ev)
        indices = etapa_3_seleccionar_M12(rangos, 2 * parejas, k, rng)      # ETAPA 3
        A, B = P[indices[0::2]], P[indices[1::2]]
        H1, H2 = RECOMBINAR[cruce](A, B, rng, inst, par)                    # ETAPA 4
        cruza = rng.random(parejas) < pc
        H1 = np.where(cruza[:, None], H1, A)
        H2 = np.where(cruza[:, None], H2, B)
        H = np.empty((2 * parejas, d))
        H[0::2], H[1::2] = H1, H2
        H = MUTAR[mutacion](H[:hijos], pm, rng, inst, par)                  # ETAPA 5
        evH = etapa_2_evaluar_M08(inst, H)                                  # repetir ETAPA 2
        evaluaciones += hijos
        P, ev = etapa_6_reemplazar_M22(P, ev, H, evH, e)                    # ETAPA 6
        generacion += 1
        registrar(float(evH["factible"].mean()))

    if evaluaciones != N + generacion * (N - e) or evaluaciones > E_max:
        raise RuntimeError("El contador de evaluaciones no coincide con N + G(N - e).")
    if mejor["precios"] is not None:
        auditoria = evaluar_escalar(inst, mejor["precios"])
        if (auditoria["violacion"] > 10 * TOL_FACTIBLE or not auditoria["en_caja"]
                or abs(auditoria["utilidad"] - mejor["utilidad"]) > 1e-8 * abs(mejor["utilidad"])):
            raise RuntimeError("La solución entregada no coincide con su evaluación escalar.")
    return {"utilidad": mejor["utilidad"] if mejor["precios"] is not None else np.nan,
            "precios": mejor["precios"], "eval_mejor": mejor["evaluaciones"], "evaluaciones": evaluaciones,
            "generaciones": generacion, "segundos": time.perf_counter() - inicio, "historial": registro,
            "poblacion": P, "fraccion_factible": float(ev["factible"].mean()), "dispersion": dispersion(P, inst),
            "en_cota": float(np.mean((P == inst.piso) | (P == inst.techo))),
            "parametros": {"cruce": cruce, "mutacion": mutacion, **par, "N": N, "k": k, "e": e, "pc": pc,
                           "pm": pm, "E_max": E_max, "semilla": semilla}}


def busqueda_aleatoria(inst: Instancia, semilla: int, evaluaciones: int, N: int = N_POBLACION,
                       por_bloque: int = N_POBLACION - E_ELITE) -> dict:
    """Muestreo uniforme en la caja con presupuesto idéntico; registra con la granularidad del AG."""
    rng = np.random.default_rng(semilla)
    mejor_V, mejor_U, mejor_x = np.inf, -np.inf, None
    usadas, trayectoria = 0, []
    bloque = N
    while usadas + bloque <= evaluaciones:
        X = inst.piso + rng.random((bloque, inst.d)) * (inst.techo - inst.piso)
        ev = etapa_2_evaluar_M08(inst, X)
        V, U = ev["V"], ev["utilidad"]
        usadas += bloque
        i = int(orden_M08(ev)[0])
        if (V[i], -U[i]) < (mejor_V, -mejor_U):
            mejor_V, mejor_U, mejor_x = float(V[i]), float(U[i]), X[i].copy()
        trayectoria.append({"evaluaciones": usadas, "mejor_factible": mejor_U if mejor_V <= 0 else np.nan})
        bloque = por_bloque
    factible = mejor_V <= 0
    return {"utilidad": mejor_U if factible else np.nan, "factible": factible, "precios": mejor_x,
            "violacion_minima": mejor_V, "evaluaciones": usadas, "trayectoria": trayectoria}


# ============================================================================
# 6. Pruebas automatizadas (se ejecutan antes de cualquier experimento)
# ============================================================================
def instancia_manual(costo, precio, q0, B, existencias, piso, techo, holgura=10.0) -> Instancia:
    """Instancia pequeña para pruebas: cada producto en su propia familia (sin escalera)."""
    d = len(costo)
    inst = ensamblar_instancia("manual", tuple(f"F{f}" for f in range(d)), np.arange(d), np.zeros(d, int),
                               costo, precio, q0, B, existencias, piso, techo, BRECHA_ESCALERA, holgura)
    validar_instancia(inst)
    return inst


def muestras_cercanas(inst, centro, n, semilla, escala=0.08):
    rng = np.random.default_rng(semilla)
    return np.clip(centro * (1 + escala * rng.uniform(-1, 1, (n, inst.d))), inst.piso, inst.techo)


def muestras_factibles(inst, n, semilla):
    X = muestras_cercanas(inst, inst.precio_actual, 50_000, semilla)
    X = X[violacion(inst, X) <= TOL_FACTIBLE]
    assert len(X) >= n, "no hay suficientes muestras factibles para la prueba"
    return X[:n]


def prueba_01():
    for malo in (replace(INST_PRUEBA, B=-INST_PRUEBA.B), replace(INST_PRUEBA, piso=INST_PRUEBA.costo * 0.9)):
        try:
            validar_instancia(malo)
        except ValueError:
            continue
        raise AssertionError("la validación aceptó una instancia inválida")


def prueba_02():
    inst = INST_PRUEBA_F
    X = np.vstack([inst.piso + np.random.default_rng(1).random((250, inst.d)) * (inst.techo - inst.piso),
                   muestras_cercanas(inst, inst.precio_actual, 250, 1)])
    U, V = utilidad(inst, X), violacion(inst, X)
    for i in range(len(X)):
        r = evaluar_escalar(inst, X[i])
        assert abs(r["utilidad"] - U[i]) <= 1e-9 * abs(U[i]) and abs(r["violacion"] - V[i]) <= 1e-12
    assert 0 < np.mean(V <= TOL_FACTIBLE) < 1, "la muestra debe mezclar factibles e inviables"


def prueba_03():
    base = dict(costo=[10.0], precio=[20.0], q0=[60.0], B=[[2.0]], existencias=[1000.0], piso=[11.0])
    r = resolver_exacto(instancia_manual(**base, techo=[60.0]))
    assert abs(r["precios"][0] - 30) < 1e-9 and abs(r["utilidad"] - 800) < 1e-7 and not r["activas"]
    r = resolver_exacto(instancia_manual(**base, techo=[25.0]))
    assert abs(r["precios"][0] - 25) < 1e-9 and abs(r["utilidad"] - 750) < 1e-7
    assert r["activas"] == [("techo", 0)] and abs(r["multiplicadores"][0] - 20) < 1e-7


def prueba_04():
    inst = instancia_manual(costo=[10.0, 10.0], precio=[20.0, 20.0], q0=[80.0, 80.0],
                            B=[[2.0, -1.0], [-1.0, 2.0]], existencias=[1000.0, 1000.0],
                            piso=[11.0, 11.0], techo=[100.0, 100.0])
    r = resolver_exacto(inst)
    assert np.allclose(r["precios"], 55, atol=1e-9) and abs(r["utilidad"] - 4050) < 1e-6


def prueba_05():
    inst = INST_PRUEBA_F
    r = resolver_exacto(inst)
    assert r["max_violacion"] <= 1e-9 and r["residuo"] <= 1e-6 * np.abs(inst.a).max()
    assert len(r["multiplicadores"]) == 0 or r["multiplicadores"].min() >= -1e-9 * max(1, np.abs(r["multiplicadores"]).max())
    for u, v in r["slsqp"].values():
        assert v > 1e-9 or u <= r["utilidad"] * (1 + 1e-9), "un arranque factible superó al certificado"
    X = muestras_cercanas(inst, r["precios"], 20_000, 2, escala=0.03)
    factibles = violacion(inst, X) <= TOL_FACTIBLE
    assert factibles.sum() > 100 and utilidad(inst, X[factibles]).max() < r["utilidad"]


def prueba_06():
    inst = INST_PRUEBA_F
    P = etapa_1_inicializar_M03(40, inst, np.random.default_rng(3))
    U = (P - inst.piso) / (inst.techo - inst.piso)
    assert np.all((U >= 0) & (U < 1))
    assert all(sorted(np.floor(U[:, j] * 40).astype(int)) == list(range(40)) for j in range(inst.d))
    assert np.array_equal(P, etapa_1_inicializar_M03(40, inst, np.random.default_rng(3)))


def prueba_07():
    ev = {"V": np.array([0.0, 0.0, 0.5, 0.1]), "utilidad": np.array([100.0, 120.0, 500.0, 50.0])}
    assert list(orden_M08(ev)) == [1, 0, 3, 2]
    penalizada = -ev["utilidad"] + 100.0 * ev["V"]          # M07 con coeficiente bajo
    assert int(np.argmin(penalizada)) == 2 and ev["V"][2] > 0


def prueba_08():
    rangos = rangos_densos_M08({"V": np.zeros(3), "utilidad": np.array([9.0, 5.0, 1.0])})
    assert list(rangos) == [0, 1, 2]
    assert torneo_desde_participantes(rangos, np.array([[2, 0, 2]]))[0] == 0
    empate = rangos_densos_M08({"V": np.zeros(3), "utilidad": np.array([7.0, 7.0, 1.0])})
    assert list(empate) == [0, 0, 1] and torneo_desde_participantes(empate, np.array([[1, 0]]))[0] == 1


def prueba_09():
    a, b = np.array([[1.0, 2, 3, 4]]), np.array([[5.0, 6, 7, 8]])
    h1, h2 = cruce_con_mascara(a, b, np.array([[1, 0, 1, 0]]))
    assert h1.tolist() == [[1, 6, 3, 8]] and h2.tolist() == [[5, 2, 7, 4]]
    A, B = np.random.default_rng(4).random((2, 200, 6))
    H1, H2 = etapa_4_recombinar_M14(A, B, np.random.default_rng(5))
    assert np.all((H1 == A) | (H1 == B)) and np.allclose(H1 + H2, A + B)


def prueba_10():
    a, b = np.array([[2.0, 10.0]]), np.array([[6.0, 2.0]])
    h1, h2 = combinacion_aritmetica(a, b, 0.25)
    assert np.allclose(h1, [[5.0, 4.0]]) and np.allclose(h2, [[3.0, 8.0]])
    inst = INST_PRUEBA
    X = muestras_factibles(inst, 2000, 6)
    H1, H2 = etapa_4_recombinar_M15(X[:1000], X[1000:], np.random.default_rng(7))
    assert np.allclose(H1 + H2, X[:1000] + X[1000:])
    assert np.all((H1 >= np.minimum(X[:1000], X[1000:]) - 1e-12) & (H1 <= np.maximum(X[:1000], X[1000:]) + 1e-12))
    assert violacion(inst, np.vstack([H1, H2])).max() <= 1e-12, "un hijo de padres factibles resultó inviable"


def prueba_11():
    inst = INST_PRUEBA
    rng = np.random.default_rng(8)
    A = inst.piso + rng.random((3000, inst.d)) * (inst.techo - inst.piso)
    B = inst.piso + rng.random((3000, inst.d)) * (inst.techo - inst.piso)
    bajo, alto, I = np.minimum(A, B), np.maximum(A, B), np.abs(A - B)
    H1, _ = etapa_4_recombinar_M16(A, B, rng, 0.0, inst.piso, inst.techo)
    assert np.all((H1 >= bajo) & (H1 <= alto))
    H1, H2 = etapa_4_recombinar_M16(A, B, rng, 0.5, inst.piso, inst.techo)
    for H in (H1, H2):
        assert np.all((H >= np.maximum(bajo - 0.5 * I, inst.piso)) & (H <= np.minimum(alto + 0.5 * I, inst.techo)))
    assert np.mean((H1 == inst.piso) | (H1 == inst.techo)) > 0, "el recorte debería dejar masa en las cotas"
    H1, H2 = etapa_4_recombinar_M16(A, A, rng, 0.5, inst.piso, inst.techo)
    assert np.array_equal(H1, A) and np.array_equal(H2, A)


def prueba_12():
    inst = INST_PRUEBA
    H = inst.piso + np.random.default_rng(9).random((4000, inst.d)) * (inst.techo - inst.piso)
    copia = H.copy()
    assert np.array_equal(etapa_5_mutar_M18(H, 0.0, np.random.default_rng(10), inst.piso, inst.techo), H)
    Y = etapa_5_mutar_M18(H, 1.0, np.random.default_rng(10), inst.piso, inst.techo)
    assert np.array_equal(H, copia) and np.all((Y >= inst.piso) & (Y <= inst.techo)) and not np.any(Y == H)
    centro, ee = (inst.piso + inst.techo) / 2, (inst.techo - inst.piso) / np.sqrt(12 * len(Y))
    assert np.all(np.abs(Y.mean(axis=0) - centro) < 4 * ee)


def prueba_13():
    inst = INST_PRUEBA
    R, centro = inst.techo - inst.piso, (inst.piso + inst.techo) / 2
    n = 20_000
    Y = etapa_5_mutar_M19(np.tile(centro, (n, 1)), 1.0, np.random.default_rng(11), inst.piso, inst.techo, 0.05)
    assert np.all((Y >= inst.piso) & (Y <= inst.techo))
    assert np.all(np.abs(Y.mean(axis=0) - centro) < 4 * 0.05 * R / np.sqrt(n))
    assert np.all(np.abs(Y.std(axis=0) / (0.05 * R) - 1) < 0.03)
    Y = etapa_5_mutar_M19(np.tile(inst.techo, (n, 1)), 1.0, np.random.default_rng(12), inst.piso, inst.techo, 0.05)
    assert np.all(np.abs(np.mean(Y == inst.techo, axis=0) - 0.5) < 0.02)


def prueba_14():
    inst = INST_PRUEBA
    centro = (inst.piso + inst.techo) / 2
    n = 200_000 // 3
    for x in (inst.piso, inst.techo, centro):
        for eta in (5, 20, 50):
            Y = etapa_5_mutar_M20(np.tile(x, (n, 1)), 1.0, np.random.default_rng(13), inst.piso, inst.techo, eta)
            assert np.all((Y >= inst.piso) & (Y <= inst.techo))
    rng = np.random.default_rng(14)
    Y5 = etapa_5_mutar_M20(np.tile(centro, (n, 1)), 1.0, rng, inst.piso, inst.techo, 5)
    Y50 = etapa_5_mutar_M20(np.tile(centro, (n, 1)), 1.0, rng, inst.piso, inst.techo, 50)
    assert np.all(Y5.std(axis=0) > Y50.std(axis=0))
    assert np.all(np.abs(Y5.mean(axis=0) - centro) < 4 * Y5.std(axis=0) / np.sqrt(n))


def prueba_15():
    inst = INST_PRUEBA_F
    par = {"alfa": 0.3, "sigma": 0.05, "eta": 20}
    iniciales = [algoritmo_genetico(inst, c, m, 21, par, E_max=100)["poblacion"] for c, m in CONFIGURACIONES]
    assert all(np.array_equal(iniciales[0], P) for P in iniciales[1:]), "la población inicial debe ser común"
    for c, m in CONFIGURACIONES:
        r = algoritmo_genetico(inst, c, m, 22, par, E_max=3000)
        assert r["evaluaciones"] == 100 + r["generaciones"] * 98 and 3000 - 98 < r["evaluaciones"] <= 3000
        serie = [h["mejor_factible"] for h in r["historial"] if not np.isnan(h["mejor_factible"])]
        assert all(b >= a for a, b in zip(serie, serie[1:])), "la mejor utilidad factible no debe bajar"
    r1 = algoritmo_genetico(inst, "M16", "M20", 23, par, E_max=3000)
    r2 = algoritmo_genetico(inst, "M16", "M20", 23, par, E_max=3000)
    assert np.array_equal(r1["precios"], r2["precios"]) and r1["utilidad"] == r2["utilidad"]
    P = np.arange(3.0)[:, None]
    ev = {"V": np.zeros(3), "utilidad": np.array([5.0, 9.0, 7.0]), "factible": np.ones(3, bool)}
    evH = {"V": np.zeros(2), "utilidad": np.array([8.0, 1.0]), "factible": np.ones(2, bool)}
    P2, ev2 = etapa_6_reemplazar_M22(P, ev, np.array([[10.0], [11.0]]), evH, 1)
    assert ev2["utilidad"].tolist() == [9.0, 8.0, 1.0] and P2[:, 0].tolist() == [1.0, 10.0, 11.0]
    assert etapa_7_parar_M26(19_994, 98, 20_000) and not etapa_7_parar_M26(19_896, 98, 20_000)


def prueba_16():
    inst = INST_PRUEBA_F
    r = busqueda_aleatoria(inst, 31, 3000)
    assert r["evaluaciones"] == 100 + 29 * 98
    p = ajuste_parejo(inst)
    assert violacion(inst, p["precios"])[0] <= TOL_FACTIBLE and p["utilidad"] >= utilidad(inst, inst.precio_actual)[0]


INST_PRUEBA = generar_instancia("T06-D", 2, "D", 7)
INST_PRUEBA_F = generar_instancia("T15-F", 5, "F", 11)

PRUEBAS = [
    ("01", "Validación de instancias", prueba_01),
    ("02", "Evaluación vectorizada idéntica a la escalar", prueba_02),
    ("03", "Un producto resuelto a mano", prueba_03),
    ("04", "Dos sustitutos simétricos resueltos a mano", prueba_04),
    ("05", "Certificado KKT en una instancia generada", prueba_05),
    ("06", "M03: hipercubo latino", prueba_06),
    ("07", "M08: prioridad de factibilidad frente a penalización", prueba_07),
    ("08", "M12: torneo y desempate", prueba_08),
    ("09", "M14: cruce por máscara", prueba_09),
    ("10", "M15: segmento, suma y factibilidad por convexidad", prueba_10),
    ("11", "M16: BLX-α", prueba_11),
    ("12", "M18: reinicio uniforme", prueba_12),
    ("13", "M19: gaussiana acotada", prueba_13),
    ("14", "M20: polinomial acotada", prueba_14),
    ("15", "Integración: presupuesto, emparejamiento, reproducibilidad y elitismo", prueba_15),
    ("16", "Líneas base", prueba_16),
]


def ejecutar_pruebas() -> pd.DataFrame:
    """Ejecuta las 16 pruebas. Si alguna falla, la excepción detiene el cuadernillo."""
    filas = []
    for numero, descripcion, funcion in PRUEBAS:
        funcion()                       # AssertionError si falla
        filas.append({"prueba": numero, "descripcion": descripcion, "resultado": "pasó"})
    tabla = pd.DataFrame(filas)
    assert len(tabla) == 16
    return tabla


# ============================================================================
# 7. Instancias, calibración y experimento
# ============================================================================
def preparar_instancias(RES: Path):
    """Genera las cinco instancias, calcula sus óptimos y medidas de dificultad."""
    instancias_obj, exactos, parejo, filas = {}, {}, {}, []
    for nombre, definicion in DEFINICION.items():
        opciones = {k: v for k, v in definicion.items() if k != "uso"}
        inst = instancias_obj[nombre] = generar_instancia(nombre, **opciones)
        ex = exactos[nombre] = resolver_exacto(inst)
        pa = parejo[nombre] = ajuste_parejo(inst)
        z0, zs = float(utilidad(inst, inst.precio_actual)[0]), ex["utilidad"]
        X = inst.piso + np.random.default_rng(0).random((20_000, inst.d)) * (inst.techo - inst.piso)
        R = np.diag(inst.techo - inst.piso)
        valores = np.linalg.eigvalsh(R @ inst.B @ R)
        tipos = [t for t, _ in ex["activas"]]
        filas.append({
            "instancia": nombre, "productos": inst.d, "sustitucion": definicion["intensidad"],
            "uso": definicion["uso"], "utilidad_actual": z0, "utilidad_optima": zs,
            "mejora_posible_pct": 100 * (zs - z0) / z0,
            "ajuste_parejo_t_pct": 100 * pa["t"], "ajuste_parejo_utilidad": pa["utilidad"],
            "brecha_actual_pct": 100 * (zs - z0) / zs, "brecha_parejo_pct": 100 * (zs - pa["utilidad"]) / zs,
            "fraccion_factible_azar": float(np.mean(violacion(inst, X) <= TOL_FACTIBLE)),
            "condicion": float(valores.max() / valores.min()),
            "activas_existencias": tipos.count("existencias"), "activas_escalera": tipos.count("escalera"),
            "activas_indice": tipos.count("índice"), "activas_techo": tipos.count("techo"),
            "activas_piso": tipos.count("piso"), "activas_demanda": tipos.count("demanda"),
            "residuo_kkt": ex["residuo"], "violacion_maxima": ex["max_violacion"],
            "segundos_exacto": ex["segundos"]})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RES / "instancias.csv", index=False)
    return instancias_obj, exactos, parejo, tabla


def celdas_calibracion(cruce: str, mutacion: str) -> list:
    nombres = [p for p in (PARAMETRO_DE[cruce], PARAMETRO_DE[mutacion]) if p]
    return [dict(zip(nombres, valores)) for valores in product(*(REJILLA[n] for n in nombres))]


def calibrar(inst_c: Instancia, z_c: float, RES: Path):
    """Calibra sólo los parámetros propios de cada configuración en C24-F (regla fijada de antemano)."""
    filas = []
    for cruce, mutacion in CONFIGURACIONES:
        for par in celdas_calibracion(cruce, mutacion):
            for s in SEMILLAS_CALIBRACION:
                r = algoritmo_genetico(inst_c, cruce, mutacion, s, par, historial=False)
                filas.append({"cruce": cruce, "mutacion": mutacion, "configuracion": f"{cruce}+{mutacion}",
                              "alfa": par.get("alfa", np.nan), "sigma": par.get("sigma", np.nan),
                              "eta": par.get("eta", np.nan), "semilla": s,
                              "brecha_pct": 100 * (z_c - r["utilidad"]) / z_c,
                              "evaluaciones": r["evaluaciones"]})
    calibracion = pd.DataFrame(filas)
    calibracion.to_csv(RES / "calibracion.csv", index=False)

    parametros, elegidas = {}, []
    claves = ["alfa", "sigma", "eta"]
    celdas = (calibracion.groupby(["cruce", "mutacion"] + claves, dropna=False)["brecha_pct"]
              .agg(mediana="median", media="mean").reset_index())
    for (cruce, mutacion), grupo in celdas.groupby(["cruce", "mutacion"]):
        orden = grupo.sort_values(["mediana", "media"])
        g = orden.iloc[0]
        parametros[(cruce, mutacion)] = {k: (int(g[k]) if k == "eta" else float(g[k]))
                                         for k in claves if not pd.isna(g[k])}
        elegidas.append({"configuracion": f"{cruce}+{mutacion}", "celdas": len(grupo),
                         **{k: g[k] for k in claves}, "brecha_mediana_pct": g["mediana"],
                         "brecha_media_pct": g["media"],
                         "segunda_mediana_pct": orden["mediana"].iloc[1] if len(grupo) > 1 else np.nan})
    tabla = pd.DataFrame(elegidas)
    tabla.to_csv(RES / "calibracion_elegidos.csv", index=False)
    return parametros, tabla, calibracion


def primera_evaluacion(evaluaciones, brechas, umbral: float) -> float:
    idx = np.flatnonzero(brechas <= umbral)
    return float(evaluaciones[idx[0]]) if len(idx) else np.nan


def experimento(instancias_obj, exactos, parametros, RES: Path):
    """Factorial 3 × 3 en cuatro instancias, treinta semillas, más búsqueda aleatoria con presupuesto idéntico."""
    filas, historiales, filas_ba, historiales_ba = [], [], [], []
    for nombre in INSTANCIAS_EVALUACION:
        inst, ex = instancias_obj[nombre], exactos[nombre]
        zs, z0 = ex["utilidad"], float(utilidad(inst, inst.precio_actual)[0])
        for cruce, mutacion in CONFIGURACIONES:
            par = parametros[(cruce, mutacion)]
            for s in SEMILLAS_EVALUACION:
                r = algoritmo_genetico(inst, cruce, mutacion, s, par)
                h = pd.DataFrame(r["historial"])
                h["brecha_pct"] = 100 * (zs - h["mejor_factible"]) / zs
                ev_h, br_h = h["evaluaciones"].to_numpy(), h["brecha_pct"].to_numpy()
                filas.append({
                    "instancia": nombre, "cruce": cruce, "mutacion": mutacion,
                    "configuracion": f"{cruce}+{mutacion}", "semilla": s, "parametros": json.dumps(par),
                    "utilidad": r["utilidad"], "brecha_pct": 100 * (zs - r["utilidad"]) / zs,
                    "mejora_capturada_pct": 100 * (r["utilidad"] - z0) / (zs - z0),
                    "brecha_5000_pct": br_h[GEN_25], "brecha_9998_pct": br_h[GEN_50],
                    "eval_1pct": primera_evaluacion(ev_h, br_h, 1.0),
                    "eval_01pct": primera_evaluacion(ev_h, br_h, 0.1),
                    "eval_mejor": r["eval_mejor"], "evaluaciones": r["evaluaciones"],
                    "fraccion_factible_final": r["fraccion_factible"], "dispersion_final": r["dispersion"],
                    "en_cota_final": r["en_cota"],
                    "distancia_precios_pp": float(100 * np.max(np.abs(r["precios"] - ex["precios"])
                                                               / inst.precio_actual)),
                    "segundos": r["segundos"]})
                h = h.assign(instancia=nombre, cruce=cruce, mutacion=mutacion, semilla=s)
                historiales.append(h[["instancia", "cruce", "mutacion", "semilla", "generacion", "evaluaciones",
                                      "brecha_pct", "fraccion_factible", "fraccion_hijos_factibles",
                                      "dispersion", "en_cota"]])
        for s in SEMILLAS_EVALUACION:
            ba = busqueda_aleatoria(inst, s, EVALUACIONES_AG)
            if ba["evaluaciones"] != EVALUACIONES_AG:
                raise RuntimeError("La búsqueda aleatoria no usó el presupuesto del AG.")
            filas_ba.append({"instancia": nombre, "semilla": s, "factible": ba["factible"],
                             "utilidad": ba["utilidad"],
                             "brecha_pct": 100 * (zs - ba["utilidad"]) / zs if ba["factible"] else np.nan,
                             "violacion_minima": ba["violacion_minima"], "evaluaciones": ba["evaluaciones"]})
            t = pd.DataFrame(ba["trayectoria"]).assign(instancia=nombre, semilla=s)
            t["brecha_pct"] = 100 * (zs - t["mejor_factible"]) / zs
            historiales_ba.append(t[["instancia", "semilla", "evaluaciones", "brecha_pct"]])

    corridas = pd.DataFrame(filas)
    historial = pd.concat(historiales, ignore_index=True)
    aleatoria = pd.DataFrame(filas_ba)
    historial_ba = pd.concat(historiales_ba, ignore_index=True)
    corridas.to_csv(RES / "corridas.csv", index=False)
    historial.to_csv(RES / "historiales.csv", index=False, float_format="%.6g")
    aleatoria.to_csv(RES / "busqueda_aleatoria.csv", index=False)
    historial_ba.to_csv(RES / "historial_busqueda_aleatoria.csv", index=False, float_format="%.6g")
    assert (corridas["evaluaciones"] == EVALUACIONES_AG).all() and (corridas["brecha_pct"] >= -1e-8).all()
    return corridas, historial, aleatoria, historial_ba


# ============================================================================
# 8. Análisis estadístico
# ============================================================================
def holm(pvalores) -> np.ndarray:
    """Valores p ajustados por el procedimiento escalonado de Holm (1979)."""
    p = np.asarray(pvalores, float)
    m, ajustados, acumulado = len(p), np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        acumulado = max(acumulado, min(1.0, (m - i) * p[j]))
        ajustados[j] = acumulado
    return ajustados


def wilcoxon_emparejado(x, y) -> float:
    return 1.0 if np.all(x == y) else float(wilcoxon(x, y).pvalue)


def comparar(M: np.ndarray, etiquetas: list) -> dict:
    """Friedman sobre las columnas de M (bloques en filas) y Wilcoxon-Holm contra la de menor rango."""
    chi2, p = friedmanchisquare(*M.T)
    rangos = rankdata(M, axis=1).mean(axis=0)
    mejor = int(np.argmin(rangos))
    otros = [j for j in range(M.shape[1]) if j != mejor]
    p_crudos = [wilcoxon_emparejado(M[:, j], M[:, mejor]) for j in otros]
    p_holm = dict(zip(otros, holm(p_crudos)))
    return {"chi2": float(chi2), "p": float(p), "rangos": dict(zip(etiquetas, rangos)),
            "mejor": etiquetas[mejor],
            "p_holm": {etiquetas[j]: float(p_holm[j]) for j in otros},
            "empatadas": [etiquetas[j] for j in otros if p_holm[j] >= 0.05]}


def matriz_bloques(df: pd.DataFrame, indice, columna: str, niveles) -> np.ndarray:
    return df.pivot_table(index=indice, columns=columna, values="brecha_pct").loc[:, list(niveles)].to_numpy()


def analizar(corridas: pd.DataFrame, historial: pd.DataFrame, historial_ba: pd.DataFrame,
             exactos: dict, RES: Path):
    """Resúmenes, pruebas por instancia, ranking global, mecanismos y caso principal."""
    # --- resumen por instancia y configuración
    def q1(x):
        return x.quantile(0.25)

    def q3(x):
        return x.quantile(0.75)

    resumen = (corridas.groupby(["instancia", "configuracion", "cruce", "mutacion"])
               .agg(brecha_mediana=("brecha_pct", "median"), brecha_q1=("brecha_pct", q1),
                    brecha_q3=("brecha_pct", q3), brecha_max=("brecha_pct", "max"),
                    brecha_min=("brecha_pct", "min"),
                    mejora_capturada_mediana=("mejora_capturada_pct", "median"),
                    brecha_5000_mediana=("brecha_5000_pct", "median"),
                    brecha_9998_mediana=("brecha_9998_pct", "median"),
                    aciertos_01=("brecha_pct", lambda x: int((x <= 0.1).sum())),
                    eval_1pct_mediana=("eval_1pct", "median"), eval_01pct_mediana=("eval_01pct", "median"),
                    dispersion_final=("dispersion_final", "median"), en_cota_final=("en_cota_final", "median"),
                    distancia_precios_pp=("distancia_precios_pp", "median"), segundos=("segundos", "median"))
               .reset_index())
    resumen["configuracion"] = pd.Categorical(resumen["configuracion"], ORDEN_CONFIG, ordered=True)
    resumen = resumen.sort_values(["instancia", "configuracion"]).reset_index(drop=True)
    resumen.to_csv(RES / "resumen.csv", index=False)

    # --- pruebas por instancia
    POR_INSTANCIA, filas_est = {}, []
    for nombre in INSTANCIAS_EVALUACION:
        df = corridas[corridas["instancia"] == nombre]
        c = POR_INSTANCIA[nombre] = comparar(matriz_bloques(df, "semilla", "configuracion", ORDEN_CONFIG),
                                             ORDEN_CONFIG)
        filas_est.append({"instancia": nombre, "prueba": "Friedman, 9 configuraciones",
                          "estadistico": c["chi2"], "p": c["p"], "mejor": c["mejor"],
                          "empatadas_con_la_mejor": " · ".join(c["empatadas"]) or "ninguna"})
        for factor, otro, niveles, otros_niveles in (("cruce", "mutacion", CRUCES, MUTACIONES),
                                                     ("mutacion", "cruce", MUTACIONES, CRUCES)):
            e = comparar(matriz_bloques(df, ["semilla", otro], factor, niveles), list(niveles))
            filas_est.append({"instancia": nombre, "prueba": f"Efecto principal de {factor}",
                              "estadistico": e["chi2"], "p": e["p"], "mejor": e["mejor"],
                              "empatadas_con_la_mejor": " · ".join(e["empatadas"]) or "ninguna"})
            for nivel in otros_niveles:
                sub = df[df[otro] == nivel]
                s_ = comparar(matriz_bloques(sub, "semilla", factor, niveles), list(niveles))
                filas_est.append({"instancia": nombre, "prueba": f"Efecto simple de {factor} con {nivel}",
                                  "estadistico": s_["chi2"], "p": s_["p"], "mejor": s_["mejor"],
                                  "empatadas_con_la_mejor": " · ".join(s_["empatadas"]) or "ninguna"})
    estadistica = pd.DataFrame(filas_est)
    estadistica.to_csv(RES / "pruebas_estadisticas.csv", index=False)

    # --- ranking global: 120 bloques (4 instancias × 30 semillas)
    M_global = np.vstack([matriz_bloques(corridas[corridas["instancia"] == n], "semilla", "configuracion",
                                         ORDEN_CONFIG) for n in INSTANCIAS_EVALUACION])
    chi2_g, p_g = friedmanchisquare(*M_global.T)
    R_g = rankdata(M_global, axis=1).mean(axis=0)
    k_g, n_g = M_global.shape[1], M_global.shape[0]
    j_mejor = int(np.argmin(R_g))
    ee_g = np.sqrt(k_g * (k_g + 1) / (6 * n_g))
    otros = [j for j in range(k_g) if j != j_mejor]
    p_crudo = [2 * (1 - norm.cdf(abs(R_g[j] - R_g[j_mejor]) / ee_g)) for j in otros]
    p_ajustado = dict(zip(otros, holm(p_crudo)))
    global_ = pd.DataFrame({
        "configuracion": ORDEN_CONFIG, "rango_medio": R_g,
        "p_holm_vs_mejor": [p_ajustado.get(j, np.nan) for j in range(k_g)],
        "brecha_maxima_pct": [corridas.loc[corridas["configuracion"] == c, "brecha_pct"].max() for c in ORDEN_CONFIG],
        "brecha_mediana_pct": [corridas.loc[corridas["configuracion"] == c, "brecha_pct"].median() for c in ORDEN_CONFIG],
        "gana_al_elegido": np.nan, "pierde_contra_el_elegido": np.nan,
    }).sort_values("rango_medio").reset_index(drop=True)

    # Regla fijada de antemano: menor rango medio; entre los empates estadísticos, la de menor brecha máxima.
    candidatas = global_[(global_["configuracion"] == ORDEN_CONFIG[j_mejor]) | (global_["p_holm_vs_mejor"] >= 0.05)]
    ELEGIDA = candidatas.sort_values(["brecha_maxima_pct", "rango_medio"]).iloc[0]["configuracion"]
    j_el = ORDEN_CONFIG.index(ELEGIDA)
    global_["gana_al_elegido"] = [float(np.mean(M_global[:, ORDEN_CONFIG.index(c)] < M_global[:, j_el]))
                                  for c in global_["configuracion"]]
    global_["pierde_contra_el_elegido"] = [float(np.mean(M_global[:, ORDEN_CONFIG.index(c)] > M_global[:, j_el]))
                                           for c in global_["configuracion"]]
    global_.to_csv(RES / "ranking_global.csv", index=False)

    # --- mecanismos (medianas por generación)
    mecanismo = (historial.groupby(["instancia", "cruce", "mutacion", "generacion"])
                 [["evaluaciones", "brecha_pct", "dispersion", "fraccion_hijos_factibles",
                   "fraccion_factible", "en_cota"]].median().reset_index())
    mecanismo.to_csv(RES / "mecanismo_medianas.csv", index=False, float_format="%.6g")

    return {"resumen": resumen, "estadistica": estadistica, "global": global_, "chi2_global": chi2_g,
            "p_global": p_g, "n_global": n_g, "elegida": ELEGIDA, "mecanismo": mecanismo,
            "POR_INSTANCIA": POR_INSTANCIA, "M_global": M_global, "candidatas": candidatas}


def caso_principal(instancias_obj, exactos, parejo, ELEGIDA: str, parametros: dict, peor: str,
                   RES: Path):
    """Precios de las doce bebidas de E12-F: óptimo, corrida del AG, ajuste parejo y redondeos."""
    inst_p, ex_p = instancias_obj["E12-F"], exactos["E12-F"]
    cruce_el, mutacion_el = ELEGIDA.split("+")
    corrida_p = algoritmo_genetico(inst_p, cruce_el, mutacion_el, SEMILLAS_EVALUACION[0],
                                   parametros[(cruce_el, mutacion_el)])
    z0_p, zs_p = float(utilidad(inst_p, inst_p.precio_actual)[0]), ex_p["utilidad"]
    activa_de = {}
    for (tipo, j), lam in zip(ex_p["activas"], ex_p["multiplicadores"]):
        if j >= 0:
            activa_de[j] = f"{tipo} (λ = {lam:,.0f})"
    q_opt = inst_p.demanda(ex_p["precios"])
    caso = pd.DataFrame({
        "producto": [inst_p.producto(j) for j in range(inst_p.d)],
        "costo": inst_p.costo, "piso": inst_p.piso, "precio_actual": inst_p.precio_actual,
        "techo": inst_p.techo, "precio_optimo": ex_p["precios"],
        "cambio_optimo_pct": 100 * (ex_p["precios"] / inst_p.precio_actual - 1),
        "precio_ag": corrida_p["precios"], "dif_ag_pesos": corrida_p["precios"] - ex_p["precios"],
        "precio_parejo": parejo["E12-F"]["precios"],
        "demanda_actual": inst_p.demanda_actual, "demanda_optima": q_opt,
        "existencias": inst_p.existencias,
        "regla_activa": [activa_de.get(j, "") for j in range(inst_p.d)]})
    caso.to_csv(RES / "caso_principal_productos.csv", index=False)

    corrida_peor = algoritmo_genetico(inst_p, *peor.split("+"), SEMILLAS_EVALUACION[0],
                                      parametros[tuple(peor.split("+"))])
    alternativas = {
        "Precios actuales (costo más 45 %)": inst_p.precio_actual,
        f"Ajuste parejo ({100 * parejo['E12-F']['t']:+.2f} %)": parejo["E12-F"]["precios"],
        f"AG {ELEGIDA}, semilla {SEMILLAS_EVALUACION[0]}": corrida_p["precios"],
        f"AG {peor}, semilla {SEMILLAS_EVALUACION[0]}": corrida_peor["precios"],
        "Óptimo redondeado a 50 centavos": np.round(ex_p["precios"] * 2) / 2,
        "Óptimo redondeado hacia abajo a 50 centavos": np.floor(ex_p["precios"] * 2) / 2,
        "Óptimo redondeado a 10 centavos": np.round(ex_p["precios"], 1),
        "Óptimo certificado": ex_p["precios"],
    }
    filas = []
    for nombre, p in alternativas.items():
        u = float(utilidad(inst_p, p)[0])
        viol = float(violacion(inst_p, p)[0])
        reglas = [f"{t} {inst_p.producto(j) if j >= 0 else ''}".strip()
                  for (t, j), g in zip(inst_p.tipo, inst_p.G @ p - inst_p.h) if g > TOL_FACTIBLE]
        filas.append({"alternativa": nombre, "utilidad": u, "brecha_pct": 100 * (zs_p - u) / zs_p,
                      "mejora_capturada_pct": 100 * (u - z0_p) / (zs_p - z0_p),
                      "factible": bool(viol <= TOL_FACTIBLE), "violacion": viol,
                      "indice_precios": float(inst_p.pesos_indice @ (p / inst_p.precio_actual)),
                      "reglas_violadas": "; ".join(reglas)})
    alt = pd.DataFrame(filas)
    alt.to_csv(RES / "caso_principal_alternativas.csv", index=False)
    return caso, alt, corrida_p


def verificar_contra_readme(instancias: pd.DataFrame, resumen: pd.DataFrame, global_: pd.DataFrame,
                            elegida: str, RES: Path) -> pd.DataFrame:
    """Compara las cifras publicadas en el README original con las de esta ejecución."""
    def brecha(cfg, inst):
        fila = resumen[(resumen["instancia"] == inst) & (resumen["configuracion"] == cfg)]
        return float(fila["brecha_mediana"].iloc[0])

    ins = instancias.set_index("instancia")
    rg = global_.set_index("configuracion")
    casos = [
        ("Mejora posible E12-D (%)", 5.68, ins.loc["E12-D", "mejora_posible_pct"], 0.006),
        ("Mejora posible E12-F (%)", 8.67, ins.loc["E12-F", "mejora_posible_pct"], 0.006),
        ("Brecha ajuste parejo E12-D (%)", 3.84, ins.loc["E12-D", "brecha_parejo_pct"], 0.006),
        ("Brecha ajuste parejo E36-F (%)", 1.86, ins.loc["E36-F", "brecha_parejo_pct"], 0.006),
        ("Brecha mediana M16+M20 E12-D (%)", 0.041, brecha("M16+M20", "E12-D"), 0.0006),
        ("Brecha mediana M16+M20 E36-F (%)", 0.331, brecha("M16+M20", "E36-F"), 0.0006),
        ("Brecha mediana M15+M18 E12-F (%)", 0.878, brecha("M15+M18", "E12-F"), 0.0006),
        ("Brecha mediana M14+M18 E36-F (%)", 1.408, brecha("M14+M18", "E36-F"), 0.0006),
        ("Rango medio global M16+M19", 2.59, rg.loc["M16+M19", "rango_medio"], 0.006),
        ("Brecha máxima global M16+M20 (%)", 0.575, rg.loc["M16+M20", "brecha_maxima_pct"], 0.0006),
        ("Brecha máxima global M14+M19 (%)", 1.72, rg.loc["M14+M19", "brecha_maxima_pct"], 0.006),
        ("Brecha máxima global M14+M20 (%)", 1.80, rg.loc["M14+M20", "brecha_maxima_pct"], 0.006),
        ("Configuración elegida", "M16+M20", elegida, None),
    ]
    filas = []
    for nombre, esperado, obtenido, tol in casos:
        if tol is None:
            ok = esperado == obtenido
            diff = np.nan
        else:
            diff = float(obtenido) - float(esperado)
            ok = abs(diff) <= tol
        filas.append({"cifra": nombre, "readme": esperado,
                      "esta_ejecucion": obtenido if isinstance(obtenido, str) else round(float(obtenido), 4),
                      "diferencia": diff, "coincide": bool(ok)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RES / "verificacion_readme.csv", index=False)
    return tabla


# ============================================================================
# 9. Figuras
# ============================================================================
def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": EJE, "axes.linewidth": 0.8, "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "axes.grid": True, "grid.color": REJILLA_COLOR, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


MILES = mticker.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:.0f} mil")
DECIMAL = mticker.FuncFormatter(lambda v, _: f"{v:g}")


def eje_log_legible(ax, marcas=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20)):
    ax.set_yscale("log")
    bajo, alto = ax.get_ylim()
    ax.yaxis.set_major_locator(mticker.FixedLocator([m for m in marcas if bajo <= m <= alto]))
    ax.yaxis.set_major_formatter(DECIMAL)
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def manijas_factoriales():
    manijas = [Line2D([], [], color=COLOR_CRUCE[c], lw=2.4, label=NOMBRE[c]) for c in CRUCES]
    manijas += [Line2D([], [], color=TINTA_2, lw=1.6, ls=TRAZO_MUTACION[m], marker=MARCA_MUTACION[m],
                       ms=5, label=NOMBRE[m]) for m in MUTACIONES]
    return manijas


def guardar(fig, carpeta: Path, nombre: str):
    ruta = carpeta / nombre
    fig.savefig(ruta)
    plt.close(fig)
    print("  figura:", ruta)


def figura_diseno(FIG: Path):
    fig = plt.figure(figsize=(11.5, 4.9))
    ax = fig.add_axes([0, 0, 0.60, 1])
    ax.set_axis_off()
    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(0, 10)
    etapas = [("1 · Inicialización", "M03 hipercubo latino", False),
              ("2 · Evaluación", "M08 prioridad de factibilidad", False),
              ("3 · Selección", "M12 torneo, k = 2", False),
              ("4 · Recombinación", "M14 · M15 · M16", True),
              ("5 · Mutación", "M18 · M19 · M20", True),
              ("6 · Reemplazo", "M22 elitista, e = 2", False),
              ("7 · Paro", "M26 presupuesto de 20 000", False)]
    for i, (titulo, metodo, comparada) in enumerate(etapas):
        y = 9.0 - i * 1.25
        color_borde, fondo = (AZUL, "#e9f1fc") if comparada else (EJE, "white")
        ax.add_patch(FancyBboxPatch((0.6, y - 0.45), 5.6, 0.9, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc=fondo, ec=color_borde, lw=1.6 if comparada else 1.0))
        ax.text(0.85, y + 0.12, titulo, fontsize=10, color=TINTA, weight="bold" if comparada else "normal",
                va="center")
        ax.text(0.85, y - 0.2, metodo, fontsize=9, color=TINTA_2, va="center")
        if i < len(etapas) - 1:
            ax.add_patch(FancyArrowPatch((3.4, y - 0.47), (3.4, y - 0.78), arrowstyle="-|>",
                                         mutation_scale=9, color=TENUE, lw=1))
    y_de = lambda i: 9.0 - i * 1.25
    ax.add_patch(FancyArrowPatch((6.22, y_de(4)), (6.22, y_de(1)), connectionstyle="arc3,rad=0.55",
                                 arrowstyle="-|>", mutation_scale=10, color=AZUL, lw=1.2))
    ax.text(7.55, y_de(2.5), "los hijos se\nevalúan con\nla etapa 2\nantes del\nreemplazo",
            fontsize=8.5, color=TINTA_2, ha="center", va="center")
    ax.add_patch(FancyArrowPatch((0.58, y_de(6)), (0.58, y_de(2)), connectionstyle="arc3,rad=-0.5",
                                 arrowstyle="-|>", mutation_scale=10, color=TENUE, lw=1.0))
    ax.text(0.05, y_de(4), "otra\ngeneración\nsi cabe en el\npresupuesto", fontsize=8, color=TINTA_2,
            ha="center", va="center", rotation=90)
    ax.text(0.6, 9.85, "Etapas fijas en gris; las dos comparadas, en azul", fontsize=9, color=TINTA_2)
    ax2 = fig.add_axes([0.63, 0.12, 0.33, 0.74])
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(2.5, -0.5)
    ax2.grid(False)
    for i, c in enumerate(CRUCES):
        for j, m in enumerate(MUTACIONES):
            ax2.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fc="white", ec=COLOR_CRUCE[c], lw=1.6))
            ax2.text(j, i - 0.08, f"{c}+{m}", ha="center", va="center", fontsize=9.5, color=TINTA)
            ax2.plot(j, i + 0.22, marker=MARCA_MUTACION[m], color=COLOR_CRUCE[c], ms=6)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([NOMBRE[m] for m in MUTACIONES], fontsize=8.5)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([NOMBRE[c] for c in CRUCES], fontsize=8.5)
    ax2.tick_params(length=0)
    ax2.xaxis.tick_top()
    for lado in ax2.spines.values():
        lado.set_visible(False)
    ax2.set_title("Diseño factorial 3 × 3: nueve configuraciones", fontsize=10, pad=34)
    guardar(fig, FIG, "ae_a04_01_diseno_factorial.png")


def figura_operadores(FIG: Path):
    rng_f = np.random.default_rng(20261016)
    a_f, b_f = np.array([[0.30, 0.70]]), np.array([[0.62, 0.42]])
    n_f = 4000
    A_f, B_f = np.repeat(a_f, n_f, 0), np.repeat(b_f, n_f, 0)
    cero, uno = np.zeros(2), np.ones(2)
    hijos = {"M14": np.vstack(etapa_4_recombinar_M14(A_f, B_f, rng_f)),
             "M15": np.vstack(etapa_4_recombinar_M15(A_f, B_f, rng_f)),
             "M16": np.vstack(etapa_4_recombinar_M16(A_f, B_f, rng_f, 0.3, cero, uno))}
    fig, ejes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    for ax, c in zip(ejes[0], CRUCES):
        H = hijos[c]
        if c == "M14":
            puntos, cuentas = np.unique(np.round(H, 6), axis=0, return_counts=True)
            ax.scatter(puntos[:, 0], puntos[:, 1], s=40 + cuentas / 12, color=COLOR_CRUCE[c], alpha=0.85, zorder=3)
            for (x, y), n in zip(puntos, cuentas):
                ax.annotate(f"{100 * n / len(H):.0f} %", (x, y), xytext=(7, -12), textcoords="offset points",
                            fontsize=8, color=TINTA_2)
        else:
            ax.scatter(H[:, 0], H[:, 1], s=3, color=COLOR_CRUCE[c], alpha=0.25, lw=0, zorder=2)
        ax.scatter([a_f[0, 0], b_f[0, 0]], [a_f[0, 1], b_f[0, 1]], s=70, marker="X", color=TINTA, zorder=4)
        ax.annotate("padre a", a_f[0], xytext=(-48, 6), textcoords="offset points", fontsize=8.5, color=TINTA)
        ax.annotate("padre b", b_f[0], xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=TINTA)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("precio 1 (normalizado a su caja)")
        ax.set_ylabel("precio 2")
        ax.set_title(NOMBRE[c] + (", α = 0.3" if c == "M16" else ""))
    n_m = 60_000
    X_m = np.full((n_m, 1), 0.88)
    mutados = {"M18": etapa_5_mutar_M18(X_m, 1.0, rng_f, np.zeros(1), np.ones(1)),
               "M19": etapa_5_mutar_M19(X_m, 1.0, rng_f, np.zeros(1), np.ones(1), 0.10),
               "M20": etapa_5_mutar_M20(X_m, 1.0, rng_f, np.zeros(1), np.ones(1), 20)}
    for ax, m in zip(ejes[1], MUTACIONES):
        y = mutados[m][:, 0]
        interior = y[y < 1]
        ax.hist(interior, bins=np.linspace(0, 1, 81), color=TINTA_2, alpha=0.75,
                weights=np.full(len(interior), 100 / n_m))
        en_techo = 100 * np.mean(y == 1)
        if en_techo > 0:
            ax.bar(1.0, en_techo, width=0.0125, color=NARANJA, align="edge")
            ax.annotate(f"{en_techo:.1f} % de las mutaciones\nqueda exactamente en el techo",
                        (1.0, 0.8 * en_techo), xytext=(0.12, 0.62), textcoords="axes fraction",
                        fontsize=8, color=TINTA_2, arrowprops=dict(arrowstyle="-", color=TINTA_2, lw=0.8))
        ax.axvline(0.88, color=TINTA, lw=1, ls="--")
        ax.text(0.86, ax.get_ylim()[1] * 0.97, "gen original", ha="right", va="top", fontsize=8, color=TINTA,
                bbox=dict(fc="white", ec="none", pad=1))
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("valor mutado (normalizado: 0 = piso, 1 = techo)")
        ax.set_ylabel("% de las mutaciones por intervalo")
        ax.set_title(NOMBRE[m] + {"M18": "", "M19": ", σ = 0.10", "M20": ", η = 20"}[m])
    fig.tight_layout()
    guardar(fig, FIG, "ae_a04_02_operadores.png")


def figura_calibracion(FIG: Path, calibracion: pd.DataFrame, parametros: dict):
    cal = (calibracion.groupby(["cruce", "mutacion", "alfa", "sigma", "eta"], dropna=False)["brecha_pct"]
           .median().reset_index())
    mapa_azul = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
    fig, ejes = plt.subplots(1, 5, figsize=(14.5, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.05, 1.05]})
    for ax, (param, mut) in zip(ejes[:2], (("sigma", "M19"), ("eta", "M20"))):
        for c in ("M14", "M15"):
            g = cal[(cal["cruce"] == c) & (cal["mutacion"] == mut)].sort_values(param)
            ax.plot(range(3), g["brecha_pct"], color=COLOR_CRUCE[c], ls=TRAZO_MUTACION[mut],
                    marker=MARCA_MUTACION[mut], ms=6, label=f"{c}+{mut}")
            i = list(REJILLA[param]).index(parametros[(c, mut)][param])
            ax.plot(i, g["brecha_pct"].iloc[i], marker="o", ms=13, mfc="none", mec=TINTA, lw=0)
        ax.set_xticks(range(3))
        ax.set_xticklabels([str(v) for v in REJILLA[param]])
        ax.set_xlabel({"sigma": "σ de M19", "eta": "η de M20"}[param])
        ax.set_ylabel("brecha mediana (%)")
        ax.legend(fontsize=8.5, loc="upper center")
    g = cal[(cal["cruce"] == "M16") & (cal["mutacion"] == "M18")].sort_values("alfa")
    ax = ejes[2]
    ax.plot(range(3), g["brecha_pct"], color=COLOR_CRUCE["M16"], ls=TRAZO_MUTACION["M18"],
            marker=MARCA_MUTACION["M18"], ms=6, label="M16+M18")
    i = list(REJILLA["alfa"]).index(parametros[("M16", "M18")]["alfa"])
    ax.plot(i, g["brecha_pct"].iloc[i], marker="o", ms=13, mfc="none", mec=TINTA, lw=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([str(v) for v in REJILLA["alfa"]])
    ax.set_xlabel("α de M16")
    ax.set_ylabel("brecha mediana (%)")
    ax.legend(fontsize=8.5, loc="upper left")
    for ax, (param, mut) in zip(ejes[3:], (("sigma", "M19"), ("eta", "M20"))):
        g = cal[(cal["cruce"] == "M16") & (cal["mutacion"] == mut)]
        Z = g.pivot(index="alfa", columns=param, values="brecha_pct").loc[list(REJILLA["alfa"]),
                                                                          list(REJILLA[param])].to_numpy()
        ax.imshow(Z, cmap=mapa_azul, norm=LogNorm(vmin=Z.min(), vmax=Z.max()), aspect="auto")
        el = parametros[("M16", mut)]
        for r_ in range(3):
            for c_ in range(3):
                elegido = REJILLA["alfa"][r_] == el["alfa"] and REJILLA[param][c_] == el[param]
                oscuro = np.log(Z[r_, c_] / Z.min()) > 0.55 * np.log(Z.max() / Z.min())
                ax.text(c_, r_, f"{Z[r_, c_]:.3f}" + (" *" if elegido else ""), ha="center", va="center",
                        fontsize=8.5, color="white" if oscuro else TINTA, weight="bold" if elegido else "normal")
        ax.set_xticks(range(3))
        ax.set_xticklabels([str(v) for v in REJILLA[param]])
        ax.set_yticks(range(3))
        ax.set_yticklabels([str(v) for v in REJILLA["alfa"]])
        ax.set_xlabel({"sigma": "σ de M19", "eta": "η de M20"}[param])
        ax.set_ylabel("α de M16")
        ax.set_title(f"M16+{mut}", fontsize=10)
        ax.grid(False)
    for ax, t in zip(ejes, ("M14 y M15 con M19", "M14 y M15 con M20", "M16 con M18")):
        ax.set_title(t, fontsize=10)
    fig.tight_layout()
    guardar(fig, FIG, "ae_a04_03_calibracion.png")


def figura_brechas(FIG: Path, resumen: pd.DataFrame, POR_INSTANCIA: dict, exactos: dict):
    mapa_azul = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
    fig, ejes = plt.subplots(1, 4, figsize=(14.5, 3.9))
    norma = LogNorm(vmin=resumen["brecha_mediana"].min(), vmax=resumen["brecha_mediana"].max())
    for ax, nombre in zip(ejes, INSTANCIAS_EVALUACION):
        r_ = resumen[resumen["instancia"] == nombre]
        Z = (r_.pivot(index="cruce", columns="mutacion", values="brecha_mediana")
             .loc[list(CRUCES), list(MUTACIONES)].to_numpy())
        ax.imshow(Z, cmap=mapa_azul, norm=norma, aspect="auto")
        c = POR_INSTANCIA[nombre]
        for i, cr in enumerate(CRUCES):
            for j, mu in enumerate(MUTACIONES):
                cfg = f"{cr}+{mu}"
                marca = " ●" if cfg == c["mejor"] else (" ≈" if cfg in c["empatadas"] else "")
                oscuro = norma(Z[i, j]) > 0.55
                ax.text(j, i, f"{Z[i, j]:.3f}{marca}", ha="center", va="center", fontsize=9,
                        color="white" if oscuro else TINTA, weight="bold" if cfg == c["mejor"] else "normal")
        ax.set_xticks(range(3))
        ax.set_xticklabels(MUTACIONES)
        ax.set_yticks(range(3))
        ax.set_yticklabels(CRUCES)
        ax.set_xlabel("mutación")
        ax.set_ylabel("recombinación")
        ax.grid(False)
        ax.set_title(f"{nombre}: óptimo {exactos[nombre]['utilidad']:,.0f} $", fontsize=10)
    fig.text(0.5, -0.03, "Brecha mediana (%) de treinta semillas. ● menor rango medio; ≈ no difiere de ella "
             "(Wilcoxon-Holm, α = 0.05). Escala de color logarítmica común.", ha="center", fontsize=9,
             color=TINTA_2)
    fig.tight_layout()
    guardar(fig, FIG, "ae_a04_04_brechas_3x3.png")


def figura_interaccion(FIG: Path, resumen: pd.DataFrame):
    fig, ejes = plt.subplots(1, 4, figsize=(14.5, 3.9))
    for ax, nombre in zip(ejes, INSTANCIAS_EVALUACION):
        r_ = resumen[resumen["instancia"] == nombre]
        for desplazamiento, c in zip((-0.08, 0.0, 0.08), CRUCES):
            g = r_[r_["cruce"] == c].set_index("mutacion").loc[list(MUTACIONES)]
            x = np.arange(3) + desplazamiento
            ax.vlines(x, g["brecha_q1"], g["brecha_q3"], color=COLOR_CRUCE[c], lw=1.2, alpha=0.7)
            ax.plot(x, g["brecha_mediana"], color=COLOR_CRUCE[c], lw=2)
            for xi, mu, yi in zip(x, MUTACIONES, g["brecha_mediana"]):
                ax.plot(xi, yi, marker=MARCA_MUTACION[mu], color=COLOR_CRUCE[c], ms=7, mec="white", mew=1.2)
        eje_log_legible(ax)
        ax.set_xticks(range(3))
        ax.set_xticklabels(MUTACIONES)
        ax.set_xlabel("mutación")
        ax.set_ylabel("brecha (%), escala log")
        ax.set_title(nombre, fontsize=10)
    manijas = [Line2D([], [], color=COLOR_CRUCE[c], lw=2, label=NOMBRE[c]) for c in CRUCES]
    ejes[0].legend(handles=manijas, fontsize=8.5, loc="upper right")
    fig.text(0.5, -0.03, "Punto: mediana de treinta semillas; barra: rango intercuartil. Si las líneas no son "
             "paralelas, la mejor recombinación depende de la mutación.", ha="center", fontsize=9, color=TINTA_2)
    fig.tight_layout()
    guardar(fig, FIG, "ae_a04_05_interaccion.png")


def figura_convergencia(FIG: Path, historial: pd.DataFrame, historial_ba: pd.DataFrame, instancias: pd.DataFrame):
    med = historial.groupby(["instancia", "cruce", "mutacion", "generacion"]).agg(
        evaluaciones=("evaluaciones", "first"), brecha=("brecha_pct", "median")).reset_index()
    med_ba = historial_ba.groupby(["instancia", "evaluaciones"])["brecha_pct"].median().reset_index()
    ins = instancias.set_index("instancia")
    fig, ejes = plt.subplots(1, 4, figsize=(14.5, 4.1))
    for ax, nombre in zip(ejes, INSTANCIAS_EVALUACION):
        for c, m in CONFIGURACIONES:
            g = med[(med["instancia"] == nombre) & (med["cruce"] == c) & (med["mutacion"] == m)]
            ax.plot(g["evaluaciones"], g["brecha"], color=COLOR_CRUCE[c], ls=TRAZO_MUTACION[m], lw=1.6)
            ax.plot(g["evaluaciones"].iloc[-1], g["brecha"].iloc[-1], marker=MARCA_MUTACION[m],
                    color=COLOR_CRUCE[c], ms=6)
        b = med_ba[med_ba["instancia"] == nombre].dropna()
        if len(b):
            ax.plot(b["evaluaciones"], b["brecha_pct"], color=TENUE, lw=1.4, ls="-.", label="búsqueda aleatoria")
        ax.axhline(ins.loc[nombre, "brecha_parejo_pct"], color=TINTA, lw=0.9, ls=(0, (1, 2)))
        ax.text(EVALUACIONES_AG, ins.loc[nombre, "brecha_parejo_pct"], "ajuste parejo ", ha="right",
                va="bottom", fontsize=8, color=TINTA)
        ax.set_yscale("log")
        ax.set_ylim(max(1e-3, med[med["instancia"] == nombre]["brecha"].min() * 0.6), 25)
        eje_log_legible(ax)
        ax.set_xlabel("evaluaciones")
        ax.set_ylabel("brecha mediana (%), escala log")
        ax.set_title(nombre, fontsize=10)
        ax.set_xticks([0, 5000, 10000, 15000, 20000])
        ax.xaxis.set_major_formatter(MILES)
    manijas = manijas_factoriales() + [Line2D([], [], color=TENUE, lw=1.4, ls="-.", label="búsqueda aleatoria")]
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=manijas, loc="lower center", ncol=7, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, -0.03, "Mediana de treinta semillas. La búsqueda aleatoria no aparece en E36-D ni en E36-F "
             "porque no encontró ninguna solución factible.", ha="center", fontsize=9, color=TINTA_2)
    guardar(fig, FIG, "ae_a04_06_convergencia.png")


def figura_mecanismo(FIG: Path, mecanismo: pd.DataFrame, resumen: pd.DataFrame):
    fig, ejes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    sub = mecanismo[mecanismo["instancia"] == "E36-F"]
    for c, m in CONFIGURACIONES:
        g = sub[(sub["cruce"] == c) & (sub["mutacion"] == m)]
        ejes[0].plot(g["evaluaciones"], g["dispersion"], color=COLOR_CRUCE[c], ls=TRAZO_MUTACION[m], lw=1.5)
        g1 = g[g["generacion"] >= 1]
        ejes[1].plot(g1["evaluaciones"], g1["fraccion_hijos_factibles"], color=COLOR_CRUCE[c],
                     ls=TRAZO_MUTACION[m], lw=1.5)
    ejes[0].set_yscale("log")
    eje_log_legible(ejes[0], marcas=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3))
    ejes[0].set_ylabel("dispersión D(P), escala log")
    ejes[0].set_title("Dispersión de la población", fontsize=10)
    ejes[1].set_ylabel("fracción de hijos factibles")
    ejes[1].set_ylim(0, 1)
    ejes[1].set_title("Hijos factibles por generación", fontsize=10)
    for ax in ejes[:2]:
        ax.set_xlabel("evaluaciones")
        ax.set_xticks([0, 5000, 10000, 15000, 20000])
        ax.xaxis.set_major_formatter(MILES)
    r36 = resumen[resumen["instancia"] == "E36-F"].sort_values("brecha_mediana")
    posiciones = np.log10(r36["brecha_mediana"].to_numpy())
    for i in range(1, len(posiciones)):
        posiciones[i] = max(posiciones[i], posiciones[i - 1] + 0.06)
    for (_, fila), y_etq in zip(r36.iterrows(), posiciones):
        c, m = fila["cruce"], fila["mutacion"]
        ejes[2].plot([0, 1], [fila["brecha_5000_mediana"], fila["brecha_mediana"]], color=COLOR_CRUCE[c],
                     ls=TRAZO_MUTACION[m], lw=1.6, marker=MARCA_MUTACION[m], ms=6)
        ejes[2].text(1.05, 10 ** y_etq, fila["configuracion"], fontsize=7.5, va="center", color=TINTA_2)
    eje_log_legible(ejes[2], marcas=(0.2, 0.3, 0.5, 1, 2, 3, 5))
    ejes[2].set_xlim(-0.15, 1.35)
    ejes[2].set_xticks([0, 1])
    ejes[2].set_xticklabels(["5 000\nevaluaciones", "19 994\nevaluaciones"])
    ejes[2].set_ylabel("brecha mediana (%), escala log")
    ejes[2].set_title("Al cuarto del presupuesto y al final", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=manijas_factoriales(), loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    guardar(fig, FIG, "ae_a04_07_mecanismo.png")


def figura_caso(FIG: Path, caso: pd.DataFrame, elegida: str, parejo_t: float, semilla: int):
    inst_nombres = caso["producto"].tolist()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    y = np.arange(len(caso))[::-1]
    cambio_ag = 100 * (caso["precio_ag"] / caso["precio_actual"] - 1)
    cambio_par = 100 * (caso["precio_parejo"] / caso["precio_actual"] - 1)
    ax.barh(y, caso["cambio_optimo_pct"], height=0.62, color=AZUL, alpha=0.9, label="óptimo certificado")
    ax.plot(cambio_ag, y, "D", color=NARANJA, ms=6.5, mec="white", mew=1, label=f"AG {elegida}, semilla {semilla}")
    ax.plot(cambio_par, y, "|", color=TINTA, ms=14, mew=2, label=f"ajuste parejo ({100 * parejo_t:+.1f} %)")
    ax.axvline(0, color=TINTA_2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(inst_nombres, fontsize=8.5)
    ax.set_xlabel("cambio respecto al precio actual (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("E12-F: precios de temporada de doce bebidas", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    guardar(fig, FIG, "ae_a04_08_caso_principal.png")


# ============================================================================
# 10. Orquestación y manifiesto
# ============================================================================
def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main():
    global RESULTADOS, FIGURAS
    ap = argparse.ArgumentParser(description="Actividad 04 · algoritmo genético para precios de temporada.")
    ap.add_argument("--salida", default="resultados_actividad04", help="carpeta de salida")
    ap.add_argument("--sin-figuras", action="store_true", help="no generar las figuras PNG")
    args = ap.parse_args()

    raiz = Path(args.salida)
    RESULTADOS, FIGURAS = raiz / "resultados", raiz / "figuras"
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    estilo()
    inicio = time.perf_counter()

    print("1. Pruebas automatizadas")
    tabla_pruebas = ejecutar_pruebas()
    tabla_pruebas.to_csv(RESULTADOS / "pruebas.csv", index=False)
    print(f"   {int((tabla_pruebas['resultado'] == 'pasó').sum())} de 16 pruebas pasaron")

    print("2. Instancias y óptimos certificados")
    instancias_obj, exactos, parejo, tabla_instancias = preparar_instancias(RESULTADOS)
    print(f"   {len(tabla_instancias)} instancias; cinco óptimos certificados")

    print("3. Calibración en C24-F (350 corridas)")
    inst_c, z_c = instancias_obj["C24-F"], exactos["C24-F"]["utilidad"]
    parametros, tabla_cal, calibracion = calibrar(inst_c, z_c, RESULTADOS)
    print(tabla_cal[["configuracion", "alfa", "sigma", "eta", "brecha_mediana_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("4. Experimento factorial (1 080 corridas + 120 búsquedas aleatorias)")
    corridas, historial, aleatoria, historial_ba = experimento(instancias_obj, exactos, parametros, RESULTADOS)
    print(f"   mediana por corrida: {corridas['segundos'].median():.3f} s")

    print("5. Análisis estadístico")
    analisis = analizar(corridas, historial, historial_ba, exactos, RESULTADOS)
    print(f"   Friedman global: chi2 = {analisis['chi2_global']:.2f}, p = {analisis['p_global']:.3g}, "
          f"n = {analisis['n_global']} bloques")
    print(f"   Configuración elegida por la regla: {analisis['elegida']}")

    peor = analisis["global"].iloc[-1]["configuracion"]
    caso, alt, corrida_p = caso_principal(instancias_obj, exactos, parejo, analisis["elegida"], parametros,
                                          peor, RESULTADOS)

    print("6. Verificación contra las cifras del README original")
    verif = verificar_contra_readme(tabla_instancias, analisis["resumen"], analisis["global"],
                                    analisis["elegida"], RESULTADOS)
    print(f"   {int(verif['coincide'].sum())} de {len(verif)} cifras coinciden")
    print(verif[~verif["coincide"]][["cifra", "readme", "esta_ejecucion"]].to_string(index=False)
          if (~verif["coincide"]).any() else "   sin discrepancias")

    if not args.sin_figuras:
        print("7. Figuras")
        figura_diseno(FIGURAS)
        figura_operadores(FIGURAS)
        figura_calibracion(FIGURAS, calibracion, parametros)
        figura_brechas(FIGURAS, analisis["resumen"], analisis["POR_INSTANCIA"], exactos)
        figura_interaccion(FIGURAS, analisis["resumen"])
        figura_convergencia(FIGURAS, historial, historial_ba, tabla_instancias)
        figura_mecanismo(FIGURAS, analisis["mecanismo"], analisis["resumen"])
        figura_caso(FIGURAS, caso, analisis["elegida"], parejo["E12-F"]["t"], SEMILLAS_EVALUACION[0])

    total = time.perf_counter() - inicio
    manifiesto = {
        "actividad": "AAEE · Actividad 04 · corte 2",
        "entorno": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                    "plataforma": platform.platform()},
        "etapas_fijas": {"1": "M03", "2": "M08", "3": f"M12 (k = {K_TORNEO})",
                         "6": f"M22 (e = {E_ELITE})", "7": f"M26 (E_max = {E_MAX})"},
        "comparadas": {"4": list(CRUCES), "5": list(MUTACIONES)},
        "comunes": {"N": N_POBLACION, "k": K_TORNEO, "e": E_ELITE, "pc": P_CRUCE, "pm": "1/d",
                    "E_max": E_MAX, "evaluaciones": EVALUACIONES_AG},
        "rejilla_calibracion": REJILLA,
        "parametros_calibrados": {f"{c}+{m}": v for (c, m), v in parametros.items()},
        "semillas_calibracion": SEMILLAS_CALIBRACION, "semillas_evaluacion": SEMILLAS_EVALUACION,
        "optimos": {n: e_["utilidad"] for n, e_ in exactos.items()},
        "elegida": analisis["elegida"],
        "pruebas_pasadas": int((tabla_pruebas["resultado"] == "pasó").sum()),
        "cifras_readme_coincidentes": int(verif["coincide"].sum()),
        "segundos_totales": round(total, 1),
        "archivos": {str(r.relative_to(raiz)): sha256(r)
                     for r in sorted(list(RESULTADOS.glob("*.csv")) + list(FIGURAS.glob("*.png")))},
    }
    (RESULTADOS / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
    print(f"Listo en {total:.0f} s. Resultados en {raiz}")


if __name__ == "__main__":
    main()
