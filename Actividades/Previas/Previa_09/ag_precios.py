"""ag_precios — Algoritmo genético de siete etapas para fijar precios de temporada.

Actividad 04 · Algoritmos Evolutivos · Maestría en Inteligencia Artificial
Universidad Politécnica Metropolitana de Hidalgo · Dr. Jaime Aguilar Ortiz
Alumna: Jessica Melani Romero Lora (matrícula 25322116)

Contenido
  1. Modelo: instancias sintéticas con demanda lineal (parte 1) o de elasticidad constante (continuación).
  2. Referencias: óptimo certificado por KKT (modelo lineal), mejor conocido por multiarranque (modelo log)
     y líneas base (precios actuales, ajuste parejo, búsqueda aleatoria).
  3. Las siete etapas del algoritmo genético, con tres formas de recombinar (M14, M15, M16)
     y tres de mutar (M18, M19, M20).
  4. Estadística: Friedman, Wilcoxon con Holm, ranking global de Demšar e intervalos bootstrap.

El módulo no depende de ninguna biblioteca de algoritmos genéticos; sólo NumPy, pandas y SciPy.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import friedmanchisquare, norm, rankdata, wilcoxon

CRUCES = ("M14", "M15", "M16")
MUTACIONES = ("M18", "M19", "M20")
CONFIGURACIONES = [(c, m) for c in CRUCES for m in MUTACIONES]
ORDEN_CONFIG = [f"{c}+{m}" for c, m in CONFIGURACIONES]
NOMBRE = {"M14": "M14 uniforme", "M15": "M15 aritmética", "M16": "M16 BLX-α",
          "M18": "M18 reinicio", "M19": "M19 gaussiana", "M20": "M20 polinomial"}


# ======================================================================================
# 1. MODELO
# ======================================================================================
NIVELES = ("marca propia", "marca estándar", "marca premium")


@dataclass(frozen=True)
class Instancia:
    """Un caso de fijación de precios. Precios en pesos; demanda en unidades por temporada."""
    nombre: str
    familias: tuple
    familia: np.ndarray        # familia de cada producto
    nivel: np.ndarray          # 0 propia, 1 estándar, 2 premium
    costo: np.ndarray          # c_j
    precio_actual: np.ndarray  # P_j, regla vigente de costo más 45 %
    demanda_actual: np.ndarray # q0_j = q_j(P)
    B: np.ndarray              # pendientes: simétrica y definida positiva
    a: np.ndarray              # intercepto: q(p) = a - B p
    existencias: np.ndarray    # S_j
    piso: np.ndarray           # l_j
    techo: np.ndarray          # u_j
    G: np.ndarray              # reglas comerciales: G p <= h
    h: np.ndarray
    tipo: tuple                # (regla, producto) de cada fila de G
    brecha_escalera: float
    holgura_indice: float
    pesos_indice: np.ndarray
    modelo: str = "lineal"        # "lineal" (parte 1) o "log" (continuación, elasticidad constante)
    E: np.ndarray = None          # matriz de elasticidades E_jk (sólo modelo "log")
    G_lin: np.ndarray = None      # reglas lineales sin existencias ni demanda (sólo modelo "log")
    h_lin: np.ndarray = None

    @property
    def d(self) -> int:
        return len(self.costo)

    def producto(self, j: int) -> str:
        return f"{self.familias[self.familia[j]]} · {NIVELES[self.nivel[j]]}"

    def demanda(self, X):
        """Demanda en cada fila de X: lineal q = a - B p, o log-lineal q = q0 · exp(E ln(p/P))."""
        X = np.asarray(X, float)
        if self.modelo == "log":
            return self.demanda_actual * np.exp(np.log(X / self.precio_actual) @ self.E.T)
        return self.a - X @ self.B


def ensamblar_instancia(nombre, familias, familia, nivel, costo, precio_actual, demanda_actual, B,
                        existencias, piso, techo, brecha_escalera, holgura_indice) -> Instancia:
    """Calcula el intercepto de la demanda y escribe las reglas comerciales como G p <= h."""
    familia, nivel = np.asarray(familia, int), np.asarray(nivel, int)
    B, P = np.asarray(B, float), np.asarray(precio_actual, float)
    q0, S = np.asarray(demanda_actual, float), np.asarray(existencias, float)
    a = q0 + B @ P
    d = len(P)
    filas, lados, tipo = [], [], []
    for j in range(d):                      # existencias: a_j - B_j p <= S_j
        filas.append(-B[j] / S[j]); lados.append((S[j] - a[j]) / S[j]); tipo.append(("existencias", j))
    for j in range(d):                      # demanda no negativa: B_j p - a_j <= 0
        filas.append(B[j] / q0[j]); lados.append(a[j] / q0[j]); tipo.append(("demanda", j))
    for f in np.unique(familia):            # escalera: (1 + g) p_inferior - p_superior <= 0
        miembros = np.flatnonzero(familia == f)
        miembros = miembros[np.argsort(nivel[miembros], kind="stable")]
        for i, s in zip(miembros[:-1], miembros[1:]):
            fila = np.zeros(d); fila[i] = 1.0 + brecha_escalera; fila[s] = -1.0
            filas.append(fila / P[s]); lados.append(0.0); tipo.append(("escalera", int(s)))
    w = q0 * P / np.sum(q0 * P)             # índice de precios ponderado por el ingreso actual
    filas.append(w / P); lados.append(1.0 + holgura_indice); tipo.append(("índice", -1))
    return Instancia(nombre, tuple(familias), familia, nivel, np.asarray(costo, float), P, q0, B, a, S,
                     np.asarray(piso, float), np.asarray(techo, float), np.array(filas), np.array(lados),
                     tuple(tipo), float(brecha_escalera), float(holgura_indice), w)


def instancia_manual(costo, precio, q0, B, existencias, piso, techo, holgura=10.0) -> Instancia:
    """Instancia pequeña para pruebas: cada producto en su propia familia (sin escalera)."""
    d = len(costo)
    inst = ensamblar_instancia("manual", tuple(f"F{f}" for f in range(d)), np.arange(d), np.zeros(d, int),
                               costo, precio, q0, B, existencias, piso, techo, BRECHA_ESCALERA, holgura)
    validar_instancia(inst)
    return inst


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


MARGEN_ACTUAL = 0.45          # regla vigente: precio = costo × 1.45
MARGEN_MINIMO = 0.10          # piso: costo × 1.10
ALZA_MAXIMA = 0.20            # techo: precio actual × 1.20
BRECHA_ESCALERA = 0.08
HOLGURA_INDICE = 0.03
MULTIPLICADOR_COSTO = np.array([0.70, 1.00, 1.40])
CUOTA_NIVEL = np.array([0.35, 0.45, 0.20])
ELASTICIDAD = ((3.0, 5.5), (2.0, 3.8), (1.3, 2.4))
SUSTITUCION = {"D": (0.10, 0.05), "F": (0.30, 0.12)}   # (κ por par en la familia, suma entre familias)
EXISTENCIAS = (1.05, 1.50)


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


TOL_FACTIBLE = 1e-12   # unidades relativas de la violación; absorbe sólo el redondeo de punto flotante


def utilidad(inst: Instancia, X) -> np.ndarray:
    """Utilidad de la temporada de cada fila de X (pesos)."""
    X = np.atleast_2d(X)
    return np.sum((X - inst.costo) * inst.demanda(X), axis=1)


def violacion(inst: Instancia, X) -> np.ndarray:
    """Violación agregada V de cada fila de X; vale 0 exactamente en el conjunto factible."""
    X = np.atleast_2d(X)
    if inst.modelo == "log":     # existencias no lineales + escalera e índice (lineales)
        q = inst.demanda(X)
        exist = np.maximum(0.0, (q - inst.existencias) / inst.existencias).sum(axis=1)
        return exist + np.maximum(0.0, X @ inst.G_lin.T - inst.h_lin).sum(axis=1)
    return np.maximum(0.0, X @ inst.G.T - inst.h).sum(axis=1)


def evaluar_escalar(inst: Instancia, p) -> dict:
    """Recalcula utilidad y reglas con bucles, a partir de su significado y sin usar G ni h."""
    d = inst.d
    p = [float(v) for v in p]
    if inst.modelo == "log":
        q = [inst.demanda_actual[j] * math.exp(sum(inst.E[j, k] * math.log(p[k] / inst.precio_actual[k])
                                                    for k in range(d))) for j in range(d)]
    else:
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
        residuo = float(np.max(np.abs(g - H @ x - A[act].T @ lam))) if m else float(np.max(np.abs(g - H @ x)))
        return {"precios": x, "utilidad": float(utilidad(inst, x)[0]), "activas": [nombres[i] for i in act],
                "multiplicadores": lam, "residuo": residuo, "max_violacion": float(max(0.0, -holgura.min())),
                "iteraciones": iteracion}
    raise RuntimeError(f"{inst.nombre}: el conjunto activo no convergió; no hay certificado.")


def resolver_exacto(inst: Instancia) -> dict:
    """Óptimo global de la utilidad: SLSQP desde cuatro arranques y certificado KKT (sólo modelo lineal)."""
    if inst.modelo != "lineal":
        raise ValueError("El certificado KKT global sólo vale con utilidad cóncava (demanda lineal).")
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


def busqueda_aleatoria(inst: Instancia, semilla: int, evaluaciones: int, N: int = 100, por_bloque: int = 98,
                       z_ref: float = None) -> dict:
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


def etapa_1_inicializar_M03(N: int, inst: Instancia, rng: np.random.Generator) -> np.ndarray:
    """M03: hipercubo latino aleatorizado en la caja [piso, techo]."""
    d = inst.d
    estratos = rng.permuted(np.tile(np.arange(N), (d, 1)), axis=1).T     # cada columna, una permutación
    U = (estratos + rng.random((N, d))) / N
    return inst.piso + U * (inst.techo - inst.piso)


def etapa_2_evaluar_M08(inst: Instancia, X: np.ndarray) -> dict:
    """M08: violación y utilidad; la comparación usa la clave lexicográfica (V, -utilidad)."""
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


def torneo_desde_participantes(rangos: np.ndarray, participantes: np.ndarray) -> np.ndarray:
    """Gana el de menor rango; en empate, el primero sorteado (argmin devuelve la primera aparición)."""
    return participantes[np.arange(len(participantes)), np.argmin(rangos[participantes], axis=1)]


def etapa_3_seleccionar_M12(rangos: np.ndarray, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """M12: n torneos de tamaño k con reemplazo."""
    return torneo_desde_participantes(rangos, rng.integers(0, len(rangos), size=(n, k)))


def cruce_con_mascara(A: np.ndarray, B: np.ndarray, mascara: np.ndarray):
    """Hijo 1 toma de A donde la máscara vale 1 y de B donde vale 0; el hijo 2, al revés."""
    m = mascara.astype(bool)
    return np.where(m, A, B), np.where(m, B, A)


def etapa_4_recombinar_M14(A: np.ndarray, B: np.ndarray, rng: np.random.Generator, q: float = 0.5):
    """M14: cruce uniforme por máscara; cada gen se hereda de uno u otro padre."""
    return cruce_con_mascara(A, B, rng.random(A.shape) < q)


def combinacion_aritmetica(A: np.ndarray, B: np.ndarray, alfa: np.ndarray):
    """Hijos alfa·A + (1 - alfa)·B y (1 - alfa)·A + alfa·B; alfa es un escalar o una columna."""
    return alfa * A + (1 - alfa) * B, (1 - alfa) * A + alfa * B


def etapa_4_recombinar_M15(A: np.ndarray, B: np.ndarray, rng: np.random.Generator):
    """M15: recombinación aritmética completa, con un alfa uniforme por pareja."""
    return combinacion_aritmetica(A, B, rng.random((A.shape[0], 1)))


def etapa_4_recombinar_M16(A: np.ndarray, B: np.ndarray, rng: np.random.Generator, alfa: float,
                           piso: np.ndarray, techo: np.ndarray):
    """M16: BLX-alfa; muestreo uniforme en el intervalo de los padres ampliado alfa·I por cada lado."""
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


def etapa_5_mutar_M18(H: np.ndarray, pm: float, rng: np.random.Generator, piso, techo) -> np.ndarray:
    """M18: con probabilidad pm por gen, sortea un precio nuevo uniforme en [piso, techo]."""
    m = rng.random(H.shape) < pm
    return np.where(m, piso + rng.random(H.shape) * (techo - piso), H)


def etapa_5_mutar_M19(H: np.ndarray, pm: float, rng: np.random.Generator, piso, techo, sigma: float) -> np.ndarray:
    """M19: perturbación gaussiana con desviación sigma·(techo - piso), recortada a la caja."""
    m = rng.random(H.shape) < pm
    Y = np.clip(H + sigma * (techo - piso) * rng.standard_normal(H.shape), piso, techo)
    return np.where(m, Y, H)


def etapa_5_mutar_M20(H: np.ndarray, pm: float, rng: np.random.Generator, piso, techo, eta: float) -> np.ndarray:
    """M20: mutación polinomial acotada; el resultado queda en [piso, techo] por construcción."""
    m = rng.random(H.shape) < pm
    R = techo - piso
    d1, d2 = (H - piso) / R, (techo - H) / R
    r = rng.random(H.shape)
    e1 = eta + 1.0
    izquierda = (2 * r + (1 - 2 * r) * (1 - d1) ** e1) ** (1 / e1) - 1
    derecha = 1 - (2 * (1 - r) + 2 * (r - 0.5) * (1 - d2) ** e1) ** (1 / e1)
    Y = H + np.where(r <= 0.5, izquierda, derecha) * R
    Y = np.minimum(np.maximum(Y, piso), techo)     # sólo absorbe el error de redondeo de punto flotante
    return np.where(m, Y, H)


MUTAR = {
    "M18": lambda H, pm, rng, inst, par: etapa_5_mutar_M18(H, pm, rng, inst.piso, inst.techo),
    "M19": lambda H, pm, rng, inst, par: etapa_5_mutar_M19(H, pm, rng, inst.piso, inst.techo, par["sigma"]),
    "M20": lambda H, pm, rng, inst, par: etapa_5_mutar_M20(H, pm, rng, inst.piso, inst.techo, par["eta"]),
}
PARAMETRO_DE = {"M14": None, "M15": None, "M16": "alfa", "M18": None, "M19": "sigma", "M20": "eta"}


def etapa_6_reemplazar_M22(P: np.ndarray, ev: dict, H: np.ndarray, evH: dict, e: int):
    """M22: conserva las e mejores claves de la población actual y completa con los hijos."""
    elite = orden_M08(ev)[:e]
    P_nueva = np.vstack([P[elite], H])
    ev_nueva = {k: np.concatenate([ev[k][elite], evH[k]]) for k in ev}
    return P_nueva, ev_nueva


def etapa_7_parar_M26(evaluaciones: int, por_generacion: int, E_max: int) -> bool:
    """M26: se detiene si la siguiente generación completa no cabe en el presupuesto."""
    return evaluaciones + por_generacion > E_max


def dispersion(P: np.ndarray, inst: Instancia) -> float:
    return float(np.mean(P.std(axis=0) / (inst.techo - inst.piso)))


def algoritmo_genetico(inst: Instancia, cruce: str, mutacion: str, semilla: int, parametros: dict = None,
                       N: int = 100, k: int = 2, e: int = 2, pc: float = 0.9, pm: float = None,
                       E_max: int = 20_000, historial: bool = True) -> dict:
    """Siete etapas: M03, M08, M12, recombinación y mutación elegidas, M22 y M26."""
    par = dict(parametros or {})
    d = inst.d
    pm = 1.0 / d if pm is None else pm
    rng = np.random.default_rng(semilla)                  # un solo generador por corrida
    inicio = time.perf_counter()
    P = etapa_1_inicializar_M03(N, inst, rng)             # ETAPA 1: M03
    ev = etapa_2_evaluar_M08(inst, P)                     # ETAPA 2: M08
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
                             "fraccion_factible": float(ev["factible"].mean()), "fraccion_hijos_factibles": frac_hijos,
                             "dispersion": dispersion(P, inst), "en_cota": float(en_cota)})

    registrar(np.nan)
    while not etapa_7_parar_M26(evaluaciones, hijos, E_max):                   # ETAPA 7: M26
        rangos = rangos_densos_M08(ev)
        indices = etapa_3_seleccionar_M12(rangos, 2 * parejas, k, rng)         # ETAPA 3: M12
        A, B = P[indices[0::2]], P[indices[1::2]]
        H1, H2 = RECOMBINAR[cruce](A, B, rng, inst, par)                       # ETAPA 4: M14, M15 o M16
        cruza = rng.random(parejas) < pc
        H1 = np.where(cruza[:, None], H1, A)
        H2 = np.where(cruza[:, None], H2, B)
        H = np.empty((2 * parejas, d))
        H[0::2], H[1::2] = H1, H2
        H = MUTAR[mutacion](H[:hijos], pm, rng, inst, par)                     # ETAPA 5: M18, M19 o M20
        evH = etapa_2_evaluar_M08(inst, H)                                     # repetir ETAPA 2
        evaluaciones += hijos
        P, ev = etapa_6_reemplazar_M22(P, ev, H, evH, e)                       # ETAPA 6: M22
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

# ======================================================================================
# 1b. CONTINUACIÓN: DEMANDA DE ELASTICIDAD CONSTANTE (no lineal, utilidad no necesariamente cóncava)
# ======================================================================================
def a_demanda_log(inst: Instancia, nombre: str = None) -> Instancia:
    """Misma instancia con demanda log-lineal  q_j(p) = q0_j · Π_k (p_k / P_k)^E_jk.

    La matriz de elasticidades se toma de la pendiente lineal en el precio actual,
    E_jk = -B_jk · P_k / q0_j, de modo que ambas demandas coinciden en valor y derivada en p = P.
    Propia: E_jj = -ε_j < -1. Cruzada: E_jk > 0 para sustitutos. Las existencias dejan de ser
    lineales en p, así que el conjunto factible deja de ser un poliedro convexo.
    """
    E = -inst.B * inst.precio_actual[None, :] / inst.demanda_actual[:, None]
    lineales = [i for i, (t, _) in enumerate(inst.tipo) if t in ("escalera", "índice")]
    return replace(inst, nombre=nombre or inst.nombre.replace("-", "L-", 1), modelo="log", E=E,
                   G_lin=inst.G[lineales], h_lin=inst.h[lineales])


def _jacobiana_demanda_log(inst: Instancia, p: np.ndarray):
    """q(p) y su jacobiana dq_j/dp_k = q_j E_jk / p_k."""
    q = inst.demanda(p)
    return q, q[:, None] * inst.E / p[None, :]


def gradiente_utilidad_log(inst: Instancia, p: np.ndarray) -> np.ndarray:
    q, J = _jacobiana_demanda_log(inst, p)
    return q + J.T @ (p - inst.costo)


def resolver_referencia_log(inst: Instancia, n_arranques: int = 24, semilla: int = 0) -> dict:
    """Mejor solución conocida con SLSQP multiarranque (gradiente analítico).

    No es un óptimo certificado: la utilidad puede no ser cóncava. Se informa la fracción de
    arranques que convergen al mismo valor, que indica qué tan probable es que sea el óptimo global.
    """
    rng = np.random.default_rng(semilla)
    objetivo = lambda p: -utilidad(inst, p)[0]
    gradiente = lambda p: -gradiente_utilidad_log(inst, p)
    reglas = [
        {"type": "ineq", "fun": lambda p: inst.h_lin - inst.G_lin @ p, "jac": lambda p: -inst.G_lin},
        {"type": "ineq", "fun": lambda p: 1.0 - inst.demanda(p) / inst.existencias,
         "jac": lambda p: -_jacobiana_demanda_log(inst, p)[1] / inst.existencias[:, None]},
    ]
    cotas = list(zip(inst.piso, inst.techo))
    arranques = [inst.precio_actual, inst.piso, inst.techo, (inst.piso + inst.techo) / 2]
    for _ in range(n_arranques - 4):      # arranques aleatorios acercados al precio actual (que es factible)
        azar = inst.piso + rng.random(inst.d) * (inst.techo - inst.piso)
        arranques.append(inst.precio_actual + rng.random() * (azar - inst.precio_actual))
    inicio = time.perf_counter()
    candidatos = []
    for x0 in arranques:
        try:
            r = minimize(objetivo, x0, jac=gradiente, method="SLSQP", bounds=cotas, constraints=reglas,
                         options={"ftol": 1e-12, "maxiter": 2000})
        except Exception:        # un arranque que falla numéricamente se descarta, no se oculta: se cuenta
            continue
        x = np.clip(r.x, inst.piso, inst.techo)
        v = float(violacion(inst, x)[0])
        candidatos.append((float(utilidad(inst, x)[0]), v, x))
    factibles = [c for c in candidatos if c[1] <= 1e-6]      # SLSQP deja violaciones de ~1e-8; se reparan abajo
    if not factibles:
        raise RuntimeError(f"{inst.nombre}: ningún arranque de SLSQP terminó factible.")
    mejor = max(factibles, key=lambda c: c[0])
    coinciden = sum(abs(c[0] - mejor[0]) <= 1e-6 * abs(mejor[0]) for c in factibles)
    # El punto de SLSQP puede violar una regla en ~1e-8; se reduce hacia el precio actual (factible) hasta cumplir M08.
    x = mejor[2]
    if violacion(inst, x)[0] > TOL_FACTIBLE:
        for t in np.concatenate([np.geomspace(1e-9, 1e-2, 60), np.linspace(0.01, 1.0, 100)]):
            y = x + t * (inst.precio_actual - x)
            if violacion(inst, y)[0] <= TOL_FACTIBLE:
                x = y
                break
    return {"precios": x, "utilidad": float(utilidad(inst, x)[0]), "arranques": len(arranques),
            "factibles": len(factibles), "coinciden": int(coinciden),
            "segundos": time.perf_counter() - inicio,
            "activas_existencias": int(np.sum(inst.demanda(x) >= inst.existencias * (1 - 1e-6)))}


# ======================================================================================
# 4. ESTADÍSTICA
# ======================================================================================
def holm(pvalores) -> np.ndarray:
    """Valores p ajustados por el procedimiento escalonado de Holm (1979)."""
    p = np.asarray(pvalores, float)
    m, ajustados, acumulado = len(p), np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        acumulado = max(acumulado, min(1.0, (m - i) * p[j]))
        ajustados[j] = acumulado
    return ajustados


def wilcoxon_emparejado(x: np.ndarray, y: np.ndarray) -> float:
    return 1.0 if np.all(x == y) else float(wilcoxon(x, y).pvalue)


def comparar(M: np.ndarray, etiquetas: list) -> dict:
    """Friedman sobre las columnas de M (bloques en filas) y Wilcoxon-Holm de cada columna contra la de menor rango."""
    chi2, p = friedmanchisquare(*M.T)
    rangos = rankdata(M, axis=1).mean(axis=0)
    mejor = int(np.argmin(rangos))
    otros = [j for j in range(M.shape[1]) if j != mejor]
    p_crudos = [wilcoxon_emparejado(M[:, j], M[:, mejor]) for j in otros]
    p_holm = dict(zip(otros, holm(p_crudos)))
    return {"chi2": float(chi2), "p": float(p), "rangos": dict(zip(etiquetas, rangos)), "mejor": etiquetas[mejor],
            "p_holm": {etiquetas[j]: float(p_holm[j]) for j in otros},
            "empatadas": [etiquetas[j] for j in otros if p_holm[j] >= 0.05]}


def matriz_bloques(df: pd.DataFrame, indice, columna: str, niveles) -> np.ndarray:
    return df.pivot_table(index=indice, columns=columna, values="brecha_pct").loc[:, list(niveles)].to_numpy()


def ranking_demsar(M: np.ndarray, etiquetas: list) -> pd.DataFrame:
    """Rango medio por columna y valor p de Demšar (2006) contra la mejor, con Holm."""
    n, k = M.shape
    R = rankdata(M, axis=1).mean(axis=0)
    mejor = int(np.argmin(R))
    ee = math.sqrt(k * (k + 1) / (6 * n))
    otros = [j for j in range(k) if j != mejor]
    p_ajustado = dict(zip(otros, holm([2 * (1 - norm.cdf(abs(R[j] - R[mejor]) / ee)) for j in otros])))
    return pd.DataFrame({"configuracion": etiquetas, "rango_medio": R,
                         "p_holm_vs_mejor": [p_ajustado.get(j, np.nan) for j in range(k)]})


def ic_bootstrap_mediana(x, repeticiones: int = 4000, semilla: int = 0, nivel: float = 0.95):
    """Intervalo bootstrap percentil de la mediana."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(semilla)
    medianas = np.median(x[rng.integers(0, len(x), (repeticiones, len(x)))], axis=1)
    a = (1 - nivel) / 2
    return float(np.quantile(medianas, a)), float(np.quantile(medianas, 1 - a))


def vargha_delaney(x, y) -> float:
    """Probabilidad de que x < y (menor brecha es mejor) más la mitad de los empates: tamaño de efecto A12."""
    x, y = np.asarray(x), np.asarray(y)
    return float(np.mean(x[:, None] < y[None, :]) + 0.5 * np.mean(x[:, None] == y[None, :]))
