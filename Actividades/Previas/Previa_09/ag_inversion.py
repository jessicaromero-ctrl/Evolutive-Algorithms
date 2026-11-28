"""ag_inversion — Cartera estatal de inversión para la productividad de Hidalgo (datos sintéticos).

Actividad previa número 9 · Algoritmos Evolutivos · Maestría en Inteligencia Artificial
Universidad Politécnica Metropolitana de Hidalgo · Dr. Jaime Aguilar Ortiz
Alumna: Jessica Melani Romero Lora (matrícula 25322116)

El gen x[r, m] es la inversión (millones de pesos) asignada al mercado m en la región r.
Un algoritmo genético de siete etapas la distribuye para maximizar el valor agregado anual,
sujeto a presupuesto, agua disponible por región, equidad para la Valle del Mezquital,
empleo mínimo y tope por región.

El valor agregado tiene rendimientos no monótonos en escala de cuadrante: cada mercado
necesita un umbral de infraestructura (θ) antes de rendir su potencial. La función es
no cóncava, así que no hay certificado KKT global; la referencia es el mejor valor conocido.

Reutiliza de ag_precios los operadores genéricos (M03, M12, M14–M16, M18–M20, M22, M26)
porque sólo necesitan piso, techo y d. La evaluación y el bucle genético se definen aquí.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ag_precios import (TOL_FACTIBLE, RECOMBINAR, MUTAR, etapa_1_inicializar_M03, etapa_3_seleccionar_M12,
                        etapa_6_reemplazar_M22, etapa_7_parar_M26, orden_M08, rangos_densos_M08, dispersion)

REGIONES = ("Zapotlán–Tizayuca", "Pachuca–Mineral de la Reforma", "Tula–Tepeji", "Valle del Mezquital")
MERCADOS = ("Logística", "Manufactura ligera", "Agroindustria", "Turismo", "Agua y energía industrial")
R, M = len(REGIONES), len(MERCADOS)
IDX_VM = 3                       # índice de la Valle del Mezquital en REGIONES

# ----------------------------------------------------------------------------- parámetros (sintéticos)
# Valor agregado por millón de pesos invertido, en régimen pleno (a ≥ umbral).
A = np.array([[0.80, 0.70, 0.45, 0.35, 0.55],
              [0.60, 0.65, 0.40, 0.50, 0.45],
              [0.55, 0.75, 0.35, 0.30, 0.70],
              [0.40, 0.30, 0.60, 0.35, 0.25]])
# Umbral de infraestructura habilitadora (mdp) por región y mercado.
THETA = np.array([[2500, 1500, 1200, 800, 1800],
                  [2000, 1200, 1000, 900, 1500],
                  [2200, 1600, 1400, 700, 2000],
                  [2500, 1500, 1000, 800, 2500]], float)
ANCHO = 400.0                    # anchura de la transición (mdp)
TECHO_BASE = np.array([[6000, 4000, 2500, 2000, 3000],
                       [4000, 3000, 2000, 3000, 2500],
                       [3000, 4000, 2000, 1500, 3000],
                       [2000, 1500, 3000, 2500, 1500]], float)
EMPLEO_POR_MDP = np.array([[1.2, 1.8, 0.9, 0.6, 0.5],
                           [1.0, 1.5, 0.8, 1.1, 0.4],
                           [0.9, 1.9, 0.7, 0.6, 0.6],
                           [0.7, 1.3, 1.6, 0.9, 0.4]])
USO_AGUA = np.array([[0.02, 0.05, 0.10, 0.02, 0.12],
                     [0.02, 0.04, 0.08, 0.03, 0.10],
                     [0.02, 0.05, 0.10, 0.02, 0.20],
                     [0.03, 0.04, 0.25, 0.04, 0.12]])
AGUA_DISPONIBLE = np.array([900.0, 900.0, 600.0, 500.0])
PRESUPUESTO = 20_000.0           # millones de pesos
EQUIDAD_VM = 0.15                # mínimo de la inversión total para la Valle del Mezquital
TOPE_REGION = 0.40               # máximo de la inversión total por región
EMPLEO_MINIMO = 12_000.0         # empleos directos a generar


@dataclass
class Cartera:
    """Instancia de la cartera. Duck-typed para los operadores de ag_precios (d, piso, techo)."""
    nombre: str
    presupuesto: float
    A: np.ndarray
    theta: np.ndarray
    ancho: float
    techo_u: np.ndarray
    emp: np.ndarray
    uso: np.ndarray
    agua: np.ndarray
    equidad: float
    tope: float
    empleo_min: float

    @property
    def d(self) -> int:
        return R * M

    @property
    def piso(self) -> np.ndarray:
        return np.zeros(R * M)

    @property
    def techo(self) -> np.ndarray:
        return self.techo_u.ravel()


def generar_cartera(nombre: str, presupuesto: float = PRESUPUESTO, escala_agua: float = 1.0,
                    escala_U: float = 1.0, empleo_min: float = EMPLEO_MINIMO) -> Cartera:
    return Cartera(nombre, presupuesto, A.copy(), THETA.copy(), ANCHO, TECHO_BASE * escala_U,
                   EMPLEO_POR_MDP.copy(), USO_AGUA.copy(), AGUA_DISPONIBLE * escala_agua,
                   EQUIDAD_VM, TOPE_REGION, empleo_min)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def valor(c: Cartera, X) -> np.ndarray:
    """Valor agregado anual (mdp) por fila de X. Función no cóncava: s(x) = 0.35 + 0.65·σ((x−θ)/ancho)."""
    Xr = np.atleast_2d(np.asarray(X, float)).reshape(-1, R, M)
    s = 0.35 + 0.65 * _sig((Xr - c.theta) / c.ancho)
    return np.sum(c.A * Xr * s, axis=(1, 2))


def violaciones(c: Cartera, X) -> dict:
    """Violaciones relativas por restricción (cero = cumple). Todas en escala adimensional."""
    Xr = np.atleast_2d(np.asarray(X, float)).reshape(-1, R, M)
    total_r = Xr.sum(axis=2)                                             # (n, R)
    return {
        "presupuesto": np.maximum(0.0, total_r.sum(axis=1) / c.presupuesto - 1.0),
        "agua": np.maximum(0.0, (Xr * c.uso).sum(axis=2) / c.agua - 1.0).sum(axis=1),
        "equidad_vm": np.maximum(0.0, 1.0 - Xr[:, IDX_VM, :].sum(axis=1) / (c.equidad * c.presupuesto)),
        "empleo": np.maximum(0.0, 1.0 - (Xr * c.emp).sum(axis=(1, 2)) / c.empleo_min),
        "tope_region": np.maximum(0.0, total_r / (c.tope * c.presupuesto) - 1.0).sum(axis=1),
    }


def violacion(c: Cartera, X) -> np.ndarray:
    return sum(violaciones(c, X).values())


def _restricciones_lineales(c: Cartera):
    """Filas (Ca, cv) con Ca·x + cv ≥ 0 para cada restricción lineal."""
    filas, cv = [-np.ones(R * M)], [c.presupuesto]
    for r in range(R):
        fila = np.zeros((R, M)); fila[r, :] = -c.uso[r]
        filas.append(fila.ravel()); cv.append(c.agua[r])
    fila = np.zeros((R, M)); fila[IDX_VM, :] = 1.0
    filas.append(fila.ravel()); cv.append(-c.equidad * c.presupuesto)
    filas.append(c.emp.ravel()); cv.append(-c.empleo_min)
    for r in range(R):
        fila = np.zeros((R, M)); fila[r, :] = -1.0
        filas.append(fila.ravel()); cv.append(c.tope * c.presupuesto)
    return np.array(filas), np.array(cv)


def _evaluar(c: Cartera, X) -> dict:
    V = violacion(c, X)
    V = np.where(V <= TOL_FACTIBLE, 0.0, V)
    return {"V": V, "utilidad": valor(c, X), "factible": V == 0.0}


def resolver_referencia(c: Cartera, n_arranques: int = 40, semilla: int = 0) -> dict:
    """Mejor valor conocido por SLSQP multiarranque (gradiente analítico). No es óptimo certificado."""
    rng = np.random.default_rng(semilla)
    Ca, cv = _restricciones_lineales(c)

    def objetivo(x):
        return -valor(c, x)[0]

    def gradiente(x):
        Xr = x.reshape(R, M)
        sg = _sig((Xr - c.theta) / c.ancho)
        s = 0.35 + 0.65 * sg
        ds = 0.65 * sg * (1 - sg) / c.ancho
        return -(c.A * (s + Xr * ds)).ravel()

    restr = [{"type": "ineq", "fun": lambda x: Ca @ x + cv, "jac": lambda x: Ca}]
    cotas = list(zip(c.piso, c.techo))
    arranques = [0.5 * c.techo, np.zeros(R * M) + 1.0]
    arranques += [c.piso + rng.random(R * M) * (c.techo - c.piso) for _ in range(n_arranques - 2)]
    inicio, candidatos = time.perf_counter(), []
    for x0 in arranques:
        try:
            r = minimize(objetivo, x0, jac=gradiente, method="SLSQP", bounds=cotas, constraints=restr,
                         options={"ftol": 1e-12, "maxiter": 1000})
        except Exception:
            continue
        x = np.clip(r.x, c.piso, c.techo)
        v = float(violacion(c, x)[0])
        candidatos.append((float(valor(c, x)[0]), v, x))
    factibles = [q for q in candidatos if q[1] <= 1e-9]
    if not factibles:
        raise RuntimeError(f"{c.nombre}: ningún arranque de SLSQP terminó factible.")
    mejor = max(factibles, key=lambda q: q[0])
    coinciden = sum(abs(q[0] - mejor[0]) <= 1e-6 * abs(mejor[0]) for q in factibles)
    return {"x": mejor[2], "valor": mejor[0], "violacion": mejor[1], "arranques": len(arranques),
            "factibles": len(factibles), "coinciden": int(coinciden), "segundos": time.perf_counter() - inicio}


def reparto_proporcional(c: Cartera) -> dict:
    """Línea base sin análisis: reparte el presupuesto en proporción a la capacidad de cada gen."""
    x = c.presupuesto * c.techo / c.techo.sum()
    return {"x": x, "valor": float(valor(c, x)[0]), "violacion": float(violacion(c, x)[0])}


def busqueda_aleatoria(c: Cartera, semilla: int, evaluaciones: int, N: int = 100, por_bloque: int = 98) -> dict:
    """Muestreo uniforme en la caja con el mismo presupuesto de evaluaciones que el AG."""
    rng = np.random.default_rng(semilla)
    mejor_V, mejor_U, mejor_x = np.inf, -np.inf, None
    usadas, bloque = 0, N
    while usadas + bloque <= evaluaciones:
        X = c.piso + rng.random((bloque, c.d)) * (c.techo - c.piso)
        ev = _evaluar(c, X)
        usadas += bloque
        i = int(orden_M08(ev)[0])
        if (ev["V"][i], -ev["utilidad"][i]) < (mejor_V, -mejor_U):
            mejor_V, mejor_U, mejor_x = float(ev["V"][i]), float(ev["utilidad"][i]), X[i].copy()
        bloque = por_bloque
    factible = mejor_V <= 0
    return {"valor": mejor_U if factible else np.nan, "factible": bool(factible), "x": mejor_x,
            "violacion_minima": mejor_V, "evaluaciones": usadas}


def algoritmo_genetico(c: Cartera, cruce: str, mutacion: str, semilla: int, parametros: dict = None,
                       N: int = 100, k: int = 2, e: int = 2, pc: float = 0.9, pm: float = None,
                       E_max: int = 20_000, historial: bool = True) -> dict:
    """Siete etapas: M03, M08, M12, recombinación y mutación elegidas, M22 y M26."""
    par = dict(parametros or {})
    d = c.d
    pm = 1.0 / d if pm is None else pm
    rng = np.random.default_rng(semilla)
    inicio = time.perf_counter()
    P = etapa_1_inicializar_M03(N, c, rng)                                  # ETAPA 1
    ev = _evaluar(c, P)                                                     # ETAPA 2
    evaluaciones, generacion = N, 0
    hijos, parejas = N - e, (N - e + 1) // 2
    mejor = {"valor": -np.inf, "x": None, "evaluaciones": None}
    registro, clave_anterior = [], (np.inf, np.inf)

    def registrar(frac_hijos: float):
        nonlocal clave_anterior
        i = int(orden_M08(ev)[0])
        clave = (float(ev["V"][i]), float(-ev["utilidad"][i]))
        if clave > clave_anterior:
            raise RuntimeError("La mejor clave empeoró: el elitismo no se cumplió.")
        clave_anterior = clave
        if ev["factible"][i] and ev["utilidad"][i] > mejor["valor"]:
            mejor.update(valor=float(ev["utilidad"][i]), x=P[i].copy(), evaluaciones=evaluaciones)
        if historial:
            registro.append({"generacion": generacion, "evaluaciones": evaluaciones,
                             "mejor_factible": mejor["valor"] if mejor["x"] is not None else np.nan,
                             "fraccion_factible": float(ev["factible"].mean()),
                             "fraccion_hijos_factibles": frac_hijos, "dispersion": dispersion(P, c),
                             "en_cota": float(np.mean((P == c.piso) | (P == c.techo)))})

    registrar(np.nan)
    while not etapa_7_parar_M26(evaluaciones, hijos, E_max):                # ETAPA 7
        rangos = rangos_densos_M08(ev)
        indices = etapa_3_seleccionar_M12(rangos, 2 * parejas, k, rng)      # ETAPA 3
        A_, B_ = P[indices[0::2]], P[indices[1::2]]
        H1, H2 = RECOMBINAR[cruce](A_, B_, rng, c, par)                     # ETAPA 4
        cruza = rng.random(parejas) < pc
        H1 = np.where(cruza[:, None], H1, A_)
        H2 = np.where(cruza[:, None], H2, B_)
        H = np.empty((2 * parejas, d))
        H[0::2], H[1::2] = H1, H2
        H = MUTAR[mutacion](H[:hijos], pm, rng, c, par)                     # ETAPA 5
        evH = _evaluar(c, H)                                                # ETAPA 2 de los hijos
        evaluaciones += hijos
        P, ev = etapa_6_reemplazar_M22(P, ev, H, evH, e)                    # ETAPA 6
        generacion += 1
        registrar(float(evH["factible"].mean()))

    if evaluaciones != N + generacion * (N - e) or evaluaciones > E_max:
        raise RuntimeError("El contador de evaluaciones no coincide con N + G(N - e).")
    if mejor["x"] is not None:
        if violacion(c, mejor["x"])[0] > 10 * TOL_FACTIBLE or abs(valor(c, mejor["x"])[0] - mejor["valor"]) > 1e-8 * abs(mejor["valor"]):
            raise RuntimeError("La solución entregada no coincide con su evaluación directa.")
    return {"valor": mejor["valor"] if mejor["x"] is not None else np.nan, "x": mejor["x"],
            "eval_mejor": mejor["evaluaciones"], "evaluaciones": evaluaciones, "generaciones": generacion,
            "segundos": time.perf_counter() - inicio, "historial": registro, "poblacion": P,
            "fraccion_factible": float(ev["factible"].mean()), "dispersion": dispersion(P, c),
            "en_cota": float(np.mean((P == c.piso) | (P == c.techo))),
            "parametros": {"cruce": cruce, "mutacion": mutacion, **par, "N": N, "k": k, "e": e,
                           "pc": pc, "pm": pm, "E_max": E_max, "semilla": semilla}}


def resumen_por_sector(c: Cartera, x) -> "pd.DataFrame":
    import pandas as pd
    Xr = np.asarray(x, float).reshape(R, M)
    filas = []
    for r in range(R):
        for m in range(M):
            filas.append({"region": REGIONES[r], "mercado": MERCADOS[m], "inversion_mdp": Xr[r, m],
                          "empleos": Xr[r, m] * c.emp[r, m], "agua_uso": Xr[r, m] * c.uso[r, m],
                          "valor_agregado_mdp": c.A[r, m] * Xr[r, m] *
                          (0.35 + 0.65 / (1 + np.exp(-np.clip((Xr[r, m] - c.theta[r, m]) / c.ancho, -500, 500))))})
    return pd.DataFrame(filas)


# ======================================================================================
# Algoritmo elegido: evolución diferencial con reglas de factibilidad (Deb, 2000)
# ======================================================================================
def regla_factibilidad(Va: float, Ua: float, Vb: float, Ub: float) -> bool:
    """True si la solución a supera a b: factible y mejor valor; si ambas son infactibles, menor violación."""
    fa, fb = Va <= TOL_FACTIBLE, Vb <= TOL_FACTIBLE
    if fa and fb:
        return Ua > Ub
    if fa != fb:
        return fa
    return Va < Vb


MUTACIONES_DE = ("rand1", "best1", "ctb1")
CRUCES_DE = ("bin", "exp", "arit")
NOMBRE_MUT = {"rand1": "M-rand/1", "best1": "M-best/1", "ctb1": "M-current-to-best/1"}
NOMBRE_CRUCE = {"bin": "binomial", "exp": "exponencial", "arit": "aritmética"}


def _indice_mejor(V: np.ndarray, U: np.ndarray) -> int:
    """Mejor individuo según la regla de factibilidad (mayor valor entre factibles; si no hay, menor violación)."""
    ok = V <= TOL_FACTIBLE
    return int(np.argmax(np.where(ok, U, -np.inf))) if ok.any() else int(np.argmin(V))


def _mutante(P, i, mejor, F, rng, mutacion):
    NP, d = P.shape
    otros = [j for j in range(NP) if j != i]
    if mutacion == "rand1":
        a, b, q = rng.choice(otros, 3, replace=False)
        return P[a] + F * (P[b] - P[q])
    if mutacion == "best1":
        a, b = rng.choice([j for j in otros if j != mejor], 2, replace=False)
        return P[mejor] + F * (P[a] - P[b])
    if mutacion == "ctb1":
        a, b = rng.choice([j for j in otros if j != mejor], 2, replace=False)
        return P[i] + F * (P[mejor] - P[i]) + F * (P[a] - P[b])
    raise ValueError(mutacion)


def _cruce(x, v, CR, rng, cruce):
    d = x.size
    if cruce == "bin":
        m = rng.random(d) < CR
        m[rng.integers(d)] = True
        return np.where(m, v, x)
    if cruce == "exp":
        u = x.copy(); j = rng.integers(d); L = 0
        while True:
            u[j] = v[j]; j = (j + 1) % d; L += 1
            if L >= d or rng.random() >= CR:
                break
        return u
    if cruce == "arit":
        lam = rng.random()
        return lam * v + (1 - lam) * x
    raise ValueError(cruce)


def evolucion_diferencial(c: Cartera, semilla: int, E_max: int = 20_000, NP: int = 60,
                          F: float = 0.4, CR: float = 0.5, mutacion: str = "rand1",
                          cruce: str = "bin") -> dict:
    """Evolución diferencial (Storn y Price, 1997) con la regla de factibilidad de Deb (2000).

    mutacion ∈ {rand1, best1, ctb1}; cruce ∈ {bin, exp, arit}. Un único presupuesto de E_max
    evaluaciones de la función, igual para todas las configuraciones."""
    rng = np.random.default_rng(semilla)
    d = c.d
    inicio = time.perf_counter()
    P = c.piso + rng.random((NP, d)) * (c.techo - c.piso)
    V, U = violacion(c, P), valor(c, P)
    n = NP
    historial = []
    while n < E_max:
        for i in range(NP):
            if n >= E_max:
                break
            mejor = _indice_mejor(V, U)
            v = _mutante(P, i, mejor, F, rng, mutacion)
            u = np.clip(_cruce(P[i], v, CR, rng, cruce), c.piso, c.techo)
            Vu, Uu = violacion(c, u[None])[0], valor(c, u[None])[0]
            n += 1
            if regla_factibilidad(Vu, Uu, V[i], U[i]):
                P[i], V[i], U[i] = u, Vu, Uu
        ok = V <= TOL_FACTIBLE
        historial.append({"evaluaciones": n,
                          "mejor_factible": float(U[ok].max()) if ok.any() else np.nan,
                          "fraccion_factible": float(ok.mean())})
    ok = V <= TOL_FACTIBLE
    if ok.any():
        i = int(np.argmax(np.where(ok, U, -np.inf)))
        x, valor_mejor = P[i].copy(), float(U[i])
    else:
        x, valor_mejor = None, np.nan
    return {"valor": valor_mejor, "x": x, "evaluaciones": n, "segundos": time.perf_counter() - inicio,
            "historial": historial, "fraccion_factible": float(ok.mean()),
            "parametros": {"algoritmo": f"DE {NOMBRE_MUT[mutacion]} + cruce {NOMBRE_CRUCE[cruce]}",
                           "mutacion": mutacion, "cruce": cruce, "NP": NP, "F": F, "CR": CR,
                           "E_max": E_max, "semilla": semilla}}
