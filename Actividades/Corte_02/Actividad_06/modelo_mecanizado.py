"""Optimización de condiciones de corte en torneado (modelo de Gilbert con restricciones).

Variables de decisión (x = (v, f, a)):
  v  velocidad de corte (m/min)
  f  avance (mm/rev)
  a  profundidad de corte (mm)

Costo de producción por pieza (Gilbert, 1950):
  C(x) = M·t_m + (t_m / T(x))·(M·t_c + C_t)
  t_m = (s/a)·π·D·L / (1000·v·f)                 tiempo de maquinado total (min); s/a pasadas
  T(x) = (C_v / (v·f^x·a^y))^(1/n)               vida de herramienta, ecuación de Taylor generalizada (min)

Restricciones (g(x) ≤ 0):
  Ra(f) ≤ Ra_max          rugosidad superficial, Ra = 1000·f² / (32·r_ε)   (µm)
  P(x)  ≤ P_max           potencia de corte, P = k_c·a·f·v / 60 000        (kW)
  T_min ≤ T(x)            vida mínima de herramienta para estabilidad del proceso (min)

Los coeficientes son valores de trabajo representativos de torneado de acero con herramienta de carburo.
No son datos de un fabricante; el estudio los trata como supuestos y evalúa su incertidumbre.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- parámetros del proceso
@dataclass(frozen=True)
class Parametros:
    D: float = 80.0          # diámetro de la pieza (mm)
    s: float = 4.0           # sobrematerial radial a retirar (mm); el número de pasadas es s/a
    L: float = 200.0         # longitud de maquinado (mm)
    M: float = 1.2           # costo de máquina y mano de obra (€/min)
    t_c: float = 2.0         # tiempo de cambio de herramienta (min)
    C_t: float = 4.0         # costo de una arista de corte (€)
    r_eps: float = 0.8       # radio de punta de la herramienta (mm)
    k_c: float = 2000.0      # presión específica de corte (N/mm²)
    n: float = 0.25          # exponente de Taylor
    x_f: float = 0.6         # exponente del avance en Taylor
    y_a: float = 0.3         # exponente de la profundidad en Taylor
    C_v: float = 0.0         # constante de Taylor (se calibra en __post_init__ si es 0)
    Ra_max: float = 2.0      # rugosidad máxima admisible (µm)
    P_max: float = 6.0       # potencia máxima de la máquina (kW)
    T_min: float = 15.0     # vida mínima de herramienta (min)
    cotas: tuple = ((60.0, 350.0), (0.05, 0.40), (0.5, 3.0))   # (v, f, a); a ≤ 3 mm por geometría del inserto

    def __post_init__(self):
        if self.C_v == 0.0:
            # Calibración: a v = 150 m/min, f = 0.2 mm/rev, a = 2 mm la vida de referencia es 30 min.
            C = 150.0 * 30.0 ** self.n * 0.2 ** self.x_f * 2.0 ** self.y_a
            object.__setattr__(self, "C_v", C)


PAR = Parametros()
NOMBRES = ("v (m/min)", "f (mm/rev)", "a (mm)")


# ----------------------------------------------------------------------------- funciones del modelo
def vida_herramienta(x, p: Parametros = PAR):
    v, f, a = _separa(x)
    return (p.C_v / (v * f ** p.x_f * a ** p.y_a)) ** (1.0 / p.n)


def tiempo_maquinado(x, p: Parametros = PAR):
    v, f, a = _separa(x)
    return (p.s / a) * math.pi * p.D * p.L / (1000.0 * v * f)


def costo(x, p: Parametros = PAR):
    t_m = tiempo_maquinado(x, p)
    T = vida_herramienta(x, p)
    return p.M * t_m + (t_m / T) * (p.M * p.t_c + p.C_t)


def rugosidad(x, p: Parametros = PAR):
    _, f, _ = _separa(x)
    return 1000.0 * f ** 2 / (32.0 * p.r_eps)


def potencia(x, p: Parametros = PAR):
    v, f, a = _separa(x)
    return p.k_c * a * f * v / 60_000.0


def restricciones(x, p: Parametros = PAR):
    """Vector g(x) ≤ 0 de restricciones normalizadas (relativas al límite)."""
    x = np.asarray(x, float)
    return np.array([
        rugosidad(x, p) / p.Ra_max - 1.0,
        potencia(x, p) / p.P_max - 1.0,
        p.T_min / vida_herramienta(x, p) - 1.0,
    ])


def violacion(x, p: Parametros = PAR) -> float:
    """Violación total: suma de las partes positivas de g(x). Cero indica factibilidad."""
    return float(np.sum(np.maximum(0.0, restricciones(x, p))))


def _separa(x):
    x = np.asarray(x, float)
    return x[..., 0], x[..., 1], x[..., 2]


def cotas(p: Parametros = PAR):
    lo = np.array([c[0] for c in p.cotas]); hi = np.array([c[1] for c in p.cotas])
    return lo, hi


# ----------------------------------------------------------------------------- referencia de referencia
def referencia(p: Parametros = PAR, malla: int = 40, arranques: int = 12) -> dict:
    """Mejor solución conocida: malla exhaustiva en la caja y refinamiento SLSQP desde los mejores puntos.
    No es un certificado de optimalidad; se reporta la distancia entre el refinamiento y la malla."""
    lo, hi = cotas(p)
    g = [np.linspace(lo[i], hi[i], malla) for i in range(3)]
    V, F, A = np.meshgrid(*g, indexing="ij")
    X = np.stack([V.ravel(), F.ravel(), A.ravel()], axis=1)
    C = costo(X, p); G = _restr_vec(X, p)
    viol = np.maximum(0, G).sum(axis=1)
    factible = viol <= 0
    Cf = np.where(factible, C, np.inf)
    orden = np.argsort(Cf)[:arranques]
    mejor = {"costo": np.inf, "x": None}
    cons = [{"type": "ineq", "fun": lambda x: -restricciones(x, p)}]
    for i in orden:
        if not np.isfinite(Cf[i]):
            break
        r = minimize(lambda x: costo(x, p), X[i], method="SLSQP", bounds=list(zip(lo, hi)), constraints=cons,
                     options={"ftol": 1e-12, "maxiter": 500})
        xr = np.clip(r.x, lo, hi)
        if violacion(xr, p) <= 1e-9 and costo(xr, p) < mejor["costo"]:
            mejor = {"costo": float(costo(xr, p)), "x": xr}
    return {"costo": mejor["costo"], "x": mejor["x"], "malla_factibles": int(factible.sum()),
            "malla_mejor": float(Cf.min()), "malla_x": X[int(np.argmin(Cf))]}


def _restr_vec(X, p):
    v, f, a = X[:, 0], X[:, 1], X[:, 2]
    T = (p.C_v / (v * f ** p.x_f * a ** p.y_a)) ** (1.0 / p.n)
    Ra = 1000.0 * f ** 2 / (32.0 * p.r_eps)
    P = p.k_c * a * f * v / 60_000.0
    return np.stack([Ra / p.Ra_max - 1.0, P / p.P_max - 1.0, p.T_min / T - 1.0], axis=1)


def perturbar(p: Parametros, rng, cv=0.10, cr=0.05) -> Parametros:
    """Parámetros con incertidumbre: coeficiente de Taylor y k_c ±cv (lognormal), rugosidad ±cr."""
    return replace(p, C_v=p.C_v * float(np.exp(rng.normal(0, cv))),
                   k_c=p.k_c * float(np.exp(rng.normal(0, cv))),
                   Ra_max=p.Ra_max, r_eps=p.r_eps * float(np.exp(rng.normal(0, cr))))
