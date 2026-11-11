"""moea_hiper — Optimización multiobjetivo de hiperparámetros de una red neuronal (MLP).

Problema: minimizar simultáneamente
  f1 = 1 − exactitud (validación cruzada estratificada de 3 pliegues), error de generalización;
  f2 = costo de entrenamiento en MFLOPs (estimado analíticamente: 6 · parámetros · n_train · épocas);
  f3 = número de parámetros del modelo (tamaño).
Datos: Breast Cancer Wisconsin (scikit-learn), clasificación binaria, 569 muestras, 30 variables.

Espacio de decisión discreto (960 configuraciones distintas) con codificación entera de 6 genes:
  g0 ∈ {0,1}: número de capas ocultas L = g0 + 1
  g1 ∈ {0..3}: unidades de la primera capa  U = (8, 16, 32, 64)
  g2 ∈ {0..3}: unidades de la segunda capa (ignorado si L = 1; se fija en 0 para evitar duplicados)
  g3 ∈ {0..3}: tasa de aprendizaje     (1e-4, 1e-3, 1e-2, 1e-1)
  g4 ∈ {0..3}: regularización L2 alpha (1e-5, 1e-4, 1e-3, 1e-2)
  g5 ∈ {0..2}: tamaño de lote          (32, 64, 256)

Como el espacio es pequeño, se evalúa completo una vez (tabla de verdad) y los algoritmos consultan
esa tabla, que es idéntica a volver a entrenar: la prueba de reproducibilidad lo verifica.
El frente de Pareto exacto se obtiene a partir de la tabla completa.
"""
from __future__ import annotations

import time
import warnings
from itertools import product

import numpy as np

warnings.filterwarnings("ignore")

CARDINALIDAD = (2, 4, 4, 4, 4, 3)
UNIDADES = (8, 16, 32, 64)
LEARNING_RATE = (1e-4, 1e-3, 1e-2, 1e-1)
ALPHA = (1e-5, 1e-4, 1e-3, 1e-2)
BATCH = (32, 64, 256)
EPOCAS = 60
N_PLIEGUES = 3
N_ENTRADA = 30
N_TRAIN = 379                    # tamaño nominal del conjunto de entrenamiento por pliegue
NOMBRES_OBJ = ("error", "MFLOPs", "parámetros")


# ----------------------------------------------------------------------------- decodificación
def decodificar(g) -> dict:
    """Genotipo → hiperparámetros (fenotipo)."""
    g = tuple(int(v) for v in g)
    L = g[0] + 1
    capas = (UNIDADES[g[1]],) if L == 1 else (UNIDADES[g[1]], UNIDADES[g[2]])
    return {"capas": capas, "lr": LEARNING_RATE[g[3]], "alpha": ALPHA[g[4]], "batch": BATCH[g[5]]}


def canonico(g) -> tuple:
    """Fija g2 = 0 cuando L = 1, para que configuraciones iguales tengan un solo genotipo."""
    g = [int(v) for v in g]
    if g[0] == 0:
        g[2] = 0
    return tuple(g)


def espacio_completo() -> list:
    """Todos los genotipos canónicos distintos (960)."""
    return sorted({canonico(g) for g in product(*(range(c) for c in CARDINALIDAD))})


def parametros(capas, n_in=N_ENTRADA, n_out=1) -> int:
    """Número de pesos y sesgos de un MLP con las capas dadas (igual que scikit-learn)."""
    tamanos = (n_in,) + tuple(capas) + (n_out,)
    return int(sum((a + 1) * b for a, b in zip(tamanos[:-1], tamanos[1:])))


def flops_entrenamiento(n_param: int) -> float:
    """Estimación de FLOPs para entrenar el modelo en un pliegue: 6 por parámetro, muestra y época."""
    return 6.0 * n_param * N_TRAIN * EPOCAS / 1e6


# ----------------------------------------------------------------------------- evaluación real
_DATOS = None


def _datos():
    global _DATOS
    if _DATOS is None:
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        _DATOS = (X, y)
    return _DATOS


def entrenar_y_evaluar(g) -> tuple:
    """Entrena el MLP con validación cruzada real y devuelve (f1, f2, f3, exactitud)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = _datos()
    hp = decodificar(g)
    modelo = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=hp["capas"], learning_rate_init=hp["lr"], alpha=hp["alpha"],
        batch_size=hp["batch"], max_iter=EPOCAS, tol=0.0, n_iter_no_change=10**6, random_state=0))
    cv = StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=0)
    exactitud = float(np.mean(cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")))
    p = parametros(hp["capas"])
    return (1.0 - exactitud, flops_entrenamiento(p), float(p), exactitud)


class Tabla:
    """Evaluaciones exhaustivas. Cada consulta cuenta como una evaluación del algoritmo."""

    def __init__(self, registros: dict):
        self.r = registros
        self.llamadas = 0

    def __call__(self, g):
        self.llamadas += 1
        return self.r[canonico(g)]

    def reiniciar(self):
        self.llamadas = 0


# ----------------------------------------------------------------------------- dominancia y métricas
def domina(a, b) -> bool:
    """a domina a b (minimización): no es peor en ningún objetivo y es mejor en alguno."""
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def frente_no_dominado(F: np.ndarray) -> np.ndarray:
    """Índices de las soluciones no dominadas."""
    F = np.asarray(F, float)
    n = len(F); mascara = np.ones(n, bool)
    for i in range(n):
        if not mascara[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dom.any():
            mascara[i] = False
    return np.flatnonzero(mascara)


def nds(F: np.ndarray) -> list:
    """Ordenamiento por no dominancia rápido (Deb et al., 2002): lista de frentes (listas de índices)."""
    F = np.asarray(F, float); n = len(F)
    S = [[] for _ in range(n)]; cuenta = np.zeros(n, int); frentes = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if domina(F[p], F[q]):
                S[p].append(q)
            elif domina(F[q], F[p]):
                cuenta[p] += 1
        if cuenta[p] == 0:
            frentes[0].append(p)
    i = 0
    while frentes[i]:
        siguiente = []
        for p in frentes[i]:
            for q in S[p]:
                cuenta[q] -= 1
                if cuenta[q] == 0:
                    siguiente.append(q)
        i += 1
        frentes.append(siguiente)
    return [f for f in frentes if f]


def distancia_crowding(F: np.ndarray, idx) -> np.ndarray:
    idx = list(idx); k = len(idx); d = np.zeros(k)
    if k <= 2:
        d[:] = np.inf; return d
    sub = F[idx]
    for m in range(F.shape[1]):
        orden = np.argsort(sub[:, m]); rango = sub[orden[-1], m] - sub[orden[0], m]
        d[orden[0]] = d[orden[-1]] = np.inf
        if rango <= 0:
            continue
        for j in range(1, k - 1):
            d[orden[j]] += (sub[orden[j + 1], m] - sub[orden[j - 1], m]) / rango
    return d


def hipervolumen(P: np.ndarray, ref: np.ndarray) -> float:
    """Hipervolumen de un conjunto de puntos (minimización) respecto al punto de referencia ref.
    Algoritmo de rebanadas en 2D y 3D, exacto para cualquier número de puntos."""
    P = np.asarray(P, float); ref = np.asarray(ref, float)
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[frente_no_dominado(P)]
    if P.shape[1] == 2:
        return _hv2d(P, ref)
    P = P[np.argsort(P[:, 2])]
    total = 0.0
    for i in range(len(P)):
        z_sig = P[i + 1, 2] if i + 1 < len(P) else ref[2]
        if z_sig > P[i, 2]:
            total += _hv2d(P[: i + 1, :2], ref[:2]) * (z_sig - P[i, 2])
    return float(total)


def _hv2d(P, ref) -> float:
    P = P[np.argsort(P[:, 0])]
    area, mejor_y = 0.0, ref[1]
    for x, y in P:
        if y < mejor_y:
            area += (ref[0] - x) * (mejor_y - y)
            mejor_y = y
    return float(area)


def igd(aproximacion: np.ndarray, referencia: np.ndarray) -> float:
    """Distancia generacional invertida: media de la distancia de cada punto del frente exacto al más cercano."""
    A = np.asarray(aproximacion, float); R = np.asarray(referencia, float)
    d = np.sqrt(((R[:, None, :] - A[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return float(d.mean())


# ----------------------------------------------------------------------------- operadores
def cruce_uniforme(a, b, rng, pc=0.9):
    a, b = np.array(a), np.array(b)
    if rng.random() >= pc:
        return a.copy(), b.copy()
    m = rng.random(len(a)) < 0.5
    return np.where(m, a, b), np.where(m, b, a)


def mutar(g, rng, pm):
    g = np.array(g)
    for j in range(len(g)):
        if rng.random() < pm:
            g[j] = rng.integers(CARDINALIDAD[j])
    return canonico(g)


def muestra_aleatoria(rng):
    return canonico([rng.integers(c) for c in CARDINALIDAD])


# ----------------------------------------------------------------------------- algoritmos
class Registro:
    """Guarda cada evaluación (genotipo, objetivos) para calcular métricas a lo largo del presupuesto."""

    def __init__(self, evaluar, presupuesto):
        self.evaluar, self.presupuesto = evaluar, presupuesto
        self.genotipos, self.objs = [], []

    def __call__(self, g):
        if len(self.genotipos) >= self.presupuesto:
            raise StopIteration
        f = self.evaluar(g)[:3]
        self.genotipos.append(canonico(g)); self.objs.append(f)
        return np.array(f)

    @property
    def agotado(self):
        return len(self.genotipos) >= self.presupuesto


def nsga2(evaluar, presupuesto, semilla, N=30, pm=None):
    rng = np.random.default_rng(semilla)
    pm = 1.0 / len(CARDINALIDAD) if pm is None else pm
    R = Registro(evaluar, presupuesto)
    try:
        P = [muestra_aleatoria(rng) for _ in range(N)]
        F = np.array([R(g) for g in P])
        while not R.agotado:
            frentes = nds(F); rango = np.zeros(len(P)); crowd = np.zeros(len(P))
            for r, fr in enumerate(frentes):
                rango[fr] = r; crowd[fr] = distancia_crowding(F, fr)
            hijos = []
            while len(hijos) < N:
                i, j = rng.integers(len(P), size=2)
                padre1 = P[i] if (rango[i], -crowd[i]) < (rango[j], -crowd[j]) else P[j]
                i, j = rng.integers(len(P), size=2)
                padre2 = P[i] if (rango[i], -crowd[i]) < (rango[j], -crowd[j]) else P[j]
                h1, h2 = cruce_uniforme(padre1, padre2, rng)
                hijos += [mutar(h1, rng, pm), mutar(h2, rng, pm)]
            hijos = hijos[:N]
            Fh = []
            for g in hijos:
                Fh.append(R(g))
            Fh = np.array(Fh)
            Q, FQ = P + hijos, np.vstack([F, Fh])
            nuevo = []
            for fr in nds(FQ):
                if len(nuevo) + len(fr) <= N:
                    nuevo += fr
                else:
                    d = distancia_crowding(FQ, fr)
                    nuevo += [fr[k] for k in np.argsort(-d)[: N - len(nuevo)]]
                    break
            P = [Q[k] for k in nuevo]; F = FQ[nuevo]
    except StopIteration:
        pass
    return R


def _pesos(H, m=3):
    """Vectores de pesos uniformes en el simplex con H divisiones (Zhang y Li, 2007)."""
    vs = []
    for c in product(range(H + 1), repeat=m):
        if sum(c) == H:
            vs.append(np.array(c, float) / H)
    return np.array(vs)


def moead(evaluar, presupuesto, semilla, H=6, T=5, nr=2, pm=None):
    """MOEA/D con descomposición de Tchebycheff (Zhang y Li, 2007)."""
    rng = np.random.default_rng(semilla)
    pm = 1.0 / len(CARDINALIDAD) if pm is None else pm
    W = _pesos(H); K = len(W)
    vec = np.argsort(np.linalg.norm(W[:, None] - W[None], axis=2), axis=1)[:, :T]
    R = Registro(evaluar, presupuesto)
    try:
        X = [muestra_aleatoria(rng) for _ in range(K)]
        F = np.array([R(g) for g in X])
        z = F.min(axis=0).copy()
        def tch(f, w):
            return float(np.max(w * np.abs(f - z) + 1e-12 * 0))
        while not R.agotado:
            for i in range(K):
                vecinos = vec[i]
                a, b = rng.choice(vecinos, 2, replace=False)
                h1, _ = cruce_uniforme(X[a], X[b], rng)
                h = mutar(h1, rng, pm)
                fh = R(h)
                z = np.minimum(z, fh)
                for j in vecinos[rng.permutation(T)][:nr]:
                    if tch(fh, W[j]) <= tch(F[j], W[j]):
                        X[j] = h; F[j] = fh
    except StopIteration:
        pass
    return R


def aleatoria(evaluar, presupuesto, semilla):
    rng = np.random.default_rng(semilla)
    R = Registro(evaluar, presupuesto)
    try:
        while not R.agotado:
            R(muestra_aleatoria(rng))
    except StopIteration:
        pass
    return R


ALGORITMOS = {"NSGA-II": nsga2, "MOEA/D": moead, "Aleatoria": aleatoria}


# ----------------------------------------------------------------------------- normalización y métricas
def normalizar(F, minimo, maximo):
    return (np.asarray(F, float) - minimo) / (maximo - minimo)


def metricas(objs_evaluadas, frente_exacto_norm, minimo, maximo, ref=1.1):
    """HV y IGD sobre el conjunto de soluciones evaluadas (normalizadas con el espacio completo)."""
    N = normalizar(objs_evaluadas, minimo, maximo)
    no_dom = N[frente_no_dominado(N)]
    hv = hipervolumen(no_dom, np.full(3, ref))
    d = igd(no_dom, frente_exacto_norm)
    return hv, d
