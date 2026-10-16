#!/usr/bin/env python3
"""
Evolución Diferencial: Implementación y Ejemplos Prácticos
===========================================================
Alumna: Jessica Melani Romero Lora
Materia: Algoritmos Evolutivos
Profesor: Dr. Jaime Aguilar Ortiz
Universidad Politécnica Metropolitana del Estado de Hidalgo
Maestría en Inteligencia Artificial

Este archivo contiene:
  1. Implementación genérica de Evolución Diferencial con múltiples variantes
  2. Ejemplo 1: Optimización de funciones benchmark (Rosenbrock, Rastrigin, Ackley)
  3. Ejemplo 2: Diseño óptimo de un resorte de tensión/compresión
  4. Ejemplo 3: Entrenamiento de una red neuronal simple (clasificación)
  5. Ejemplo 4: Ajuste de parámetros de un modelo PID

Ejecución:
  python3 evolucion_diferencial.py
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
import time


# ============================================================================
# 1. IMPLEMENTACIÓN GENÉRICA DE EVOLUCIÓN DIFERENCIAL
# ============================================================================

class EvolucionDiferencial:
    """
    Implementación de Evolución Diferencial con soporte para
    múltiples estrategias de mutación.

    Parámetros
    ----------
    func : callable
        Función objetivo a minimizar. Recibe un vector np.ndarray.
    bounds : list of tuples
        Límites (min, max) para cada dimensión.
    NP : int
        Tamaño de la población (por defecto 10 * dimensión).
    F : float
        Factor de escala de mutación, rango típico [0.4, 1.0].
    CR : float
        Probabilidad de cruzamiento, rango típico [0.1, 0.9].
    strategy : str
        Estrategia de mutación: 'rand/1', 'best/1', 'rand/2',
        'current-to-best/1'.
    max_gen : int
        Número máximo de generaciones.
    tol : float
        Tolerancia para criterio de convergencia.
    seed : int or None
        Semilla para reproducibilidad.
    """

    STRATEGIES = ['rand/1', 'best/1', 'rand/2', 'current-to-best/1']

    def __init__(self, func: Callable, bounds: List[Tuple[float, float]],
                 NP: Optional[int] = None, F: float = 0.8, CR: float = 0.9,
                 strategy: str = 'rand/1', max_gen: int = 500,
                 tol: float = 1e-8, seed: Optional[int] = None):
        self.func = func
        self.bounds = np.array(bounds)
        self.D = len(bounds)
        self.NP = NP if NP else 10 * self.D
        self.F = F
        self.CR = CR
        self.max_gen = max_gen
        self.tol = tol
        self.rng = np.random.default_rng(seed)

        if strategy not in self.STRATEGIES:
            raise ValueError(f"Estrategia '{strategy}' no soportada. "
                             f"Use: {self.STRATEGIES}")
        self.strategy = strategy

        # Historial para análisis posterior
        self.history: List[float] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = np.inf
        self.evaluations: int = 0

    def _init_population(self) -> np.ndarray:
        """Genera población inicial con distribución uniforme en los límites."""
        pop = self.rng.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.NP, self.D)
        )
        return pop

    def _mutate(self, pop: np.ndarray, idx: int,
                best_idx: int) -> np.ndarray:
        """Aplica la estrategia de mutación seleccionada."""
        idxs = [i for i in range(self.NP) if i != idx]

        if self.strategy == 'rand/1':
            r1, r2, r3 = self.rng.choice(idxs, 3, replace=False)
            v = pop[r1] + self.F * (pop[r2] - pop[r3])

        elif self.strategy == 'best/1':
            r1, r2 = self.rng.choice(idxs, 2, replace=False)
            v = pop[best_idx] + self.F * (pop[r1] - pop[r2])

        elif self.strategy == 'rand/2':
            r1, r2, r3, r4, r5 = self.rng.choice(idxs, 5, replace=False)
            v = pop[r1] + self.F * (pop[r2] - pop[r3]) + \
                self.F * (pop[r4] - pop[r5])

        elif self.strategy == 'current-to-best/1':
            r1, r2 = self.rng.choice(idxs, 2, replace=False)
            v = pop[idx] + self.F * (pop[best_idx] - pop[idx]) + \
                self.F * (pop[r1] - pop[r2])

        return v

    def _crossover(self, target: np.ndarray,
                   donor: np.ndarray) -> np.ndarray:
        """Cruzamiento binomial."""
        trial = target.copy()
        j_rand = self.rng.integers(0, self.D)
        for j in range(self.D):
            if self.rng.random() < self.CR or j == j_rand:
                trial[j] = donor[j]
        return trial

    def _repair(self, vector: np.ndarray) -> np.ndarray:
        """Repara componentes fuera de los límites (bounce-back)."""
        for j in range(self.D):
            lo, hi = self.bounds[j]
            if vector[j] < lo or vector[j] > hi:
                vector[j] = lo + self.rng.random() * (hi - lo)
        return vector

    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Ejecuta el algoritmo de evolución diferencial.

        Retorna
        -------
        best_solution : np.ndarray
            Mejor solución encontrada.
        best_fitness : float
            Valor de la función objetivo en la mejor solución.
        """
        pop = self._init_population()
        fitness = np.array([self.func(ind) for ind in pop])
        self.evaluations = self.NP

        best_idx = np.argmin(fitness)
        self.best_solution = pop[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.history = [self.best_fitness]

        for gen in range(1, self.max_gen + 1):
            for i in range(self.NP):
                # Mutación
                donor = self._mutate(pop, i, best_idx)
                donor = self._repair(donor)

                # Cruzamiento
                trial = self._crossover(pop[i], donor)

                # Selección greedy
                trial_fit = self.func(trial)
                self.evaluations += 1

                if trial_fit <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit

                    if trial_fit < self.best_fitness:
                        self.best_fitness = trial_fit
                        self.best_solution = trial.copy()
                        best_idx = i

            self.history.append(self.best_fitness)

            if verbose and gen % 50 == 0:
                print(f"  Gen {gen:4d} | Mejor: {self.best_fitness:.6e} "
                      f"| Media: {np.mean(fitness):.6e}")

            # Criterio de convergencia
            if len(self.history) > 20:
                delta = abs(self.history[-20] - self.history[-1])
                if delta < self.tol:
                    if verbose:
                        print(f"  Convergencia alcanzada en gen {gen}")
                    break

        return self.best_solution, self.best_fitness


# ============================================================================
# 2. EJEMPLO 1: FUNCIONES BENCHMARK
# ============================================================================

def rosenbrock(x: np.ndarray) -> float:
    """Función de Rosenbrock (mínimo global en x = [1, 1, ..., 1])."""
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
               for i in range(len(x) - 1))


def rastrigin(x: np.ndarray) -> float:
    """Función de Rastrigin (mínimo global en x = [0, 0, ..., 0])."""
    n = len(x)
    return 10 * n + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)


def ackley(x: np.ndarray) -> float:
    """Función de Ackley (mínimo global en x = [0, 0, ..., 0])."""
    n = len(x)
    s1 = -0.2 * np.sqrt(np.sum(x**2) / n)
    s2 = np.sum(np.cos(2 * np.pi * x)) / n
    return -20 * np.exp(s1) - np.exp(s2) + 20 + np.e


def ejemplo_benchmark():
    """Optimiza funciones benchmark con diferentes estrategias DE."""
    print("=" * 70)
    print("EJEMPLO 1: Optimización de Funciones Benchmark")
    print("=" * 70)

    benchmarks = [
        ("Rosenbrock", rosenbrock, [(-5, 10)] * 10, "x* = [1,...,1], f* = 0"),
        ("Rastrigin",  rastrigin,  [(-5.12, 5.12)] * 10, "x* = [0,...,0], f* = 0"),
        ("Ackley",     ackley,     [(-5, 5)] * 10, "x* = [0,...,0], f* = 0"),
    ]

    strategies = ['rand/1', 'best/1', 'current-to-best/1']

    for name, func, bounds, opt_str in benchmarks:
        print(f"\n{'─' * 60}")
        print(f"  {name} (D=10) — Óptimo conocido: {opt_str}")
        print(f"{'─' * 60}")

        for strat in strategies:
            de = EvolucionDiferencial(
                func=func, bounds=bounds, NP=50, F=0.8, CR=0.9,
                strategy=strat, max_gen=1000, seed=42
            )
            sol, fit = de.optimize()
            print(f"  DE/{strat:20s} → f(x*) = {fit:12.6e}  "
                  f"({de.evaluations:,} evaluaciones)")


# ============================================================================
# 3. EJEMPLO 2: DISEÑO ÓPTIMO DE RESORTE
# ============================================================================

def spring_design(x: np.ndarray) -> float:
    """
    Problema de diseño de resorte de tensión/compresión.
    Variables: x = [d, D, N] (diámetro del alambre, diámetro medio, número de espiras)
    Objetivo: minimizar el peso del resorte.
    Restricciones manejadas con penalización exterior.
    """
    d, D, N = x[0], x[1], x[2]

    # Función objetivo: peso
    f = (N + 2) * D * d**2

    # Restricciones (g <= 0)
    g1 = 1 - (D**3 * N) / (71785 * d**4)
    g2 = (4 * D**2 - d * D) / (12566 * (D * d**3 - d**4)) + 1 / (5108 * d**2) - 1
    g3 = 1 - 140.45 * d / (D**2 * N)
    g4 = (d + D) / 1.5 - 1

    # Penalización cuadrática
    penalty = 1e6 * (max(0, g1)**2 + max(0, g2)**2 +
                     max(0, g3)**2 + max(0, g4)**2)

    return f + penalty


def ejemplo_resorte():
    """Optimiza el diseño de un resorte con DE."""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Diseño Óptimo de Resorte de Tensión/Compresión")
    print("=" * 70)

    bounds = [(0.05, 2.0), (0.25, 1.3), (2.0, 15.0)]
    print("  Variables: d (diám. alambre), D (diám. medio), N (espiras)")
    print(f"  Límites: d∈[0.05,2.0], D∈[0.25,1.3], N∈[2,15]")
    print(f"  Referencia: f* ≈ 0.012665 (literatura)")

    results = []
    for strat in ['rand/1', 'best/1', 'current-to-best/1']:
        fits = []
        for trial in range(30):
            de = EvolucionDiferencial(
                func=spring_design, bounds=bounds,
                NP=30, F=0.7, CR=0.9, strategy=strat,
                max_gen=500, seed=trial
            )
            _, fit = de.optimize()
            fits.append(fit)

        fits = np.array(fits)
        results.append((strat, fits))
        print(f"\n  DE/{strat}:")
        print(f"    Mejor:     {fits.min():.6f}")
        print(f"    Media:     {fits.mean():.6f}")
        print(f"    Desv. Est: {fits.std():.6f}")
        print(f"    Peor:      {fits.max():.6f}")

    # Mostrar mejor solución encontrada
    de = EvolucionDiferencial(
        func=spring_design, bounds=bounds,
        NP=50, F=0.7, CR=0.9, strategy='rand/1',
        max_gen=1000, seed=42
    )
    sol, fit = de.optimize()
    print(f"\n  Mejor solución global encontrada:")
    print(f"    d = {sol[0]:.6f}, D = {sol[1]:.6f}, N = {sol[2]:.6f}")
    print(f"    Peso = {fit:.6f}")


# ============================================================================
# 4. EJEMPLO 3: ENTRENAMIENTO DE RED NEURONAL SIMPLE
# ============================================================================

def ejemplo_red_neuronal():
    """
    Usa DE para entrenar una red neuronal simple (1 capa oculta)
    en un problema de clasificación XOR extendido.
    """
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Entrenamiento de Red Neuronal con DE")
    print("=" * 70)

    # Datos XOR extendido (con ruido)
    rng = np.random.default_rng(42)
    n_samples = 200
    X = rng.uniform(-1, 1, (n_samples, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(float)
    # Añadir ruido al 5% de las etiquetas
    flip = rng.choice(n_samples, n_samples // 20, replace=False)
    y[flip] = 1 - y[flip]

    # Arquitectura: 2 → 5 → 1
    n_in, n_hidden, n_out = 2, 5, 1
    # Total de pesos: (2*5 + 5) + (5*1 + 1) = 21
    n_weights = (n_in * n_hidden + n_hidden) + (n_hidden * n_out + n_out)

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def nn_loss(w: np.ndarray) -> float:
        """Error cuadrático medio de la red neuronal."""
        idx = 0
        # Capa 1
        W1 = w[idx:idx + n_in * n_hidden].reshape(n_in, n_hidden)
        idx += n_in * n_hidden
        b1 = w[idx:idx + n_hidden]
        idx += n_hidden
        # Capa 2
        W2 = w[idx:idx + n_hidden * n_out].reshape(n_hidden, n_out)
        idx += n_hidden * n_out
        b2 = w[idx:idx + n_out]

        # Forward pass
        h = sigmoid(X @ W1 + b1)
        out = sigmoid(h @ W2 + b2).flatten()

        # MSE
        return np.mean((out - y) ** 2)

    bounds = [(-5.0, 5.0)] * n_weights

    print(f"  Arquitectura: {n_in} → {n_hidden} → {n_out}")
    print(f"  Pesos totales: {n_weights}")
    print(f"  Muestras: {n_samples} (XOR extendido con 5% ruido)")

    de = EvolucionDiferencial(
        func=nn_loss, bounds=bounds, NP=80,
        F=0.6, CR=0.9, strategy='current-to-best/1',
        max_gen=300, seed=42
    )
    sol, fit = de.optimize(verbose=True)

    # Evaluar precisión final
    idx = 0
    W1 = sol[:n_in * n_hidden].reshape(n_in, n_hidden)
    idx = n_in * n_hidden
    b1 = sol[idx:idx + n_hidden]
    idx += n_hidden
    W2 = sol[idx:idx + n_hidden * n_out].reshape(n_hidden, n_out)
    idx += n_hidden * n_out
    b2 = sol[idx:idx + n_out]

    h = sigmoid(X @ W1 + b1)
    pred = (sigmoid(h @ W2 + b2).flatten() > 0.5).astype(float)
    accuracy = np.mean(pred == y) * 100

    print(f"\n  MSE final:  {fit:.6f}")
    print(f"  Precisión:  {accuracy:.1f}%")
    print(f"  Evaluaciones: {de.evaluations:,}")


# ============================================================================
# 5. EJEMPLO 4: AJUSTE DE CONTROLADOR PID
# ============================================================================

def ejemplo_pid():
    """
    Usa DE para ajustar los parámetros Kp, Ki, Kd de un controlador PID
    simulado en un sistema de segundo orden.
    """
    print("\n" + "=" * 70)
    print("EJEMPLO 4: Ajuste de Controlador PID con DE")
    print("=" * 70)

    def simulate_pid(params: np.ndarray) -> float:
        """
        Simula un controlador PID en un sistema de segundo orden.
        Planta: G(s) = 1 / (s^2 + 2*zeta*wn*s + wn^2)
        Discretizado con Euler, dt = 0.01.

        Retorna el Integral del Error Absoluto por Tiempo (ITAE).
        """
        Kp, Ki, Kd = params
        dt = 0.01
        t_final = 5.0
        steps = int(t_final / dt)

        # Parámetros de la planta (segundo orden subamortiguado)
        wn = 5.0   # frecuencia natural
        zeta = 0.3  # factor de amortiguamiento

        # Estado de la planta [y, dy/dt]
        state = np.array([0.0, 0.0])
        setpoint = 1.0

        integral_error = 0.0
        prev_error = setpoint
        itae = 0.0  # Integral Time Absolute Error

        for k in range(steps):
            t = k * dt
            error = setpoint - state[0]

            integral_error += error * dt
            derivative = (error - prev_error) / dt

            u = Kp * error + Ki * integral_error + Kd * derivative
            u = np.clip(u, -100, 100)  # Saturación del actuador

            # Ecuación de la planta: d²y/dt² = wn²(u - y) - 2*zeta*wn*dy/dt
            ddx = wn**2 * (u - state[0]) - 2 * zeta * wn * state[1]
            state[1] += ddx * dt
            state[0] += state[1] * dt

            prev_error = error
            itae += t * abs(error) * dt

        # Penalizar sobreimpulso excesivo
        return itae

    bounds = [(0.1, 50.0), (0.0, 30.0), (0.0, 10.0)]
    print("  Planta: G(s) = 1/(s² + 3s + 25)")
    print("  Variables: Kp ∈ [0.1, 50], Ki ∈ [0, 30], Kd ∈ [0, 10]")
    print("  Criterio: minimizar ITAE (Integral Time Absolute Error)")

    de = EvolucionDiferencial(
        func=simulate_pid, bounds=bounds,
        NP=40, F=0.7, CR=0.8, strategy='rand/1',
        max_gen=200, seed=42
    )
    sol, fit = de.optimize(verbose=True)

    print(f"\n  Parámetros óptimos:")
    print(f"    Kp = {sol[0]:.4f}")
    print(f"    Ki = {sol[1]:.4f}")
    print(f"    Kd = {sol[2]:.4f}")
    print(f"  ITAE = {fit:.6f}")
    print(f"  Evaluaciones: {de.evaluations:,}")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == '__main__':
    t0 = time.time()

    ejemplo_benchmark()
    ejemplo_resorte()
    ejemplo_red_neuronal()
    ejemplo_pid()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Tiempo total de ejecución: {elapsed:.1f} segundos")
    print("=" * 70)
