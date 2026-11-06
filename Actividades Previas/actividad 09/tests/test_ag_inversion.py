"""Pruebas de ag_inversion: modelo, no concavidad, restricciones, operadores y bucle genético."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ag_inversion import (R, M, PRESUPUESTO, IDX_VM, generar_cartera, valor, violacion, violaciones,  # noqa: E402
                          resolver_referencia, reparto_proporcional, algoritmo_genetico, busqueda_aleatoria,
                          evolucion_diferencial, regla_factibilidad)
from ag_precios import TOL_FACTIBLE, etapa_1_inicializar_M03  # noqa: E402


@pytest.fixture(scope="module")
def c():
    return generar_cartera("T-base")


def test_01_dimensiones(c):
    assert c.d == R * M == 20 and c.piso.shape == (20,) and np.all(c.techo > 0)


def test_02_valor_creciente_en_cada_gen(c):
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = c.piso + rng.random(c.d) * (c.techo - c.piso)
        j = rng.integers(c.d)
        y = x.copy(); y[j] = min(c.techo[j], x[j] + 100.0)
        assert valor(c, y)[0] >= valor(c, x)[0] - 1e-9


def test_03_no_concava_por_umbral(c):
    """Con un solo gen: f(θ) debe quedar por debajo del promedio de f(0) y f(2θ) (no concavidad)."""
    j = 0
    def f(t):
        x = np.zeros(c.d); x[j] = t
        return valor(c, x)[0]
    th = c.theta.ravel()[j]
    assert f(th) < 0.5 * (f(0.0) + f(2 * th))


def test_04_presupuesto_y_equidad(c):
    x = np.zeros(c.d); x[:] = c.techo * 0.9
    v = violaciones(c, x)
    assert v["presupuesto"][0] > 0 and v["tope_region"][0] > 0
    y = c.techo * (PRESUPUESTO / c.techo.sum())          # reparto proporcional
    y = y.reshape(R, M); y[IDX_VM] = 0.0; y = y.ravel()
    assert violaciones(c, y)["equidad_vm"][0] > 0


def test_05_agua_limita(c):
    x = np.zeros((R, M)); x[IDX_VM, 2] = 5000.0          # agroindustria en el Mezquital
    assert violaciones(c, x.ravel())["agua"][0] > 0


def test_06_reparto_factible(c):
    rp = reparto_proporcional(c)
    assert rp["violacion"] <= TOL_FACTIBLE and rp["valor"] > 0


def test_07_m03_en_la_caja(c):
    P = etapa_1_inicializar_M03(100, c, np.random.default_rng(3))
    assert np.all(P >= c.piso) and np.all(P <= c.techo)


def test_08_referencia_factible_y_no_menor_que_reparto(c):
    ref = resolver_referencia(c, n_arranques=12, semilla=2)
    assert ref["violacion"] <= 1e-9
    assert ref["valor"] >= reparto_proporcional(c)["valor"] - 1e-6


def test_09_ga_contador_reproducible_y_factible(c):
    a = algoritmo_genetico(c, "M16", "M20", 5, {"alfa": 0.3, "eta": 20}, E_max=3000)
    b = algoritmo_genetico(c, "M16", "M20", 5, {"alfa": 0.3, "eta": 20}, E_max=3000)
    assert a["evaluaciones"] == 100 + a["generaciones"] * 98 and a["evaluaciones"] <= 3000
    assert np.array_equal(a["x"], b["x"]) and a["valor"] == b["valor"]
    serie = [h["mejor_factible"] for h in a["historial"] if not np.isnan(h["mejor_factible"])]
    assert all(y >= x for x, y in zip(serie, serie[1:]))


def test_10_busqueda_aleatoria_presupuesto(c):
    r = busqueda_aleatoria(c, 4, 3000)
    assert r["evaluaciones"] == 100 + 29 * 98


def test_11_regla_de_factibilidad():
    assert regla_factibilidad(0.0, 10.0, 0.0, 9.0)            # ambas factibles: gana la de mayor valor
    assert regla_factibilidad(0.0, 1.0, 0.5, 100.0)           # factible gana a infactible
    assert regla_factibilidad(0.1, 0.0, 0.5, 100.0)           # entre infactibles, menor violación
    assert not regla_factibilidad(0.5, 100.0, 0.0, 1.0)


def test_12_de_factible_presupuesto_y_reproducible(c):
    a = evolucion_diferencial(c, 7, E_max=4000)
    b = evolucion_diferencial(c, 7, E_max=4000)
    assert a["evaluaciones"] == 4000 and a["x"] is not None
    assert violacion(c, a["x"])[0] <= TOL_FACTIBLE
    assert a["valor"] == b["valor"] and np.array_equal(a["x"], b["x"])
    serie = [h["mejor_factible"] for h in a["historial"] if not np.isnan(h["mejor_factible"])]
    assert all(y >= x for x, y in zip(serie, serie[1:]))


@pytest.mark.parametrize("mut", ["rand1", "best1", "ctb1"])
@pytest.mark.parametrize("cru", ["bin", "exp", "arit"])
def test_13_variantes_de_factibles(c, mut, cru):
    r = evolucion_diferencial(c, 11, E_max=2500, mutacion=mut, cruce=cru)
    assert r["evaluaciones"] == 2500 and r["x"] is not None
    assert violacion(c, r["x"])[0] <= TOL_FACTIBLE
    assert np.all(r["x"] >= c.piso) and np.all(r["x"] <= c.techo)


def test_14_cruce_exponencial_copia_segmento():
    from ag_inversion import _cruce
    rng = np.random.default_rng(0)
    x = np.zeros(10); v = np.ones(10)
    u = _cruce(x, v, 0.5, rng, "exp")
    assert set(np.unique(u)) <= {0.0, 1.0} and u.sum() >= 1
    # segmento contiguo cíclico: los unos forman un bloque (con envoltura)
    idx = np.flatnonzero(u)
    assert len(idx) == 10 or np.all(np.diff(idx) == 1) or (idx[0] == 0 and idx[-1] == 9)
