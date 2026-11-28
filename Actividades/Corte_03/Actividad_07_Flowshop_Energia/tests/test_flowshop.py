"""Pruebas: modelo flow shop, operadores de permutación, ordenamiento no dominado, métricas y reproducibilidad."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowshop_modelo import generar, evaluar, frente_no_dominado, frente_exacto, Instancia  # noqa: E402
from flowshop_moea import ox, pmx, nds, hipervolumen, igd, nsga2, aleatoria  # noqa: E402


def energia_manual(inst, perm):
    n, m = inst.n, inst.m
    C = np.zeros((n, m)); S = np.zeros((n, m))
    for j, job in enumerate(perm):
        for i in range(m):
            ini = max(C[j - 1, i] if j > 0 else 0.0, C[j, i - 1] if i > 0 else 0.0)
            S[j, i] = ini; C[j, i] = ini + inst.p[job, i]
    E = 0.0
    for i in range(m):
        E += inst.P_act[i] * sum(inst.p[job, i] for job in perm) / 60.0
        E += inst.E_arr[i]
        for j in range(1, n):
            E += min(inst.P_idle[i] * (S[j, i] - C[j - 1, i]) / 60.0, inst.E_arr[i])
    return E


@pytest.fixture(scope="module")
def inst():
    return generar("T", 9, 20262900)


def test_01_cmax_a_mano(inst):
    C = np.zeros((9, 4))
    for j in range(9):
        for i in range(4):
            C[j, i] = max(C[j - 1, i] if j > 0 else 0, C[j, i - 1] if i > 0 else 0) + inst.p[j, i]
    assert abs(evaluar(inst, np.arange(9))[0, 0] - C[-1, -1]) < 1e-9


def test_02_energia_igual_a_calculo_independiente(inst):
    rng = np.random.default_rng(7)
    for _ in range(20):
        perm = rng.permutation(9)
        assert abs(evaluar(inst, perm)[0, 2] - energia_manual(inst, perm)) < 1e-9


def test_03_tardanza_no_negativa(inst):
    rng = np.random.default_rng(8)
    F = evaluar(inst, np.array([rng.permutation(9) for _ in range(200)]))
    assert np.all(F[:, 1] >= 0)


def test_04_operadores_producen_permutaciones(inst):
    rng = np.random.default_rng(9)
    for _ in range(500):
        a, b = rng.permutation(9), rng.permutation(9)
        assert sorted(ox(a, b, rng)) == list(range(9))
        assert sorted(pmx(a, b, rng)) == list(range(9))


def test_05_nds_coincide_con_fuerza_bruta():
    dom = lambda a, b: np.all(a <= b) and np.any(a < b)
    for seed in range(4):
        F = np.random.default_rng(seed).integers(0, 6, (60, 3)).astype(float)
        fr = nds(F); rest = set(range(60)); k = 0
        while rest:
            actual = [i for i in rest if not any(dom(F[j], F[i]) for j in rest if j != i)]
            assert sorted(fr[k]) == sorted(actual)
            rest -= set(actual); k += 1
        assert k == len(fr)


def test_06_frente_no_dominado_sin_duplicados():
    F = np.array([[1, 2, 3], [1, 2, 3], [2, 1, 3], [3, 3, 3]], float)
    idx = frente_no_dominado(F)
    assert sorted(map(tuple, F[idx])) == [(1, 2, 3), (2, 1, 3)]


def test_07_hipervolumen_y_igd():
    assert abs(hipervolumen(np.array([[0.5, 0.5, 0.5]]), np.array([1.0, 1.0, 1.0])) - 0.125) < 1e-12
    R = np.array([[0, 1, 0.5], [1, 0, 0.5]], float)
    assert igd(np.vstack([R, [[0.9, 0.9, 0.9]]]), R) == 0.0


def test_08_presupuesto_y_reproducibilidad(inst):
    a = nsga2(inst, 900, 3, cruce="PMX"); b = nsga2(inst, 900, 3, cruce="PMX")
    assert len(a.perms) == 900 and a.perms == b.perms
    c = aleatoria(inst, 500, 3)
    assert len(c.perms) == 500


def test_09_frente_exacto_pequeno_es_consistente():
    pequena = generar("P", 5, 1)
    fr, X, F = frente_exacto(pequena)
    assert len(F) == 120 and len(fr) >= 1
    assert np.all(F[:, 1] >= 0)
