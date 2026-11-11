"""Pruebas: modelo de torneado, regla de factibilidad, evolución diferencial y referencia."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modelo_mecanizado import (PAR, costo, restricciones, violacion, rugosidad, potencia, vida_herramienta,  # noqa
                               tiempo_maquinado, referencia, perturbar, cotas)
from evolucion_diferencial import regla_factibilidad, de_clasico, jade, _corrige

_REF = None


def ref():
    global _REF
    if _REF is None:
        _REF = referencia(malla=40, arranques=12)
    return _REF


def test_01_ra_y_potencia_a_mano():
    x = np.array([150.0, 0.2, 2.0])
    assert abs(rugosidad(x) - 1.5625) < 1e-9                 # 1000·0.04 / (32·0.8)
    assert abs(potencia(x) - 2.0) < 1e-9                     # 2000·2·0.2·150 / 60000
    assert abs(vida_herramienta(x) - 30.0) < 1e-6            # calibración de C_v


def test_02_tiempo_de_maquinado_a_mano():
    x = np.array([150.0, 0.2, 2.0])
    esperado = (4.0 / 2.0) * np.pi * 80 * 200 / (1000 * 150 * 0.2)
    assert abs(tiempo_maquinado(x) - esperado) < 1e-12


def test_03_costo_incluye_herramienta():
    x = np.array([150.0, 0.2, 2.0])
    assert costo(x) > PAR.M * tiempo_maquinado(x)


def test_04_restricciones_signo():
    assert violacion(np.array([60.0, 0.05, 0.5])) == 0.0     # punto conservador: factible
    assert violacion(np.array([350.0, 0.4, 3.0])) > 0        # punto agresivo: infactible


def test_05_referencia_factible_y_en_frontera():
    r = ref()
    assert violacion(r["x"]) <= 1e-9
    assert abs(restricciones(r["x"])[0]) < 1e-6              # rugosidad activa
    assert abs(r["x"][2] - 3.0) < 1e-9                       # profundidad en el límite del inserto
    assert r["costo"] <= r["malla_mejor"] + 1e-9


def test_06_perturbacion_cambia_parametros():
    q = perturbar(PAR, np.random.default_rng(0))
    assert q.C_v != PAR.C_v and q.k_c != PAR.k_c


def test_07_regla_factibilidad():
    assert regla_factibilidad(0.0, 3.0, 0.0, 4.0)
    assert regla_factibilidad(0.0, 9.0, 0.5, 1.0)            # factible gana a infactible
    assert regla_factibilidad(0.2, 9.0, 0.5, 1.0)            # entre infactibles, menor violación


def test_08_correccion_de_limites():
    lo, hi = np.array([0.0]), np.array([10.0])
    assert _corrige(np.array([12.0]), np.array([4.0]), lo, hi)[0] == 7.0
    assert _corrige(np.array([-2.0]), np.array([4.0]), lo, hi)[0] == 2.0


@pytest.mark.parametrize("alg", [de_clasico, jade])
def test_09_presupuesto_factibilidad_reproducibilidad(alg):
    a = alg(5, 400); b = alg(5, 400)
    assert a.evaluaciones == 400 and a.mejor_violacion <= 1e-12
    assert a.mejor_costo == b.mejor_costo
    lo, hi = cotas()
    assert np.all(a.mejor_x >= lo - 1e-12) and np.all(a.mejor_x <= hi + 1e-12)
    serie = [c for _, c in a.historial if not np.isnan(c)]
    assert all(y <= x + 1e-12 for x, y in zip(serie, serie[1:]))
