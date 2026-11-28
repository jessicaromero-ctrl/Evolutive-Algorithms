"""Experimento factorial 3 × 3 de evolución diferencial con reglas de factibilidad.

Tres estrategias de mutación (rand/1, best/1, current-to-best/1) × tres cruces (binomial, exponencial,
aritmética). Cada configuración calibra sus parámetros (F, y CR cuando aplica) en C-calib.

Uso:  python experimento.py      (≈ 35 min en un núcleo; escribe resultados/*.csv y manifiesto.json)
"""
import hashlib, json, platform, time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import friedmanchisquare, wilcoxon

from ag_precios import (comparar, matriz_bloques, ranking_demsar, ic_bootstrap_mediana, vargha_delaney, holm)
from ag_inversion import (generar_cartera, resolver_referencia, reparto_proporcional, busqueda_aleatoria,
                          algoritmo_genetico, evolucion_diferencial, resumen_por_sector,
                          MUTACIONES_DE, CRUCES_DE, NOMBRE_MUT, NOMBRE_CRUCE)

INICIO = time.perf_counter()
RES = Path("resultados"); RES.mkdir(exist_ok=True)
E_MAX = 20_000
EVALS_BA = 100 + ((E_MAX - 100) // 98) * 98
SEM_CAL = list(range(20262500, 20262506))
SEM_EVAL = list(range(20262517, 20262547))
ARRANQUES_REF = 200
GRID_F = (0.4, 0.6, 0.8)
GRID_CR = (0.5, 0.9)
PARAM_AG_PREVIO = {"alfa": 0.3, "eta": 20}
CONFIGS = [(m, c) for m in MUTACIONES_DE for c in CRUCES_DE]
ORDEN = [f"{m}+{c}" for m, c in CONFIGS]
EVAL = ["E-base", "E-escasez", "E-amplia"]

INSTANCIAS = {
    "C-calib": generar_cartera("C-calib", escala_agua=0.85),
    "E-base": generar_cartera("E-base"),
    "E-escasez": generar_cartera("E-escasez", escala_agua=0.7, empleo_min=13_000),
    "E-amplia": generar_cartera("E-amplia", presupuesto=28_000, escala_U=1.3),
}

# ---------------------------------------------------------------- referencias y líneas base
REF, PARE, filas_ref = {}, {}, []
for n, c in INSTANCIAS.items():
    REF[n] = resolver_referencia(c, n_arranques=ARRANQUES_REF, semilla=20262600)
    PARE[n] = reparto_proporcional(c)
    filas_ref.append({"instancia": n, "presupuesto": c.presupuesto, "empleo_min": c.empleo_min,
                      "valor_reparto": PARE[n]["valor"], "referencia": REF[n]["valor"],
                      "arranques_factibles": REF[n]["factibles"], "arranques_en_el_mejor": REF[n]["coinciden"],
                      "brecha_reparto_pct": 100 * (REF[n]["valor"] - PARE[n]["valor"]) / REF[n]["valor"]})
pd.DataFrame(filas_ref).to_csv(RES / "referencias.csv", index=False)
print(pd.DataFrame(filas_ref).round(2).to_string(index=False), flush=True)

# ---------------------------------------------------------------- calibración por configuración (C-calib)
filas_cal = []
for m, cr_ in CONFIGS:
    cruce_cr = GRID_CR if cr_ != "arit" else (None,)          # CR no interviene en el cruce aritmético
    for F, CR in product(GRID_F, cruce_cr):
        for s in SEM_CAL:
            r = evolucion_diferencial(INSTANCIAS["C-calib"], s, E_max=E_MAX, F=F,
                                      CR=CR if CR is not None else 0.5, mutacion=m, cruce=cr_)
            filas_cal.append({"configuracion": f"{m}+{cr_}", "F": F, "CR": CR if CR is not None else np.nan,
                              "semilla": s, "brecha_pct": 100 * (REF["C-calib"]["valor"] - r["valor"]) / REF["C-calib"]["valor"]})
    print("calibrada", m, cr_, flush=True)
cal = pd.DataFrame(filas_cal); cal.to_csv(RES / "calibracion.csv", index=False)
agg = cal.groupby(["configuracion", "F", "CR"], dropna=False)["brecha_pct"].agg(mediana="median", media="mean").reset_index()
PARAM = {}
for cfg, g in agg.groupby("configuracion"):
    o = g.sort_values(["mediana", "media"]).iloc[0]
    PARAM[cfg] = {"F": float(o["F"]), "CR": None if pd.isna(o["CR"]) else float(o["CR"]),
                  "brecha_calibracion": float(o["mediana"])}
pd.DataFrame([{"configuracion": k, **v} for k, v in PARAM.items()]).to_csv(RES / "calibracion_elegidos.csv", index=False)
print("\nParámetros elegidos:", json.dumps(PARAM, indent=1), flush=True)

# ---------------------------------------------------------------- experimento principal
crudas, filas_ba, conv = {}, [], []
for n in EVAL:
    c = INSTANCIAS[n]
    for s in SEM_EVAL:
        for m, cr_ in CONFIGS:
            p = PARAM[f"{m}+{cr_}"]
            r = evolucion_diferencial(c, s, E_max=E_MAX, F=p["F"], CR=p["CR"] if p["CR"] is not None else 0.5,
                                      mutacion=m, cruce=cr_)
            crudas[("DE", n, f"{m}+{cr_}", s)] = r
            if n == "E-base":
                for h in r["historial"]:
                    conv.append({"configuracion": f"{m}+{cr_}", "semilla": s, **h})
        crudas[("AG", n, "AG previo M16+M20", s)] = algoritmo_genetico(
            c, "M16", "M20", s, PARAM_AG_PREVIO, E_max=E_MAX, historial=False)
        b = busqueda_aleatoria(c, s, EVALS_BA)
        filas_ba.append({"instancia": n, "semilla": s, "valor": b["valor"], "factible": b["factible"],
                         "evaluaciones": b["evaluaciones"]})
    print("evaluada", n, flush=True)
ba = pd.DataFrame(filas_ba)
pd.DataFrame(conv).to_csv(RES / "convergencia_E-base.csv", index=False, float_format="%.6g")

ZREF = {}
for n in EVAL:
    cand = [REF[n]["valor"], PARE[n]["valor"]]
    cand += [r["valor"] for (alg, nn, cfg, s), r in crudas.items() if nn == n and not np.isnan(r["valor"])]
    cand += ba.loc[(ba.instancia == n) & ba.factible, "valor"].tolist()
    ZREF[n] = max(cand)

filas = []
for (alg, n, cfg, s), r in crudas.items():
    zs = ZREF[n]
    filas.append({"algoritmo": alg, "instancia": n, "configuracion": cfg, "semilla": s,
                  "mutacion": cfg.split("+")[0] if alg == "DE" else "AG previo",
                  "cruce": cfg.split("+")[1] if alg == "DE" else "M16", "valor": r["valor"],
                  "brecha_pct": 100 * (zs - r["valor"]) / zs,
                  "captura_pct": 100 * (r["valor"] - PARE[n]["valor"]) / (zs - PARE[n]["valor"]),
                  "fraccion_factible_final": r["fraccion_factible"], "segundos": r["segundos"]})
corridas = pd.DataFrame(filas)
corridas.to_csv(RES / "corridas.csv", index=False)
ba["brecha_pct"] = np.where(ba.factible, 100 * (ba.instancia.map(ZREF) - ba.valor) / ba.instancia.map(ZREF), np.nan)
ba.to_csv(RES / "busqueda_aleatoria.csv", index=False)
pd.DataFrame([{"instancia": n, "zref": ZREF[n], "slsqp": REF[n]["valor"], "reparto": PARE[n]["valor"],
               "mejor_de": corridas[(corridas.algoritmo == "DE") & (corridas.instancia == n)].valor.max(),
               "mejor_ag_previo": corridas[(corridas.algoritmo == "AG") & (corridas.instancia == n)].valor.max()}
              for n in EVAL]).to_csv(RES / "zref.csv", index=False)

# ---------------------------------------------------------------- estadística
DE = corridas[corridas.algoritmo == "DE"]
resumen = (DE.groupby(["instancia", "configuracion", "mutacion", "cruce"])
           .agg(brecha_mediana=("brecha_pct", "median"), brecha_max=("brecha_pct", "max"),
                captura_mediana=("captura_pct", "median")).reset_index())
resumen.to_csv(RES / "resumen.csv", index=False)

estad, por_inst = [], {}
for n in EVAL:
    df = DE[DE.instancia == n]
    cc = por_inst[n] = comparar(matriz_bloques(df, "semilla", "configuracion", ORDEN), ORDEN)
    estad.append({"instancia": n, "prueba": "Friedman, 9 configuraciones", "estadistico": cc["chi2"], "p": cc["p"],
                  "mejor": cc["mejor"], "empatadas_con_la_mejor": " · ".join(cc["empatadas"]) or "ninguna"})
    for factor, otro, niveles in (("mutacion", "cruce", MUTACIONES_DE), ("cruce", "mutacion", CRUCES_DE)):
        e = comparar(matriz_bloques(df, ["semilla", otro], factor, list(niveles)), list(niveles))
        estad.append({"instancia": n, "prueba": f"Efecto principal de {factor}", "estadistico": e["chi2"], "p": e["p"],
                      "mejor": e["mejor"], "empatadas_con_la_mejor": " · ".join(e["empatadas"]) or "ninguna"})
M_ = np.vstack([matriz_bloques(DE[DE.instancia == n], "semilla", "configuracion", ORDEN) for n in EVAL])
chi2_g, p_g = friedmanchisquare(*M_.T)
g = ranking_demsar(M_, ORDEN)
g["brecha_maxima_pct"] = [DE.loc[DE.configuracion == c_, "brecha_pct"].max() for c_ in ORDEN]
g["captura_mediana_pct"] = [DE.loc[DE.configuracion == c_, "captura_pct"].median() for c_ in ORDEN]
ics = [ic_bootstrap_mediana(DE.loc[DE.configuracion == c_, "captura_pct"]) for c_ in ORDEN]
g["captura_ic95_bajo"] = [a for a, _ in ics]; g["captura_ic95_alto"] = [b for _, b in ics]
g = g.sort_values("rango_medio").reset_index(drop=True)
mejor_cfg = g.iloc[0]["configuracion"]
cand = g[(g.configuracion == mejor_cfg) | (g.p_holm_vs_mejor >= 0.05)]
elegida = cand.sort_values(["brecha_maxima_pct", "rango_medio"]).iloc[0]["configuracion"]
j_el = ORDEN.index(elegida)
g["a12_vs_elegida"] = [vargha_delaney(M_[:, ORDEN.index(c_)], M_[:, j_el]) for c_ in g.configuracion]
g.to_csv(RES / "ranking_global.csv", index=False)
pd.DataFrame(estad).to_csv(RES / "pruebas_estadisticas.csv", index=False)
print(f"\nFriedman global: chi2 = {chi2_g:.2f}, p = {p_g:.3g}, bloques = {M_.shape[0]}")
print(g.round(3).to_string(index=False))
print("Elegida:", elegida, "· grupo superior:", list(cand.configuracion), flush=True)

# DE elegida frente al AG previo (emparejado por semilla)
pares = []
for n in EVAL:
    d_ = DE[(DE.configuracion == elegida) & (DE.instancia == n)].sort_values("semilla")
    a_ = corridas[(corridas.algoritmo == "AG") & (corridas.instancia == n)].sort_values("semilla")
    w = wilcoxon(d_.brecha_pct.values, a_.brecha_pct.values)
    pares.append({"instancia": n, "brecha_mediana_elegida": d_.brecha_pct.median(),
                  "brecha_mediana_AG_previo": a_.brecha_pct.median(),
                  "A12_elegida_mejor": vargha_delaney(d_.brecha_pct.values, a_.brecha_pct.values), "p_wilcoxon": w.pvalue})
pd.DataFrame(pares).to_csv(RES / "elegida_vs_ag_previo.csv", index=False)
print(pd.DataFrame(pares).to_string(index=False), flush=True)

# cartera de la mejor corrida de la configuración elegida en E-base
mejor = max((r for (alg, n, cfg, s), r in crudas.items() if alg == "DE" and n == "E-base" and cfg == elegida),
            key=lambda r: r["valor"])
resumen_por_sector(INSTANCIAS["E-base"], mejor["x"]).to_csv(RES / "cartera_E-base_elegida.csv", index=False)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifiesto = {"actividad": "Actividad previa número 9 · cartera estatal de inversión, Hidalgo (datos sintéticos)",
              "alumna": "Jessica Melani Romero Lora (25322116)",
              "entorno": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                          "scipy": scipy.__version__, "plataforma": platform.platform()},
              "parametros_calibrados": PARAM, "elegida": elegida, "zref": ZREF,
              "semillas_evaluacion": SEM_EVAL, "semillas_calibracion": SEM_CAL, "arranques_referencia": ARRANQUES_REF,
              "segundos_totales": time.perf_counter() - INICIO,
              "archivos": {str(p): sha(p) for p in sorted(RES.glob("*.csv"))}}
(RES / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False))
print(f"\nListo en {manifiesto['segundos_totales']:.0f} s", flush=True)
