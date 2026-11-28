"""Experimento: DE clásica frente a JADE en torneado, y análisis de robustez del punto de operación.

Uso: python experimento.py   (≈ 1 minuto en un núcleo)

1. Convergencia: 30 semillas emparejadas, presupuestos de 300, 600 y 1 000 evaluaciones.
   Métricas: brecha final respecto a la referencia, evaluaciones necesarias para llegar a 0.1 % de brecha.
2. Robustez: 5 000 perturbaciones de coeficientes (±10 % de Taylor y k_c, ±5 % de radio de punta),
   comunes para todos los candidatos. Se re-optimiza cada combinación de márgenes en rugosidad y vida
   mínima, y se calcula la probabilidad de incumplimiento frente a su costo.
"""
import hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon
from dataclasses import replace

from modelo_mecanizado import PAR, referencia, costo, restricciones, perturbar, potencia, vida_herramienta
from evolucion_diferencial import ALGORITMOS

INICIO = time.perf_counter()
RES = Path("resultados"); RES.mkdir(exist_ok=True)
PRESUPUESTOS = (300, 600, 1000)
SEMILLAS = list(range(20262800, 20262830))
UMBRAL = 0.1          # brecha objetivo (%)
N_ROBUSTEZ = 5000

ref = referencia(malla=60, arranques=20)
X_REF, C_REF = ref["x"], ref["costo"]
pd.DataFrame([{"costo_referencia": C_REF, "v": X_REF[0], "f": X_REF[1], "a": X_REF[2],
               "Ra": 1000 * X_REF[1] ** 2 / (32 * PAR.r_eps), "P_kW": float(potencia(X_REF)),
               "T_min": float(vida_herramienta(X_REF))}]).to_csv(RES / "referencia.csv", index=False, float_format="%.6g")

# ------------------------------------------------------------------ 1. convergencia
filas, filas_hist = [], []
for B in PRESUPUESTOS:
    for nombre, alg in ALGORITMOS.items():
        for s in SEMILLAS:
            r = alg(s, B)
            factible = r.mejor_violacion <= 1e-12
            brecha = 100 * (r.mejor_costo - C_REF) / C_REF if factible else np.nan
            evs = next((e for e, c in r.historial if not np.isnan(c) and 100 * (c - C_REF) / C_REF <= UMBRAL), np.nan)
            filas.append({"presupuesto": B, "algoritmo": nombre, "semilla": s, "costo": r.mejor_costo,
                          "factible": factible, "brecha_pct": brecha, "evaluaciones_a_0_1pct": evs,
                          "segundos": r.segundos})
            for e, c in r.historial:
                if not np.isnan(c):
                    filas_hist.append({"presupuesto": B, "algoritmo": nombre, "semilla": s, "evaluaciones": e,
                                       "brecha_pct": 100 * (c - C_REF) / C_REF})
    print("listo presupuesto", B, flush=True)
conv = pd.DataFrame(filas); conv.to_csv(RES / "convergencia.csv", index=False, float_format="%.6g")
pd.DataFrame(filas_hist).to_csv(RES / "historial.csv", index=False, float_format="%.6g")

def holm(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); aj = np.empty(m); cur = 0.0
    for r_, i in enumerate(o):
        cur = max(cur, (m - r_) * p[i]); aj[i] = min(1.0, cur)
    return aj

est = []
for B in PRESUPUESTOS:
    a = conv[(conv.presupuesto == B) & (conv.algoritmo == "JADE")].sort_values("semilla")
    b = conv[(conv.presupuesto == B) & (conv.algoritmo == "DE clásico")].sort_values("semilla")
    xa, xb = a.brecha_pct.values, b.brecha_pct.values
    p = wilcoxon(xa, xb).pvalue
    A12 = float(np.mean(xa[:, None] < xb[None, :]) + 0.5 * np.mean(xa[:, None] == xb[None, :]))
    est.append({"presupuesto": B, "mediana_JADE": np.median(xa), "mediana_DE": np.median(xb),
                "p": p, "A12_JADE_mejor": A12,
                "evals_mediana_JADE": np.nanmedian(a.evaluaciones_a_0_1pct),
                "evals_mediana_DE": np.nanmedian(b.evaluaciones_a_0_1pct),
                "alcanza_0_1pct_JADE": int(np.sum(a.evaluaciones_a_0_1pct.notna())),
                "alcanza_0_1pct_DE": int(np.sum(b.evaluaciones_a_0_1pct.notna()))})
est = pd.DataFrame(est); est["p_holm"] = holm(est.p.values)
est.to_csv(RES / "estadistica.csv", index=False, float_format="%.6g")
print(est.round(4).to_string(index=False), flush=True)

# ------------------------------------------------------------------ 2. robustez y costo de la confiabilidad
# Mismas K perturbaciones (números aleatorios comunes) para todos los candidatos: comparación justa.
rng = np.random.default_rng(20262850)
PERTURBACIONES = [perturbar(PAR, rng) for _ in range(N_ROBUSTEZ)]
MARGENES_RA = (0.00, 0.05, 0.10, 0.15, 0.20)  # fracción de margen sobre Ra_max
MARGENES_T = (15.0, 18.0, 21.0, 24.0, 27.0, 30.0)   # vida mínima de herramienta (min)
filas_rob = []
for mr in MARGENES_RA:
    for Tm in MARGENES_T:
        base = replace(PAR, Ra_max=PAR.Ra_max * (1 - mr), T_min=Tm)
        xr = referencia(p=base, malla=60, arranques=20)["x"]
        G = np.array([restricciones(xr, q) for q in PERTURBACIONES])   # cumplimiento frente a los límites reales
        costos = np.array([costo(xr, q) for q in PERTURBACIONES])
        falla = np.any(G > 0, axis=1)
        filas_rob.append({"margen_Ra_pct": 100 * mr, "T_min": Tm, "v": xr[0], "f": xr[1], "a": xr[2],
                          "costo_nominal": costo(xr, PAR), "costo_p95_incertidumbre": float(np.quantile(costos, 0.95)),
                          "prob_incumplimiento_pct": 100 * falla.mean(),
                          "prob_Ra_pct": 100 * np.mean(G[:, 0] > 0), "prob_T_pct": 100 * np.mean(G[:, 2] > 0),
                          "prob_P_pct": 100 * np.mean(G[:, 1] > 0)})
rob = pd.DataFrame(filas_rob)
rob.to_csv(RES / "robustez.csv", index=False, float_format="%.6g")
# recomendación: menor costo nominal con incumplimiento ≤ 5 %
ok = rob[rob.prob_incumplimiento_pct <= 5.0]
recomendado = ok.sort_values("costo_nominal").iloc[0] if len(ok) else rob.sort_values("prob_incumplimiento_pct").iloc[0]
recomendado.to_frame().T.to_csv(RES / "robustez_recomendado.csv", index=False, float_format="%.6g")
print(rob.round(3).to_string(index=False), flush=True)
print("Recomendado (incumplimiento ≤ 5 %):", recomendado.to_dict(), flush=True)

# ------------------------------------------------------------------ manifiesto
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
man = {"actividad": "Aplicación industrial: optimización de condiciones de corte en torneado con evolución diferencial",
       "alumna": "Jessica Melani Romero Lora (25322116)",
       "entorno": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scipy": scipy.__version__, "plataforma": platform.platform()},
       "parametros_modelo": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PAR.__dict__.items()},
       "presupuestos": list(PRESUPUESTOS), "semillas": SEMILLAS, "umbral_pct": UMBRAL,
       "costo_referencia": C_REF, "x_referencia": X_REF.tolist(), "n_robustez": N_ROBUSTEZ,
       "segundos": time.perf_counter() - INICIO,
       "archivos": {str(p): sha(p) for p in sorted(RES.glob("*.csv"))}}
(RES / "manifiesto.json").write_text(json.dumps(man, indent=2, ensure_ascii=False, default=str))
print(f"Listo en {man['segundos']:.0f} s", flush=True)
