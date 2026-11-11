"""Experimento: NSGA-II, MOEA/D y búsqueda aleatoria con presupuestos idénticos, frente al frente exacto.

Uso: python experimento.py   (requiere resultados/tabla_exhaustiva.csv; ≈ 1 minuto)
"""
import hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon
from moea_hiper import Tabla, ALGORITMOS, frente_no_dominado, normalizar, metricas, canonico

INICIO = time.perf_counter()
RES = Path("resultados"); RES.mkdir(exist_ok=True)
PRESUPUESTOS = (100, 200, 300)
SEMILLAS = list(range(20262700, 20262730))
REF = 1.1

t = pd.read_csv(RES / "tabla_exhaustiva.csv")
registros = {canonico([r.g0, r.g1, r.g2, r.g3, r.g4, r.g5]): (1 - r.exactitud, r.mflops, float(r.parametros), r.exactitud)
             for r in t.itertuples()}
claves = list(registros.keys())
OBJ = np.array([registros[k][:3] for k in claves])
MIN, MAX = OBJ.min(axis=0), OBJ.max(axis=0)
ON = normalizar(OBJ, MIN, MAX)
idx_front = frente_no_dominado(OBJ)
# Los genotipos que sólo difieren en alpha tienen los mismos objetivos: el frente tiene vectores repetidos.
# Las métricas de recuperación cuentan vectores de objetivos distintos del frente exacto.
clave_vec = lambda v: tuple(np.round(v, 10))
vectores_frente = sorted({clave_vec(OBJ[i]) for i in idx_front})
FRENTE = normalizar(np.array(vectores_frente), MIN, MAX)
vec_a_idx = {}
for i in idx_front:
    vec_a_idx.setdefault(clave_vec(OBJ[i]), []).append(i)
frente_df = t.iloc[idx_front].copy().sort_values("error")
frente_df["vector_repetido"] = frente_df.duplicated(subset=["error", "mflops", "parametros"], keep="first")
frente_df["vector_repetido"] = frente_df["vector_repetido"].map({True: "sí (difiere sólo en alpha)", False: "no"})
frente_df["error_norm"] = normalizar(frente_df[["error", "mflops", "parametros"]].values, MIN, MAX)[:, 0]
frente_df["mflops_norm"] = normalizar(frente_df[["error", "mflops", "parametros"]].values, MIN, MAX)[:, 1]
frente_df["parametros_norm"] = normalizar(frente_df[["error", "mflops", "parametros"]].values, MIN, MAX)[:, 2]
frente_df.to_csv(RES / "frente_exacto.csv", index=False, float_format="%.6g")
frente_df[frente_df.vector_repetido.str.startswith("no")].to_csv(RES / "frente_exacto_distinto.csv", index=False, float_format="%.6g")
print(f"Espacio: {len(claves)} configuraciones; frente exacto: {len(idx_front)} genotipos no dominados, "
      f"{len(vectores_frente)} vectores de objetivos distintos", flush=True)

# punto de compromiso: mínima distancia al punto ideal (0,0,0) en el espacio normalizado
rodilla = int(np.argmin(np.linalg.norm(FRENTE, axis=1)))
ref_k = np.array(vectores_frente[rodilla])
recom = frente_df[frente_df.vector_repetido.str.startswith("no")]
recom = recom.iloc[[int(np.argmin(np.linalg.norm(normalizar(recom[["error", "mflops", "parametros"]].values, MIN, MAX), axis=1)))]]
recom.to_csv(RES / "recomendada.csv", index=False, float_format="%.6g")

def holm(p):
    p = np.asarray(p); m = len(p); orden = np.argsort(p); aj = np.empty(m); cur = 0.0
    for r, i in enumerate(orden):
        cur = max(cur, (m - r) * p[i]); aj[i] = min(1.0, cur)
    return aj

filas_corr, filas_conv, filas_est = [], [], []
for B in PRESUPUESTOS:
    corr_B = []
    for nombre, alg in ALGORITMOS.items():
        for s in SEMILLAS:
            R = alg(Tabla(registros), B, s)
            objs = np.array([registros[g][:3] for g in R.genotipos])
            hv, d = metricas(objs, FRENTE, MIN, MAX, REF)
            vecs_hallados = {clave_vec(registros[g][:3]) for g in R.genotipos}
            recup = len(vecs_hallados & set(vectores_frente))
            fila = {"presupuesto": B, "algoritmo": nombre, "semilla": s, "evaluaciones": len(R.genotipos),
                    "HV": hv, "IGD": d, "recuperados": recup, "fraccion_frente": recup / len(vectores_frente)}
            corr_B.append(fila); filas_corr.append(fila)
            for k in range(20, B + 1, 20):
                hv_k, _ = metricas(objs[:k], FRENTE, MIN, MAX, REF)
                filas_conv.append({"presupuesto": B, "algoritmo": nombre, "semilla": s, "evaluaciones": k, "HV": hv_k})
    cb = pd.DataFrame(corr_B)
    pares = [("NSGA-II", "MOEA/D"), ("NSGA-II", "Aleatoria"), ("MOEA/D", "Aleatoria")]
    tmp = []
    for a, b in pares:
        for met, alto in (("HV", True), ("IGD", False)):
            xa = cb[cb.algoritmo == a].sort_values("semilla")[met].values
            xb = cb[cb.algoritmo == b].sort_values("semilla")[met].values
            p = wilcoxon(xa, xb).pvalue if np.any(xa != xb) else 1.0
            eq = np.mean(xa[:, None] == xb[None, :])
            A12 = float(np.mean(xa[:, None] > xb[None, :]) + 0.5 * eq) if alto else float(np.mean(xa[:, None] < xb[None, :]) + 0.5 * eq)
            tmp.append({"presupuesto": B, "comparacion": f"{a} vs {b}", "metrica": met,
                        "mediana_a": float(np.median(xa)), "mediana_b": float(np.median(xb)), "p": float(p), "A12_a_mejor": A12})
    tmp = pd.DataFrame(tmp)
    tmp["p_holm"] = holm(tmp.p.values)
    filas_est += tmp.to_dict("records")
    print(f"\nPresupuesto {B}:", flush=True)
    print(cb.groupby("algoritmo")[["HV", "IGD", "fraccion_frente"]].median().round(4).to_string(), flush=True)

corr = pd.DataFrame(filas_corr); corr.to_csv(RES / "corridas.csv", index=False, float_format="%.6g")
pd.DataFrame(filas_conv).to_csv(RES / "convergencia.csv", index=False, float_format="%.6g")
pd.DataFrame(filas_est).to_csv(RES / "estadistica.csv", index=False, float_format="%.6g")

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
man = {"actividad": "Optimización multiobjetivo de hiperparámetros de un MLP con MOEA",
       "alumna": "Jessica Melani Romero Lora (25322116)",
       "entorno": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scipy": scipy.__version__, "plataforma": platform.platform()},
       "presupuestos": list(PRESUPUESTOS), "semillas": SEMILLAS, "espacio": len(claves),
       "frente_exacto_genotipos": len(idx_front), "frente_exacto_vectores": len(vectores_frente), "normalizacion_min": MIN.tolist(), "normalizacion_max": MAX.tolist(),
       "referencia_HV": REF, "segundos": time.perf_counter() - INICIO,
       "archivos": {str(p): sha(p) for p in sorted(RES.glob("*.csv"))}}
(RES / "manifiesto.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
print(f"\nListo en {man['segundos']:.0f} s", flush=True)
