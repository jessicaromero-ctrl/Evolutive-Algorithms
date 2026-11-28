"""Experimento: NSGA-II (OX y PMX) frente a búsqueda aleatoria en flow shop con energía.

Instancia A (n = 9): frente de Pareto exacto por enumeración de 9! = 362 880 secuencias.
Instancia B (n = 25): referencia de mejor conocido = frente no dominado de 20 corridas largas de NSGA-II.
Protocolo: 30 semillas emparejadas, presupuesto idéntico por algoritmo, métricas normalizadas.
Uso: python experimento.py   (≈ 3 min en un núcleo)
"""
import hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon

from flowshop_modelo import generar, frente_exacto, frente_no_dominado, evaluar, NOMBRES_OBJ, MAQUINAS
from flowshop_moea import ALGORITMOS, nsga2, objetivos_de, hipervolumen, igd

INICIO = time.perf_counter()
RES = Path("resultados"); RES.mkdir(exist_ok=True)
SEMILLAS = list(range(20262900, 20262930))
PRES_A = 6000
PRES_B = 20000
REF_CORRIDAS = 20
PRES_REF = 100000
REF_HV = 1.1


def clave(v):
    return tuple(np.round(v, 9))


def normalizador(Fref):
    lo, hi = Fref.min(axis=0), Fref.max(axis=0)
    return lo, hi, (lambda F: (np.asarray(F, float) - lo) / (hi - lo))


def evalua_algoritmo(nombre, alg, inst, presupuesto, semilla, ref_norm, norm, claves_ref, ref_nd):
    R = alg(inst, presupuesto, semilla)
    O = objetivos_de(R)
    ND = O[frente_no_dominado(O)]
    ND_n = norm(ND)
    hv = hipervolumen(ND_n, np.full(3, REF_HV))
    d = igd(ND_n, ref_norm)
    hallados = len({clave(v) for v in O} & claves_ref)
    return {"algoritmo": nombre, "semilla": semilla, "evaluaciones": len(R.perms), "HV": hv,
            "IGD": d, "recuperados": hallados, "fraccion_frente": hallados / len(claves_ref),
            "n_no_dominadas": len(ND)}


def estadistica(df, metrica, mejor_alto):
    filas = []
    pares = [("NSGA-II (OX)", "NSGA-II (PMX)"), ("NSGA-II (OX)", "Aleatoria"), ("NSGA-II (PMX)", "Aleatoria")]
    for a, b in pares:
        xa = df[df.algoritmo == a].sort_values("semilla")[metrica].values
        xb = df[df.algoritmo == b].sort_values("semilla")[metrica].values
        p = wilcoxon(xa, xb).pvalue if np.any(xa != xb) else 1.0
        eq = np.mean(xa[:, None] == xb[None, :])
        A12 = float(np.mean(xa[:, None] > xb[None, :]) + 0.5 * eq) if mejor_alto else float(np.mean(xa[:, None] < xb[None, :]) + 0.5 * eq)
        filas.append({"comparacion": f"{a} vs {b}", "metrica": metrica, "mediana_a": float(np.median(xa)),
                      "mediana_b": float(np.median(xb)), "p": float(p), "A12_a_mejor": A12})
    t = pd.DataFrame(filas)
    o = np.argsort(t.p.values); aj = np.empty(len(t)); cur = 0.0
    for r_, i in enumerate(o):
        cur = max(cur, (len(t) - r_) * t.p.values[i]); aj[i] = min(1.0, cur)
    t["p_holm"] = aj
    return t


# ------------------------------------------------------------------ instancia A: frente exacto
A = generar("A-envasado-9", 9, 20262900)
frente_A, X_A, F_A = frente_exacto(A)
lo_A, hi_A, norm_A = normalizador(F_A)
ref_A = norm_A(frente_A)
claves_A = {clave(v) for v in frente_A}
HV_EXACTO_A = hipervolumen(ref_A, np.full(3, REF_HV))
pd.DataFrame(np.column_stack([frente_A, X_A]), columns=list(NOMBRES_OBJ) + [f"pos{k+1}" for k in range(A.n)]) \
    .astype({f"pos{k+1}": int for k in range(A.n)}).to_csv(RES / "frente_exacto_A.csv", index=False, float_format="%.6g")
print(f"Instancia A: {len(frente_A)} puntos en el frente exacto; HV exacto (normalizado) = {HV_EXACTO_A:.4f}", flush=True)

# ------------------------------------------------------------------ instancia B: referencia de mejor conocido
B = generar("B-envasado-25", 25, 20262925)
ref_B_union = []
for s in range(REF_CORRIDAS):
    R = nsga2(B, PRES_REF, 20262950 + s, cruce="OX" if s % 2 == 0 else "PMX")
    ref_B_union.append(objetivos_de(R))
U = np.vstack(ref_B_union)
frente_B = U[frente_no_dominado(U)]
lo_B, hi_B, norm_B = normalizador(frente_B)
ref_B = norm_B(frente_B)
claves_B = {clave(v) for v in frente_B}
print(f"Instancia B: referencia con {len(frente_B)} puntos no dominados (de {REF_CORRIDAS} corridas de {PRES_REF} evaluaciones)", flush=True)

# ------------------------------------------------------------------ comparación
# Las soluciones se guardan crudas; la normalización de cada instancia usa el rango de TODAS las soluciones
# evaluadas (referencia y algoritmos), para que ninguna solución quede fuera de la caja de hipervolumen.
crudas = []
for nombre, alg in ALGORITMOS.items():
    for s in SEMILLAS:
        R = alg(A, PRES_A, s); O = objetivos_de(R)
        crudas.append(("A (n=9, exacta)", nombre, s, O, len(R.perms)))
for nombre, alg in ALGORITMOS.items():
    for s in SEMILLAS:
        R = alg(B, PRES_B, s); O = objetivos_de(R)
        crudas.append(("B (n=25, referencia)", nombre, s, O, len(R.perms)))
print("Corridas completadas:", len(crudas), flush=True)

filas = []
for inst_nom, claves_ref, ref_F in (("A (n=9, exacta)", claves_A, frente_A), ("B (n=25, referencia)", claves_B, frente_B)):
    todas = np.vstack([ref_F] + [c[3] for c in crudas if c[0] == inst_nom])
    lo, hi = todas.min(axis=0), todas.max(axis=0)
    norm = lambda F, lo=lo, hi=hi: (np.asarray(F, float) - lo) / (hi - lo)
    ref_n = norm(ref_F)
    for (ins, nombre, s, O, n_eval) in [c for c in crudas if c[0] == inst_nom]:
        ND_n = norm(O[frente_no_dominado(O)])
        hallados = len({clave(v) for v in O} & claves_ref)
        filas.append({"instancia": ins, "algoritmo": nombre, "semilla": s, "evaluaciones": n_eval,
                      "HV": hipervolumen(ND_n, np.full(3, REF_HV)), "IGD": igd(ND_n, ref_n),
                      "recuperados": hallados, "fraccion_frente": hallados / len(claves_ref),
                      "n_no_dominadas": len(ND_n)})
corr = pd.DataFrame(filas)
corr.to_csv(RES / "corridas.csv", index=False, float_format="%.6g")

est = []
for inst_nom, df in corr.groupby("instancia"):
    for met, alto in (("HV", True), ("IGD", False), ("fraccion_frente", True)):
        t = estadistica(df, met, alto); t.insert(0, "instancia", inst_nom); est.append(t)
est = pd.concat(est, ignore_index=True)
est.to_csv(RES / "estadistica.csv", index=False, float_format="%.6g")
print(est.round(4).to_string(index=False), flush=True)

resumen = corr.groupby(["instancia", "algoritmo"]).agg(
    HV_mediana=("HV", "median"), IGD_mediana=("IGD", "median"), IGD_q1=("IGD", lambda x: x.quantile(.25)),
    IGD_q3=("IGD", lambda x: x.quantile(.75)), recuperados_media=("recuperados", "mean"),
    fraccion_mediana=("fraccion_frente", "median"), n_nd_mediana=("n_no_dominadas", "median")).reset_index()
resumen.to_csv(RES / "resumen.csv", index=False, float_format="%.6g")
print(resumen.round(4).to_string(index=False), flush=True)

# ------------------------------------------------------------------ puntos de decisión (instancia A)
ideal = frente_A.min(axis=0); nadir = frente_A.max(axis=0)
dist = np.linalg.norm((frente_A - ideal) / (nadir - ideal), axis=1)
dec = []
for etiqueta, idx in (("mínimo C_max", np.argmin(frente_A[:, 0])), ("mínima tardanza", np.argmin(frente_A[:, 1])),
                      ("mínima energía", np.argmin(frente_A[:, 2])), ("compromiso (más cercano al ideal)", np.argmin(dist))):
    dec.append({"criterio": etiqueta, "C_max": frente_A[idx, 0], "tardanza": frente_A[idx, 1], "energia": frente_A[idx, 2],
                "secuencia": " → ".join(f"T{int(j)+1}" for j in X_A[idx])})
pd.DataFrame(dec).to_csv(RES / "decision_A.csv", index=False, float_format="%.6g")
print(pd.DataFrame(dec).to_string(index=False), flush=True)

# correlación entre objetivos en el espacio completo (instancia A)
Fall = evaluar(A, X_A[:1]) if False else None
man = {"actividad": "Flow shop de permutación con energía: NSGA-II frente a búsqueda aleatoria",
       "alumna": "Jessica Melani Romero Lora (25322116)",
       "entorno": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                   "scipy": scipy.__version__, "plataforma": platform.platform()},
       "maquinas": list(MAQUINAS), "semillas": SEMILLAS, "presupuesto_A": PRES_A, "presupuesto_B": PRES_B,
       "corridas_referencia_B": REF_CORRIDAS, "presupuesto_referencia_B": PRES_REF, "HV_ref": REF_HV,
       "puntos_frente_A": int(len(frente_A)), "puntos_referencia_B": int(len(frente_B)),
       "segundos": time.perf_counter() - INICIO,
       "archivos": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RES.glob("*.csv"))}}
(RES / "manifiesto.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
print(f"Listo en {man['segundos']:.0f} s", flush=True)
