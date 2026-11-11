"""Genera las figuras y las tablas auxiliares del informe a partir de resultados/ y del modelo.
Escribe resultados/frecuencia_hallazgo.csv y resultados/recuperacion.csv (vectores del frente encontrados
según la evaluación, con presupuesto 300) y figuras/*.png."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moea_hiper import Tabla, ALGORITMOS, canonico, frente_no_dominado, normalizar, hipervolumen, igd

RES, FIG = Path("resultados"), Path("figuras"); FIG.mkdir(exist_ok=True)
AZUL, NARANJA, VERDE, GRIS, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b1", "#1f1f1f"
COL_ALG = {"NSGA-II": AZUL, "MOEA/D": NARANJA, "Aleatoria": VERDE}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

t = pd.read_csv(RES / "tabla_exhaustiva.csv")
reg = {canonico([r.g0, r.g1, r.g2, r.g3, r.g4, r.g5]): (1 - r.exactitud, r.mflops, float(r.parametros), r.exactitud)
       for r in t.itertuples()}
OBJ = np.array([reg[k][:3] for k in reg]); MIN, MAX = OBJ.min(0), OBJ.max(0)
idx = frente_no_dominado(OBJ)
vecs = sorted({tuple(np.round(OBJ[i], 10)) for i in idx}, key=lambda v: v[0])
FRENTE = normalizar(np.array(vecs), MIN, MAX)
nombre_v = {}
for v in vecs:
    r = t[(np.round(1 - t.exactitud, 10) == v[0]) & (np.round(t.mflops, 10) == v[1])].iloc[0]
    nombre_v[v] = f"{r.capas} · lr {r.lr:g} · lote {r.batch}"
SEM = list(range(20262700, 20262730))

# ---- tablas auxiliares: frecuencia de hallazgo por punto del frente y recuperación por evaluación
filas_f, filas_r = [], []
for B in (100, 200, 300):
    for alg, fn in ALGORITMOS.items():
        cuenta = np.zeros(len(vecs))
        for s in SEM:
            R = fn(Tabla(reg), B, s)
            hall = {tuple(np.round(reg[g][:3], 10)) for g in R.genotipos}
            cuenta += [v in hall for v in vecs]
            for k in range(20, B + 1, 20):
                h = {tuple(np.round(reg[g][:3], 10)) for g in R.genotipos[:k]}
                filas_r.append({"presupuesto": B, "algoritmo": alg, "semilla": s, "evaluaciones": k,
                                "vectores_recuperados": sum(v in h for v in vecs)})
        for v, c in zip(vecs, cuenta):
            filas_f.append({"presupuesto": B, "algoritmo": alg, "punto": nombre_v[v], "frecuencia": c / len(SEM)})
pd.DataFrame(filas_f).to_csv(RES / "frecuencia_hallazgo.csv", index=False, float_format="%.4g")
rec = pd.DataFrame(filas_r); rec.to_csv(RES / "recuperacion.csv", index=False)

# ---- Figura 1: espacio completo y frente exacto
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
err = OBJ[:, 0] * 100
for a, j, xl in ((ax[0], 1, "costo de entrenamiento (MFLOPs, log)"), (ax[1], 2, "parámetros del modelo (log)")):
    a.scatter(OBJ[:, j], err, s=10, color=GRIS, label="espacio completo (960)")
    fv = np.array(vecs)
    a.scatter(fv[:, j], fv[:, 0] * 100, s=60, color=NARANJA, edgecolor=TINTA, lw=0.6, zorder=3, label="frente exacto (4 vectores)")
    desplaz = [(6, 9), (6, -13), (-6, 10), (6, -4)]
    for k, v in enumerate(sorted(vecs, key=lambda v: v[j])):
        lab = nombre_v[v].split(" · ")[0]
        a.annotate(lab, (v[j], v[0] * 100), xytext=desplaz[k % 4], textcoords="offset points", fontsize=7, color=TINTA,
                   ha="left" if desplaz[k % 4][0] > 0 else "right")
    a.set_xscale("log"); a.set_xlabel(xl); a.set_ylabel("error de validación cruzada (%)")
    marcas = [30, 50, 100, 200, 500, 1000] if j == 1 else [200, 500, 1000, 2000, 5000]
    marcas = [m for m in marcas if a.get_xlim()[0] <= m <= a.get_xlim()[1]] if False else marcas
    a.set_xticks(marcas if j == 1 else [200, 500, 1000, 2000, 5000])
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].legend(frameon=False, fontsize=7.2, loc="upper right")
ax[0].set_title("Espacio de decisión y frente de Pareto exacto")
ax[1].set_title("Mismo espacio frente al tamaño del modelo")
fig.tight_layout(); fig.savefig(FIG / "fig1_frente_exacto.png"); plt.close(fig)

# ---- Figura 2: recuperación de vectores del frente a lo largo de la búsqueda
fig, ax = plt.subplots(figsize=(6.4, 3.4))
g = rec[rec.presupuesto == 300].groupby(["algoritmo", "evaluaciones"]).vectores_recuperados.agg(
    mediana="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)).reset_index()
for alg, col in COL_ALG.items():
    h = g[g.algoritmo == alg]
    ax.plot(h.evaluaciones, h.mediana, color=col, lw=2, label=alg)
    ax.fill_between(h.evaluaciones, h.q1, h.q3, color=col, alpha=0.15)
ax.set_xlabel("evaluaciones"); ax.set_ylabel("vectores del frente exacto encontrados (de 4)")
ax.set_ylim(0, 4.3); ax.set_yticks(range(5)); ax.legend(frameon=False, fontsize=8)
ax.set_title("Recuperación del frente a lo largo de la búsqueda (mediana y RIQ)")
fig.tight_layout(); fig.savefig(FIG / "fig2_recuperacion.png"); plt.close(fig)

# ---- Figura 3: frecuencia de hallazgo por punto del frente (presupuesto 300)
fr = pd.DataFrame(filas_f); fr = fr[fr.presupuesto == 300]
puntos = list(dict.fromkeys(fr.punto))
fig, ax = plt.subplots(figsize=(7.4, 3.4))
ancho = 0.26
for k, (alg, col) in enumerate(COL_ALG.items()):
    vals = [fr[(fr.algoritmo == alg) & (fr.punto == p)].frecuencia.values[0] for p in puntos]
    ax.bar(np.arange(len(puntos)) + (k - 1) * ancho, vals, width=ancho, color=col, label=alg)
ax.set_xticks(range(len(puntos))); ax.set_xticklabels(puntos, fontsize=7.5)
ax.set_ylabel("frecuencia de hallazgo (30 semillas)"); ax.set_ylim(0, 1.08)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
ax.set_title("Qué puntos del frente encuentra cada método (300 evaluaciones)")
fig.tight_layout(); fig.savefig(FIG / "fig3_frecuencia_hallazgo.png"); plt.close(fig)

# ---- Figura 4: distribución del IGD y de la brecha de hipervolumen (presupuesto 300)
c = pd.read_csv(RES / "corridas.csv"); c = c[c.presupuesto == 300]
hv_ex = hipervolumen(FRENTE, np.full(3, 1.1))
c = c.assign(brecha_HV=100 * (hv_ex - c.HV) / hv_ex)
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
for a, met, tit in ((ax[0], "IGD", "IGD (menor es mejor)"), (ax[1], "brecha_HV", "brecha de hipervolumen, % (menor es mejor)")):
    datos = [c[c.algoritmo == alg][met].values for alg in COL_ALG]
    bp = a.boxplot(datos, patch_artist=True, widths=0.55, medianprops=dict(color=TINTA, lw=1.2),
                   flierprops=dict(ms=2.5, mfc=GRIS, mec="none"))
    for b_, col in zip(bp["boxes"], COL_ALG.values()):
        b_.set_facecolor(col); b_.set_alpha(0.7)
    a.set_xticks(range(1, 4)); a.set_xticklabels(list(COL_ALG), fontsize=8); a.set_title(tit, fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig4_distribucion_metricas.png"); plt.close(fig)

# ---- Figura 5: convergencia del hipervolumen relativo a la referencia exacta (presupuesto 300)
conv = pd.read_csv(RES / "convergencia.csv"); conv = conv[conv.presupuesto == 300]
conv = conv.assign(brecha=100 * (hv_ex - conv.HV) / hv_ex)
fig, ax = plt.subplots(figsize=(6.4, 3.3))
for alg, col in COL_ALG.items():
    h = conv[conv.algoritmo == alg].groupby("evaluaciones").brecha.median().reset_index()
    ax.plot(h.evaluaciones, h.brecha, color=col, lw=2, label=alg)
ax.set_xlabel("evaluaciones"); ax.set_ylabel("brecha de hipervolumen (%), mediana")
ax.set_title("Convergencia del hipervolumen frente al frente exacto"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig5_convergencia_hv.png"); plt.close(fig)
print("figuras:", sorted(p.name for p in FIG.glob("*.png")))
