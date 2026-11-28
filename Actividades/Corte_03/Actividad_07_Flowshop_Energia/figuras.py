"""Figuras del informe a partir de resultados/ y del modelo. Escribe figuras/*.png."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from flowshop_modelo import generar, evaluar, MAQUINAS

RES, FIG = Path("resultados"), Path("figuras"); FIG.mkdir(exist_ok=True)
AZUL, NARANJA, VERDE, GRIS, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b1", "#1f1f1f"
COL = {"NSGA-II (OX)": AZUL, "NSGA-II (PMX)": NARANJA, "Aleatoria": VERDE}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

# ---- Figura 1: esquema del flow shop
fig, ax = plt.subplots(figsize=(8.6, 2.6)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
xs = [18, 40, 62, 84]
for k, (x, nombre) in enumerate(zip(xs, MAQUINAS)):
    ax.add_patch(FancyBboxPatch((x - 7, 10), 14, 11, boxstyle="round,pad=0.4", fc="#e9f1fc", ec=AZUL, lw=1.4))
    ax.text(x, 15.5, nombre, ha="center", va="center", fontsize=8.5, weight="bold", color=TINTA)
    ax.text(x, 6.5, f"máquina M{k+1}", ha="center", va="center", fontsize=7.5, color=TINTA)
    if k < 3:
        ax.annotate("", xy=(xs[k + 1] - 7.6, 15.5), xytext=(x + 7.6, 15.5), arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=1.1))
for k, lote in enumerate(["L3", "L7", "L9", "L1", "L5"]):
    ax.add_patch(Rectangle((2 + k * 3.2, 25), 2.6, 3.2, fc=[NARANJA, AZUL, VERDE, GRIS, AZUL][k], ec=TINTA, lw=0.6, alpha=0.85))
    ax.text(3.3 + k * 3.2, 26.6, lote, ha="center", va="center", fontsize=6.5, color="white" if k in (0, 1, 2) else TINTA)
ax.text(2, 29.5, "cola de lotes (secuencia π)", fontsize=8, color=TINTA)
ax.text(50, 1.2, "Todos los lotes siguen la misma ruta: M1 → M2 → M3 → M4", ha="center", fontsize=8, color=TINTA)
fig.savefig(FIG / "fig1_esquema_flowshop.png"); plt.close(fig)

# ---- Figura 2: diagramas de Gantt de dos secuencias de la instancia A
A = generar("A-envasado-9", 9, 20262900)
dec = pd.read_csv(RES / "decision_A.csv")
def gantt(seq):
    n, m = A.n, A.m
    C = np.zeros((n, m)); S = np.zeros((n, m))
    for j, job in enumerate(seq):
        for i in range(m):
            S[j, i] = max(C[j - 1, i] if j > 0 else 0, C[j, i - 1] if i > 0 else 0); C[j, i] = S[j, i] + A.p[job, i]
    return S, C
fig, ejes = plt.subplots(1, 2, figsize=(10.0, 3.2), sharey=True)
for a_, fila, titulo in ((ejes[0], dec.iloc[0], "Mínimo C_max"), (ejes[1], dec.iloc[2], "Mínima energía")):
    seq = [int(s.strip()[1:]) - 1 for s in fila.secuencia.split("→")]
    S, C = gantt(seq)
    colores = plt.cm.tab10(np.linspace(0, 1, A.n))
    for i in range(A.m):
        for j, job in enumerate(seq):
            a_.barh(i, A.p[job, i], left=S[j, i], height=0.6, color=colores[job], edgecolor="white", lw=0.6)
            a_.text(S[j, i] + A.p[job, i] / 2, i, f"T{job+1}", ha="center", va="center", fontsize=6.5, color="white")
    a_.set_yticks(range(A.m)); a_.set_yticklabels(MAQUINAS, fontsize=8); a_.invert_yaxis()
    a_.set_xlabel("tiempo (min)"); a_.set_title(f"{titulo}: C_max = {C[-1,-1]:.0f} min, E = {fila.energia:.1f} kWh", fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "fig2_gantt.png"); plt.close(fig)

# ---- Figura 3: frente exacto (instancia A) proyectado, con muestra del espacio
F_exacto = pd.read_csv(RES / "frente_exacto_A.csv")
fr = F_exacto[["C_max (min)", "Tardanza ponderada", "Energía (kWh)"]].values
muestra = []
rng = np.random.default_rng(0)
muestra = evaluar(A, np.array([rng.permutation(A.n) for _ in range(20000)]))
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
pares = [(0, 1, "C_max (min)", "Tardanza ponderada"), (0, 2, "C_max (min)", "Energía (kWh)")]
for a_, (i, j, xl, yl) in zip(ax, pares):
    a_.scatter(muestra[:, i], muestra[:, j], s=5, color=GRIS, alpha=0.5, label="muestra aleatoria (20 000)")
    orden = np.argsort(fr[:, i])
    a_.scatter(fr[:, i], fr[:, j], s=28, color=NARANJA, edgecolor=TINTA, lw=0.4, zorder=3, label="frente exacto (28)")
    a_.set_xlabel(xl); a_.set_ylabel(yl)
ax[0].legend(frameon=False, fontsize=7.2, loc="upper right")
ax[0].set_title("Proyección C_max–tardanza"); ax[1].set_title("Proyección C_max–energía")
fig.tight_layout(); fig.savefig(FIG / "fig3_frente_exacto.png"); plt.close(fig)

# ---- Figura 4: comparación de métodos (IGD y HV) por instancia
corr = pd.read_csv(RES / "corridas.csv")
fig, ax = plt.subplots(2, 2, figsize=(8.6, 6.0))
for fila, inst in enumerate(corr.instancia.unique()):
    for col, met in enumerate(("IGD", "HV")):
        a_ = ax[fila, col]
        datos = [corr[(corr.instancia == inst) & (corr.algoritmo == alg)][met].values for alg in COL]
        bp = a_.boxplot(datos, patch_artist=True, widths=0.55, medianprops=dict(color=TINTA, lw=1.2),
                        flierprops=dict(ms=2.5, mfc=GRIS, mec="none"))
        for b_, col_ in zip(bp["boxes"], COL.values()):
            b_.set_facecolor(col_); b_.set_alpha(0.7)
        a_.set_xticks(range(1, 4)); a_.set_xticklabels(["OX", "PMX", "Aleatoria"], fontsize=8)
        a_.set_title(f"{inst.split(' (')[0]}: {'IGD (menor es mejor)' if met == 'IGD' else 'hipervolumen (mayor es mejor)'}", fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "fig4_comparacion_metricas.png"); plt.close(fig)
print(sorted(p.name for p in FIG.glob("*.png")))
