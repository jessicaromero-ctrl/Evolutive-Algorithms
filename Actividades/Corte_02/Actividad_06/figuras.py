"""Figuras del informe a partir de resultados/ y del modelo. Escribe figuras/*.png."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modelo_mecanizado import PAR, costo, tiempo_maquinado, vida_herramienta, rugosidad, potencia, restricciones, referencia

RES, FIG = Path("resultados"), Path("figuras"); FIG.mkdir(exist_ok=True)
AZUL, NARANJA, GRIS, TINTA, VERDE = "#2a78d6", "#eb6834", "#9b9a95", "#1f1f1f", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})
X_REF = referencia(malla=40, arranques=12)["x"]

# ---- Figura 1: compromiso de costo frente a la velocidad de corte
v = np.linspace(60, 300, 400)
x = np.stack([v, np.full_like(v, X_REF[1]), np.full_like(v, X_REF[2])], axis=1)
tm = np.array([tiempo_maquinado(q) for q in x]); T = np.array([vida_herramienta(q) for q in x])
C = np.array([costo(q) for q in x])
t_maq = PAR.M * tm; t_herr = (tm / T) * (PAR.M * PAR.t_c + PAR.C_t)
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
ax[0].plot(v, t_maq, color=AZUL, lw=2, label="costo de maquinado $M·t_m$")
ax[0].plot(v, t_herr, color=NARANJA, lw=2, label="costo de herramienta")
ax[0].plot(v, C, color=TINTA, lw=1.4, ls="--", label="costo total")
ax[0].axvline(X_REF[0], color=GRIS, lw=1); ax[0].text(X_REF[0] + 3, 0.5, f"óptimo\n{X_REF[0]:.0f} m/min", fontsize=7.5, color=TINTA)
ax[0].set_xlabel("velocidad de corte v (m/min)"); ax[0].set_ylabel("€ por pieza"); ax[0].set_ylim(0, 12)
ax[0].set_title("Compromiso de costo a f y a fijos"); ax[0].legend(frameon=False, fontsize=7.5)
ax[1].plot(v, T, color=NARANJA, lw=2); ax[1].axhline(PAR.T_min, color=TINTA, ls=":", lw=1.2)
ax[1].text(70, PAR.T_min + 1.2, "vida mínima T_min", fontsize=7.5)
ax[1].set_yscale("log"); ax[1].set_xlabel("velocidad de corte v (m/min)"); ax[1].set_ylabel("vida de herramienta (min)")
ax[1].set_title("La vida decrece rápido con v (Taylor)")
fig.tight_layout(); fig.savefig(FIG / "fig1_compromiso_velocidad.png"); plt.close(fig)

# ---- Figura 2: convergencia de DE clásica y JADE
hist = pd.read_csv(RES / "historial.csv")
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=False)
for a_, B in zip(ax, (300, 600, 1000)):
    for alg, col in (("DE clásico", NARANJA), ("JADE", AZUL)):
        h = hist[(hist.presupuesto == B) & (hist.algoritmo == alg)]
        g = h.groupby("evaluaciones").brecha_pct.agg(mediana="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75)).reset_index()
        a_.plot(g.evaluaciones, g.mediana.clip(lower=1e-3), color=col, lw=2, label=alg)
        a_.fill_between(g.evaluaciones, g.q1.clip(lower=1e-3), g.q3.clip(lower=1e-3), color=col, alpha=0.15)
    a_.set_yscale("log"); a_.axhline(0.1, color=GRIS, ls=":", lw=1)
    a_.set_xlabel("evaluaciones"); a_.set_title(f"Presupuesto {B}")
    a_.set_ylabel("brecha respecto a la referencia (%)")
ax[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig2_convergencia.png"); plt.close(fig)

# ---- Figura 3: distribución final de la brecha (presupuesto 1000)
conv = pd.read_csv(RES / "convergencia.csv"); c1 = conv[(conv.presupuesto == 1000)]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
datos = [c1[c1.algoritmo == a].brecha_pct.dropna().values for a in ("DE clásico", "JADE")]
bp = ax.boxplot(datos, patch_artist=True, widths=0.5, medianprops=dict(color=TINTA), flierprops=dict(ms=2.5, mfc=GRIS, mec="none"))
for b_, col in zip(bp["boxes"], (NARANJA, AZUL)):
    b_.set_facecolor(col); b_.set_alpha(0.7)
ax.set_yscale("log"); ax.set_xticks([1, 2]); ax.set_xticklabels(["DE clásico", "JADE"])
ax.set_ylabel("brecha final (%), escala log"); ax.set_title("Brecha final con 1 000 evaluaciones (30 semillas)")
est = pd.read_csv(RES / "estadistica.csv").set_index("presupuesto")
ax.text(1.5, 0.11, f"Holm p = {est.loc[1000,'p_holm']:.3f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig3_distribucion_final.png"); plt.close(fig)

# ---- Figura 4: costo de la confiabilidad (frontera de costo frente a probabilidad de incumplimiento)
rob = pd.read_csv(RES / "robustez.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.8))
cmap = plt.get_cmap("Blues")
for k, mr in enumerate(sorted(rob.margen_Ra_pct.unique())):
    s = rob[rob.margen_Ra_pct == mr].sort_values("T_min")
    ax.plot(s.prob_incumplimiento_pct, s.costo_nominal, marker="o", ms=4, lw=1.2, color=cmap(0.35 + 0.13 * k),
            label=f"margen Ra {mr:g} %")
    for _, r in s.iterrows():
        if r.T_min in (15.0, 30.0):
            ax.annotate(f"T={r.T_min:g}", (r.prob_incumplimiento_pct, r.costo_nominal), xytext=(3, 3),
                        textcoords="offset points", fontsize=6.5, color=TINTA)
ax.axvline(5, color=GRIS, ls=":", lw=1); ax.text(5.5, rob.costo_nominal.max() - 0.03, "criterio 5 %", fontsize=7.5, color=TINTA)
rec = pd.read_csv(RES / "robustez_recomendado.csv").iloc[0]
ax.scatter([rec.prob_incumplimiento_pct], [rec.costo_nominal], s=110, facecolor="none", edgecolor=NARANJA, lw=2, zorder=5)
ax.annotate("recomendado", (rec.prob_incumplimiento_pct, rec.costo_nominal), xytext=(12, -12), textcoords="offset points", fontsize=8, color=NARANJA)
ax.set_xlabel("probabilidad de incumplir alguna restricción (%)"); ax.set_ylabel("costo nominal (€/pieza)")
ax.set_title("Costo de la confiabilidad"); ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig4_costo_confiabilidad.png"); plt.close(fig)

# ---- Figura 5: región factible en (v, f) con a = 3 mm
vv = np.linspace(60, 300, 260); ff = np.linspace(0.05, 0.40, 260)
V, F = np.meshgrid(vv, ff)
Xg = np.stack([V.ravel(), F.ravel(), np.full(V.size, 3.0)], axis=1)
G = np.array([restricciones(q) for q in Xg]).reshape(V.shape + (3,))
Cg = np.array([costo(q) for q in Xg]).reshape(V.shape)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ok = np.all(G <= 0, axis=2)
ax.contourf(V, F, ok.astype(float), levels=[-0.5, 0.5, 1.5], colors=["#f2f2ef", "#dbeafb"])
cs = ax.contour(V, F, Cg, levels=np.linspace(Cg[ok].min(), Cg[ok].max(), 9), colors=TINTA, linewidths=0.6)
ax.clabel(cs, fontsize=6.5, fmt="%.2f €")
ax.contour(V, F, G[:, :, 0], levels=[0], colors=AZUL, linewidths=1.8)
ax.contour(V, F, G[:, :, 2], levels=[0], colors=NARANJA, linewidths=1.8, linestyles="--")
ax.plot(X_REF[0], X_REF[1], "o", ms=7, mfc="white", mec=TINTA, mew=1.5, zorder=5)
ax.annotate("óptimo nominal", (X_REF[0], X_REF[1]), xytext=(12, -14), textcoords="offset points", fontsize=8)
ax.set_xlabel("velocidad de corte v (m/min)"); ax.set_ylabel("avance f (mm/rev)")
ax.set_title("Región factible con a = 3 mm (zona celeste)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=AZUL, lw=2, label="límite de rugosidad Ra"),
                   Line2D([], [], color=NARANJA, lw=2, ls="--", label="límite de vida mínima T_min")],
          frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(FIG / "fig5_region_factible.png"); plt.close(fig)
print(sorted(p.name for p in FIG.glob("*.png")))
