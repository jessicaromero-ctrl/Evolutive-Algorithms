"""Genera las figuras del informe a partir de resultados/*.csv y del modelo (sin recalcular el experimento)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from ag_inversion import (generar_cartera, valor, reparto_proporcional, REGIONES, MERCADOS, R, M, A, THETA, ANCHO)

RES, FIG = Path("resultados"), Path("figuras"); FIG.mkdir(exist_ok=True)
AZUL, NARANJA, VERDE, GRIS, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b86", "#1f1f1f"
COL_MERC = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#d9a21b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})
MAPA = LinearSegmentedColormap.from_list("m", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
corr = pd.read_csv(RES / "corridas.csv")
de = corr[corr.algoritmo == "DE"]
ORDEN = [f"{m}+{c}" for m in ("rand1", "best1", "ctb1") for c in ("bin", "exp", "arit")]
ETQ = {"rand1": "rand/1", "best1": "best/1", "ctb1": "cur-to-best/1", "bin": "binomial", "exp": "exponencial", "arit": "aritmética"}
def etq(cfg): m, c = cfg.split("+"); return f"{ETQ[m]}\n{ETQ[c]}"

# ---- Figura 1: el modelo: umbral de infraestructura y productividad media
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.0))
x = np.linspace(0, 6000, 600)
th, a = THETA[0, 0], A[0, 0]
s = 0.35 + 0.65 / (1 + np.exp(-(x - th) / ANCHO))
ax[0].plot(x, a * x * s, color=AZUL, lw=2, label="valor agregado A·x·s(x)")
ax[0].plot(x, a * x, color=GRIS, lw=1.2, ls="--", label="referencia lineal A·x")
ax[0].axvline(th, color=NARANJA, lw=1); ax[0].text(th + 80, ax[0].get_ylim()[1] * 0.08, f"umbral θ = {th:,.0f} mdp", color=NARANJA, fontsize=8)
ax[0].set_xlabel("inversión en un mercado (mdp)"); ax[0].set_ylabel("valor agregado anual (mdp)")
ax[0].set_title("Logística en Zapotlán–Tizayuca"); ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
prod = a * s
ax[1].plot(x[1:], prod[1:], color=AZUL, lw=2)
ax[1].axvline(th, color=NARANJA, lw=1)
ax[1].set_xlabel("inversión en un mercado (mdp)"); ax[1].set_ylabel("valor agregado por mdp")
ax[1].set_title("Productividad media: no monótona en el umbral")
ax[1].text(4200, prod[-1] * 0.5, "mismo potencial,\nmenos rendimiento\npor debajo del umbral", fontsize=7.5, color=TINTA)
fig.tight_layout(); fig.savefig(FIG / "fig1_modelo_umbral.png"); plt.close(fig)

# ---- Figura 2: diseño factorial y calibración
fig, ax = plt.subplots(figsize=(7.4, 3.6))
cal = pd.read_csv(RES / "calibracion_elegidos.csv").set_index("configuracion")
for i, m in enumerate(("rand1", "best1", "ctb1")):
    for j, c in enumerate(("bin", "exp", "arit")):
        cfg = f"{m}+{c}"; p = cal.loc[cfg]
        col = [AZUL, NARANJA, VERDE][j]
        ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fc="white", ec=col, lw=1.8))
        par = f"F = {p.F:g}" + ("" if pd.isna(p.CR) else f", CR = {p.CR:g}")
        ax.text(j, i - 0.15, f"{ETQ[m]} + {ETQ[c]}", ha="center", va="center", fontsize=8.5, weight="bold", color=TINTA)
        ax.text(j, i + 0.17, f"{par}\nbrecha calibr. {p.brecha_calibracion:.2f} %", ha="center", va="center", fontsize=7.2, color=TINTA)
ax.set_xlim(-0.5, 2.5); ax.set_ylim(2.5, -0.5); ax.set_xticks(range(3)); ax.set_xticklabels(["cruce binomial", "cruce exponencial", "cruce aritmética"])
ax.set_yticks(range(3)); ax.set_yticklabels(["mutación rand/1", "mutación best/1", "mutación cur-to-best/1"])
ax.tick_params(length=0); ax.xaxis.tick_top(); ax.set_title("Diseño factorial 3 × 3 con parámetros calibrados por configuración", pad=28)
for s_ in ax.spines.values(): s_.set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig2_diseno_factorial.png"); plt.close(fig)

# ---- Figura 3: brecha mediana por configuración en cada instancia (mapas 3 × 3)
res = pd.read_csv(RES / "resumen.csv")
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.6))
nor = matplotlib.colors.LogNorm(vmin=res.brecha_mediana.min(), vmax=res.brecha_mediana.max())
rank = pd.read_csv(RES / "ranking_global.csv").set_index("configuracion")
for ax, n in zip(axs, ["E-base", "E-escasez", "E-amplia"]):
    Z = np.zeros((3, 3))
    for i, m in enumerate(("rand1", "best1", "ctb1")):
        for j, c in enumerate(("bin", "exp", "arit")):
            Z[i, j] = res[(res.instancia == n) & (res.configuracion == f"{m}+{c}")].brecha_mediana.iloc[0]
    ax.imshow(Z, cmap=MAPA, norm=nor)
    for i in range(3):
        for j in range(3):
            cfg = f"{('rand1','best1','ctb1')[i]}+{('bin','exp','arit')[j]}"
            ax.text(j, i, f"{Z[i, j]:.2f} %", ha="center", va="center", fontsize=8.5,
                    color="white" if nor(Z[i, j]) > 0.6 else TINTA, weight="bold" if rank.loc[cfg, "rango_medio"] == rank.rango_medio.min() else "normal")
    ax.set_xticks(range(3)); ax.set_xticklabels(["binomial", "exponencial", "aritmética"], fontsize=7.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(["rand/1", "best/1", "cur-to-best"], fontsize=7.5)
    ax.set_title(n); ax.grid(False)
fig.text(0.5, -0.03, "Brecha mediana frente al mejor conocido, 30 semillas. Negrita: menor rango medio global.", ha="center", fontsize=7.5, color=GRIS)
fig.tight_layout(); fig.savefig(FIG / "fig3_brechas_3x3.png"); plt.close(fig)

# ---- Figura 4: distribución y convergencia (E-base)
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})
d_eb = de[de.instancia == "E-base"]
datos = [d_eb.loc[d_eb.configuracion == c, "brecha_pct"].clip(lower=1e-3) for c in ORDEN]
bp = ax[0].boxplot(datos, patch_artist=True, widths=0.6, flierprops=dict(ms=2, mfc=GRIS, mec="none"),
                   medianprops=dict(color=TINTA, lw=1.2))
for b, c in zip(bp["boxes"], ORDEN):
    b.set_facecolor({"bin": AZUL, "exp": NARANJA, "arit": VERDE}[c.split("+")[1]]); b.set_alpha(0.75)
ax[0].set_yscale("log"); ax[0].set_xticks(range(1, 10)); ax[0].set_xticklabels([etq(c) for c in ORDEN], rotation=60, ha="right", fontsize=6.5)
ax[0].set_ylabel("brecha (%) · escala log"); ax[0].set_title("Distribución de la brecha, E-base")
conv = pd.read_csv(RES / "convergencia_E-base.csv")
cv = conv.groupby(["configuracion", "evaluaciones"])["mejor_factible"].median().reset_index()
estilo = {"rand1": "-", "best1": "--", "ctb1": ":"}
for c in ORDEN:
    g = cv[cv.configuracion == c]
    ax[1].plot(g.evaluaciones, g.mejor_factible, ls=estilo[c.split("+")[0]], color={"bin": AZUL, "exp": NARANJA, "arit": VERDE}[c.split("+")[1]], lw=1.4)
ax[1].axhline(pd.read_csv(RES / "zref.csv").set_index("instancia").loc["E-base", "zref"], color=TINTA, lw=0.8, ls="-.")
ax[1].text(500, pd.read_csv(RES / "zref.csv").set_index("instancia").loc["E-base", "zref"] * 1.002, "mejor conocido", fontsize=7.5)
ax[1].set_ylim(9000, 13600); ax[1].set_xlabel("evaluaciones"); ax[1].set_ylabel("mejor valor factible (mdp/año)")
ax[1].set_title("Convergencia (mediana de 30 semillas)")
leyenda = [Patch(color=AZUL, label="binomial"), Patch(color=NARANJA, label="exponencial"), Patch(color=VERDE, label="aritmética")]
leyenda += [plt.Line2D([], [], color=GRIS, ls=estilo[k], label=ETQ[k]) for k in estilo]
ax[1].legend(handles=leyenda, frameon=False, fontsize=6.8, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "fig4_distribucion_convergencia.png"); plt.close(fig)

# ---- Figura 5: efectos principales y restricciones de la solución elegida
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.4))
for k, (factor, niveles, col) in enumerate([("mutacion", ("rand1", "best1", "ctb1"), AZUL), ("cruce", ("bin", "exp", "arit"), NARANJA)]):
    medias, ics = [], []
    for nv in niveles:
        v = de[de[factor] == nv].brecha_pct.values
        medias.append(np.median(v)); ics.append(np.quantile(v, [0.25, 0.75]))
    xs = np.arange(3)
    ax[k].bar(xs, medias, color=col, alpha=0.8, width=0.55)
    for xi, (lo, hi) in zip(xs, ics):
        ax[k].plot([xi, xi], [lo, hi], color=TINTA, lw=1)
    ax[k].set_xticks(xs); ax[k].set_xticklabels([ETQ[n] for n in niveles], fontsize=8)
    ax[k].set_ylabel("brecha mediana (%)"); ax[k].set_title(f"Efecto de la {factor}\n(barras: mediana; líneas: IQR)", fontsize=9)
# restricciones de la solución elegida
cartera = pd.read_csv(RES / "cartera_E-base_elegida.csv")
c = generar_cartera("E-base")
x = cartera.inversion_mdp.values.reshape(R, M)
uso = (x * c.uso).sum(axis=1) / c.agua
tope = x.sum(axis=1) / (c.tope * c.presupuesto)
eq = x[3].sum() / (c.equidad * c.presupuesto)
emp = cartera.empleos.sum() / c.empleo_min
etiquetas = ["agua", "tope región\n(máx. Z-T)", "equidad VM", "empleo"]
valores = [uso.max(), tope.max(), eq, emp]
ax[2].barh(range(4), valores, color=[NARANJA if v > 0.97 else VERDE for v in valores], height=0.55)
ax[2].axvline(1, color=TINTA, lw=1, ls="--")
ax[2].set_yticks(range(4)); ax[2].set_yticklabels(etiquetas, fontsize=8); ax[2].invert_yaxis()
ax[2].set_xlabel("uso relativo del límite (1 = activo)"); ax[2].set_xlim(0, 1.25)
ax[2].set_title("Restricciones de la cartera elegida\n(mejor corrida, E-base)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig5_efectos_restricciones.png"); plt.close(fig)

# ---- Figura 6: cartera elegida frente al reparto proporcional
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5), gridspec_kw={"width_ratios": [1, 1.1]})
rp = reparto_proporcional(c)["x"].reshape(R, M)
cortos = ["Z-T", "PM", "TT", "VM"]
for k, (mat, titulo) in enumerate([(rp, "Reparto proporcional a la capacidad"), (x, "Cartera elegida (DE, mejor corrida)")]):
    ax_ = ax[0] if k == 0 else ax[1]
    bot = np.zeros(R)
    for m in range(M):
        ax_.bar(range(R), mat[:, m], bottom=bot, color=COL_MERC[m], width=0.6, label=MERCADOS[m] if k == 1 else None)
        bot += mat[:, m]
    ax_.set_xticks(range(R)); ax_.set_xticklabels(cortos); ax_.set_ylabel("inversión (mdp)")
    ax_.set_title(titulo, fontsize=9)
    ax_.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f} mil"))
ax[1].legend(frameon=False, fontsize=7, loc="upper right")
fig.text(0.5, -0.03, "Z-T: Zapotlán–Tizayuca · PM: Pachuca–Mineral de la Reforma · TT: Tula–Tepeji · VM: Valle del Mezquital",
         ha="center", fontsize=7.5, color=GRIS)
fig.tight_layout(); fig.savefig(FIG / "fig6_cartera_vs_reparto.png"); plt.close(fig)
print("figuras:", sorted(p.name for p in FIG.glob("*.png")))
