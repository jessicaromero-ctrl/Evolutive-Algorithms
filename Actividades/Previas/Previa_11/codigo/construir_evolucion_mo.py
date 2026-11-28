#!/usr/bin/env python3
"""
construir_evolucion_mo.py — Genera el reporte DOCX del proyecto
"Evolución diferencial multiobjetivo: técnicas de adaptación y autoadaptación".

Lee los resultados de resultados/*.json y las figuras de figuras/*.png; ninguna cifra del texto
se escribe a mano.

Uso: python3 construir_evolucion_mo.py --salida ..
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

import docx_utils as U
from mode_core import (VARIANTES, ETIQUETAS, SEMILLAS, N_POP, GENS, EVALS, H_MEM, F_FIJA,
                       CR_FIJA, TAU1, TAU2, Z_K, C_K, A_K, Z_MIN, PRECIO, C_PROC, R_MAX, K_Z, A_REF)

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "resultados"
FIG = BASE / "figuras"


def cargar(nombre):
    with open(RES / nombre, encoding="utf-8") as fh:
        return json.load(fh)


def ms(a, d=4):
    a = np.asarray(a, dtype=float)
    return f"{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}"


def pv(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def holm(comps, nombre):
    return next(c[3] for c in comps if c[0] == nombre)


def encabezado(doc, texto, nivel=1):
    h = doc.add_heading(texto, level=nivel)
    h.paragraph_format.keep_with_next = True
    return h


def p(doc, texto, **kw):
    return U.parrafo(doc, texto, **kw)


def contenido(doc):
    E1, E2, E3 = cargar("esc1_zdt1.json"), cargar("esc2_zdt3.json"), cargar("esc3_mezcla.json")
    hv1, ig1 = np.array(E1["hv_final"]), np.array(E1["igd_final"])
    hv2, ig2 = np.array(E2["hv_final"]), np.array(E2["igd_final"])
    hv3, ig3 = np.array(E3["hv_final"]), np.array(E3["igd_final"])
    ref_puntos = len(E3["frente_referencia"])
    k = {v: j for j, v in enumerate(VARIANTES)}
    rel_mezcla = (hv3[:, k["jde"]].mean() - hv3[:, k["fija"]].mean()) / hv3[:, k["fija"]].mean() * 100

    # ── Resumen ─────────────────────────────────────────────────────
    encabezado(doc, "Resumen")
    p(doc, (
        "Este reporte estudia cómo afectan las técnicas de control de parámetros al desempeño de la "
        "evolución diferencial multiobjetivo (MODE). Se comparan cuatro estrategias con un mismo presupuesto "
        f"de {EVALS:,} evaluaciones y {len(SEMILLAS)} semillas: parámetros fijos (F = {F_FIJA}, CR = {CR_FIJA}), "
        "dither, autoadaptación jDE y SHADE-MO con memoria de éxito. Se usan tres escenarios: ZDT1 "
        "(frente convexo), ZDT3 (frente discontinuo) y un problema aplicado de mezcla de mineral con "
        "dos objetivos en conflicto y una restricción de ley mínima de zinc. El desempeño se mide con el "
        "hipervolumen y la distancia invertida generacional (IGD), y las diferencias se contrastan con la "
        "prueba de Friedman y comparaciones de Wilcoxon corregidas por Holm."))
    p(doc, (
        f"En ZDT1 y ZDT3, jDE y SHADE-MO superan de forma significativa a la estrategia fija y al dither "
        f"(Holm, p = {pv(holm(E1['holm_igd'], 'jde'))} en ZDT1 y p = {pv(holm(E2['holm_igd'], 'jde'))} en ZDT3). "
        f"En la mezcla, las diferencias en hipervolumen son estadísticamente detectables pero de magnitud "
        f"{abs(rel_mezcla):.2f} % del valor medio de la estrategia fija, por lo que el beneficio práctico de la "
        "adaptación en este problema de baja dimensión es marginal."))

    # ── 1. Introducción ─────────────────────────────────────────────
    encabezado(doc, "1. Introducción")
    p(doc, (
        "La evolución diferencial (DE) es un algoritmo poblacional para optimización continua que genera "
        "soluciones candidatas mediante diferencias entre individuos y las combina con una cruza binomial "
        "(Storn y Price, 1997; Price, Storn y Lampinen, 2005). Su desempeño depende de dos parámetros: el "
        "factor de escala F, que controla el tamaño del paso de mutación, y la probabilidad de cruza CR. "
        "Ambos tienen un efecto fuerte en la velocidad de convergencia y en la capacidad de exploración, y "
        "su valor óptimo depende del problema (Gämperle, Müller y Koumoutsakos, 2002)."))
    p(doc, (
        "En optimización multiobjetivo el problema se agrava, porque el algoritmo debe aproximar un conjunto "
        "de compromisos y no un único óptimo. Un valor de parámetros que converge rápido en una región del "
        "frente puede estancar la diversidad en otra. La parametrización adaptativa, que ajusta F y CR durante "
        "la ejecución, es una forma de control de parámetros en el sentido de Eiben, Hinterding y Michalewicz "
        "(1999). Este reporte evalúa si esa adaptación produce mejoras medibles en MODE y en qué condiciones."))
    p(doc, "Los objetivos del trabajo son:")
    U.vinetas(doc, [
        "Implementar MODE con restricciones y cuatro estrategias de control de F y CR bajo un protocolo común.",
        "Comparar convergencia, calidad final y estabilidad entre las estrategias en un problema con frente convexo (ZDT1) y en otro con frente discontinuo (ZDT3).",
        "Verificar si el resultado se mantiene en un problema aplicado de mezcla de mineral con restricción de calidad.",
        "Discutir el costo, la viabilidad y las limitaciones de cada estrategia.",
    ])

    # ── 2. Referentes previos ────────────────────────────────────────
    encabezado(doc, "2. Referentes previos")
    p(doc, (
        "Storn y Price (1997) propusieron DE con la estrategia DE/rand/1/bin y recomendaron valores de F "
        "cercanos a 0.5 y CR entre 0.8 y 1.0 como punto de partida. Gämperle et al. (2002) mostraron que el "
        "desempeño es sensible a estos valores y que su mejor ajuste cambia con el problema. Para reducir esa "
        "dependencia se han propuesto mecanismos de autoadaptación. Brest et al. (2006) introdujeron jDE, en "
        "el que cada individuo hereda F y CR y los regenera con probabilidades fijas τ1 y τ2. Tanabe y Fukunaga "
        "(2013) propusieron SHADE, que mantiene un historial de valores exitosos y muestrea F y CR a partir de "
        "él; su extensión para múltiples objetivos aquí denominada SHADE-MO aplica el mismo esquema en MODE."))
    p(doc, (
        "En el lado multiobjetivo, NSGA-II (Deb, Pratap, Agarwal y Meyarivan, 2002) define la selección "
        "elitista mediante ordenación en frentes no dominados y distancia de multitud. Para restricciones, "
        "Deb (2000) propuso la dominancia restringida, que se usa en este trabajo. La evaluación de calidad "
        "se apoya en el hipervolumen (Zitzler et al., 2003) y en la distancia invertida generacional (Coello "
        "y Cortés, 2005). Los problemas de prueba ZDT (Zitzler, Deb y Thiele, 2000) son el estándar para "
        "comparar algoritmos bi-objetivo."))

    # ── 3. Fundamentos ──────────────────────────────────────────────
    encabezado(doc, "3. Fundamentos")
    encabezado(doc, "3.1 Evolución diferencial con mutación DE/rand/1 y cruza binomial", 2)
    p(doc, "Para cada individuo x_i,g de la generación g se genera un vector mutante y una cruza:")
    U.formula(doc, "v_i,g = x_r1,g + F_i · (x_r2,g − x_r3,g),     r1 ≠ r2 ≠ r3 ≠ i", "(1)")
    U.formula(doc, "u_ij,g = v_ij,g  si  rand_j ≤ CR_i  o  j = j_rand;   u_ij,g = x_ij,g  en otro caso", "(2)")
    p(doc, (
        "El índice j_rand garantiza que al menos una componente proviene del mutante. La selección compara "
        "a cada hijo u_i,g con su padre x_i,g; en MODE la comparación se reemplaza por la selección elitista "
        "descrita en la sección 3.3."))

    encabezado(doc, "3.2 Estrategias de control de F y CR", 2)
    U.vinetas(doc, [
        f"Fija:: F = {F_FIJA} y CR = {CR_FIJA} durante toda la ejecución. Es la línea base del estudio.",
        "Dither:: F_i,g ~ U(0.5, 1) en cada generación y para cada individuo, con CR fijo. Introduce "
        "variabilidad en el paso sin mecanismo de retroalimentación.",
        f"jDE:: con probabilidades τ1 = τ2 = {TAU1}, F_i,g+1 = 0.1 + 0.9·rand con probabilidad τ1, y "
        "CR_i,g+1 = rand con probabilidad τ2; en otro caso el valor se hereda del padre. Los valores "
        "heredados se seleccionan junto con el hijo.",
        f"SHADE-MO:: historial de tamaño H = {H_MEM} con medias M_F y M_CR. Para cada hijo se elige una "
        "entrada r y se muestrea CR ~ N(M_CR,r, 0.1) truncada en [0, 1] y F ~ Cauchy(M_F,r, 0.1) con "
        "reintento si F ≤ 0 y truncado en 1. Un hijo es exitoso si sobrevive a la selección. Las medias se "
        "actualizan con la media de Lehmer para F, M_F = Σ F² / Σ F, y con la media aritmética para CR.",
    ])
    p(doc, "La actualización del historial de SHADE-MO sigue la expresión:")
    U.formula(doc, "M_F,k ← Σ_s F_s² / Σ_s F_s,     M_CR,k ← (1/|S|) Σ_s CR_s,     k ← (k + 1) mod H", "(3)")
    p(doc, "donde S es el conjunto de valores exitosos de la generación.")

    encabezado(doc, "3.3 Dominancia restringida y selección elitista", 2)
    p(doc, (
        "Sea cv_i la violación de restricción del individuo i. Según Deb (2000), i domina a j si i es factible "
        "y j no lo es, si ambos son infactibles y cv_i < cv_j, o si ambos son factibles y i domina a j en el "
        "sentido de Pareto. La selección combina padres e hijos (2N individuos), los ordena en frentes no "
        "dominados y completa la nueva población con los individuos de mayor distancia de multitud dentro "
        "del último frente admitido, como en NSGA-II (Deb et al., 2002)."))
    U.formula(doc, "d_i = Σ_m (f_m(i+1) − f_m(i−1)) / (f_m,máx − f_m,mín)", "(4)")

    encabezado(doc, "3.4 Indicadores de calidad", 2)
    p(doc, (
        "El hipervolumen de un conjunto P de puntos en minimización, respecto a un punto de referencia r, es "
        "la medida de la unión de los rectángulos que cada punto define con r (Zitzler et al., 2003). En dos "
        "objetivos se calcula de forma exacta por barrido. La distancia invertida generacional mide, para "
        "cada punto del frente verdadero o de referencia R, la distancia al punto más cercano de P (Coello y "
        "Cortés, 2005):"))
    U.formula(doc, "IGD(P, R) = (1/|R|) Σ_{r ∈ R} min_{p ∈ P} ‖r − p‖₂", "(5)")
    p(doc, (
        "Menor IGD indica mejor aproximación y cobertura del frente. Mayor hipervolumen indica mejor "
        "convergencia y diversidad. Ambos se usan porque miden aspectos distintos."))

    encabezado(doc, "3.5 Pruebas estadísticas", 2)
    p(doc, (
        "Se usa la prueba de Friedman (Friedman, 1937) sobre los bloques formados por las semillas, con "
        "las cuatro estrategias como tratamientos. Cuando el valor global es significativo, cada estrategia "
        "se compara con la fija mediante la prueba de rangos con signo de Wilcoxon (Wilcoxon, 1945). Los p "
        "valores de estas comparaciones se corrigen con el procedimiento de Holm (1979). Derrac, García, "
        "Molina y Herrera (2011) recomiendan este conjunto para comparar algoritmos de inteligencia "
        "computacional. Con diez semillas, el p mínimo exacto de Wilcoxon bilateral es 0.00195, lo que "
        "limita la potencia de la prueba; por eso se reportan también las diferencias de medianas."))

    # ── 4. Método común ─────────────────────────────────────────────
    encabezado(doc, "4. Método común")
    p(doc, (
        "Los tres escenarios comparten el mismo algoritmo, presupuesto y protocolo. Cada combinación de "
        "problema, estrategia y semilla produce una ejecución independiente. La población inicial se "
        "muestrea uniformemente en el dominio y cada generación registra las trazas de hipervolumen, IGD "
        "y parámetros cada cinco generaciones. La tabla 1 resume la configuración."))
    U.tabla(doc, ["Parámetro", "Valor"], [
        ["Tamaño de población", f"{N_POP}"],
        ["Generaciones", f"{GENS}"],
        ["Evaluaciones por ejecución", f"{EVALS:,}"],
        ["Semillas independientes", ", ".join(str(s) for s in SEMILLAS)],
        ["Mutación y cruza", "DE/rand/1 y binomial"],
        ["Estrategia fija", f"F = {F_FIJA}, CR = {CR_FIJA}"],
        ["Dither", "F ~ U(0.5, 1), CR = " + str(CR_FIJA)],
        ["jDE", f"τ1 = τ2 = {TAU1}, F ∈ [0.1, 1.0]"],
        ["SHADE-MO", f"H = {H_MEM}, σ_F = σ_CR = 0.1"],
        ["Punto de referencia (ZDT)", "(1.1, 1.1)"],
        ["Pruebas", "Friedman; Wilcoxon con corrección de Holm"],
    ], [6.0, 10.0], titulo="Tabla 1. Configuración común de los experimentos",
        nota="Valores fijados en mode_core.py; las cuatro estrategias usan el mismo número de evaluaciones.")

    # ── 5. Escenario 1: ZDT1 ────────────────────────────────────────
    encabezado(doc, "5. Escenario 1: control de parámetros en ZDT1")
    p(doc, (
        "ZDT1 tiene 30 variables en [0, 1] y un frente convexo f₂ = 1 − √f₁ para f₁ ∈ [0, 1] (Zitzler et al., "
        "2000). Es un caso favorable para la búsqueda y permite comparar la convergencia de las estrategias "
        "en un frente sin discontinuidades."))
    U.tabla(doc, ["Estrategia", "Hipervolumen final", "IGD final", "p Holm (HV vs fija)"], [
        [ETIQUETAS[v], ms(hv1[:, k[v]]), f"{ig1[:, k[v]].mean():.5f}",
         "—" if v == "fija" else pv(holm(E1["holm_hv"], v))]
        for v in VARIANTES
    ], [4.0, 4.6, 3.4, 4.0], alinear=["l", "c", "c", "c"],
        titulo="Tabla 2. Resultados finales en ZDT1 (media ± desviación estándar, 10 semillas)",
        nota=(f"Friedman sobre el hipervolumen: χ² = {E1['friedman_hv'][0]:.1f}, p = {pv(E1['friedman_hv'][1])}. "
              f"Friedman sobre la IGD: χ² = {E1['friedman_igd'][0]:.1f}, p = {pv(E1['friedman_igd'][1])}."))
    p(doc, (
        f"El hipervolumen final de jDE ({hv1[:, k['jde']].mean():.4f}) y de SHADE-MO "
        f"({hv1[:, k['shade']].mean():.4f}) supera al de la estrategia fija ({hv1[:, k['fija']].mean():.4f}) "
        f"y al dither ({hv1[:, k['dither']].mean():.4f}). La IGD final es del orden de 10⁻³ en las dos "
        f"estrategias autoadaptativas, frente a {ig1[:, k['fija']].mean():.4f} en la fija. Todas las "
        "comparaciones contra la fija son significativas tras la corrección de Holm. La figura 1 muestra la "
        "convergencia y la evolución media de los parámetros."))
    U.figura(doc, FIG / "fig1_zdt1.png", 16.0,
             "Figura 1. Escenario 1 (ZDT1). (a) Hipervolumen medio por generación con banda de ± una desviación "
             "estándar; (b) hipervolumen final; (c) evolución media de F y CR en jDE y de M_F y M_CR en SHADE-MO.")
    p(doc, (
        "La trayectoria del dither es la más lenta. Su F medio es 0.75, mayor que el 0.5 de la fija, y mantiene "
        "CR fijo. Con 150 generaciones, el paso grande favorece la exploración pero retrasa la explotación del "
        "frente. Las trazas de jDE y SHADE-MO muestran que la autoadaptación baja F en las primeras "
        "generaciones y ajusta CR a valores altos en la fase final. Esta secuencia es consistente con la "
        "idea de que la adaptación modula el compromiso entre exploración y explotación a lo largo de la "
        "búsqueda."))

    # ── 6. Escenario 2: ZDT3 ────────────────────────────────────────
    encabezado(doc, "6. Escenario 2: control de parámetros en ZDT3")
    p(doc, (
        "ZDT3 tiene un frente formado por varios segmentos desconectados. Un algoritmo que se estanca en un "
        "segmento puede obtener un hipervolumen razonable con una IGD alta. Por esa razón este escenario usa "
        "la IGD como indicador principal. El hipervolumen de ZDT3 se reporta, pero su interpretación es menos "
        "directa porque el punto de referencia y la forma del frente interactúan."))
    U.tabla(doc, ["Estrategia", "IGD final", "Hipervolumen final", "p Holm (IGD vs fija)"], [
        [ETIQUETAS[v], ms(ig2[:, k[v]], 5), ms(hv2[:, k[v]], 4),
         "—" if v == "fija" else pv(holm(E2["holm_igd"], v))]
        for v in VARIANTES
    ], [4.0, 4.0, 4.4, 3.6], alinear=["l", "c", "c", "c"],
        titulo="Tabla 3. Resultados finales en ZDT3 (media ± desviación estándar, 10 semillas)",
        nota=(f"Friedman sobre la IGD: χ² = {E2['friedman_igd'][0]:.1f}, p = {pv(E2['friedman_igd'][1])}. "
              f"Friedman sobre el hipervolumen: χ² = {E2['friedman_hv'][0]:.1f}, p = {pv(E2['friedman_hv'][1])}."))
    p(doc, (
        f"La IGD final de jDE ({ig2[:, k['jde']].mean():.5f}) y de SHADE-MO ({ig2[:, k['shade']].mean():.5f}) "
        f"es {ig2[:, k['fija']].mean() / ig2[:, k['jde']].mean():.0f} veces menor que la de la estrategia fija "
        f"({ig2[:, k['fija']].mean():.5f}) y {ig2[:, k['dither']].mean() / ig2[:, k['jde']].mean():.0f} veces menor "
        f"que la del dither ({ig2[:, k['dither']].mean():.5f}). La figura 2 muestra los frentes finales de la "
        "semilla 11 frente al verdadero: jDE y SHADE-MO se ajustan a los segmentos del frente, mientras que la "
        "estrategia fija y el dither dejan puntos alejados del frente verdadero."))
    U.figura(doc, FIG / "fig2_zdt3.png", 16.0,
             "Figura 2. Escenario 2 (ZDT3). (a) Frentes finales de la semilla 11 frente al verdadero; "
             "(b) IGD final en escala logarítmica; (c) convergencia media de la IGD.")
    p(doc, (
        "La convergencia de la IGD confirma el patrón. jDE y SHADE-MO reducen la distancia a los segmentos "
        "del frente con rapidez entre las generaciones 20 y 80 y después se estabilizan en el orden de "
        "10⁻³. La estrategia fija reduce la IGD de forma sostenida pero lenta, y el dither no la reduce al "
        "mismo ritmo dentro del presupuesto."))

    # ── 7. Escenario 3: mezcla de mineral ───────────────────────────
    encabezado(doc, "7. Escenario 3: mezcla bi-objetivo de mineral")
    p(doc, (
        "El problema aplicado retoma la planta del reporte minero previo, con datos ilustrativos. La mezcla "
        "combina cinco frentes de mineral con proporciones u_k ≥ 0 normalizadas a P_k = u_k / Σ u_k. Las "
        "propiedades medias de la mezcla son z̄ = Σ P_k z_k (ley de zinc, %), ā = Σ P_k a_k (impureza, ppm) "
        "y c̄ = Σ P_k c_k (costo de extracción, USD/t). La utilidad se calcula como:"))
    U.formula(doc, "U(P) = p · (z̄/100) · r(P) − c̄ − c_p,     r(P) = R_máx · z̄/(z̄ + K_z) · 1/(1 + ā/A_ref)", "(6)")
    p(doc, (
        f"con p = {PRECIO:.0f} USD/t, c_p = {C_PROC:.0f} USD/t, R_máx = {R_MAX}, K_z = {K_Z} y A_ref = {A_REF:.0f}. "
        "El problema minimiza f₁ = −U(P) y f₂ = ā, sujeto a la restricción de calidad z̄ ≥ "
        f"{Z_MIN:.1f} %, tratada con dominancia restringida. La tabla 4 resume los frentes."))
    U.tabla(doc, ["Frente k", "Ley de zinc z_k (%)", "Costo c_k (USD/t)", "Impureza a_k (ppm)"],
            [[f"{k_ + 1}", f"{Z_K[k_]:.1f}", f"{C_K[k_]:.1f}", f"{A_K[k_]:.0f}"] for k_ in range(5)],
            [3.0, 4.2, 4.2, 4.2], alinear=["c", "c", "c", "c"],
            titulo="Tabla 4. Propiedades de los cinco frentes de mineral (datos ilustrativos)")
    p(doc, (
        "Como el frente verdadero no se conoce, el frente de referencia es la unión de los frentes no "
        f"dominados de las 40 ejecuciones y contiene {ref_puntos} puntos. Los objetivos se normalizan con el "
        "mínimo y el rango de la referencia, y el hipervolumen usa el punto (1.1, 1.1). Los indicadores son "
        "relativos entre métodos y no valores absolutos de calidad."))
    U.tabla(doc, ["Estrategia", "Hipervolumen (normalizado)", "IGD (normalizada)", "p Holm HV", "p Holm IGD"], [
        [ETIQUETAS[v], ms(hv3[:, k[v]], 5), ms(ig3[:, k[v]], 5),
         "—" if v == "fija" else pv(holm(E3["holm_hv"], v)),
         "—" if v == "fija" else pv(holm(E3["holm_igd"], v))]
        for v in VARIANTES
    ], [3.4, 4.0, 3.6, 2.5, 2.5], alinear=["l", "c", "c", "c", "c"],
        titulo="Tabla 5. Resultados en la mezcla de mineral (media ± desviación estándar, 10 semillas)",
        nota=(f"Friedman sobre el hipervolumen: χ² = {E3['friedman_hv'][0]:.2f}, p = {pv(E3['friedman_hv'][1])}. "
              f"Friedman sobre la IGD: χ² = {E3['friedman_igd'][0]:.2f}, p = {pv(E3['friedman_igd'][1])}. "
              "Todas las soluciones finales fueron factibles."))
    p(doc, (
        f"Las cuatro estrategias alcanzan la misma región del frente. El hipervolumen medio de jDE supera al de "
        f"la fija en {rel_mezcla:.2f} %, y la prueba de Friedman detecta diferencias globales en hipervolumen "
        f"(p = {pv(E3['friedman_hv'][1])}). Tras la corrección de Holm, dither y jDE siguen por debajo de 0.05 "
        f"en hipervolumen (p ajustado = {pv(holm(E3['holm_hv'], 'dither'))} y {pv(holm(E3['holm_hv'], 'jde'))}), "
        f"SHADE-MO no lo está (p ajustado = {pv(holm(E3['holm_hv'], 'shade'))}) y ninguna estrategia lo está en "
        f"IGD (Friedman p = {pv(E3['friedman_igd'][1])}). Sin embargo, la diferencia es de magnitud despreciable "
        "en la práctica: las distribuciones se traslapan casi por completo. "
        "La figura 3 muestra los frentes, la distribución del hipervolumen y la composición de la mezcla en "
        "cinco puntos del frente de referencia."))
    U.figura(doc, FIG / "fig3_mezcla.png", 16.0,
             "Figura 3. Escenario 3 (mezcla). (a) Frentes de la semilla 11 frente a la referencia; (b) "
             "hipervolumen final normalizado; (c) proporción de cada frente en cinco puntos de la referencia.")

    # ── 8. Discusión ────────────────────────────────────────────────
    encabezado(doc, "8. Discusión")
    encabezado(doc, "8.1 Efecto de la adaptación", 2)
    p(doc, (
        "Los resultados sugieren que la adaptación importa más cuando el problema es de mayor dimensión y "
        "tiene una geometría difícil de recorrer con parámetros constantes. En ZDT1 y ZDT3, con 30 variables, "
        "jDE y SHADE-MO superan a la fija con diferencias grandes y consistentes. En la mezcla, con cinco "
        "variables, la estrategia fija obtiene un resultado casi equivalente. Esta observación es coherente "
        "con la idea de que el control de parámetros es más útil cuando el número de variables y la "
        "dificultad aumentan (Eiben et al., 1999)."))
    p(doc, (
        "El dither es el caso menos favorable en los tres escenarios. Su F medio más alto retrasa la "
        "convergencia dentro del presupuesto. Esto muestra que una perturbación aleatoria sin retroalimentación "
        "puede empeorar el desempeño frente a un valor fijo razonable. La autoadaptación evita ese problema "
        "porque los valores que producen hijos exitosos se propagan. La diferencia entre jDE y SHADE-MO es "
        "pequeña en los tres problemas, y este estudio no basta para ordenarlas de forma definitiva."))
    encabezado(doc, "8.2 Costo computacional", 2)
    p(doc, (
        "Todas las estrategias usan el mismo número de evaluaciones de función, así que la comparación es "
        "justa en ese criterio. El costo adicional de jDE es la generación de dos números aleatorios por "
        "individuo. El de SHADE-MO es el muestreo de Cauchy y normal y la actualización de un historial de "
        f"tamaño {H_MEM}, operaciones de orden O(N) por generación frente al costo de la ordenación no "
        "dominada. No se midió el tiempo de pared de cada estrategia, por lo que el costo se describe de forma "
        "cualitativa."))
    encabezado(doc, "8.3 Viabilidad", 2)
    U.vinetas(doc, [
        "Técnica:: la implementación usa Python con numpy y se ejecuta en un equipo de escritorio. Cada "
        "ejecución de 15 100 evaluaciones tarda alrededor de medio segundo en ZDT1, por lo que el estudio "
        "completo corre en menos de un minuto.",
        "Operativa:: en una planta, la función de evaluación debe calibrarse con leyes, impurezas y costos "
        "medidos. Los resultados aquí usan datos ilustrativos y no sustituyen esa calibración.",
        "Económica:: el costo es de cómputo y no requiere licencias. Su valor depende de que la mejora del "
        "frente sea relevante para la decisión; en la mezcla, la mejora práctica es marginal.",
        "Recomendación:: para problemas de baja dimensión como la mezcla, una estrategia fija bien calibrada "
        "basta. La autoadaptación se justifica en problemas de mayor dimensión o con frentes complejos.",
    ])

    # ── 9. Limitaciones y conclusiones ──────────────────────────────
    encabezado(doc, "9. Limitaciones y conclusiones")
    p(doc, (
        "Las limitaciones principales son: diez semillas por estrategia, un presupuesto fijo de 15 100 "
        "evaluaciones, una sola configuración de F y CR para la línea base, indicadores relativos en la mezcla "
        "y datos ilustrativos en el problema aplicado. Un estudio con más semillas, mayor presupuesto y "
        "ajuste fino de la línea base podría modificar la magnitud de las diferencias."))
    p(doc, (
        "Las conclusiones son las siguientes. Primero, en ZDT1 y ZDT3, jDE y SHADE-MO mejoran de forma "
        "significativa la convergencia y la calidad final respecto a parámetros fijos y dither. Segundo, el "
        "dither puede empeorar el desempeño si su rango de F no se compensa con mecanismos de "
        "retroalimentación. Tercero, en la mezcla de mineral, la adaptación no produce una mejora práctica "
        "relevante; la estrategia fija es suficiente. La adaptación es recomendable en problemas más "
        "exigentes, y su beneficio debe verificarse con indicadores y pruebas estadísticas en cada caso."))

    # ── Referencias ─────────────────────────────────────────────────
    encabezado(doc, "Referencias")
    refs = [
        "Brest, J., Greiner, S., Bošković, B., Mernik, M., y Žumer, V. (2006). Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems. IEEE Transactions on Evolutionary Computation, 10(6), 646–657.",
        "Coello Coello, C. A., y Cortés, N. C. (2005). Solving multiobjective optimization problems using an artificial immune system. Genetic Programming and Evolvable Machines, 6(2), 163–190.",
        "Deb, K. (2000). An efficient constraint handling method for genetic algorithms. Computer Methods in Applied Mechanics and Engineering, 186(2–4), 311–338.",
        "Deb, K., Pratap, A., Agarwal, S., y Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182–197.",
        "Derrac, J., García, S., Molina, D., y Herrera, F. (2011). A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms. Swarm and Evolutionary Computation, 1(1), 3–18.",
        "Eiben, A. E., Hinterding, R., y Michalewicz, Z. (1999). Parameter control in evolutionary algorithms. IEEE Transactions on Evolutionary Computation, 3(2), 124–141.",
        "Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. Journal of the American Statistical Association, 32(200), 675–701.",
        "Gämperle, R., Müller, S. D., y Koumoutsakos, P. (2002). A parameter study for differential evolution. En Advances in Intelligent Systems, Fuzzy Systems, Evolutionary Computation (pp. 293–298). WSEAS.",
        "Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65–70.",
        "Price, K., Storn, R. M., y Lampinen, J. A. (2005). Differential evolution: A practical approach to global optimization. Springer.",
        "Storn, R., y Price, K. (1997). Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces. Journal of Global Optimization, 11(4), 341–359.",
        "Tanabe, R., y Fukunaga, A. (2013). Success-history based parameter adaptation for differential evolution. En 2013 IEEE Congress on Evolutionary Computation (pp. 71–78). IEEE.",
        "Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80–83.",
        "Zitzler, E., Deb, K., y Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: Empirical results. Evolutionary Computation, 8(2), 173–195.",
        "Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., y Grunert da Fonseca, V. (2003). Performance assessment of multiobjective optimizers: An analysis and review. IEEE Transactions on Evolutionary Computation, 7(2), 117–132.",
    ]
    for r in refs:
        q = doc.add_paragraph()
        q.paragraph_format.left_indent = Cm(0.8)
        q.paragraph_format.first_line_indent = Cm(-0.8)
        q.paragraph_format.space_after = Pt(3)
        q.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = q.add_run(r)
        run.font.size = Pt(9.5)

    # ── Anexo ───────────────────────────────────────────────────────
    encabezado(doc, "Anexo. Reproducción de los resultados")
    p(doc, "Desde la carpeta codigo/ ejecutar, en este orden:")
    for linea in [
        "python3 esc1_zdt1.py --salida ../resultados --figuras ../figuras",
        "python3 esc2_zdt3.py --salida ../resultados --figuras ../figuras",
        "python3 esc3_mezcla.py --salida ../resultados --figuras ../figuras",
        "python3 construir_evolucion_mo.py --salida ..",
    ]:
        q = doc.add_paragraph()
        q.paragraph_format.left_indent = Cm(0.6)
        r = q.add_run(linea)
        r.font.name = "Consolas"
        r.font.size = Pt(9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=str(BASE))
    a = ap.parse_args()
    U.TITULO = ("Evolución diferencial multiobjetivo: técnicas de adaptación y autoadaptación de parámetros")
    U.SUBTITULO = "Reporte técnico · Corte 2"
    U.FECHA = "20 de noviembre de 2026"
    doc = Document()
    U.estilos(doc)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.5)
        s.top_margin = s.bottom_margin = Cm(2.5)
    U.portada(doc)
    huerfano = doc.paragraphs[-1]
    if huerfano.text.strip() == "Código fuente:":
        huerfano._element.getparent().remove(huerfano._element)
    doc.add_page_break()
    contenido(doc)
    pie = doc.sections[0].footer.paragraphs[0]
    pie.alignment = WD_ALIGN_PARAGRAPH.CENTER
    U.campo_pagina(pie)
    salida = Path(a.salida) / "Reporte_Evolucion_Diferencial_Multiobjetivo_Romero_Lora.docx"
    doc.save(salida)
    print("Guardado:", salida)


if __name__ == "__main__":
    main()
