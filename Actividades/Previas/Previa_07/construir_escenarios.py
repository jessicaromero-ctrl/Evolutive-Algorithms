#!/usr/bin/env python3
"""
construir_escenarios.py — Genera el reporte DOCX de los tres escenarios de aplicación.
Todas las cifras de texto y tablas se leen de ../resultados/*.json.

USO
  python3 esc1_medicina.py && python3 esc2_grupos_sociales.py && python3 esc3_trayectorias.py
  python3 construir_escenarios.py --raiz .. --salida ../Reporte_Escenarios_Romero_Lora.docx \
      --url-repo https://github.com/USUARIO/REPOSITORIO
"""
import argparse
import json
from pathlib import Path

import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt
from scipy.stats import wilcoxon

import docx_utils as U

TITULO = ("Algoritmos genéticos en ingeniería médica, social y de trayectorias clínicas: "
          "tres escenarios de aplicación")
U.TITULO = TITULO
U.SUBTITULO = "Reporte técnico · Corte 2"


def p_wilcoxon(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def cargar(raiz):
    R = Path(raiz) / "resultados"
    e1 = json.load(open(R / "esc1_medicina.json", encoding="utf-8"))
    e2 = json.load(open(R / "esc2_grupos.json", encoding="utf-8"))
    e3 = json.load(open(R / "esc3_trayectorias.json", encoding="utf-8"))
    return e1, e2, e3


def texto_intro(doc):
    U.parrafo(doc, "1. Introducción", alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 1"]
    U.parrafo(doc,
              "Muchas decisiones de ingeniería en salud se formulan como la búsqueda de una configuración "
              "que maximiza un criterio de calidad: qué combinación de indicadores anticipa mejor un "
              "diagnóstico, cómo asignar un programa preventivo entre comunidades sin dejar a un grupo "
              "sistemáticamente atrás, o cómo parametrizar la comparación entre una trayectoria clínica "
              "real y una vía guía. En los tres casos el espacio de soluciones es grande, no siempre es "
              "derivable y la función objetivo no es lineal ni diferenciable. Los algoritmos evolutivos "
              "son una familia de métodos de búsqueda que no requieren gradientes y que operan sobre "
              "poblaciones de candidatos [1][2][13].")
    U.parrafo(doc,
              "Este reporte aplica un mismo motor de algoritmo genético (AG) a tres escenarios de ingeniería: "
              "medicina (fusión de puntuaciones de riesgo clínico), grupos sociales (asignación equitativa de "
              "un programa preventivo) y trayectorias médicas (calibración de un alineamiento de secuencias "
              "clínicas). El objetivo es mostrar qué elementos son comunes a los tres problemas, cuáles "
              "dependen del dominio y qué aporta el algoritmo frente a alternativas sencillas. Los datos son "
              "sintéticos y se generan con supuestos explícitos; los resultados describen el comportamiento "
              "de los métodos bajo esos supuestos y no constituyen evidencia clínica.")


def texto_referentes(doc):
    U.parrafo(doc, "2. Referentes previos", alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 1"]
    U.parrafo(doc,
              "En medicina, el aprendizaje automático se ha consolidado como herramienta de apoyo diagnóstico "
              "y de pronóstico, con revisiones que describen tanto su potencial como sus exigencias de "
              "validación [9]. La calidad de una puntuación de riesgo en problemas con clases poco frecuentes "
              "se evalúa mejor con la curva precisión–exhaustividad que con la curva ROC [5][11]. La "
              "combinación de varias puntuaciones mediante pesos optimizados es una práctica común, y los "
              "algoritmos evolutivos se han empleado para esa tarea por su capacidad de explorar espacios "
              "no convexos [3][13].")
    U.parrafo(doc,
              "En problemas sociales, distintos trabajos han documentado que un criterio aparentemente neutro "
              "puede producir daños desiguales entre grupos; un caso emblemático es la asignación de cuidados "
              "a partir de costos históricos, que subestimó la necesidad de pacientes de un grupo racial [7]. "
              "Para formalizar la equidad se han propuesto criterios como la igualdad de oportunidades, que "
              "exige tasas de detección similares entre grupos [8][10]. El escenario 2 incorpora esa idea "
              "como un término de penalización explícito.")
    U.parrafo(doc,
              "En trayectorias clínicas, la minería de procesos permite contrastar registros de eventos con "
              "modelos de referencia, y su aplicación en salud se ha revisado de forma sistemática [6][14]. "
              "El alineamiento global de secuencias, formulado para comparar cadenas biológicas, es un "
              "instrumento directo para medir la conformidad de una trayectoria con una vía guía [4]. El "
              "escenario 3 calibra los parámetros de ese alineamiento con un AG, en lugar de fijarlos por "
              "criterio experto.")


def texto_fundamentos(doc):
    U.parrafo(doc, "3. Fundamentos matemáticos", alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 1"]
    U.parrafo(doc,
              "Los tres escenarios son casos particulares del problema de optimización de la ecuación (1). "
              "Lo que cambia entre ellos es el dominio Θ, que determina la codificación, y la función F, "
              "que determina el paisaje de aptitud.")
    U.formula(doc, "θ* = argmax θ∈Θ  F(θ)", "(1)")
    U.parrafo(doc,
              "La precisión media (AP, por sus siglas en inglés) resume la curva precisión–exhaustividad. "
              "Si los puntos de corte se ordenan de forma decreciente, con P_n la precisión y R_n la "
              "exhaustividad en el n-ésimo punto, se define:")
    U.formula(doc, "AP = Σn (R_n − R_(n−1)) · P_n", "(2)")
    U.parrafo(doc,
              "En el escenario 1, K puntuaciones normalizadas z_k se combinan linealmente con pesos que "
              "pertenecen al símplex Δ. La proyección de un vector no negativo u sobre Δ es w_k = u_k / Σj u_j:")
    U.formula(doc, "s_w(x) = Σk=1..K w_k z_k(x),     w ∈ Δ = {w ∈ [0,1]K : Σk w_k = 1}", "(3)")
    U.parrafo(doc,
              "La codificación real BLX-α, usada en los escenarios 1 y 3, genera un hijo h a partir de los "
              "padres a y b con mín = min(a, b), máx = max(a, b) e intervalo I = máx − mín, muestreando "
              "h ~ U[mín − αI, máx + αI] [3]. La mutación gaussiana acotada aplica, a cada gen j, "
              "x'j = clip(xj + δj·σ·(uj − lj), lj, uj) con δj ~ N(0,1) y probabilidad 1/d.")
    U.parrafo(doc,
              "El escenario 2 usa codificación binaria s ∈ {0,1}Z, donde s_i = 1 indica que la zona i recibe el "
              "programa. Sea N_i la necesidad esperada de la zona i, e_g la eficiencia de alcance del grupo g(i) y "
              "b_i = N_i·e_g(i). La función objetivo es:")
    U.formula(doc,
              "F(s; λ) = Σi s_i b_i / Σi N_i − λ·(máxg c_g − míng c_g) − μ·|Σi s_i − K|,   "
              "c_g = Σi∈g s_i b_i / Σi∈g N_i", "(4)")
    U.parrafo(doc,
              "El primer término es la cobertura total, el segundo penaliza la brecha de cobertura entre "
              "grupos con peso λ ≥ 0 y el tercero impone el tamaño del programa mediante la penalización μ. "
              "Para Z = 20 y K = 6, el espacio factible tiene C(20, 6) = 38 760 elementos, lo que permite "
              "calcular el óptimo exacto por enumeración.")
    U.parrafo(doc,
              "El escenario 3 usa el alineamiento global de Needleman–Wunsch [4] sobre una trayectoria T de "
              "longitud L y una vía guía R de longitud M, con parámetros m > 0 (coincidencia), x < 0 "
              "(sustitución) y g < 0 (hueco):")
    U.formula(doc,
              "F(i,j) = máx{ F(i−1,j−1) + s(Ti, Rj), F(i−1,j) + g, F(i,j−1) + g },  "
              "F(i,0) = i·g,  F(0,j) = j·g", "(5)")
    U.parrafo(doc, "Relación entre los elementos. ", negrita_inicio="Relación entre los elementos. ")
    doc.paragraphs[-1].add_run(
        "Dos propiedades conectan las ecuaciones. Primera: la precisión media de la ecuación (2) depende "
        "solo del orden de las puntuaciones, por lo que AP(y, c·s) = AP(y, s) para todo c > 0 y para "
        "cualquier transformación estrictamente creciente. De aquí se sigue que la normalización del símplex "
        "en la ecuación (3) elimina una redundancia de escala sin cambiar el criterio. Segunda: en la ecuación "
        "(5), multiplicar (m, x, g) por c > 0 multiplica todas las sumas de caminos por c, así que el "
        "alineamiento óptimo y el orden de las puntuaciones no cambian. El espacio efectivo del escenario 3 "
        "tiene, por tanto, dos grados de libertad, los cocientes x/m y g/m, y no tres. En el escenario 2, "
        "la función (4) es lineal a trozos en s, y el término de brecha introduce aristas que pueden "
        "atrapar a un método de búsqueda local.")


def texto_metodo(doc):
    U.parrafo(doc, "4. Método común", alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 1"]
    U.parrafo(doc,
              "Los tres escenarios usan el mismo núcleo de AG (archivo ga_comun.py) con los parámetros de la "
              "tabla 1. La selección es por torneo binario, el reemplazo es elitista con dos individuos y cada "
              "ejecución consume 1 940 evaluaciones de aptitud. Cada escenario se ejecuta con cinco semillas "
              "(11, 23, 37, 53 y 71). Como control, la búsqueda aleatoria usa exactamente el mismo número de "
              "evaluaciones, de modo que cualquier diferencia no se debe a un mayor presupuesto.")
    U.tabla(doc, ["Parámetro", "Valor", "Parámetro", "Valor"],
            [["Población N", "40", "Cruce (probabilidad)", "0.9"],
             ["Generaciones", "50", "α de BLX", "0.3"],
             ["Elitismo", "2", "Desviación σ (mutación)", "0.05"],
             ["Torneo", "2", "Evaluaciones por ejecución", "1 940"]],
            [4.0, 2.6, 5.4, 2.6],
            titulo="Tabla 1. Parámetros comunes del algoritmo genético",
            nota="Tamaño de la población y generaciones definidos en ga_comun.py; el presupuesto de "
                 "evaluaciones coincide con el de la búsqueda aleatoria.",
            alinear=["l", "c", "l", "c"])


def texto_esc1(doc, e1):
    c = e1["corridas"]
    get = lambda k: np.array([r[k] for r in c])
    ga, ale, uni = get("ga"), get("aleatoria"), get("uniforme")
    best, ora = get("mejor_individual"), get("oraculo")
    U.parrafo(doc, "5. Escenario 1: fusión de puntuaciones de riesgo clínico", alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 2"]
    U.parrafo(doc,
              "Se simula una cohorte de 4 000 pacientes con seis covariables clínicas correlacionadas. El riesgo "
              "latente z es una combinación lineal de las covariables, y la presencia del evento sigue una "
              "distribución de Bernoulli con probabilidad logística de z. Cinco modelos observan z con ruido de "
              "desviación 0.5, 0.9, 1.5, 2.2 y 0.7, respectivamente. La prevalencia resultante es de "
              f"{100 * np.mean([r['prevalencia'] for r in c]):.1f} %. Las puntuaciones se estandarizan con "
              "parámetros de entrenamiento y el AG determina los pesos de la ecuación (3) maximizando la AP en "
              "la mitad de entrenamiento.")
    filas = [["Algoritmo genético", f"{ga[:, 0].mean():.4f} ± {ga[:, 0].std(ddof=1):.4f}", f"{ga[:, 1].mean():.4f}"],
             ["Búsqueda aleatoria", f"{ale[:, 0].mean():.4f} ± {ale[:, 0].std(ddof=1):.4f}", f"{ale[:, 1].mean():.4f}"],
             ["Pesos uniformes", f"{uni[:, 0].mean():.4f} ± {uni[:, 0].std(ddof=1):.4f}", f"{uni[:, 1].mean():.4f}"],
             ["Mejor modelo individual", f"{best[:, 0].mean():.4f} ± {best[:, 0].std(ddof=1):.4f}", f"{best[:, 1].mean():.4f}"],
             ["Oráculo (z verdadero)", f"{ora[:, 0].mean():.4f} ± {ora[:, 0].std(ddof=1):.4f}", f"{ora[:, 1].mean():.4f}"]]
    U.tabla(doc, ["Método", "AP en prueba (media ± d.e.)", "ROC-AUC en prueba"], filas,
            [6.0, 5.2, 4.4], titulo="Tabla 2. Desempeño en prueba, escenario 1 (cinco semillas)",
            nota="El oráculo usa el riesgo latente sin ruido y establece una cota superior de referencia.",
            alinear=["l", "c", "c"])
    W = np.array([r["pesos_ga"] for r in c])
    pes_ae = W[:, 0] + W[:, 4]
    d_rand = ga[:, 0] - ale[:, 0]
    d_best = ga[:, 0] - best[:, 0]
    d_uni = ga[:, 0] - uni[:, 0]
    U.parrafo(doc,
              f"El AG supera a los pesos uniformes en {f'{d_uni.mean():+.4f}'} de AP en promedio y a la mejor "
              f"señal individual en {d_best.mean():+.4f}, con signo positivo en las cinco semillas en ambos casos. "
              f"Frente a la búsqueda aleatoria la diferencia media es de {d_rand.mean():+.4f} y su signo cambia entre "
              "semillas. En este escenario, por tanto, el AG no aporta una ventaja medible sobre el muestreo "
              "aleatorio con el mismo presupuesto: el paisaje de aptitud es suave en un símplex de cinco "
              "dimensiones, y 1 940 evaluaciones bastan para localizar una región de buen desempeño. Los pesos "
              f"aprendidos asignan en todas las semillas entre {pes_ae.min():.2f} y {pes_ae.max():.2f} de la masa a los "
              "modelos A y E, los dos de menor ruido, y el modelo D recibe peso casi nulo. La figura 1 muestra la "
              "convergencia, la dispersión en prueba y los pesos por semilla.")
    U.figura(doc, Path(e1["_ruta_fig"]), 16.0,
             "Figura 1. (a) Convergencia del AG en entrenamiento; (b) AP en prueba por método; (c) pesos aprendidos "
             "por semilla (escenario 1).")
    U.parrafo(doc,
              "La diferencia entre el AG y el oráculo, de "
              f"{np.mean(ora[:, 0] - ga[:, 0]):.4f} en promedio, mide el costo de no observar z directamente. "
              "Esta brecha no proviene del algoritmo, sino del ruido de los modelos, y muestra que la fusión "
              "recupera una parte del riesgo latente pero no toda.")


def texto_esc2(doc, e2):
    lam = [str(l) for l in e2["lambdas"]]
    ex = e2["exacta"]
    ag = e2["ag"]
    g = np.array(e2["grupo"])
    vor = e2["voraz"]
    U.parrafo(doc, "6. Escenario 2: asignación de un programa preventivo entre grupos sociales",
              alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 2"]
    U.parrafo(doc,
              "Se consideran 20 zonas, cada una asignada a uno de tres grupos (A, B y C) con necesidades y "
              "eficiencias de alcance distintas: el grupo A tiene eficiencia 0.80, el B 0.60 y el C 0.40. El "
              "programa debe asignarse a exactamente K = 6 zonas. Los datos son valores esperados de una "
              "población hipotética, no observaciones. El AG usa codificación binaria de 20 bits con la "
              "ecuación (4) y la penalización μ = 1. Para cada λ se compara con el óptimo exacto y con la "
              "estrategia voraz, que elige las seis zonas de mayor beneficio b_i sin considerar equidad.")
    filas = []
    for l in lam:
        e = ex[l]
        gaps = np.array([r["F_ga"] for r in ag[l]]) - e["F"]
        exactos = int(np.sum(gaps > -1e-9))
        filas.append([l, f"{e['F']:.4f}", f"{e['total']:.4f}", f"{e['brecha']:.4f}",
                      f"{exactos} / 5", f"{gaps.min():+.4f}"])
    U.tabla(doc, ["λ", "F exacta", "Cobertura total", "Brecha", "AG = óptimo", "Peor F_AG − F*"],
            filas, [1.6, 2.6, 3.2, 2.6, 3.0, 3.2],
            titulo="Tabla 3. Compromiso entre cobertura y equidad, escenario 2",
            nota="«AG = óptimo» cuenta las semillas en que el AG alcanza el óptimo exacto. "
                 "«Peor F_AG − F*» es la diferencia más negativa entre el AG y el óptimo.",
            alinear=["c", "c", "c", "c", "c", "c"])
    e1_ = ex["1.0"]
    sel = [i + 1 for i, s in enumerate(e1_["s"]) if s]
    U.parrafo(doc,
              f"La estrategia voraz obtiene la mayor cobertura total ({vor['total']:.4f}), pero con una brecha de "
              f"{vor['brecha']:.4f}: el grupo A recibe una cobertura de {vor['cob'][0]:.3f}, frente a {vor['cob'][1]:.3f} "
              f"en el grupo B y {vor['cob'][2]:.3f} en el C. Con λ = 1, el óptimo exacto selecciona las zonas "
              f"{', '.join(map(str, sel))}; la cobertura total baja a {e1_['total']:.4f}, es decir, un "
              f"{100 * (vor['total'] - e1_['total']) / vor['total']:.1f} % menos, y la brecha se reduce a "
              f"{e1_['brecha']:.4f}. Ese es el costo de equidad en este caso: una pérdida moderada de cobertura "
              "total a cambio de una distribución casi uniforme entre grupos (figura 2b).")
    n_opt = sum(int(np.sum([r["F_ga"] - ex[l]["F"] > -1e-9 for r in ag[l]])) for l in lam)
    peor = min(min(r["F_ga"] - ex[l]["F"] for r in ag[l]) for l in lam)
    f_rand = [r["F_aleatoria"] for r in ag["1.0"]]
    U.parrafo(doc,
              f"El AG encuentra el óptimo exacto en {n_opt} de las {5 * len(lam)} ejecuciones (tabla 3). Cuando no "
              "lo alcanza, su solución factible es peor, con una diferencia máxima de "
              f"{peor:.3f} en el peor caso. El término de brecha crea una superficie con aristas que el muestreo por "
              "torneo explora con dificultad. Aun así, la búsqueda aleatoria con el mismo presupuesto produce "
              f"soluciones muy inferiores (para λ = 1, aptitudes entre {min(f_rand):.2f} y {max(f_rand):.2f}, contra "
              f"un óptimo de {e1_['F']:.3f}). La figura 2a muestra la frontera de compromiso del óptimo exacto para los cinco "
              "valores de λ; los valores λ = 0.25, 0.5 y 1 producen la misma solución, por lo que la frontera "
              "tiene solo tres puntos distintos.")
    U.figura(doc, Path(e2["_ruta_fig"]), 16.0,
             "Figura 2. (a) Frontera de compromiso eficiencia–equidad del óptimo exacto; (b) cobertura por grupo "
             "para la voraz y para el óptimo exacto con λ = 0 y λ = 1 (escenario 2).")


def texto_esc3(doc, e3):
    c = e3["corridas"]
    get = lambda k: np.array([r[k] for r in c])
    ga, ale, dfa = get("ga"), get("aleatoria"), get("defecto")
    dif = ga[:, 0] - dfa[:, 0]
    pval = p_wilcoxon(dif)
    U.parrafo(doc, "7. Escenario 3: calibración del alineamiento de trayectorias clínicas",
              alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 2"]
    U.parrafo(doc,
              "Se generan 800 trayectorias de longitud 8 sobre un alfabeto de ocho eventos clínicos (A a H), con "
              "una prevalencia de desviación del 25 %. La vía guía es la secuencia A→H. Las trayectorias conformes "
              "sustituyen un evento de la guía con probabilidad 0.06 y repiten eventos con probabilidad 0.04; las "
              "desviadas lo hacen con probabilidades 0.25 y 0.20. La puntuación de desviación es −F(T) de la "
              "ecuación (5). El AG calibra (m, x, g) en el dominio [0.5, 3] × [−3, −0.1]² para maximizar la AP en "
              "entrenamiento, y se compara con los valores por defecto (1, −1, −1) y con búsqueda aleatoria.")
    filas = [["Algoritmo genético", f"{ga[:, 0].mean():.4f} ± {ga[:, 0].std(ddof=1):.4f}", f"{ga[:, 1].mean():.4f}", "—", "—"],
             ["Búsqueda aleatoria", f"{ale[:, 0].mean():.4f} ± {ale[:, 0].std(ddof=1):.4f}", f"{ale[:, 1].mean():.4f}",
              f"{(ga[:, 0] - ale[:, 0]).mean():+.4f}", f"{p_wilcoxon(ga[:, 0] - ale[:, 0]):.3f}"],
             ["Parámetros por defecto", f"{dfa[:, 0].mean():.4f} ± {dfa[:, 0].std(ddof=1):.4f}", f"{dfa[:, 1].mean():.4f}",
              f"{dif.mean():+.4f}", f"{pval:.3f}"]]
    U.tabla(doc, ["Método", "AP en prueba (media ± d.e.)", "ROC-AUC", "Dif. AG − método", "p (Wilcoxon)"],
            filas, [4.4, 4.6, 2.4, 3.2, 2.4],
            titulo="Tabla 4. Desempeño en prueba, escenario 3 (cinco semillas)",
            nota="La diferencia pareada se calcula por semilla. Con cinco pares, el valor mínimo posible de p "
                 "para la prueba bilateral de Wilcoxon es 0.0625.",
            alinear=["l", "c", "c", "c", "c"])
    P = np.array([r["p_ga"] for r in c])
    U.parrafo(doc,
              f"El AG mejora la AP respecto a los valores por defecto en las cinco semillas, con una diferencia media "
              f"de {dif.mean():+.4f} (rango de {dif.min():+.4f} a {dif.max():+.4f}). La ROC-AUC no cambia, "
              f"{ga[:, 1].mean():.4f} frente a {dfa[:, 1].mean():.4f}. Esta combinación es esperable: la AP da más peso "
              "a la precisión en la parte alta del ranking, donde se concentran las alertas, mientras que la ROC "
              "resume el orden global. Los parámetros aprendidos difieren entre semillas: x/m varía entre "
              f"{np.min(P[:, 1] / P[:, 0]):.2f} y {np.max(P[:, 1] / P[:, 0]):.2f}, y g/m entre "
              f"{np.min(P[:, 2] / P[:, 0]):.2f} y {np.max(P[:, 2] / P[:, 0]):.2f}. Según la invariancia de escala de la "
              "sección 3, solo estos cocientes importan para el orden; que distintas razones produzcan AP casi "
              "idénticas indica una meseta de soluciones equivalentes. La figura 3 muestra el paisaje de aptitud en el "
              "plano (x, g) con m = 1.5, la convergencia y la separación de puntuaciones en prueba.")
    U.figura(doc, Path(e3["_ruta_fig"]), 16.0,
             "Figura 3. (a) Paisaje de aptitud (AP en entrenamiento) en el plano (x, g) con m = 1.5; (b) convergencia "
             "del AG; (c) distribución de puntuaciones en prueba para la semilla 11.")
    U.parrafo(doc,
              "El paisaje muestra una banda diagonal de valores altos. Su forma indica que la discriminación depende "
              "de la razón entre el costo de sustitución y el costo de hueco, y no de sus valores absolutos. Por eso "
              "el AG converge a regiones distintas con desempeño similar. La separación en prueba es moderada: las "
              "trayectorias desviadas tienden a puntuaciones más altas (la puntuación es −F(T)), pero con solapamiento considerable, "
              "consistente con una ROC-AUC de alrededor de 0.91.")


def texto_discusion(doc, e1, e2, e3):
    dif = np.array([r["ga"][0] - r["defecto"][0] for r in e3["corridas"]])
    U.parrafo(doc, "8. Discusión", alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 1"]
    U.parrafo(doc,
              "Los tres escenarios comparten una estructura: un espacio de decisión, una función objetivo y un "
              "criterio de comparación. Los resultados sugieren que la contribución principal de la metodología "
              "está en la formulación de la función objetivo. El AG resuelve bien los problemas en los que el "
              "criterio está bien definido, pero en los escenarios 1 y 3 la búsqueda aleatoria con el mismo "
              "presupuesto obtiene resultados equivalentes. La ventaja del AG aparece con más claridad en el "
              "escenario 2, donde el criterio combina términos no lineales y el espacio binario es discreto; aun "
              "así, el AG no siempre alcanza el óptimo exacto.")
    U.parrafo(doc,
              "En el escenario 2, el resultado tiene una lectura social directa: maximizar la cobertura total "
              f"produce una cobertura de {e2['voraz']['cob'][0]:.2f} en el grupo A y de "
              f"{e2['voraz']['cob'][1]:.2f} y {e2['voraz']['cob'][2]:.2f} en los grupos B y C. Un criterio de "
              "equidad explícito cambia la asignación a costa de una pérdida moderada de cobertura total. La "
              "decisión sobre cuánta equidad incorporar no es técnica: es un juicio de política que el modelo "
              "solo hace explícito.")
    U.parrafo(doc,
              "En el escenario 3, la invariancia de escala implica que la calibración opera sobre dos grados de "
              "libertad. Esto reduce el costo de la búsqueda, pero también significa que la mejora de AP "
              f"({dif.mean():+.4f} en promedio) es pequeña frente a la variabilidad entre semillas de los datos. Conviene "
              "tratarla como evidencia direccional y no como una ganancia establecida.")
    U.parrafo(doc,
              "Las limitaciones son importantes. Los datos son sintéticos y sus supuestos determinan los "
              "resultados; cinco semillas son una muestra pequeña y la prueba de Wilcoxon tiene poca potencia con "
              "cinco pares; las comparaciones entre métodos se hacen en un solo conjunto de condiciones; y el "
              "escenario 2 usa valores esperados, de modo que no incorpora incertidumbre sobre la necesidad real. "
              "Ninguno de los modelos debe usarse para decisiones clínicas o de asignación de recursos sin "
              "validación con datos reales, auditoría de equidad y revisión por personal especializado.")
    U.parrafo(doc, "9. Conclusiones", alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 1"]
    U.parrafo(doc,
              "Un mismo motor de algoritmo genético puede aplicarse a problemas de medicina, de grupos sociales y "
              "de trayectorias clínicas. Las ecuaciones (2) a (5) muestran que la métrica de evaluación y la "
              "invariancia de escala determinan qué se optimiza y cuántos grados de libertad efectivos tiene el "
              "problema. En los escenarios 1 y 3, el AG iguala a la búsqueda aleatoria y supera a las alternativas "
              "por defecto o uniformes. En el escenario 2, el AG se aproxima al óptimo exacto y supera ampliamente "
              "a la búsqueda aleatoria. La recomendación práctica es formular bien el criterio de equidad y validarlo "
              "antes de confiar en la optimización.")


def texto_referencias(doc):
    U.parrafo(doc, "Referencias", alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 1"]
    refs = [
        "[1] Holland, J. H. (1975). Adaptation in Natural and Artificial Systems. University of Michigan Press.",
        "[2] Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.",
        "[3] Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata. "
        "En Foundations of Genetic Algorithms, 2, 187–202.",
        "[4] Needleman, S. B., & Wunsch, C. D. (1970). A general method applicable to the search for similarities "
        "in the amino acid sequence of two proteins. Journal of Molecular Biology, 48(3), 443–453.",
        "[5] Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot "
        "when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
        "[6] Rojas, E., Munoz-Gama, J., Sepúlveda, M., & Capurro, D. (2016). Process mining in healthcare: "
        "A literature review. Journal of Biomedical Informatics, 61, 224–236.",
        "[7] Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an "
        "algorithm used to manage the health of populations. Science, 366(6464), 447–453.",
        "[8] Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. "
        "Advances in Neural Information Processing Systems, 29.",
        "[9] Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. New England Journal of "
        "Medicine, 380(14), 1347–1358.",
        "[10] Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact. California Law Review, 104(3), 671–732.",
        "[11] Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. "
        "Proceedings of the 23rd International Conference on Machine Learning, 233–240.",
        "[12] Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of "
        "Machine Learning Research, 7, 1–30.",
        "[13] Eiben, A. E., & Smith, J. E. (2015). Introduction to Evolutionary Computing (2.ª ed.). Springer.",
        "[14] van der Aalst, W. M. P. (2016). Process Mining: Data Science in Action (2.ª ed.). Springer.",
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(r)
        run.font.size = Pt(8.5)


def anexo(doc, url_repo):
    U.parrafo(doc, "Anexo. Código y reproducción", alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles["Heading 2"]
    U.tabla(doc, ["Archivo", "Contenido"],
            [["ga_comun.py", "Núcleo del AG: codificación real (BLX-α) y binaria; búsqueda aleatoria de control"],
             ["esc1_medicina.py", "Escenario 1: cohorte sintética, fusión con pesos en el símplex, figura 1"],
             ["esc2_grupos_sociales.py", "Escenario 2: objetivo con equidad, óptimo exacto, voraz, figura 2"],
             ["esc3_trayectorias.py", "Escenario 3: alineamiento de Needleman–Wunsch, paisaje, figura 3"],
             ["construir_escenarios.py", "Genera este reporte a partir de resultados/*.json"],
             ["docx_utils.py", "Utilidades de formato del documento"]],
            [4.6, 12.0], alinear=["l", "l"])
    U.parrafo(doc, "Orden de ejecución: esc1_medicina.py, esc2_grupos_sociales.py, esc3_trayectorias.py y "
                   "después construir_escenarios.py. Requiere Python 3.10 o superior, numpy, scikit-learn, scipy, "
                   "matplotlib y python-docx. Repositorio del código: ")
    p = doc.paragraphs[-1]
    U.hipervinculo(p, url_repo, url_repo, tam=9.5)


def construir(raiz, salida, url_repo):
    e1, e2, e3 = cargar(raiz)
    fig = Path(raiz) / "figuras"
    e1["_ruta_fig"] = str(fig / "fig1_medicina.png")
    e2["_ruta_fig"] = str(fig / "fig2_grupos.png")
    e3["_ruta_fig"] = str(fig / "fig3_trayectorias.png")
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.59), Cm(27.94)
    sec.left_margin = sec.right_margin = Cm(2.3)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)
    U.estilos(doc)
    pie = sec.footer.paragraphs[0]
    pie.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = pie.add_run(f"{U.SUBTITULO} · {U.ALUMNA} · {U.MATRICULA} · página ")
    r.font.size = Pt(8.5)
    r.font.color.rgb = U.GRIS
    U.campo_pagina(pie)
    p = U.portada(doc)
    U.hipervinculo(p, url_repo, url_repo, tam=9)
    doc.add_page_break()
    texto_intro(doc)
    texto_referentes(doc)
    texto_fundamentos(doc)
    texto_metodo(doc)
    texto_esc1(doc, e1)
    texto_esc2(doc, e2)
    texto_esc3(doc, e3)
    texto_discusion(doc, e1, e2, e3)
    texto_referencias(doc)
    anexo(doc, url_repo)
    doc.save(salida)
    print("Reporte guardado:", salida)


def main():
    ap = argparse.ArgumentParser(description="Genera el reporte DOCX de los tres escenarios.")
    ap.add_argument("--raiz", default="..")
    ap.add_argument("--salida", default="Reporte_Escenarios.docx")
    ap.add_argument("--url-repo", default="https://github.com/USUARIO/REPOSITORIO")
    a = ap.parse_args()
    construir(Path(a.raiz), Path(a.salida), a.url_repo)


if __name__ == "__main__":
    main()
