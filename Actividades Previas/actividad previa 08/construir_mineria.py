#!/usr/bin/env python3
"""
construir_mineria.py — Genera el reporte DOCX de las aplicaciones de algoritmos genéticos en
minería polimetálica con enfoque regional (Pachuca de Soto, Hidalgo). Las cifras del texto
y de las tablas se leen de ../resultados/*.json.

USO
  python3 esc1_mezcla.py && python3 esc2_secuenciamiento.py && python3 esc3_molienda.py
  python3 construir_mineria.py --raiz .. --salida ../Reporte_Mineria_Hidalgo_Romero_Lora.docx \
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

TITULO = ("Algoritmos genéticos para la producción minera polimetálica: mezcla, secuenciamiento "
          "y molienda con enfoque regional en Hidalgo")
U.TITULO = TITULO
U.SUBTITULO = "Reporte técnico · Corte 2"

H1, H2 = "Heading 1", "Heading 2"


def cabecera(doc, texto, nivel):
    U.parrafo(doc, texto, alineado=WD_ALIGN_PARAGRAPH.LEFT)
    doc.paragraphs[-1].style = doc.styles[nivel]


def p_wilcoxon(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def cargar(raiz):
    R = Path(raiz) / "resultados"
    return (json.load(open(R / "esc1_mezcla.json", encoding="utf-8")),
            json.load(open(R / "esc2_secuenciamiento.json", encoding="utf-8")),
            json.load(open(R / "esc3_molienda.json", encoding="utf-8")))


# ── Secciones ──────────────────────────────────────────────────

def texto_intro(doc):
    cabecera(doc, "1. Introducción y contexto regional", H1)
    U.parrafo(doc,
              "La minería metálica en México enfrenta un problema de decisión con horizontes distintos: la mezcla "
              "diaria de mineral que llega a la planta, la secuencia de frentes que se extraen en los próximos años y "
              "los ajustes de operación del circuito de molienda. Cada decisión combina restricciones físicas, "
              "económicas y de calidad, y su efecto sobre el valor del negocio es no lineal. Los algoritmos "
              "evolutivos son adecuados para estas decisiones cuando el modelo de planta es complejo o solo puede "
              "evaluarse por simulación [1][2][13].")
    U.parrafo(doc,
              "Hidalgo es un estado con tradición minera. El distrito de Zimapán, en el norte, concentra operaciones de "
              "zinc, plomo y plata. En 2021 la empresa SantaCruz Silver adquirió la planta de Zimapán, con capacidad "
              "de 75 000 toneladas secas por mes y circuitos de zinc, plomo y cobre, por un monto de US$20 millones "
              "más IVA de US$3.2 millones [15]. La inversión privada comprometida en el estado asciende a MX$147.7 "
              "mil millones en 130 proyectos, de los cuales 75 están en operación, 23 en construcción y 22 en "
              "planeación o trámite. Ese monto se concentra sobre todo en cemento, infraestructura y manufactura "
              "[16]. En el ámbito nacional, el valor de la producción minera alcanzó en 2025 un máximo histórico de "
              "MX$379.3 mil millones, con un aumento de 21.2 % en valor y un volumen prácticamente sin cambio; el "
              "valor de la plata creció 34.9 % [17]. Esa combinación, precios altos y volumen estable, favorece las "
              "inversiones que mejoran la eficiencia de la operación existente sobre las que amplían la capacidad.")
    U.parrafo(doc,
              "Este reporte sostiene que, en ese contexto, la optimización algorítmica de la operación es una "
              "inversión de costo relativamente bajo frente a la infraestructura, y que su valor depende de la "
              "calidad del modelo y de la validación con datos de planta. Se desarrollan tres escenarios: mezcla de "
              "mineral de cinco frentes, secuenciamiento de bloques con capacidad y precedencias y ajuste de "
              "molienda. Los parámetros son ilustrativos y no provienen de ninguna operación real; el objetivo es "
              "analizar la estructura del problema y el comportamiento de los métodos.")


def texto_referentes(doc):
    cabecera(doc, "2. Referentes previos", H1)
    U.parrafo(doc,
              "La optimización de mezclas de mineral se formula clásicamente como un programa lineal cuando la "
              "recuperación es constante [3]. Cuando la recuperación depende de la ley y de las impurezas, el "
              "problema se vuelve no lineal y se han usado métodos metaheurísticos para resolverlo [4]. La "
              "evolución diferencial es un método de búsqueda sin gradientes que se usa como referencia en "
              "problemas continuos de este tipo [5].")
    U.parrafo(doc,
              "El secuenciamiento de extracción con restricciones de capacidad y precedencia es un problema entero "
              "que puede resolverse con programación lineal entera mixta [6][7]. En problemas de tamaño industrial "
              "los algoritmos genéticos se han aplicado para obtener buenas soluciones en tiempos cortos, a costa "
              "de garantías de optimalidad [8][9]. La reparación de soluciones infactibles es una práctica "
              "frecuente para mantener la factibilidad durante la búsqueda [10].")
    U.parrafo(doc,
              "El control de los circuitos de molienda se ha abordado con modelos de respuesta y optimización de "
              "consigna, combinados con restricciones de tamaño de partícula y potencia [11][12]. En Hidalgo, la "
              "formación de especialistas en minería y metalurgia se realiza en instituciones como la Universidad "
              "Autónoma del Estado de Hidalgo, que participa en comités educativos de la industria [18].")


def texto_fundamentos(doc, e1, e2, e3):
    cabecera(doc, "3. Fundamentos matemáticos", H1)
    U.parrafo(doc,
              "Los tres escenarios son instancias del problema general de la ecuación (1). Lo que cambia es el "
              "dominio, que determina la codificación, y la función de valor, que determina la dificultad de la "
              "búsqueda. La decisión de método depende de la estructura de esa función, por lo que conviene revisar "
              "antes la formulación de cada problema.")
    U.formula(doc, "x* = argmax x∈X  V(x)   sujeto a   g(x) ≤ 0", "(1)")
    cabecera(doc, "3.1 Mezcla de mineral (escenario 1)", H2)
    U.parrafo(doc,
              "Sean p_k la proporción del frente k en la alimentación, z_k su ley de zinc (%), a_k su impureza "
              "equivalente (ppm) y c_k su costo de extracción y acarreo (USD/t). Con recuperación constante R0, el "
              "problema es lineal:")
    U.formula(doc, "max Σk p_k (P·z_k/100·R0 − c_k − C_proc)   s.a.  Σk p_k = 1,  Σk z_k p_k ≥ Z_min,  Σk a_k p_k ≤ A_max,  p ≥ 0", "(2)")
    U.parrafo(doc,
              "Cuando la recuperación depende de la ley media z̄ = Σ p_k z_k y de la impureza media ā = Σ p_k a_k, "
              "la utilidad deja de ser lineal. Usamos una forma hiperbólica en ambas variables:")
    U.formula(doc, "r(z̄, ā) = R_max · z̄/(z̄ + K_z) · 1/(1 + ā/A_ref),   U(p) = P·(z̄/100)·r − Σk p_k c_k − C_proc", "(3)")
    U.parrafo(doc,
              "Las restricciones de calidad se imponen con penalización lineal: U_pen(p) = U(p) − 500·máx(0, Z_min − z̄) "
              "− 0.5·máx(0, ā − A_max). El AG usa u ∈ [0,1]^5 y proyecta u al símplex con w_k = u_k/Σ_j u_j, como en "
              "los escenarios de fusión del reporte previo de este proyecto.")
    cabecera(doc, "3.2 Secuenciamiento de frentes (escenario 2)", H2)
    U.parrafo(doc,
              "Sean τ_j el tonelaje del bloque j (kt), v_j su valor neto (MUSD), r la tasa de descuento y x_jt ∈ {0,1} "
              "la decisión de extraer el bloque j en el periodo t. El modelo entero es:")
    U.formula(doc, "max Σj Σt v_j (1+r)^(−t) x_jt   s.a.  Σt x_jt ≤ 1,   Σj τ_j x_jt ≤ C,   Σ_{τ<t} x_aτ ≥ x_bt", "(4)")
    U.parrafo(doc,
              "La restricción de precedencia exige que el bloque a se extraiga antes que el bloque b. El AG usa un gen "
              "real g_j ∈ [0, T+1) por bloque y decodifica t_j = ⌊g_j⌋, donde t_j = 0 significa no extracción. La "
              "reparación elimina, en cada periodo con sobrecarga, el bloque de menor valor descontado, y después "
              "elimina las extracciones que violan precedencia. Con reparación, cada solución evaluada es factible.")
    cabecera(doc, "3.3 Ajuste de molienda (escenario 3)", H2)
    U.parrafo(doc,
              "Sean n la fracción de velocidad crítica, J la carga de bolas (%) y Q la alimentación (t/h). Usamos un "
              "modelo sustituto con las siguientes relaciones, cuyos coeficientes son ilustrativos:")
    U.formula(doc, "P80 = P_A (Q/120)^0.5 e^(−2.2(n−0.70)) (J/32)^(−0.6),   P = P_B n^1.5 (J/32)^0.9 (0.6 + 0.4Q/120)", "(5)")
    U.formula(doc, "R = R_max [1 − e^(−(P_ref/P80)^1.5)],   V = Q (V_T·R − C_M) − C_E·P", "(6)")
    U.parrafo(doc,
              "El problema es maximizar V sujeto a P80 ≤ 150 µm y P ≤ 1000 kW, con una tolerancia relativa de 0.5 % en "
              "ambas restricciones para la evaluación de factibilidad. La ecuación (5) implica que un aumento de la "
              "velocidad reduce el tamaño de partícula pero eleva la potencia, y que el aumento de Q eleva el "
              "rendimiento pero engruesa el producto; el óptimo se encuentra en la frontera definida por ambas "
              "restricciones.")
    cabecera(doc, "3.4 Relación entre los elementos y sensibilidad económica", H2)
    z_lp = float(np.array(e1["LP"]["p"]) @ np.array(e1["parametros"]["Z_K"]))
    ton_anual = 75000 * 12
    sens = 2800.0 * (z_lp / 100.0) * 0.01 * ton_anual
    U.parrafo(doc,
              "Dos relaciones conectan los escenarios con la decisión de inversión. La primera es que la ecuación "
              "(2) es exacta cuando la recuperación es constante, de modo que un solver lineal da el óptimo global; "
              "la ecuación (3) rompe esa linealidad y, con ella, la garantía de optimalidad. La segunda es de "
              "orden económico: la ley media de la mezcla óptima, z̄ = "
              f"{z_lp:.2f} %, define la sensibilidad de la utilidad a la recuperación. Para la planta de Zimapán "
              f"(75 000 t/mes, es decir {ton_anual:,} t/año), un punto porcentual de recuperación de zinc vale del orden "
              f"de USD {sens / 1e6:.2f} millones al año en valor bruto, con precio de referencia de USD 2 800/t. Ese "
              "número no incluye costos ni supone un precio ni una ley reales; sirve para dimensionar el orden de "
              "magnitud frente a inversiones de instrumentación o de software de optimización.")
    U.formula(doc, "ΔV_anual = Δr · T_anual · P · z̄/100",
              "(7)")


def texto_metodo(doc):
    cabecera(doc, "4. Método común", H1)
    U.parrafo(doc,
              "Los tres escenarios usan el núcleo de algoritmo genético del proyecto (ga_comun.py), con 40 individuos, "
              "50 generaciones, elitismo de dos individuos, torneo binario, cruce BLX-α con α = 0.3 y mutación gaussiana "
              "con σ = 0.05 (escenarios 1 y 3) y, en el escenario 2, codificación real con decodificación por piso. "
              "Cada ejecución consume 1 940 evaluaciones. La búsqueda aleatoria de control usa el mismo número, y en "
              "el escenario 1 se añade evolución diferencial de SciPy como referencia de un método robusto sin "
              "presupuesto fijo. Se usan cinco semillas por escenario (11, 23, 37, 53, 71). Como referencia exacta se "
              "usa programación lineal (escenario 1, modelo lineal), MILP con HiGHS (escenario 2) y malla densa con "
              "refinamiento local (escenario 3).")


def texto_esc1(doc, e1):
    c = e1["corridas"]
    lp = e1["LP"]["utilidad"]
    gl = np.array([r["lineal"]["ga_utilidad"] for r in c])
    nl = {k: np.array([r["no_lineal"][k][0] for r in c]) for k in ["ga", "de", "aleatoria", "lp_en_no_lineal"]}
    mejor = c[0]["no_lineal"]["mejor_pura"]
    cabecera(doc, "5. Escenario 1: mezcla de mineral de cinco frentes", H1)
    U.parrafo(doc,
              "Se consideran cinco frentes con leyes de zinc entre 4.1 % y 11.4 %, costos entre 12 y 24 USD/t e "
              "impurezas entre 80 y 1 200 ppm. La planta exige una ley media mínima de 6.0 % y una impureza media "
              "máxima de 400 ppm. Estos valores son ilustrativos. Primero se resuelve el modelo lineal con "
              "programación lineal, que da el óptimo exacto de "
              f"USD {lp:.2f}/t. Luego se resuelve el modelo no lineal de la ecuación (3), donde el AG y la evolución "
              "diferencial se comparan con la búsqueda aleatoria y con la solución lineal evaluada en el modelo no lineal.")
    U.parrafo(doc,
              f"En el modelo lineal, el AG alcanza una utilidad media de {gl.mean():.2f} USD/t, con desviación de "
              f"{gl.std(ddof=1):.2f}, es decir, {lp - gl.mean():.2f} USD/t por debajo del óptimo exacto ("
              f"{100 * (lp - gl.mean()) / lp:.1f} %). La causa es la penalización blanda: las soluciones del AG quedan en "
              "el interior de la región cercana a la restricción de impureza, en lugar de apoyarse sobre ella. Esto "
              "muestra un punto de diseño importante: cuando el modelo es lineal y el solver existe, la programación "
              "lineal es preferible. El AG se justifica cuando la ecuación (3) impide esa formulación.")
    U.parrafo(doc,
              f"En el modelo no lineal, la mejor solución es una mezcla pura del frente 2, con utilidad de {mejor:.2f} USD/t. "
              f"El AG llega a esa solución en {int(np.sum(nl['ga'] > mejor - 1e-3))} de cinco semillas y la evolución "
              f"diferencial en {int(np.sum(nl['de'] > mejor - 1e-3))} de cinco; la búsqueda aleatoria alcanza una "
              f"utilidad media de {nl['aleatoria'].mean():.2f} USD/t. La solución lineal, evaluada en el modelo no lineal, "
              f"da {nl['lp_en_no_lineal'].mean():.2f} USD/t: la pérdida de "
              f"{mejor - nl['lp_en_no_lineal'].mean():.2f} USD/t es el costo de usar un modelo lineal cuando la "
              "recuperación depende de la mezcla. La figura 1 resume la convergencia, la distribución de utilidades y la "
              "composición de la mezcla.")
    U.figura(doc, Path(e1["_ruta_fig"]), 16.0,
             "Figura 1. (a) Convergencia del AG en el modelo lineal frente al óptimo LP; (b) utilidad en el modelo no "
             "lineal por método; (c) proporciones de la mezcla por semilla.")


def texto_esc2(doc, e2):
    v = e2["variantes"]
    milp = e2["milp_vpn"]
    rep = np.array([r["ga_vpn"] for r in v["reparacion"]])
    brech = milp - rep
    deb = v["penalizacion_debil"]
    fue = v["penalizacion_fuerte"]
    cabecera(doc, "6. Escenario 2: secuenciamiento de frentes con capacidad y precedencias", H1)
    U.parrafo(doc,
              "Se consideran 15 bloques, cuatro periodos, una tasa de descuento de 10 % y una capacidad de planta de "
              "260 kt por periodo. Tres pares de precedencia obligan a extraer ciertos bloques después de otros. El "
              "MILP resuelto con HiGHS da un VPN óptimo de "
              f"{milp:.2f} MUSD y extrae 13 de 15 bloques. Es el valor de referencia exacto del problema.")
    U.parrafo(doc,
              f"El AG con reparación obtiene un VPN medio de {rep.mean():.2f} MUSD, con brecha media de "
              f"{brech.mean():.2f} MUSD ({100 * brech.mean() / milp:.1f} %) y factibilidad en las cinco semillas. Con "
              "penalización débil (0.5 MUSD por kt excedido), el AG encuentra soluciones de VPN mayor que el óptimo "
              "en algunas semillas, pero son infactibles: en "
              f"{sum(1 for r in deb if not r['ga_factible'])} de cinco ejecuciones la mejor solución viola la capacidad "
              "o la precedencia, lo que la vuelve inutilizable. Con penalización fuerte (5 MUSD por kt), todas las "
              "soluciones son factibles, pero el VPN medio cae a "
              f"{np.mean([r['ga_vpn'] for r in fue]):.2f} MUSD: la penalización fuerte hace que la búsqueda rechace "
              "casi todas las soluciones cercanas a la frontera de capacidad, que es donde está el óptimo. Este resultado "
              "justifica la reparación como operador de factibilidad frente a la penalización.")
    U.parrafo(doc,
              "La figura 2 muestra la secuencia óptima, la carga por periodo frente a la capacidad, y la convergencia "
              "del AG con reparación. El óptimo usa la capacidad casi completa en los periodos 1 a 3. Para la planta, "
              "la lección operativa es que la ganancia de secuenciar bien proviene de nivelar la carga, no de "
              "maximizar la extracción de un solo periodo.")
    U.figura(doc, Path(e2["_ruta_fig"]), 16.0,
             "Figura 2. (a) Secuencia óptima por bloque y periodo (MILP); (b) carga por periodo frente a la capacidad; "
             "(c) convergencia del AG con reparación frente al óptimo.")


def texto_esc3(doc, e3):
    ref = e3["referencia"]
    dfl = e3["defecto"]
    c = e3["corridas"]
    ga = np.array([r["ga_v"] for r in c])
    nm = np.array([r["nm_v"] for r in c])
    al = np.array([r["aleatoria_v"] for r in c])
    gan = ref["v"] - dfl["v"]
    cabecera(doc, "7. Escenario 3: ajuste de la molienda con restricciones de planta", H1)
    U.parrafo(doc,
              "El escenario 3 representa el ajuste de consignas de un circuito de molienda con un modelo sustituto. El "
              f"ajuste por defecto (velocidad {dfl['x'][0]:.2f}, carga de bolas {dfl['x'][1]:.0f} %, alimentación "
              f"{dfl['x'][2]:.0f} t/h) produce un P80 de {dfl['p80']:.1f} µm y una potencia de {dfl['pot']:.0f} kW, "
              "dentro de especificación, con un valor de "
              f"USD {dfl['v']:,.0f}/h. La referencia de malla densa con refinamiento local ajusta la velocidad al límite "
              f"superior, la alimentación al límite de {ref['x'][2]:.0f} t/h y la potencia al límite de 1000 kW, con un "
              f"valor de USD {ref['v']:,.0f}/h. La mejora es de USD {gan:,.0f}/h, es decir {100 * gan / dfl['v']:.1f} %.")
    U.parrafo(doc,
              f"El AG encuentra una solución de {ga.mean():,.0f} USD/h en promedio, con cuatro de cinco semillas dentro "
              f"de 0.1 % de la referencia. La semilla 71 queda en {ga[4]:,.0f} USD/h, 1.3 % por debajo, porque la "
              "búsqueda se estanca cerca de una frontera. La búsqueda aleatoria llega a un valor medio de "
              f"{al.mean():,.0f} USD/h, unos {100 * (ref['v'] - al.mean()) / ref['v']:.1f} % por debajo de la referencia. "
              f"Nelder–Mead, iniciado en la configuración por defecto, converge a la referencia en las cinco semillas "
              f"({nm.mean():,.1f} USD/h). En esta superficie, suave y de tres variables, un método local con buen punto de "
              "partida supera al AG. El AG es más útil cuando la superficie es multimodal, por ejemplo cuando el "
              "modelo se calibra con datos ruidosos y aparecen varios máximos locales. La figura 3 muestra el valor en "
              "el plano (n, J) con las dos fronteras de restricción, la convergencia y la comparación por método.")
    U.figura(doc, Path(e3["_ruta_fig"]), 16.0,
             "Figura 3. (a) Valor en el plano (n, J) para Q = 130 t/h; la zona gris es infactible; (b) convergencia del "
             "AG frente a la referencia; (c) valor en factible por método.")


def texto_inversion(doc, e1, e2, e3):
    cabecera(doc, "8. Aplicación industrial y necesidades de inversión en Hidalgo", H1)
    dv = e2["variantes"]["reparacion"]
    brecha_rep = e2["milp_vpn"] - np.mean([r["ga_vpn"] for r in dv])
    ref = e3["referencia"]["v"]
    dfl = e3["defecto"]["v"]
    U.parrafo(doc,
              "La optimización de operación compite por recursos con inversiones de otra naturaleza, como la "
              "ampliación de la planta, nuevos frentes o instrumentación. Los resultados sugieren una secuencia de "
              "decisiones razonable. Primero, la optimización de la mezcla y de la molienda exige datos de planta de "
              "calidad: leyes por frente, recuperaciones por ley y potencia en tiempo real. Sin esos datos, el modelo "
              "es una hipótesis y no una herramienta. Segundo, la optimización de secuencia requiere una base de "
              "planeación minera confiable, mientras que la de molienda puede arrancar con sensores y un modelo "
              "sustituto calibrado con historiales de operación.")
    U.parrafo(doc,
              f"El orden de magnitud de los beneficios es el siguiente. En el escenario 2, la brecha media entre el AG con "
              f"reparación y el óptimo exacto es de {brecha_rep:.2f} MUSD sobre cuatro periodos, pero la diferencia entre "
              "el óptimo y una secuencia no optimizada puede ser mayor, y no fue calculada en este estudio. En el escenario "
              f"3, el ajuste de consignas aporta USD {ref - dfl:,.0f}/h en el modelo sustituto. Con el número de horas "
              "de operación de una planta de este tamaño, la cifra no es despreciable frente al costo de un "
              "sistema de cómputo y de sensores. Como cálculo de dimensionamiento, la ecuación (7) con los supuestos del "
              "escenario 1 muestra que cada punto de recuperación vale del orden de USD 2 millones al año en la planta "
              "de referencia. La decisión de inversión debe basarse en la mejora demostrada con datos propios.")
    U.parrafo(doc,
              "Para el estado, las necesidades principales son tres. La primera es capital humano en ingeniería de "
              "procesos y ciencia de datos con capacidad para calibrar modelos de planta. La segunda es infraestructura "
              "de energía y agua, dado que los circuitos de molienda concentran el consumo eléctrico; los cambios de "
              "consigna solo son valiosos si la red y el suministro de agua lo permiten. La tercera es una cadena de "
              "proveedores de instrumentación y software en la región, que reduzca el costo de implantación. El "
              "crecimiento de la inversión privada en parques industriales, con inversiones públicas de MX$52.8 millones "
              "en modernización de parques en 2026 [19], abre la posibilidad de que estos servicios se instalen cerca "
              "de los centros mineros y de manufactura.")
    U.parrafo(doc,
              "Una limitación es que el estado no tiene una base pública y actualizada de producción minera por "
              "municipio que permita calibrar los modelos con datos locales. Este estudio no sustituye esa información. "
              "Un paso útil sería que las empresas y las instituciones académicas compartieran series de planta bajo "
              "acuerdos de confidencialidad, de modo que los modelos se validen con datos reales antes de decidir "
              "inversiones.")


def texto_discusion(doc):
    cabecera(doc, "9. Discusión", H1)
    U.parrafo(doc,
              "Tres resultados son transferibles a cualquier operación de minería metálica. Primero, la formulación "
              "importa más que el método: cuando la función es lineal, un solver exacto supera a cualquier metaheurística; "
              "cuando es no lineal, el AG y la evolución diferencial alcanzan el mismo óptimo y la búsqueda aleatoria no. "
              "Segundo, la factibilidad debe gestionarse explícitamente. La penalización débil produce soluciones "
              "inutilizables y la penalización fuerte bloquea la búsqueda. La reparación resulta más estable en el "
              "problema de secuencia. Tercero, la molienda es un caso en el que un método local bien inicializado es "
              "competitivo, y el AG sirve más para explorar cuando se reconoce multimodalidad.")
    U.parrafo(doc,
              "Las limitaciones son claras. Todos los parámetros son ilustrativos y los datos no provienen de una "
              "operación real. Las comparaciones usan cinco semillas. La optimización no modela incertidumbre geológica, "
              "variabilidad de la ley en el tiempo ni fallas de equipo. Los resultados describen métodos y estructuras "
              "de problema; no son recomendaciones de operación.")


def texto_conclusiones(doc):
    cabecera(doc, "10. Conclusiones", H1)
    U.parrafo(doc,
              "Los algoritmos genéticos son útiles en tres problemas de minería polimetálica cuando la formulación es "
              "no lineal, la secuencia tiene restricciones enteras o el modelo de molienda se evalúa por simulación. En "
              "los casos lineales y convexos, la programación lineal o entera ofrece el óptimo global con menor costo. "
              "Para Hidalgo, la oportunidad está en combinar la calidad de datos de planta con modelos de optimización "
              "y en formar capacidad local para mantenerlos. La inversión en esa capacidad es pequeña frente a la "
              "infraestructura y puede justificarse con una prueba piloto en una planta de la región.")


def texto_referencias(doc):
    cabecera(doc, "Referencias", H1)
    refs = [
        "[1] Holland, J. H. (1975). Adaptation in Natural and Artificial Systems. University of Michigan Press.",
        "[2] Eiben, A. E., & Smith, J. E. (2015). Introduction to Evolutionary Computing (2.ª ed.). Springer.",
        "[3] Dantzig, G. B. (1963). Linear Programming and Extensions. Princeton University Press.",
        "[4] Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.",
        "[5] Storn, R., & Price, K. (1997). Differential evolution: A simple and efficient heuristic for global "
        "optimization over continuous spaces. Journal of Global Optimization, 11(4), 341–359.",
        "[6] Lerchs, H., & Grossmann, I. F. (1965). Optimum design of open-pit mines. Transactions of the CIM, 58, 47–54.",
        "[7] Chicoisne, R., Espinoza, D., Goycoolea, M., Moreno, E., & Rubio, E. (2012). A new algorithm for the "
        "open-pit mine scheduling problem. Operations Research, 60(3), 517–528.",
        "[8] Goldberg, D. E., & Lingle, R. (1985). Alleles, loci, and the traveling salesman problem. En Proceedings "
        "of the First International Conference on Genetic Algorithms, 154–159.",
        "[9] Lamghari, A., & Dimitrakopoulos, R. (2012). A diversified Tabu search approach for the open-pit mine "
        "production scheduling problem with metal uncertainty. European Journal of Operational Research, 222(3), 642–652.",
        "[10] Michalewicz, Z., & Schoenauer, M. (1996). Evolutionary algorithms for constrained parameter optimization "
        "problems. Evolutionary Computation, 4(1), 1–32.",
        "[11] Austin, L. G., Klimpel, R. R., & Luckie, P. T. (1984). Process Engineering of Size Reduction: Ball Milling. "
        "SME-AIME.",
        "[12] Napier-Munn, T. J., Morrell, S., Morrison, R. D., & Kojovic, T. (1996). Mineral Comminution Circuits: "
        "Their Operation and Optimisation. Julius Kruttschnitt Mineral Research Centre.",
        "[13] Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms. Wiley.",
        "[14] Mitchell, M. (1996). An Introduction to Genetic Algorithms. MIT Press.",
        "[15] NS Energy (2021). SantaCruz Silver enters definitive agreement to acquire the Zimapán mine and arranges "
        "financing for transaction. https://www.nsenergybusiness.com/news/santacruz-silver-enters-definitive-agreement-to-acquire-the-zimapan-mine-and-arranges-financing-for-transaction/",
        "[16] Mexico Business News (2026). Hidalgo private investment reaches MX$147.7 billion. "
        "https://mexicobusiness.news/trade-and-investment/news/hidalgo-private-investment-reaches-mx1477-billion",
        "[17] Mexico Business News (2026). Mining value up 21% in 2025 despite flat volume: CAMIMEX. "
        "https://mexicobusiness.news/mining/news/mining-value-21-2025-despite-flat-volume-camimex",
        "[18] Universidad Autónoma del Estado de Hidalgo (2025). UAEH se incorpora al Comité Educativo de Minería México 2025. "
        "https://www.uaeh.edu.mx/noticias/9861",
        "[19] Clúster Industrial (2026). Hidalgo destina MX$52.8 mdp en 2026 para modernizar parques industriales clave. "
        "https://clusterindustrial.com.mx/hidalgo-destina-mas-de-52-8-mdp-en-2026-para-modernizar-parques-industriales-clave/",
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
    cabecera(doc, "Anexo. Código y reproducción", H2)
    U.tabla(doc, ["Archivo", "Contenido"],
            [["ga_comun.py", "Núcleo del AG (codificación real y binaria) y búsqueda aleatoria de control"],
             ["esc1_mezcla.py", "Escenario 1: LP, modelo no lineal de recuperación, AG y DE; figura 1"],
             ["esc2_secuenciamiento.py", "Escenario 2: MILP exacto, AG con reparación y penalizaciones; figura 2"],
             ["esc3_molienda.py", "Escenario 3: modelo sustituto, malla de referencia, AG, NM y aleatoria; figura 3"],
             ["construir_mineria.py", "Genera este reporte a partir de resultados/*.json"],
             ["docx_utils.py", "Utilidades de formato del documento"]],
            [4.6, 12.0], alinear=["l", "l"])
    U.parrafo(doc, "Orden de ejecución: esc1_mezcla.py, esc2_secuenciamiento.py, esc3_molienda.py y después "
                   "construir_mineria.py. Requiere Python 3.10 o superior, numpy, scipy, matplotlib y python-docx. "
                   "Repositorio del código: ")
    U.hipervinculo(doc.paragraphs[-1], url_repo, url_repo, tam=9.5)


def construir(raiz, salida, url_repo):
    e1, e2, e3 = cargar(raiz)
    fig = Path(raiz) / "figuras"
    e1["_ruta_fig"] = str(fig / "fig1_mezcla.png")
    e2["_ruta_fig"] = str(fig / "fig2_secuenciamiento.png")
    e3["_ruta_fig"] = str(fig / "fig3_molienda.png")
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
    texto_fundamentos(doc, e1, e2, e3)
    texto_metodo(doc)
    texto_esc1(doc, e1)
    texto_esc2(doc, e2)
    texto_esc3(doc, e3)
    texto_inversion(doc, e1, e2, e3)
    texto_discusion(doc)
    texto_conclusiones(doc)
    texto_referencias(doc)
    anexo(doc, url_repo)
    doc.save(salida)
    print("Reporte guardado:", salida)


def main():
    ap = argparse.ArgumentParser(description="Genera el reporte DOCX de minería polimetálica.")
    ap.add_argument("--raiz", default="..")
    ap.add_argument("--salida", default="Reporte_Mineria.docx")
    ap.add_argument("--url-repo", default="https://github.com/USUARIO/REPOSITORIO")
    a = ap.parse_args()
    construir(Path(a.raiz), Path(a.salida), a.url_repo)


if __name__ == "__main__":
    main()
