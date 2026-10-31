<div align="center">

**UNIVERSIDAD POLITÉCNICA METROPOLITANA DE HIDALGO**

Maestría en Inteligencia Artificial

---

# Algoritmos genéticos para la producción minera polimetálica: mezcla, secuenciamiento y molienda con enfoque regional en Hidalgo

**Reporte técnico · Corte 2**

Materia: **Algoritmos Evolutivos**

| | |
|---|---|
| **Alumna** | Jessica Melani Romero Lora |
| **Matrícula** | 25322116 |
| **Catedrático** | Dr. Jaime Aguilar Ortiz |
| **Modalidad** | Individual |
| **Periodo** | Septiembre–diciembre de 2026 |
| **Fecha de entrega** | 21 de octubre de 2026 |

</div>

---

## Resumen

Este repositorio estudia cómo aplicar algoritmos genéticos (AG) a tres decisiones de operación de una mina polimetálica de zinc, plomo y plata, con referencia al contexto de Zimapán y de Pachuca de Soto, Hidalgo. Las decisiones son la mezcla de mineral que alimenta la planta, la secuencia de extracción de bloques bajo restricciones de capacidad y el ajuste de consignas de un circuito de molienda.

Cada decisión se formula como un problema de optimización con restricciones. Para cada una se resuelve el problema con un AG y, cuando existe, se compara con una referencia exacta (programación lineal o entera mixta) o con un método de búsqueda robusto. Los resultados muestran que la elección del método depende de la estructura del problema: cuando el modelo es lineal o entero, un solver exacto es preferible; cuando la función de valor es no lineal, o cuando hay restricciones que impiden una formulación clásica, el AG es competitivo, siempre que la factibilidad se gestione explícitamente mediante reparación.

Los datos son ilustrativos. Los coeficientes tienen orden de magnitud realista y no provienen de ninguna operación. El contexto regional sí se apoya en fuentes públicas citadas en la sección 3.

Resultados principales:

- **Mezcla de mineral.** En el modelo lineal, el AG alcanza una utilidad media de 155.01 USD/t, frente al óptimo exacto de 160.47 USD/t obtenido por programación lineal. En el modelo no lineal, la mejor solución vale 90.63 USD/t: la evolución diferencial la encuentra en las cinco semillas y el AG en cuatro de cinco; la búsqueda aleatoria promedia 86.24 USD/t.
- **Secuenciamiento de frentes.** El MILP óptimo tiene un VPN de 131.19 MUSD. El AG con reparación promedia 126.85 MUSD y es factible en las cinco semillas. Con penalización fuerte, todas las soluciones son factibles, pero el VPN medio baja a 111.55 MUSD.
- **Ajuste de molienda.** La referencia da 5 582.40 USD/h, frente a 3 854.66 USD/h con la configuración por defecto, una mejora de 44.8 %. Nelder–Mead recupera la referencia en las cinco semillas; el AG la alcanza en cuatro de cinco y la búsqueda aleatoria promedia 5 480.12 USD/h.

---

## Índice

1. [Tema y motivación](#1-tema-y-motivación)
2. [Aplicación: el problema operativo](#2-aplicación-el-problema-operativo)
3. [Contexto regional en Hidalgo](#3-contexto-regional-en-hidalgo)
4. [Objetivos](#4-objetivos)
5. [Solución propuesta](#5-solución-propuesta)
6. [Fundamentos matemáticos](#6-fundamentos-matemáticos)
7. [Escenario 1: mezcla de mineral](#7-escenario-1-mezcla-de-mineral)
8. [Escenario 2: secuenciamiento de frentes](#8-escenario-2-secuenciamiento-de-frentes)
9. [Escenario 3: ajuste de molienda](#9-escenario-3-ajuste-de-molienda)
10. [Viabilidad](#10-viabilidad)
11. [Limitaciones](#11-limitaciones)
12. [Estructura del repositorio y reproducción](#12-estructura-del-repositorio-y-reproducción)
13. [Referencias](#13-referencias)

---

## 1. Tema y motivación

Una planta concentradora de zinc y plomo convierte mineral extraído en concentrados vendibles. Entre la mina y el concentrado hay muchas decisiones: qué frentes se mezclan para alimentar la planta, en qué periodo se extrae cada bloque y cómo se ajusta la molienda que reduce el mineral a tamaño de liberación. Cada decisión condiciona la recuperación metalúrgica, el consumo de energía y el valor económico del negocio.

En la práctica, estas decisiones suelen tomarse con reglas de experiencia y con hojas de cálculo. Esas herramientas funcionan cuando las relaciones son sencillas, pero se vuelven insuficientes cuando la recuperación depende de la mezcla, cuando la secuencia tiene restricciones entre bloques o cuando el circuito de molienda opera cerca de sus límites de potencia y de tamaño de producto. En esos casos, un problema de optimización con restricciones describe mejor la decisión que una regla.

Los algoritmos genéticos son una familia de métodos de búsqueda que no requieren derivadas, operan sobre poblaciones de candidatos y se adaptan a funciones no lineales y discontinuas [1][2][4]. Su uso en diseño y operación industrial está documentado desde hace décadas [8][13][14]. Este reporte no pretende demostrar que un AG sea mejor que las técnicas clásicas en todos los casos. Su propósito es establecer cuándo conviene cada una y qué precauciones exige la aplicación.

## 2. Aplicación: el problema operativo

El estudio se organiza en tres decisiones de operación que cubren distintos horizontes de planeación:

| Horizonte | Decisión | Naturaleza del problema | Método de referencia |
|---|---|---|---|
| Diario | Proporción de cinco frentes en la alimentación | Continuo, con restricciones de calidad | Programación lineal (modelo lineal) |
| Anual o multianual | Periodo de extracción de 15 bloques | Entero, con capacidad y precedencias | MILP con HiGHS |
| Horario o por turno | Velocidad, carga de bolas y alimentación del molino | Continuo, no lineal, con simulación | Malla densa y refinamiento local |

Los tres problemas comparten una estructura: una función de valor que se quiere maximizar, restricciones que deben cumplirse y una evaluación que puede hacerse con un modelo, aunque sea simplificado. Esa estructura es la que hace pertinente un método de búsqueda.

## 3. Contexto regional en Hidalgo

Hidalgo tiene una tradición minera antigua y, en el norte del estado, un distrito de zinc, plomo y plata. La planta de Zimapán tiene una capacidad de 75 000 toneladas secas por mes y circuitos de zinc, plomo y cobre. SantaCruz Silver la adquirió en 2021 por US$20 millones, más IVA de US$3.2 millones [15].

La inversión privada comprometida en el estado suma MX$147.7 mil millones en 130 proyectos. De ellos, 75 están en operación, 23 en construcción y 22 en planeación o trámite. Esa inversión se concentra en cemento, infraestructura y manufactura, y la manufactura creció 7.9 % en el primer trimestre de 2026 [16]. El estado destina además MX$52.8 millones en 2026 a la modernización de parques industriales [19].

A escala nacional, el valor de la producción minera alcanzó en 2025 un máximo histórico de MX$379.3 mil millones, con un aumento de 21.2 % en valor y un volumen prácticamente sin cambio. El valor de la plata creció 34.9 % [17]. Con precios altos y volumen estable, la eficiencia de la operación existente es una fuente de valor que compite en costo con la expansión de capacidad. La formación de especialistas en la región también cuenta: la Universidad Autónoma del Estado de Hidalgo participa en comités educativos de la minería [18].

## 4. Objetivos

**Objetivo general.** Evaluar la aplicabilidad de algoritmos genéticos a tres decisiones de operación de una mina polimetálica, comparando su desempeño con referencias exactas o robustas, y derivar criterios de viabilidad técnica y económica para su adopción en Hidalgo.

**Objetivos específicos.**

1. Formular cada decisión como un problema de optimización con función objetivo, restricciones y dominio explícitos.
2. Diseñar una codificación del cromosoma adecuada a cada dominio: real con decodificación, real en el símplex y real con reparación.
3. Comparar el AG con una referencia exacta cuando exista (programación lineal o MILP) y con búsqueda aleatoria a igual presupuesto de evaluaciones.
4. Identificar en qué condiciones el AG aporta valor frente a los métodos clásicos, y cómo debe tratarse la factibilidad.
5. Estimar el orden de magnitud del valor económico de la optimización y señalar las necesidades de datos, capital humano e infraestructura para implantarla.

## 5. Solución propuesta

La solución combina un núcleo de AG común con una formulación específica por escenario. El núcleo usa población de 40 individuos, 50 generaciones, elitismo de dos individuos y torneo binario. La selección por torneo favorece a los individuos de mayor aptitud sin eliminar a los demás. Para la comparación se usa también evolución diferencial [5], un método de búsqueda continua muy utilizado como referencia robusta. El elitismo conserva las dos mejores soluciones en cada generación, lo que garantiza que la aptitud máxima no empeore. Cada ejecución consume 1 940 evaluaciones de la función objetivo, y la búsqueda aleatoria de control usa exactamente el mismo número.

Los operadores de variación dependen del dominio. En codificación real se usa el cruce BLX-α [3], que muestrea hijos en un intervalo ampliado alrededor de los padres, y la mutación gaussiana acotada al dominio. Para la secuencia de bloques, cada gen real se decodifica como un periodo entero, y una reparación determinista elimina las extracciones que violan capacidad o precedencia. La reparación sigue la idea de mantener la factibilidad en la búsqueda, que se discute como alternativa a la penalización en la literatura de optimización con restricciones [10]. Para la mezcla, el vector de decisión se proyecta al símplex, de modo que las proporciones sumen uno sin necesidad de penalizar esa igualdad.

Este diseño tiene una consecuencia práctica. Un AG no garantiza el óptimo, pero sí permite evaluar cualquier modelo que pueda simularse, incluidos los no lineales y los que no tienen una formulación cerrada. Por eso el reporte compara siempre el AG con una referencia exacta cuando existe, y con un método de búsqueda robusto cuando no.

## 6. Fundamentos matemáticos

Los tres escenarios son instancias del problema general

$$
x^{*} = \arg\max_{x \in \mathcal{X}} \; V(x) \quad \text{s.a.} \quad g(x) \le 0 \tag{1}
$$

donde $V$ es el valor del negocio, $\mathcal{X}$ el dominio de decisión y $g$ las restricciones de calidad, potencia, capacidad o precedencia.

### 6.1 Mezcla de mineral

Sean $p_k$ las proporciones de los cinco frentes en la alimentación, $z_k$ sus leyes de zinc (%), $a_k$ sus impurezas equivalentes (ppm) y $c_k$ sus costos de extracción y acarreo (USD/t). Con recuperación constante $R_0$, el problema es lineal:

$$
\max_{p} \; \sum_{k=1}^{5} p_k \left( \frac{P\, z_k}{100}\, R_0 - c_k - C_{\text{proc}} \right)
\quad \text{s.a.} \quad \sum_{k} p_k = 1, \quad \sum_{k} z_k p_k \ge Z_{\min}, \quad \sum_{k} a_k p_k \le A_{\max}, \quad p \ge 0 \tag{2}
$$

Cuando la recuperación depende de la mezcla, el modelo deja de ser lineal. Con $\bar z = \sum_k p_k z_k$ y $\bar a = \sum_k p_k a_k$, usamos la forma hiperbólica

$$
r(\bar z, \bar a) = R_{\max}\,\frac{\bar z}{\bar z + K_z}\,\frac{1}{1 + \bar a / A_{\text{ref}}},
\qquad
U(p) = P\,\frac{\bar z}{100}\, r(\bar z, \bar a) - \sum_{k} p_k c_k - C_{\text{proc}} \tag{3}
$$

Las restricciones de calidad se imponen con penalización lineal,

$$
U_{\text{pen}}(p) = U(p) - 500\,\max(0,\, Z_{\min} - \bar z) - 0.5\,\max(0,\, \bar a - A_{\max}),
$$

y el AG opera sobre $u \in [0,1]^5$ con la proyección $p_k = u_k / \sum_j u_j$.

### 6.2 Secuenciamiento de frentes

Sean $\tau_j$ el tonelaje del bloque $j$ (kt), $v_j$ su valor neto (MUSD), $r$ la tasa de descuento y $x_{jt} \in \{0,1\}$ la decisión de extraer el bloque $j$ en el periodo $t$. El modelo entero es

$$
\max \sum_{j}\sum_{t} \frac{v_j}{(1+r)^{t}}\, x_{jt}
\quad \text{s.a.} \quad
\begin{aligned}
&\textstyle\sum_{t} x_{jt} \le 1, \\
&\textstyle\sum_{j} \tau_j\, x_{jt} \le C, \\
&\textstyle\sum_{\tau < t} x_{a\tau} \ge x_{bt} \quad \forall (a,b) \in \mathcal{P}
\end{aligned} \tag{4}
$$

donde $C$ es la capacidad de planta por periodo y $\mathcal{P}$ el conjunto de pares de precedencia. La restricción de precedencia exige que el bloque $a$ se extraiga antes que $b$.

El AG usa un gen real $g_j \in [0, T+1)$ por bloque y decodifica $t_j = \lfloor g_j \rfloor$, con $t_j = 0$ para no extracción. La reparación elimina, en cada periodo con sobrecarga, el bloque de menor valor descontado $v_j(1+r)^{-t}$, y después elimina las extracciones que violan precedencia. Con reparación, toda solución evaluada es factible.

### 6.3 Ajuste de molienda

Sean $n$ la fracción de velocidad crítica, $J$ la carga de bolas (%) y $Q$ la alimentación (t/h). El modelo sustituto es

$$
P_{80} = P_A \left(\frac{Q}{120}\right)^{0.5} e^{-2.2\,(n - 0.70)} \left(\frac{J}{32}\right)^{-0.6},
\qquad
P = P_B\, n^{1.5} \left(\frac{J}{32}\right)^{0.9} \left(0.6 + 0.4\,\frac{Q}{120}\right) \tag{5}
$$

$$
R = R_{\max}\left[1 - e^{-(P_{\text{ref}}/P_{80})^{1.5}}\right],
\qquad
V = Q\,(V_T\, R - C_M) - C_E\, P \tag{6}
$$

con $P_A = 180$, $P_B = 1200$, $R_{\max} = 0.92$, $P_{\text{ref}} = 150$, $V_T = 85$ USD/t, $C_M = 12$ USD/t y $C_E = 0.12$ USD/kWh. El problema consiste en maximizar $V$ sujeto a $P_{80} \le 150\ \mu\text{m}$ y $P \le 1000$ kW, con tolerancia relativa de 0.5 % en la evaluación de factibilidad.

### 6.4 Sensibilidad económica

Con $T_{\text{anual}}$ toneladas de alimentación al año, precio $P$ y ley media $\bar z$, una variación $\Delta r$ de la recuperación de zinc produce

$$
\Delta V_{\text{anual}} = \Delta r \cdot T_{\text{anual}} \cdot P \cdot \frac{\bar z}{100} \tag{7}
$$

Con la mezcla lineal óptima, $\bar z = 7.67\,\%$ ($0.8889 \times 7.2 + 0.1111 \times 11.4$), $P = 2\,800$ USD/t, $\Delta r = 0.01$ y $T_{\text{anual}} = 900\,000$ t (equivalente a 75 000 t/mes), la ecuación (7) da

$$
\Delta V_{\text{anual}} = 0.01 \times 900\,000 \times 2\,800 \times \frac{7.67}{100} \approx 1.93 \times 10^{6}\ \text{USD/año}.
$$

Es una cifra de dimensionamiento en valor bruto. No descuenta costos ni representa un pronóstico.

## 7. Escenario 1: mezcla de mineral

**Diseño.** Cinco frentes con leyes de zinc de 9.5, 7.2, 5.8, 11.4 y 4.1 %, costos de 18, 15, 14, 24 y 12 USD/t e impurezas de 900, 300, 150, 1 200 y 80 ppm. La planta exige una ley media mínima de 6.0 % y una impureza media máxima de 400 ppm.

**Resultados.** La tabla resume la utilidad por método. Las cifras son medias de cinco semillas cuando corresponde.

| Método | Modelo | Utilidad (USD/t) | Observación |
|---|---|---|---|
| Programación lineal | Lineal | 160.47 | Óptimo exacto; $p = (0,\ 0.8889,\ 0,\ 0.1111,\ 0)$ |
| AG | Lineal | 155.01 ± 1.36 | Brecha media de 5.45 USD/t (3.4 %) |
| AG | No lineal | 90.33 | Mezcla pura del frente 2 en cuatro de cinco semillas |
| Evolución diferencial | No lineal | 90.63 | Mejor solución; igual en las cinco semillas |
| Búsqueda aleatoria | No lineal | 86.24 | Media de cinco semillas |
| Solución lineal evaluada | No lineal | 88.12 | Pérdida de 2.51 USD/t frente al óptimo no lineal |

**Interpretación.** En el modelo lineal, el AG no alcanza el vértice óptimo. Sus soluciones están cerca de la restricción de impureza, pero la penalización blanda no las fuerza a apoyarse sobre ella. Por eso la brecha es pequeña pero sistemática. La lección es que, si el modelo es lineal, la programación lineal debe preferirse: da el óptimo global con menor costo computacional.

En el modelo no lineal, la mejor solución es la mezcla pura del frente 2. La evolución diferencial la encuentra en todas las semillas, y el AG en cuatro de cinco. La solución óptima del modelo lineal, evaluada en el modelo no lineal, pierde 2.51 USD/t. Ese es el costo de ignorar la dependencia de la recuperación respecto a la mezcla. El AG se justifica cuando el modelo no admite una formulación lineal, y aquí encuentra la solución que la formulación lineal no ve.

![Figura 1. Escenario 1](figuras/fig1_mezcla.png)

*Figura 1. (a) Convergencia del AG en el modelo lineal frente al óptimo de programación lineal; (b) utilidad en el modelo no lineal por método; (c) composición de la mezcla por semilla.*

## 8. Escenario 2: secuenciamiento de frentes

**Diseño.** El secuenciamiento de bloques en minería a cielo abierto es un problema clásico de optimización entera, con formulaciones exactas que escalan a instancias industriales [6][7]. Quince bloques, cuatro periodos, tasa de descuento de 10 %, capacidad de 260 kt por periodo y tres pares de precedencia en índices base cero: $(0 \to 3)$, $(2 \to 7)$ y $(5 \to 11)$. Los tonelajes, leyes y valores se generan con semilla 2024.

**Resultados.**

| Variante del AG | VPN por semilla (MUSD) | Factibles | Brecha media frente al MILP |
|---|---|---|---|
| MILP (referencia exacta) | 131.19 | — | — |
| Con reparación | 124.34, 127.86, 129.84, 122.45, 129.76 | 5 de 5 | 4.34 MUSD (3.3 %) |
| Penalización débil (0.5 MUSD/kt) | 124.61, 122.22, 120.14, 129.90, 131.88 | 3 de 5 | No aplica: dos soluciones infactibles |
| Penalización fuerte (5 MUSD/kt) | 103.36, 104.88, 122.48, 122.54, 104.50 | 5 de 5 | 19.64 MUSD |

**Interpretación.** La penalización débil produce las mejores aptitudes, pero en dos de cinco semillas la mejor solución viola capacidad o precedencia, y por tanto no puede usarse. La penalización fuerte elimina las infactibles, pero también bloquea la búsqueda cerca de la frontera de capacidad, donde se encuentra el óptimo; el VPN medio cae a 111.55 MUSD. La reparación mantiene la factibilidad en todas las evaluaciones y deja explorar esa frontera. Este es el resultado de diseño más transferible del reporte: en problemas con restricciones duras, la forma de tratar la factibilidad pesa más que el tamaño del modelo. Esta observación es coherente con el enfoque de manejo de restricciones en algoritmos evolutivos [10].

Cuando la ley de cada bloque es incierta, la secuencia debe evaluarse con escenarios, y las búsquedas tabú y metaheurísticas se han aplicado a esa variante del problema [9]. La secuencia óptima reparte la carga casi por completo en los tres primeros periodos y deja el cuarto con el menor tonelaje. La ganancia proviene de nivelar la carga, no de maximizar la extracción en un solo periodo.

![Figura 2. Escenario 2](figuras/fig2_secuenciamiento.png)

*Figura 2. (a) Secuencia óptima por bloque y periodo (MILP); (b) carga por periodo frente a la capacidad, con la solución MILP y la del AG con reparación en la semilla 11; (c) convergencia del AG con reparación frente al óptimo.*

## 9. Escenario 3: ajuste de molienda

**Diseño.** Modelo sustituto de las ecuaciones (5) y (6) con restricciones de tamaño de producto ($P_{80} \le 150\ \mu$m) y de potencia ($P \le 1\,000$ kW). La configuración por defecto es $(n, J, Q) = (0.78,\ 30,\ 100)$, dentro de especificación.

**Resultados.**

| Método | Valor (USD/h) | Observación |
|---|---|---|
| Referencia (malla densa y refinamiento) | 5 582.40 | $n = 0.85$, $J = 30.8$, $Q = 150$ t/h; potencia en el límite |
| Configuración por defecto | 3 854.66 | $P_{80} = 143$ µm, $P = 728$ kW |
| Nelder–Mead desde el defecto | 5 582.40 | Igual a la referencia en las cinco semillas |
| AG (media de cinco semillas) | 5 567.73 | Cuatro de cinco dentro de 0.1 %; semilla 71 en 5 511 |
| Búsqueda aleatoria (media) | 5 480.12 | 1.8 % por debajo de la referencia |

**Interpretación.** La mejora sobre la configuración por defecto es de 1 728 USD/h, un 44.8 %. Se obtiene subiendo la velocidad y la alimentación hasta las fronteras de potencia y de tamaño de producto, que son donde está el óptimo. En esta superficie suave y de tres variables, un método local con buen punto de partida recupera el óptimo en las cinco semillas, mientras que el AG lo logra en cuatro. La ventaja del AG aparece cuando la superficie tiene varios máximos locales, por ejemplo si el modelo se calibra con datos ruidosos. El procesamiento de minerales y el ajuste de molinos se describen con modelos de tamaño de partícula y de potencia en la literatura metalúrgica clásica [11][12]. Por eso, en la práctica conviene combinar ambos: el AG para explorar y un refinamiento local para pulir.

![Figura 3. Escenario 3](figuras/fig3_molienda.png)

*Figura 3. (a) Valor en el plano $(n, J)$ para $Q = 130$ t/h; la zona gris es infactible; (b) convergencia del AG frente a la referencia; (c) valor en factible por método.*

## 10. Viabilidad

**Viabilidad técnica.** Los tres escenarios usan variables que una planta ya mide o puede medir: leyes por frente, consumo de potencia, tamaño de producto y tonelajes por bloque. La optimización no requiere cambios en el equipo, sino un modelo calibrado y una interfaz para aplicar las consignas. El riesgo técnico principal es la calidad del modelo sustituto. Un modelo mal calibrado produce óptimos mal ubicados, así que la validación con datos de planta es un paso obligatorio antes de operar con las recomendaciones.

**Viabilidad económica.** La ecuación (7) muestra que un punto porcentual de recuperación de zinc, en una planta del tamaño de Zimapán, equivale a del orden de 1.93 millones de dólares al año en valor bruto, con supuestos de precio y ley ilustrativos. Ese orden de magnitud es suficiente para justificar un piloto de instrumentación y optimización, cuyo costo es pequeño frente a una ampliación de planta. La decisión final debe basarse en mejoras medidas con datos propios y no en esta cifra.

**Viabilidad operativa.** Un sistema de optimización solo aporta valor si el personal de planta lo usa. Esto exige que las recomendaciones sean explicables, que la factibilidad esté garantizada y que el operador pueda anular la consigna. La reparación en la secuenciación y la restricción dura en la molienda responden a esa exigencia: nunca se recomienda una solución que viole el límite de capacidad o de potencia.

**Viabilidad regional.** Hidalgo tiene tres condiciones favorables y tres necesidades. Las condiciones favorables son la base minera existente en Zimapán [15], la inversión privada comprometida en 130 proyectos [16] y una institución formadora de capital humano en minería [18]. Las necesidades son:

- **Capital humano** en procesos mineros y ciencia de datos, capaz de calibrar y mantener modelos de planta. Es la necesidad más crítica, porque sin ella el sistema no sobrevive a la primera rotación de personal.
- **Infraestructura de energía y agua.** La molienda concentra el consumo eléctrico, y los cambios de consigna solo generan valor si el suministro de energía y agua lo permite.
- **Proveedores regionales** de instrumentación, telemetría y software, que reduzcan el costo de implantación y el tiempo de respuesta ante fallas.

Las inversiones de modernización de parques industriales en Hidalgo, de MX$52.8 millones en 2026 [19], no atienden directamente la minería, pero indica que el estado busca atraer proveedores industriales a la región, lo que puede beneficiar a esta cadena.

**Riesgos.** Los principales riesgos son la falta de datos confiables, la dependencia de un modelo sustituto no validado y la resistencia al cambio en la operación. Cada uno tiene una mitigación: un período de captura de datos antes de optimizar, validación con mediciones cruzadas y un esquema de operación en el que la recomendación se presenta al operador antes de aplicarse.

## 11. Limitaciones

- **Datos ilustrativos.** Ningún coeficiente proviene de una operación real. Los resultados describen métodos y estructuras de problema, no recomendaciones de operación.
- **Muestra pequeña.** Se usan cinco semillas por escenario. Con cinco pares, la prueba de Wilcoxon bilateral no puede producir un valor de p menor que 0.0625, por lo que no se presentan pruebas de hipótesis concluyentes.
- **Sin incertidumbre.** No se modelan la variabilidad geológica de la ley, las fallas de equipo ni los cambios de precio.
- **Dependencia de la penalización.** Los resultados del escenario 2 muestran que la elección de penalización cambia el resultado de forma importante. La reparación es más estable, pero no se explora de forma exhaustiva el espacio de parámetros de penalización.
- **Sin validación regional.** No existe una base pública y actualizada de producción minera por municipio que permita calibrar los modelos con datos de Hidalgo.

## 12. Estructura del repositorio y reproducción

```
.
├── README.md
├── Reporte_Mineria_Hidalgo_Romero_Lora.docx
├── codigo/
│   ├── ga_comun.py               Núcleo del AG (codificación real) y búsqueda aleatoria de control
│   ├── esc1_mezcla.py            Escenario 1: programación lineal, modelo no lineal, AG y evolución diferencial
│   ├── esc2_secuenciamiento.py   Escenario 2: MILP, AG con reparación y variantes con penalización
│   ├── esc3_molienda.py          Escenario 3: modelo sustituto, malla de referencia, AG, Nelder–Mead
│   ├── construir_mineria.py      Genera el reporte DOCX a partir de resultados/*.json
│   └── docx_utils.py             Utilidades de formato del documento
├── resultados/
│   ├── esc1_mezcla.json
│   ├── esc2_secuenciamiento.json
│   └── esc3_molienda.json
└── figuras/
    ├── fig1_mezcla.png
    ├── fig2_secuenciamiento.png
    └── fig3_molienda.png
```

Requisitos: Python 3.10 o superior, `numpy`, `scipy` 1.9 o superior (se usa `scipy.optimize.milp`), `matplotlib` y `python-docx`.

Ejecución desde la carpeta `codigo/`:

```bash
python3 esc1_mezcla.py
python3 esc2_secuenciamiento.py
python3 esc3_molienda.py
python3 construir_mineria.py --raiz .. \
        --salida ../Reporte_Mineria_Hidalgo_Romero_Lora.docx \
        --url-repo https://github.com/USUARIO/REPOSITORIO
```

El tiempo total de ejecución es de unos 15 segundos. Las semillas son explícitas (11, 23, 37, 53 y 71; 2024 para los datos del escenario 2). La reproducibilidad bit a bit no se ha verificado en otra plataforma.

## 13. Referencias

[1] Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.

[2] Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2.ª ed.). Springer.

[3] Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata. En *Foundations of Genetic Algorithms*, 2, 187–202.

[4] Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.

[5] Storn, R., & Price, K. (1997). Differential evolution: A simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359.

[6] Lerchs, H., & Grossmann, I. F. (1965). Optimum design of open-pit mines. *Transactions of the CIM*, 58, 47–54.

[7] Chicoisne, R., Espinoza, D., Goycoolea, M., Moreno, E., & Rubio, E. (2012). A new algorithm for the open-pit mine scheduling problem. *Operations Research*, 60(3), 517–528.

[8] Goldberg, D. E., & Lingle, R. (1985). Alleles, loci, and the traveling salesman problem. En *Proceedings of the First International Conference on Genetic Algorithms*, 154–159.

[9] Lamghari, A., & Dimitrakopoulos, R. (2012). A diversified Tabu search approach for the open-pit mine production scheduling problem with metal uncertainty. *European Journal of Operational Research*, 222(3), 642–652.

[10] Michalewicz, Z., & Schoenauer, M. (1996). Evolutionary algorithms for constrained parameter optimization problems. *Evolutionary Computation*, 4(1), 1–32.

[11] Austin, L. G., Klimpel, R. R., & Luckie, P. T. (1984). *Process Engineering of Size Reduction: Ball Milling*. SME-AIME.

[12] Napier-Munn, T. J., Morrell, S., Morrison, R. D., & Kojovic, T. (1996). *Mineral Comminution Circuits: Their Operation and Optimisation*. Julius Kruttschnitt Mineral Research Centre.

[13] Deb, K. (2001). *Multi-Objective Optimization using Evolutionary Algorithms*. Wiley.

[14] Mitchell, M. (1996). *An Introduction to Genetic Algorithms*. MIT Press.

[15] NS Energy (2021). SantaCruz Silver enters definitive agreement to acquire the Zimapán mine and arranges financing for transaction. https://www.nsenergybusiness.com/news/santacruz-silver-enters-definitive-agreement-to-acquire-the-zimapan-mine-and-arranges-financing-for-transaction/

[16] Mexico Business News (2026). Hidalgo private investment reaches MX$147.7 billion. https://mexicobusiness.news/trade-and-investment/news/hidalgo-private-investment-reaches-mx1477-billion

[17] Mexico Business News (2026). Mining value up 21% in 2025 despite flat volume: CAMIMEX. https://mexicobusiness.news/mining/news/mining-value-21-2025-despite-flat-volume-camimex

[18] Universidad Autónoma del Estado de Hidalgo (2025). UAEH se incorpora al Comité Educativo de Minería México 2025. https://www.uaeh.edu.mx/noticias/9861

[19] Clúster Industrial (2026). Hidalgo destina MX$52.8 mdp en 2026 para modernizar parques industriales clave. https://clusterindustrial.com.mx/hidalgo-destina-mas-de-52-8-mdp-en-2026-para-modernizar-parques-industriales-clave/
