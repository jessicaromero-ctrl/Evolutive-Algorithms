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

Este repositorio aplica algoritmos genéticos (AG) a tres decisiones de operación de una mina polimetálica de zinc, plomo y plata, con referencia al contexto de Pachuca de Soto y del estado de Hidalgo. El primer escenario optimiza la mezcla de mineral de cinco frentes, con una formulación lineal que tiene solución exacta por programación lineal y una formulación no lineal en la recuperación. El segundo escenario secuencia quince bloques en cuatro periodos con capacidad de planta y restricciones de precedencia, y se compara con un modelo entero mixto resuelto de forma exacta. El tercero ajusta consignas de un circuito de molienda con restricciones de tamaño de producto y de potencia.

Todos los datos son ilustrativos. Los coeficientes se eligen para tener orden de magnitud realista y no provienen de ninguna operación. El contexto regional sí se apoya en fuentes públicas: la planta de Zimapán con capacidad de 75 000 t/mes, la inversión privada comprometida en el estado y el valor récord de la producción minera nacional en 2025.

Resultados principales:

1. **Mezcla.** En el modelo lineal, el AG alcanza una utilidad media de 155.0 USD/t frente al óptimo exacto de 160.5 USD/t. En el modelo no lineal, el AG y la evolución diferencial coinciden en la mejor solución (90.63 USD/t), y la búsqueda aleatoria queda en 86.2 USD/t en promedio.
2. **Secuencia.** El MILP óptimo tiene un VPN de 131.19 MUSD. El AG con reparación alcanza 126.85 MUSD en promedio, con factibilidad en las cinco semillas. Con penalización débil, dos de cinco mejores soluciones son infactibles; con penalización fuerte, todas son factibles pero el VPN medio baja a 111.6 MUSD.
3. **Molienda.** El ajuste óptimo de referencia da 5 582 USD/h, frente a 3 855 USD/h con la configuración por defecto (+44.8 %). Nelder–Mead desde el punto por defecto recupera la referencia en las cinco semillas, mientras que el AG la alcanza en cuatro de cinco y la búsqueda aleatoria queda en 5 480 USD/h en promedio.

La conclusión principal es que la elección del método depende de la estructura del problema. La programación lineal y entera son exactas y deben preferirse cuando el modelo lo permite. El AG aporta valor en formulaciones no lineales o con restricciones que impiden esas técnicas, siempre que la factibilidad se gestione explícitamente, con reparación en lugar de penalización blanda o fuerte.

---

## Índice

1. [Estructura del repositorio](#1-estructura-del-repositorio)
2. [Requisitos y reproducción](#2-requisitos-y-reproducción)
3. [Contexto regional](#3-contexto-regional)
4. [Fundamentos matemáticos](#4-fundamentos-matemáticos)
5. [Método común](#5-método-común)
6. [Escenario 1: mezcla de mineral](#6-escenario-1-mezcla-de-mineral)
7. [Escenario 2: secuenciamiento de frentes](#7-escenario-2-secuenciamiento-de-frentes)
8. [Escenario 3: ajuste de molienda](#8-escenario-3-ajuste-de-molienda)
9. [Necesidades de inversión y aplicación industrial](#9-necesidades-de-inversión-y-aplicación-industrial)
10. [Limitaciones](#10-limitaciones)
11. [Referencias](#11-referencias)

---

## 1. Estructura del repositorio

```
.
├── README.md
├── Reporte_Mineria_Hidalgo_Romero_Lora.docx
├── codigo/
│   ├── ga_comun.py               Núcleo del AG (codificación real y binaria) y búsqueda aleatoria
│   ├── esc1_mezcla.py            Escenario 1: LP, modelo no lineal, AG, DE
│   ├── esc2_secuenciamiento.py   Escenario 2: MILP, AG con reparación y penalizaciones
│   ├── esc3_molienda.py          Escenario 3: modelo sustituto, referencia, AG, Nelder–Mead
│   ├── construir_mineria.py      Genera el reporte DOCX
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

## 2. Requisitos y reproducción

Requisitos: Python 3.10 o superior, `numpy`, `scipy` (1.9 o superior, con `scipy.optimize.milp`), `matplotlib` y `python-docx`.

Desde la carpeta `codigo/`:

```bash
python3 esc1_mezcla.py
python3 esc2_secuenciamiento.py
python3 esc3_molienda.py
python3 construir_mineria.py --raiz .. \
        --salida ../Reporte_Mineria_Hidalgo_Romero_Lora.docx \
        --url-repo https://github.com/USUARIO/REPOSITORIO
```

El tiempo total de ejecución es de aproximadamente 15 segundos. Las semillas son explícitas (11, 23, 37, 53 y 71; 2024 para los datos del escenario 2). No se ha verificado la reproducibilidad bit a bit en otra plataforma.

## 3. Contexto regional

La planta de Zimapán, en el norte del estado, tiene una capacidad de 75 000 toneladas secas por mes y circuitos de zinc, plomo y cobre. SantaCruz Silver la adquirió en 2021 por US$20 millones, más IVA de US$3.2 millones [15]. La inversión privada comprometida en Hidalgo suma MX$147.7 mil millones en 130 proyectos, de los cuales 75 están en operación, 23 en construcción y 22 en planeación o trámite [16]. El valor de la producción minera nacional alcanzó en 2025 un máximo de MX$379.3 mil millones, con un aumento de 21.2 % en valor y un volumen prácticamente estable, y el valor de la plata creció 34.9 % [17]. El estado destina MX$52.8 millones en 2026 a la modernización de parques industriales [19].

Estas cifras motivan el enfoque: con precios altos y volumen estable, la eficiencia de la operación existente es una vía de creación de valor con menor costo que la expansión.

## 4. Fundamentos matemáticos

Los tres escenarios son instancias de

$$
x^{*} = \operatorname*{arg\,max}_{x \in \mathcal{X}} \; V(x) \quad \text{s.a.} \quad g(x) \le 0 \tag{1}
$$

**Mezcla lineal (escenario 1).** Con proporciones $p_k$ de cinco frentes, leyes $z_k$, impurezas $a_k$ y costos $c_k$, y recuperación constante $R_0$:

$$
\max_{p} \; \sum_{k} p_k \left( \frac{P\, z_k}{100}\, R_0 - c_k - C_{\text{proc}} \right)
\quad \text{s.a.} \quad \sum_k p_k = 1,\;\; \sum_k z_k p_k \ge Z_{\min},\;\; \sum_k a_k p_k \le A_{\max},\;\; p \ge 0 \tag{2}
$$

**Mezcla no lineal (escenario 1).** Con $\bar z = \sum_k p_k z_k$ y $\bar a = \sum_k p_k a_k$, la recuperación depende de la mezcla:

$$
r(\bar z, \bar a) = R_{\max}\,\frac{\bar z}{\bar z + K_z}\,\frac{1}{1 + \bar a / A_{\text{ref}}},
\qquad
U(p) = P\,\frac{\bar z}{100}\, r(\bar z,\bar a) - \sum_k p_k c_k - C_{\text{proc}} \tag{3}
$$

Las restricciones de calidad se imponen con penalización: $U_{\text{pen}} = U - 500\,\max(0, Z_{\min} - \bar z) - 0.5\,\max(0, \bar a - A_{\max})$. El AG trabaja sobre $u \in [0,1]^5$ y proyecta al símplex con $p_k = u_k / \sum_j u_j$.

**Secuenciamiento (escenario 2).** Con $\tau_j$ el tonelaje del bloque $j$, $v_j$ su valor neto, $r$ la tasa de descuento y $x_{jt} \in \{0,1\}$:

$$
\max \sum_{j}\sum_{t} \frac{v_j}{(1+r)^{t}}\, x_{jt}
\quad \text{s.a.} \quad \sum_t x_{jt} \le 1,\quad \sum_j \tau_j x_{jt} \le C,\quad \sum_{\tau < t} x_{a\tau} \ge x_{bt} \;\;\forall (a,b) \in \mathcal{P} \tag{4}
$$

El AG usa un gen real $g_j \in [0, T+1)$ por bloque y decodifica $t_j = \lfloor g_j \rfloor$. La **reparación** elimina, en cada periodo con sobrecarga, el bloque de menor valor descontado $v_j (1+r)^{-t}$, y después elimina las extracciones que violan precedencia. Con reparación, toda solución evaluada es factible.

**Molienda (escenario 3).** Con velocidad relativa $n$, carga de bolas $J$ (%) y alimentación $Q$ (t/h):

$$
P_{80} = P_A \left(\frac{Q}{120}\right)^{0.5} e^{-2.2(n - 0.70)} \left(\frac{J}{32}\right)^{-0.6},
\qquad
P = P_B\, n^{1.5} \left(\frac{J}{32}\right)^{0.9} \left(0.6 + 0.4\,\frac{Q}{120}\right) \tag{5}
$$

$$
R = R_{\max}\left[1 - e^{-(P_{\text{ref}}/P_{80})^{1.5}}\right],
\qquad
V = Q\,(V_T R - C_M) - C_E\, P \tag{6}
$$

El problema es maximizar $V$ con $P_{80} \le 150\ \mu\text{m}$ y $P \le 1000$ kW, con tolerancia relativa de 0.5 % en la factibilidad.

**Sensibilidad económica (dimensionamiento).** Con $T_{\text{anual}}$ toneladas de alimentación al año, precio $P$ y ley media $\bar z$, una variación $\Delta r$ de la recuperación de zinc produce

$$
\Delta V_{\text{anual}} = \Delta r \cdot T_{\text{anual}} \cdot P \cdot \frac{\bar z}{100} \tag{7}
$$

Con $\bar z = 7.67\,\%$ (mezcla lineal óptima), $P = $ USD 2 800/t, $\Delta r = 0.01$ y $T_{\text{anual}} = 900\,000$ t (75 000 t/mes), se obtiene $\Delta V_{\text{anual}} \approx$ USD 1.9 millones por punto porcentual de recuperación. Es una cifra de dimensionamiento, no un pronóstico.

## 5. Método común

Los tres escenarios usan el núcleo `ga_comun.py`: 40 individuos, 50 generaciones, elitismo de dos individuos, torneo binario, cruce BLX-α con $\alpha = 0.3$ y mutación gaussiana con $\sigma = 0.05$ en codificación real. Cada ejecución consume 1 940 evaluaciones y la búsqueda aleatoria de control usa el mismo número. El escenario 1 incluye evolución diferencial de SciPy como referencia de un método robusto (sin presupuesto fijo, unas 18 000 evaluaciones). Las referencias exactas son programación lineal (escenario 1, modelo lineal), MILP con HiGHS (escenario 2) y malla densa con refinamiento local (escenario 3).

## 6. Escenario 1: mezcla de mineral

**Diseño.** Cinco frentes con leyes de zinc de 9.5, 7.2, 5.8, 11.4 y 4.1 %, costos de 18, 15, 14, 24 y 12 USD/t, e impurezas de 900, 300, 150, 1200 y 80 ppm. Requisitos de planta: ley media mínima de 6.0 % e impureza media máxima de 400 ppm.

**Resultados.**

| Método | Modelo | Utilidad (USD/t) | Observación |
|---|---|---|---|
| Programación lineal | Lineal | 160.47 | Óptimo exacto; $p = (0, 0.889, 0, 0.111, 0)$ |
| AG | Lineal | 155.01 ± 1.36 | Brecha de 5.5 USD/t (3.4 %) |
| AG | No lineal | 90.63 | Mezcla pura del frente 2 en 4 de 5 semillas |
| Evolución diferencial | No lineal | 90.63 | Igual que el AG en las cinco semillas |
| Búsqueda aleatoria | No lineal | 86.24 | Media de cinco semillas |
| Solución lineal evaluada | No lineal | 88.12 | Pérdida de 2.5 USD/t frente al óptimo no lineal |

En el modelo lineal, la brecha del AG proviene de la penalización blanda: sus soluciones quedan en una región cercana a la restricción de impureza, no sobre ella. Cuando el modelo es lineal, la programación lineal es preferible. En el modelo no lineal, la solución lineal pierde 2.5 USD/t, lo que cuantifica el costo de ignorar la dependencia de la recuperación respecto a la mezcla.

![Figura 1](figuras/fig1_mezcla.png)

*Figura 1. (a) Convergencia del AG en el modelo lineal frente al óptimo LP; (b) utilidad no lineal por método; (c) proporciones de la mezcla por semilla.*

## 7. Escenario 2: secuenciamiento de frentes

**Diseño.** Quince bloques, cuatro periodos, tasa de descuento de 10 %, capacidad de 260 kt por periodo y tres pares de precedencia $(0 \to 3)$, $(2 \to 7)$ y $(5 \to 11)$ en índices base 0. Los tonelajes, leyes y valores se generan con semilla 2024.

**Resultados.**

| Variante del AG | VPN por semilla (MUSD) | Factibles | Brecha media frente al MILP |
|---|---|---|---|
| MILP (referencia exacta) | 131.19 | — | — |
| Con reparación | 124.34, 127.86, 129.84, 122.45, 129.76 | 5 / 5 | 4.34 MUSD (3.3 %) |
| Penalización débil (0.5 MUSD/kt) | 124.61, 122.22, 120.14, 129.90, 131.88 | 3 / 5 | — (dos soluciones infactibles) |
| Penalización fuerte (5 MUSD/kt) | 103.36, 104.88, 122.48, 122.54, 104.50 | 5 / 5 | 19.6 MUSD |

La penalización débil permite VPN mayores que el óptimo, pero infactibles. La penalización fuerte bloquea la exploración de la frontera de capacidad, donde está el óptimo. La reparación es el operador que mantiene factibilidad sin cerrar la búsqueda.

![Figura 2](figuras/fig2_secuenciamiento.png)

*Figura 2. (a) Secuencia óptima (MILP); (b) carga por periodo frente a la capacidad; (c) convergencia del AG con reparación.*

## 8. Escenario 3: ajuste de molienda

**Diseño.** Modelo sustituto de las ecuaciones (5) y (6) con restricciones $P_{80} \le 150\ \mu$m y $P \le 1000$ kW. La configuración por defecto es $(n, J, Q) = (0.78, 30, 100)$.

**Resultados.**

| Método | Valor (USD/h) | Observación |
|---|---|---|
| Referencia (malla + refinamiento) | 5 582.40 | $n = 0.85$, $J = 30.8$, $Q = 150$ t/h; potencia en el límite |
| Por defecto | 3 854.66 | $P_{80} = 143$ µm, $P = 728$ kW |
| Nelder–Mead desde el defecto | 5 582.40 | Igual a la referencia en las cinco semillas |
| AG (media de cinco semillas) | 5 567.7 | Cuatro de cinco dentro de 0.1 %; semilla 71 en 5 511 |
| Búsqueda aleatoria (media) | 5 480.1 | 1.8 % bajo la referencia |

La mejora sobre la configuración por defecto es de USD 1 728/h (44.8 %). En esta superficie suave y de tres variables, un método local con buen punto de partida supera al AG. La ventaja del AG aparece cuando hay varios máximos locales, por ejemplo al calibrar el modelo con datos ruidosos.

![Figura 3](figuras/fig3_molienda.png)

*Figura 3. (a) Valor en el plano (n, J) para Q = 130 t/h; la zona gris es infactible; (b) convergencia del AG frente a la referencia; (c) valor en factible por método.*

## 9. Necesidades de inversión y aplicación industrial

El valor de los escenarios depende de datos de planta: leyes por frente, recuperaciones por ley, potencia en línea y un modelo de planeación minera confiable. Sin esos datos, los modelos son hipótesis. Por eso la primera inversión recomendable es de datos y capacitación, no de cómputo.

Tres necesidades se desprenden del análisis. La primera es capital humano en procesos mineros y ciencia de datos, que pueda calibrar y mantener modelos de planta. La segunda es infraestructura de energía y agua, porque la molienda concentra el consumo eléctrico y los cambios de consigna solo generan valor si el suministro lo permite. La tercera es una cadena regional de proveedores de instrumentación y software que reduzca el costo de implantación.

El dimensionamiento de la ecuación (7) indica que un punto porcentual de recuperación en una planta del tamaño de Zimapán vale del orden de USD 1.9 millones al año en valor bruto, con supuestos ilustrativos. Esa cifra justifica pilotos de instrumentación y optimización, que son de costo bajo frente a ampliaciones. La decisión final debe basarse en mejoras medidas con datos propios.

## 10. Limitaciones

- **Datos ilustrativos.** Ningún coeficiente proviene de una operación real. Los resultados describen métodos y estructuras de problema, no recomendaciones de operación.
- **Muestra pequeña.** Cinco semillas por escenario. No se reportan pruebas de significancia por la escasa potencia con cinco pares.
- **Sin incertidumbre.** No se modelan variabilidad geológica, fallas de equipo ni cambios de precio.
- **Penalización y reparación.** Los resultados dependen de la elección de penalización. El escenario 2 muestra que la reparación es más estable, pero no se explora un espacio amplio de parámetros de penalización.
- **Sin validación regional.** No existe una base pública y actualizada de producción minera por municipio para calibrar los modelos con datos locales en Hidalgo.

## 11. Referencias

1. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
2. Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2.ª ed.). Springer.
3. Dantzig, G. B. (1963). *Linear Programming and Extensions*. Princeton University Press.
4. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
5. Storn, R., & Price, K. (1997). Differential evolution: A simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359.
6. Lerchs, H., & Grossmann, I. F. (1965). Optimum design of open-pit mines. *Transactions of the CIM*, 58, 47–54.
7. Chicoisne, R., Espinoza, D., Goycoolea, M., Moreno, E., & Rubio, E. (2012). A new algorithm for the open-pit mine scheduling problem. *Operations Research*, 60(3), 517–528.
8. Goldberg, D. E., & Lingle, R. (1985). Alleles, loci, and the traveling salesman problem. En *Proceedings of the First International Conference on Genetic Algorithms*, 154–159.
9. Lamghari, A., & Dimitrakopoulos, R. (2012). A diversified Tabu search approach for the open-pit mine production scheduling problem with metal uncertainty. *European Journal of Operational Research*, 222(3), 642–652.
10. Michalewicz, Z., & Schoenauer, M. (1996). Evolutionary algorithms for constrained parameter optimization problems. *Evolutionary Computation*, 4(1), 1–32.
11. Austin, L. G., Klimpel, R. R., & Luckie, P. T. (1984). *Process Engineering of Size Reduction: Ball Milling*. SME-AIME.
12. Napier-Munn, T. J., Morrell, S., Morrison, R. D., & Kojovic, T. (1996). *Mineral Comminution Circuits: Their Operation and Optimisation*. JKMRC.
13. Deb, K. (2001). *Multi-Objective Optimization using Evolutionary Algorithms*. Wiley.
14. Mitchell, M. (1996). *An Introduction to Genetic Algorithms*. MIT Press.
15. NS Energy (2021). SantaCruz Silver enters definitive agreement to acquire the Zimapán mine and arranges financing for transaction. https://www.nsenergybusiness.com/news/santacruz-silver-enters-definitive-agreement-to-acquire-the-zimapan-mine-and-arranges-financing-for-transaction/
16. Mexico Business News (2026). Hidalgo private investment reaches MX$147.7 billion. https://mexicobusiness.news/trade-and-investment/news/hidalgo-private-investment-reaches-mx1477-billion
17. Mexico Business News (2026). Mining value up 21% in 2025 despite flat volume: CAMIMEX. https://mexicobusiness.news/mining/news/mining-value-21-2025-despite-flat-volume-camimex
18. Universidad Autónoma del Estado de Hidalgo (2025). UAEH se incorpora al Comité Educativo de Minería México 2025. https://www.uaeh.edu.mx/noticias/9861
19. Clúster Industrial (2026). Hidalgo destina MX$52.8 mdp en 2026 para modernizar parques industriales clave. https://clusterindustrial.com.mx/hidalgo-destina-mas-de-52-8-mdp-en-2026-para-modernizar-parques-industriales-clave/
