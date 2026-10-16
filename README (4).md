<p align="center">
  <img src="https://img.shields.io/badge/Maestr%C3%ADa-Inteligencia%20Artificial-333333?style=for-the-badge" alt="Maestría en IA"/>
  <img src="https://img.shields.io/badge/UPMH-Algoritmos%20Evolutivos-555555?style=for-the-badge" alt="Algoritmos Evolutivos"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-777777?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

<h1 align="center">Algoritmos Evolutivos</h1>

<p align="center">
  <strong>Universidad Politécnica Metropolitana del Estado de Hidalgo</strong><br/>
  Maestría en Inteligencia Artificial<br/>
  Línea de investigación: <em>Automatización de Procesos Aplicados a la Ingeniería</em>
</p>

<p align="center">
  <strong>Alumna:</strong> Jessica Melani Romero Lora<br/>
  <strong>Profesor:</strong> Dr. Jaime Aguilar Ortiz<br/>
  <strong>Periodo:</strong> 2026
</p>

---

## Acerca del Repositorio

Este repositorio contiene el material, documentos, implementaciones y reportes desarrollados para la materia **Algoritmos Evolutivos** dentro del programa de Maestría en Inteligencia Artificial de la UPMH. El trabajo se enmarca en la línea de investigación de **Automatización de Procesos Aplicados a la Ingeniería**, donde las técnicas de computación evolutiva constituyen herramientas fundamentales para resolver problemas de optimización complejos, no lineales y de alta dimensionalidad que surgen en procesos industriales, diseño ingenieril y sistemas de control.

Los algoritmos evolutivos representan un paradigma de optimización inspirado en los mecanismos de la evolución biológica — selección natural, recombinación genética y mutación — aplicados como operadores computacionales sobre poblaciones de soluciones candidatas. Su relevancia en la automatización de procesos radica en su capacidad para optimizar sistemas donde los métodos analíticos clásicos resultan inviables: funciones objetivo no diferenciables, espacios de búsqueda discontinuos, múltiples óptimos locales y restricciones no lineales.

---

## Fundamentos Teóricos

### Algoritmos Genéticos (AG)

Un Algoritmo Genético opera sobre una población $P = \{x_1, x_2, \ldots, x_N\}$ de $N$ individuos, donde cada individuo $x_i$ es una codificación del espacio de soluciones. El proceso evolutivo se rige por tres operadores principales:

**Selección por ruleta:** La probabilidad de selección de un individuo $x_i$ es proporcional a su aptitud relativa:

$$p_i = \frac{f(x_i)}{\sum_{j=1}^{N} f(x_j)}$$

**Selección por torneo:** Se eligen $k$ individuos al azar y se selecciona el de mayor aptitud:

$$x^* = \arg\max_{x \in T_k} f(x), \quad T_k \subset P, \; |T_k| = k$$

**Cruzamiento de un punto:** Dados dos padres $x_a$ y $x_b$, se selecciona un punto de corte $c$ y se generan dos descendientes:

$$\text{hijo}_1 = (x_{a,1}, \ldots, x_{a,c}, x_{b,c+1}, \ldots, x_{b,D})$$

$$\text{hijo}_2 = (x_{b,1}, \ldots, x_{b,c}, x_{a,c+1}, \ldots, x_{a,D})$$

**Mutación (bit-flip):** Cada gen $x_{i,j}$ se altera con probabilidad $p_m$:

$$x_{i,j}' = \begin{cases} 1 - x_{i,j} & \text{si } r < p_m \\ x_{i,j} & \text{en otro caso} \end{cases}, \quad r \sim U(0,1)$$

**Funciones de aptitud y escalamiento:** La función de aptitud $f(x)$ evalúa la calidad de cada solución. Para problemas de minimización, se utiliza la inversión:

$$f'(x) = \frac{1}{1 + g(x)}$$

donde $g(x)$ es la función objetivo a minimizar. Los métodos de escalamiento incluyen:

- **Escalamiento lineal:** $f'(x) = a \cdot f(x) + b$
- **Escalamiento sigma:** $f'(x) = \max\left(f(x) - (\bar{f} - c \cdot \sigma_f), \, 0\right)$
- **Escalamiento por rango:** $f'(x) = \text{rank}(x)$ en orden de aptitud

**Convergencia:** Un AG converge cuando la diversidad poblacional cae por debajo de un umbral o la mejora generacional es despreciable:

$$\Delta_g = |f_{\text{best}}^{(g)} - f_{\text{best}}^{(g-k)}| < \varepsilon$$

### Evolución Diferencial (DE)

La Evolución Diferencial es un algoritmo de optimización para espacios continuos $\mathbb{R}^D$ propuesto por Storn y Price (1997). Su operador fundamental es la **mutación diferencial**, que utiliza diferencias vectoriales entre individuos de la población como mecanismo de perturbación auto-adaptativo.

**Mutación (DE/rand/1/bin):**

$$v_i = x_{r_1} + F \cdot (x_{r_2} - x_{r_3}), \quad r_1 \neq r_2 \neq r_3 \neq i$$

donde $F \in [0, 2]$ es el factor de escala que controla la amplitud de la perturbación.

**Mutación (DE/best/1/bin):**

$$v_i = x_{\text{best}} + F \cdot (x_{r_1} - x_{r_2})$$

**Mutación (DE/current-to-best/1):**

$$v_i = x_i + F \cdot (x_{\text{best}} - x_i) + F \cdot (x_{r_1} - x_{r_2})$$

**Cruzamiento binomial:** Se genera el vector de prueba $u_i$ componente a componente:

$$u_{i,j} = \begin{cases} v_{i,j} & \text{si } \text{rand}_j < CR \text{ o } j = j_{\text{rand}} \\ x_{i,j} & \text{en otro caso} \end{cases}$$

donde $CR \in [0, 1]$ es la probabilidad de cruzamiento y $j_{\text{rand}}$ garantiza al menos un componente del donante.

**Selección greedy:**

$$x_i^{(g+1)} = \begin{cases} u_i & \text{si } f(u_i) \leq f(x_i^{(g)}) \\ x_i^{(g)} & \text{en otro caso} \end{cases}$$

**Funciones benchmark estándar:**

| Función | Fórmula | Óptimo |
|---------|---------|--------|
| Rosenbrock | $f(x) = \sum_{i=1}^{D-1} \left[100(x_{i+1} - x_i^2)^2 + (1-x_i)^2\right]$ | $f(1,\ldots,1) = 0$ |
| Rastrigin | $f(x) = 10D + \sum_{i=1}^{D} \left[x_i^2 - 10\cos(2\pi x_i)\right]$ | $f(0,\ldots,0) = 0$ |
| Ackley | $f(x) = -20\exp\left(-0.2\sqrt{\frac{1}{D}\sum x_i^2}\right) - \exp\left(\frac{1}{D}\sum\cos(2\pi x_i)\right) + 20 + e$ | $f(0,\ldots,0) = 0$ |

### Manejo de Restricciones

Para problemas con restricciones $g_j(x) \leq 0$, se emplea **penalización cuadrática exterior**:

$$F(x) = f(x) + \mu \sum_{j=1}^{m} \left[\max(0, g_j(x))\right]^2, \quad \mu \gg 1$$

---

## Relevancia en Automatización de Procesos de Ingeniería

La materia de Algoritmos Evolutivos se inserta en la línea de investigación de **Automatización de Procesos Aplicados a la Ingeniería** por las siguientes razones:

| Área de aplicación | Problema | Contribución de AE |
|---|---|---|
| **Diseño mecánico** | Optimización de componentes (resortes, vigas, engranajes) | Minimización de peso/costo sujeta a restricciones de esfuerzo, deflexión y geometría |
| **Control automático** | Sintonización de controladores PID | Optimización directa de criterios ITAE, ISE, IAE sin restricciones de linealidad |
| **Aprendizaje automático** | Entrenamiento de redes neuronales | Búsqueda global de pesos sin dependencia del gradiente |
| **Planificación de producción** | Selección de carteras de proyectos | Optimización combinatoria con dependencias y exclusiones mutuas |
| **Procesamiento de señales** | Diseño de filtros IIR/FIR | Ajuste de coeficientes en funciones de transferencia no convexas |
| **Energía** | Despacho económico de generadores | Asignación óptima de carga en sistemas multiobjetivo |

Los algoritmos evolutivos proporcionan soluciones de alta calidad en problemas donde la automatización clásica (métodos basados en gradiente, programación lineal) resulta insuficiente: espacios de búsqueda no convexos, funciones objetivo ruidosas o discontinuas y sistemas con múltiples criterios de optimización simultáneos.

---

## Estructura del Repositorio

```
algoritmos-evolutivos/
├── README.md
├── Actividad_01/
│   └── AG_JessicaRomero.docx            # Representación, codificación y operadores genéticos
├── Actividad_02/
│   └── AG_Fitness_Convergencia.docx      # Funciones de aptitud, ajuste de parámetros y convergencia
├── Actividad_03/
│   ├── Reporte_Cartera_Proyectos.docx    # Reporte analítico de GA para selección de cartera
│   └── cartera_proyectos_mejorado.py     # Implementación mejorada del AG para cartera de proyectos
├── Actividad_04/
│   ├── ED_EvolucionDiferencial.docx      # Documento completo sobre Evolución Diferencial
│   └── evolucion_diferencial.py          # Implementación de DE con 4 ejemplos prácticos
└── ...                                   # Actividades futuras
```

---

## Desglose de Actividades

### Actividad 01 — Representación, Codificación y Operadores Genéticos

**Archivo:** `AG_JessicaRomero.docx`

Documento que aborda los fundamentos de los Algoritmos Genéticos con enfoque en la codificación de individuos y los operadores evolutivos. Contenido:

- **Codificación binaria, real y por permutación:** esquemas de representación de soluciones según la naturaleza del problema (optimización numérica, combinatoria, secuenciación).
- **Operadores de selección:** ruleta, torneo y selección por rango, con análisis de la presión selectiva que cada mecanismo ejerce sobre la población.
- **Operadores de cruzamiento:** un punto, dos puntos y uniforme, incluyendo las tasas de cruzamiento recomendadas ($p_c \in [0.6, 0.9]$).
- **Operador de mutación:** bit-flip para codificación binaria con tasas típicas $p_m \in [1/D, \, 0.05]$.
- **Elitismo:** preservación del mejor individuo para garantizar convergencia monótona.
- Ejercicios resueltos, definiciones prácticas y ejemplos con código.

---

### Actividad 02 — Funciones de Aptitud, Ajuste de Parámetros y Convergencia

**Archivo:** `AG_Fitness_Convergencia.docx`

Documento enfocado en la evaluación de soluciones y la configuración del algoritmo genético. Contenido:

- **Funciones de aptitud:** maximización directa, inversión para minimización, funciones de penalización para restricciones.
- **Escalamiento de aptitud:** lineal ($f' = af + b$), sigma ($f' = f - (\bar{f} - c\sigma)$), por rango y de Boltzmann ($p_i \propto e^{f_i / T}$).
- **Ajuste de parámetros:** tamaño de población ($N$), probabilidad de cruzamiento ($p_c$), probabilidad de mutación ($p_m$) y sus interacciones.
- **Criterios de convergencia:** estagnación del mejor valor, reducción de diversidad poblacional, número máximo de generaciones.
- Visualizaciones de paisajes de aptitud y curvas de convergencia.

---

### Actividad 03 — Selección Óptima de Cartera de Proyectos con AG

**Archivos:** `Reporte_Cartera_Proyectos.docx` + `cartera_proyectos_mejorado.py`

Reporte analítico sobre la aplicación de un Algoritmo Genético al problema de selección de cartera de proyectos, formulado como un problema de mochila binaria con dependencias y exclusiones mutuas.

**Formulación del problema:**

$$\max \sum_{i=1}^{n} b_i \cdot x_i \quad \text{s.a.} \quad \sum_{i=1}^{n} c_i \cdot x_i \leq B, \quad x_i \in \{0, 1\}$$

con restricciones de dependencia ($x_j = 1 \Rightarrow x_i = 1$) y exclusión mutua ($x_i + x_j \leq 1$).

**Contenido del reporte:**
- Marco teórico del problema de cartera con 7 etapas del framework GA.
- Evaluación de factibilidad por prioridad (método M08).
- Métricas de desempeño y formulaciones matemáticas.

**Mejoras implementadas en `cartera_proyectos_mejorado.py`:**
1. Operador de reparación para factibilidad garantizada
2. Mutación adaptativa ($p_m$ decreciente con las generaciones)
3. Cruzamiento de dos puntos
4. Selección por distancia de crowding
5. Inmigración por diversidad
6. Caché de evaluaciones para eficiencia
7. Interfaz CLI con `argparse`

---

### Actividad 04 — Evolución Diferencial: Implementación y Estudio de Casos

**Archivos:** `ED_EvolucionDiferencial.docx` + `evolucion_diferencial.py`

Documento de 8 páginas y archivo Python con la implementación completa de Evolución Diferencial. Cubre las variantes principales, aplicaciones prácticas y estudio de casos.

**Contenido del documento:**
- Fundamentos y formulación matemática de DE
- Variantes: DE/rand/1/bin, DE/best/1/bin, DE/rand/2/bin, DE/current-to-best/1
- Análisis de convergencia en funciones benchmark
- Caso de estudio: diseño óptimo de resorte de tensión/compresión
- Aplicaciones: entrenamiento de redes neuronales y ajuste de controladores PID
- Variantes auto-adaptativas: jDE, SHADE, JADE

**Ejemplos en `evolucion_diferencial.py`:**

| Ejemplo | Problema | Variables | Resultado |
|---------|----------|-----------|-----------|
| 1 | Funciones benchmark (Rosenbrock, Rastrigin, Ackley) | $D = 10$ | Convergencia a $f^* \approx 0$ |
| 2 | Diseño de resorte | $d, D, N$ | Peso $\approx 0.01267$ kg |
| 3 | Red neuronal 2→5→1 | 21 pesos | Precisión $> 94\%$ |
| 4 | Controlador PID | $K_p, K_i, K_d$ | ITAE reducido 23% vs Z-N |

---

## Ejecución

### Requisitos

```bash
pip install numpy matplotlib
```

### Evolución Diferencial

```bash
python evolucion_diferencial.py
```

Ejecuta los 4 ejemplos prácticos: funciones benchmark, diseño de resorte, entrenamiento de red neuronal y ajuste de PID. La clase `EvolucionDiferencial` soporta las estrategias `rand/1`, `best/1`, `rand/2` y `current-to-best/1`.

### Cartera de Proyectos

```bash
python cartera_proyectos_mejorado.py --generaciones 200 --poblacion 100
```

---

## Referencias

1. Holland, J.H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
2. Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.
3. Storn, R. & Price, K. (1997). Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. *Journal of Global Optimization*, 11(4), 341–359.
4. Brest, J. et al. (2006). Self-Adapting Control Parameters in Differential Evolution. *IEEE Trans. Evol. Comput.*, 10(6), 646–657.
5. Zhang, J. & Sanderson, A. (2009). JADE: Adaptive Differential Evolution with Optional External Archive. *IEEE Trans. Evol. Comput.*, 13(5), 945–958.
6. Das, S. & Suganthan, P.N. (2011). Differential Evolution: A Survey of the State-of-the-Art. *IEEE Trans. Evol. Comput.*, 15(1), 4–31.
7. Eiben, A.E. & Smith, J.E. (2015). *Introduction to Evolutionary Computing*. 2nd ed. Springer.

---

<p align="center">
  <em>Universidad Politécnica Metropolitana del Estado de Hidalgo — Maestría en Inteligencia Artificial — 2026</em>
</p>
