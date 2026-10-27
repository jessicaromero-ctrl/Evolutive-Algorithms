# Algoritmos Evolutivos — UPMH · Maestría en Inteligencia Artificial

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-ciencia%20de%20datos-013243?logo=numpy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![Licencia](https://img.shields.io/badge/uso-acad%C3%A9mico-lightgrey)

Este repositorio reúne el material, las implementaciones y los reportes técnicos de la materia **Algoritmos Evolutivos**, que forma parte de la Maestría en Inteligencia Artificial de la Universidad Politécnica Metropolitana de Hidalgo (UPMH). Se inscribe en la línea de investigación **Automatización de Procesos Aplicados a la Ingeniería**.

El recorrido avanza de lo conceptual a lo experimental:

1. Primero se estudian la codificación de individuos, los operadores genéticos, las funciones de aptitud y la convergencia.
2. Después se aplica un **marco común de siete etapas** a varios problemas:
   - el agente viajero;
   - una cartera de proyectos con restricciones;
   - la fijación de precios de temporada;
   - la fusión de puntuaciones clínicas, la equidad social y las trayectorias médicas;
   - la detección de fraude en reclamaciones médicas.
3. Además se incluye una implementación de **Evolución Diferencial**.

Todos los experimentos usan **datos sintéticos generados con semillas fijas**. Los resultados describen cómo se comportan los métodos bajo esos supuestos y **no constituyen evidencia clínica ni comercial**.

---

## Tabla de contenido

1. [Objetivos](#1-objetivos)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Marco de trabajo: las siete etapas de un AG](#3-marco-de-trabajo-las-siete-etapas-de-un-ag)
4. [Fundamentos matemáticos](#4-fundamentos-matemáticos)
5. [Desarrollo por actividad](#5-desarrollo-por-actividad)
6. [Resumen de resultados](#6-resumen-de-resultados)
7. [Conclusiones](#7-conclusiones)
8. [Requisitos y ejecución](#8-requisitos-y-ejecución)
9. [Autoría](#9-autoría)
10. [Referencias](#10-referencias)

---

## 1. Objetivos

### Objetivo general

Diseñar, implementar y evaluar experimentalmente algoritmos evolutivos, en especial algoritmos genéticos (AG) y evolución diferencial (DE), para resolver problemas de optimización de ingeniería. El trabajo busca justificar con evidencia cuantitativa la elección de cada operador bajo condiciones de comparación controladas: misma instancia, mismo presupuesto de evaluaciones y semillas fijas.

### Objetivos específicos

1. Formalizar la **representación** de soluciones y los **operadores genéticos**: selección, cruzamiento y mutación, en codificación binaria, real, entera y de claves aleatorias.
2. Analizar el papel de la **función de aptitud**, el escalamiento, la presión selectiva y los parámetros $(N, p_c, p_m)$ en la **convergencia**.
3. Implementar y comparar, con presupuesto idéntico, configuraciones de AG que cambian **una o varias de las siete etapas** del ciclo evolutivo.
4. Tratar **restricciones** mediante penalización ($F = J + \lambda V$), prioridad de factibilidad (orden lexicográfico) y codificaciones que garantizan la factibilidad.
5. Implementar **Evolución Diferencial** con las estrategias `rand/1`, `best/1`, `rand/2` y `current-to-best/1`, y aplicarla a funciones de prueba y a problemas de ingeniería.
6. Evaluar los resultados con **métricas reproducibles**:
   - brecha al óptimo, tasa de acierto, diversidad y evaluaciones hasta converger;
   - PR-AUC y ROC-AUC en los problemas de clasificación;
   - pruebas estadísticas no paramétricas: Friedman, Wilcoxon y Holm.
7. Contrastar el AG con **líneas base honestas** que usan el mismo presupuesto: búsqueda aleatoria, pesos uniformes, reglas vigentes, la estrategia voraz y el óptimo exacto cuando se puede calcular.

---

## 2. Estructura del repositorio

```text
Evolutive-Algorithms/
├── README.md
├── Actividad 2/
│   ├── Actividad_02_Agente_Viajero.ipynb
│   └── Corte 01-Act02- Algoritmos Evolutivos- ROMERO-LORA-SANTOS-MARTINEZ.pdf
├── Actividad 3/
│   └── ACT-03-ALG-EVOLUTIVOS-SANTOS-ROMERO-LORA-2026.pdf
├── Actividad 4/
│   └── ACT-04-ROMERO-LORA-JESSICA-MELANI/
│       ├── ACT-04-ALG-EVOL-ROMERO-LORA-JESSICA-MELANI.pdf
│       ├── ae_actividad04.py
│       └── readme.md
├── Actividad 5/
│   └── ACT-05-ALG-EVOL-CORTE-02-ROMERO-LORA-JESSICA-MELANI/
│       ├── ACT-05-ALG-EVOL-CORTE-02-ROMERO-LORA-JESSICA-MELANI.pdf
│       ├── README.md
│       ├── ga_codificacion.py
│       └── synpuf_generator_v2.py
├── Actividades Previas/
│   ├── ACT-PREV-04-ALGORITMOS-GENÉTICOS-CORTE01-JESSICA MELANI ROMERO LORA.pdf
│   ├── ACT-PREV-05-ALGORITMOS-GENÉTICOS-CORTE01-JESSICA MELANI ROMERO LORA.pdf
│   ├── ACT-PREV-06-ALGORITMOS-GENÉTICOS-CORTE02-JESSICA MELANI ROMERO LORA.pdf
│   └── previa07/
│       ├── ACT-PREV-07-ALGORITMOS-GENÉTICOS-CORTE02-JESSICA MELANI ROMERO LORA.pdf
│       ├── ga_comun.py
│       ├── esc1_medicina.py
│       ├── esc2_grupos_sociales.py
│       ├── esc3_trayectorias.py
│       ├── construir_escenarios.py
│       └── ga_codificacion.py
├── Entregas finales/
│   └── Entrega Final 1. Algoritmos_Evolutivos.pdf
└── Recursos/
    └── evolucion_diferencial.py
```

| Carpeta | Corte | Tema | Tipo de contenido |
|---|---|---|---|
| `Actividades Previas/ACT-PREV-04` | 1 | Representación de individuos, operadores genéticos y un AG mínimo en Python | Reporte PDF |
| `Actividades Previas/ACT-PREV-05` | 1 | Funciones de aptitud, escalamiento, ajuste de parámetros y convergencia | Reporte PDF |
| `Actividad 2` | 1 | Variante de AG para el **agente viajero** (TSP) con claves aleatorias | Notebook y PDF |
| `Actividad 3` | 1 | AG para la **selección de una cartera de proyectos** con restricciones | PDF |
| `Entregas finales` | 1 | Entrega final del corte 1 (versión del reporte de la Actividad 2) | PDF |
| `Actividades Previas/ACT-PREV-06` y `Recursos/` | 2 | **Evolución Diferencial**: teoría, variantes y casos de estudio | PDF y `.py` |
| `Actividad 4` | 2 | Diseño factorial 3×3 de recombinación × mutación para **precios de temporada** | `.py`, `readme.md` y PDF |
| `Actividades Previas/previa07` | 2 | Un mismo AG aplicado a **tres escenarios**: medicina, grupos sociales y trayectorias clínicas | `.py` y PDF |
| `Actividad 5` | 2 | **Tres codificaciones del cromosoma** para fusionar señales de un detector de **fraude médico** | `.py`, `README.md` y PDF |

> Los scripts generan sus tablas y figuras en carpetas de salida al ejecutarse; los comandos están en la [sección 8](#8-requisitos-y-ejecución).

---

## 3. Marco de trabajo: las siete etapas de un AG

A partir de la Actividad 2, cada algoritmo se describe como una combinación de métodos, uno por etapa, con la nomenclatura del material de la materia. Estos son los métodos que se usan en el repositorio:

| Etapa | Métodos utilizados |
|---|---|
| 1. Inicialización | M01 uniforme · M02 Bernoulli · M03 hipercubo latino (LHS) |
| 2. Evaluación | M05 directa · M07 penalización · M08 prioridad de factibilidad (lexicográfica) |
| 3. Selección | M11 ranking lineal · M12 torneo de tamaño $k$ |
| 4. Recombinación | M14 uniforme · M15 aritmética · M16 BLX-α · cruce de un punto (binaria) |
| 5. Mutación | M17 bit-flip · M18 reinicio uniforme · M19 gaussiana acotada · M20 polinomial · paso ±1 (entera) |
| 6. Reemplazo | M22 generacional elitista · M23 $(\mu+\lambda)$ |
| 7. Paro | M25 número de generaciones · M26 presupuesto de evaluaciones · M28 estancamiento |
| Métrica de diversidad | M29 dispersión poblacional |

---

## 4. Fundamentos matemáticos

Esta sección reúne solo las expresiones que aparecen en los reportes y en el código del repositorio.

### 4.1 Problema general

Todos los problemas se plantean como

$$
\theta^{*} = \arg\max_{\theta \in \Theta} F(\theta)
\qquad\text{o}\qquad
x^{*} = \arg\min_{x \in \Omega} f(x),
$$

donde el dominio $\Theta$ determina la **codificación** y la función $F$ determina el **paisaje de aptitud**.

### 4.2 Codificación y decodificación

**Binaria.** Un cromosoma $b = (b_0,\dots,b_{L-1})$ de $L$ bits se decodifica como

$$
x = \sum_{j=0}^{L-1} b_j \, 2^{L-1-j}.
$$

Ejemplo: $[1,0,1,1,0] \mapsto 22$ y $f(22)=22^2=484$.

**Código de Gray** (Actividad 5). Primero se recupera el binario con $b_0 = g_0$ y $b_j = b_{j-1} \oplus g_j$. Después, con 8 bits por gen,

$$
u_k = \frac{1}{255}\sum_{j=0}^{7} \hat b_{kj}\, 2^{7-j}.
$$

**Entera** (Actividad 5). Cada gen toma un valor $g_k \in \{0,\dots,20\}$ y se transforma con $u_k = g_k / 20$.

**Claves aleatorias** (Actividad 2). El cromosoma es $x \in [0,1)^d$ y el recorrido es $\pi = \operatorname{argsort}(x)$, rotado para empezar en la ciudad 0. Cualquier vector produce una permutación válida, así que no hace falta reparar. Sin embargo, **solo importa el orden relativo** de las claves.

**Proyección al símplex** (Actividades previa 07 y 5). Un vector no negativo $u$ se proyecta como

$$
w_k = \frac{u_k}{\sum_{m} u_m}, \qquad w_k = \tfrac{1}{K}\ \text{si}\ \textstyle\sum_m u_m = 0 .
$$

### 4.3 Inicialización

- **M01 uniforme:** $x_{ij} \sim U(\ell_j, u_j)$.
- **M02 Bernoulli** (Actividad 3): $x_{ij} \sim \text{Bernoulli}(p)$ con $p = \min\!\left(B/\sum_j c_j,\; H/\sum_j h_j\right)$.
- **M03 hipercubo latino:** con $\pi_j$ una permutación aleatoria de $\{0,\dots,N-1\}$ y $U_{ij}\sim U(0,1)$,

$$
x_{ij} = \ell_j + \frac{\pi_j(i) + U_{ij}}{N}\,(u_j - \ell_j).
$$

### 4.4 Evaluación y manejo de restricciones

**Minimización convertida en aptitud** (previa 05): $\text{fitness} = 1/f(x)$ o $\text{fitness} = C - f(x)$.

**Escalamientos** (previa 05):

| Técnica | Fórmula |
|---|---|
| Lineal | $f' = a f + b$ |
| Sigma | $f' = \max\big(f - (\mu - c\sigma),\, 0\big)$ |
| Por rango | $f' = N - \text{rango}(i)$ |
| Boltzmann | $f' = e^{f/T}$ |

**Violación de restricciones** para reglas lineales $Gx \le h$ (Actividad 4):

$$
V(x) = \sum_i \max\{0,\ (Gx - h)_i\}.
$$

En la Actividad 3 la violación se normaliza:

$$
V = \max\!\left(0,\tfrac{\text{gasto}-B}{B}\right) + \max\!\left(0,\tfrac{\text{horas}-H}{H}\right) + \sum \text{penalizaciones lógicas}.
$$

**M07 penalización:** $F = J + \lambda V$ con $\lambda = \rho \sum_j v_j$.

**M08 prioridad de factibilidad (lexicográfica):** se comparan los pares $(V, J)$. Primero gana quien tiene menor violación y, si empatan, quien tiene menor $J$ (por ejemplo $J = -Z$).

**Penalización cuadrática exterior** (Evolución Diferencial, resorte), con $\mu = 10^6$:

$$
F(x) = f(x) + \mu \sum_i \big[\max(0, g_i(x))\big]^2 .
$$

### 4.5 Selección

**Ruleta:**

$$
p_i = \frac{f_i}{\sum_{k=1}^{N} f_k}.
$$

Por ejemplo, las aptitudes $100, 200, 300, 400$ dan probabilidades de $10\%$, $20\%$, $30\%$ y $40\%$.

**Torneo (M12):** se sortean $k$ individuos y gana el de mejor aptitud. Se usa $k=2$ o $k=3$.

**Ranking lineal (M11):** si $r_i \in \{0,\dots,N-1\}$ es la posición del individuo, con 0 para el mejor, y $s \in [1,2]$ es la presión selectiva ($s = 1.7$ en la Actividad 2),

$$
p_i = \frac{1}{N}\left[s - \frac{2(s-1)\,r_i}{N-1}\right].
$$

### 4.6 Recombinación

- **Un punto:** con un punto de corte $c$, el hijo es $h = (a_{1:c},\, b_{c+1:d})$.
- **M14 uniforme:** máscara $m_j \sim \text{Bernoulli}(q)$, con $q = 0.5$; $h_j = m_j a_j + (1-m_j) b_j$.
- **M15 aritmética:** $h = \alpha a + (1-\alpha) b$, con $\alpha \sim U(0,1)$. El hijo queda sobre el segmento entre los padres.
- **M16 BLX-α:** con $I_j = |a_j - b_j|$,

$$
h_j \sim U\big[\min(a_j,b_j) - \alpha I_j,\ \max(a_j,b_j) + \alpha I_j\big].
$$

### 4.7 Mutación

- **M17 bit-flip:** $x'_j = 1 - x_j$ con probabilidad $p_m$, que suele ser $1/d$.
- **M18 reinicio uniforme:** $x'_j \sim U(\ell_j, u_j)$ con probabilidad $p_m$.
- **M19 gaussiana acotada:** con $\delta_j \sim \mathcal N(0,1)$,

$$
x'_j = \operatorname{clip}\big(x_j + \delta_j\, \sigma\,(u_j - \ell_j),\ \ell_j,\ u_j\big).
$$

- **M20 polinomial acotada** con índice $\eta$ (versión acotada de Deb, usada en `ae_actividad04.py`). Con $R_j = u_j - \ell_j$, $\delta_1 = (x_j - \ell_j)/R_j$, $\delta_2 = (u_j - x_j)/R_j$ y $r \sim U(0,1)$:

$$
\bar\delta =
\begin{cases}
\big[2r + (1-2r)(1-\delta_1)^{\eta+1}\big]^{1/(\eta+1)} - 1, & r \le 0.5,\\
1 - \big[2(1-r) + 2(r-0.5)(1-\delta_2)^{\eta+1}\big]^{1/(\eta+1)}, & r > 0.5,
\end{cases}
\qquad x'_j = x_j + \bar\delta\, R_j .
$$

El hijo queda siempre dentro de $[\ell_j, u_j]$.

En la Actividad 2 se calibró $\sigma$ midiendo la fracción de mutaciones que **cambian la ruta**. Con una máscara de Bernoulli de probabilidad $1/d$, la probabilidad de que la máscara toque al menos un gen es

$$
1 - \left(1 - \tfrac{1}{20}\right)^{20} \approx 0.6415 ,
$$

que es el techo de la tasa bruta. Como referencia, la separación media entre claves ordenadas es $1/(d+1) = 1/21 \approx 0.048$.

### 4.8 Reemplazo y paro

- **M22 generacional elitista:** se conservan los $e$ mejores (aquí $e = 2$) y se generan $N - e$ hijos por generación.
- **M23 $(\mu+\lambda)$:** sobreviven los $\mu$ mejores de la unión de padres e hijos.
- **M26 presupuesto:** las evaluaciones tras $t$ generaciones son

$$
E_t = N + t\,(N - e) \le E_{\max}.
$$

| Caso | Cálculo | Evaluaciones |
|---|---|---|
| Actividad 2 (referencia) | $100 + 500 \cdot 98$ | 49 100 |
| Actividad 4 | $100 + 203 \cdot 98$ | 19 994 |
| Actividad 5 | $60 + 85 \cdot 58$ | 4 990 |
| Previa 07 | $40 + 50 \cdot 38$ | 1 940 |

- **M28 estancamiento:** se detiene si el mejor valor no mejora en $W$ generaciones; en la Actividad 3, $W = 50$.

### 4.9 Diversidad poblacional (M29)

En codificación real (Actividad 2) se usa la dispersión

$$
D(P) = \sqrt{\frac{1}{N d}\sum_{i=1}^{N}\sum_{j=1}^{d}\left(x_{ij} - \bar x_j\right)^2 }.
$$

Para claves uniformes en $[0,1)$ su valor esperado es $\sqrt{1/12} \approx 0.2887$.

En codificación binaria (Actividad 3) se usa la diversidad de Hamming media. Si $u_i$ es el número de unos en el gen $i$,

$$
D(P) = \frac{\sum_{i=1}^{d} u_i\,(N - u_i)}{d \cdot N(N-1)/2}.
$$

### 4.10 Evolución Diferencial (DE)

Para cada vector objetivo $x_i$ de la generación $g$, con índices $r_1 \ne r_2 \ne r_3 \ne r_4 \ne r_5 \ne i$:

$$
\begin{aligned}
\textbf{rand/1:}\quad & v_i = x_{r_1} + F\,(x_{r_2} - x_{r_3})\\
\textbf{best/1:}\quad & v_i = x_{best} + F\,(x_{r_1} - x_{r_2})\\
\textbf{rand/2:}\quad & v_i = x_{r_1} + F\,(x_{r_2} - x_{r_3}) + F\,(x_{r_4} - x_{r_5})\\
\textbf{current-to-best/1:}\quad & v_i = x_i + F\,(x_{best} - x_i) + F\,(x_{r_1} - x_{r_2})
\end{aligned}
$$

**Cruzamiento binomial:**

$$
u_{ij} =
\begin{cases}
v_{ij}, & \text{si } \text{rand}_j < CR \ \text{o}\ j = j_{rand},\\
x_{ij}, & \text{en otro caso.}
\end{cases}
$$

**Selección voraz:**

$$
x_i^{(g+1)} =
\begin{cases}
u_i, & f(u_i) \le f(x_i),\\
x_i, & \text{en otro caso.}
\end{cases}
$$

**Funciones de prueba** ($D = 10$):

$$
\begin{aligned}
\text{Rosenbrock: } & f(x)=\sum_{i=1}^{D-1}\left[100\,(x_{i+1}-x_i^2)^2 + (1-x_i)^2\right]\\
\text{Rastrigin: } & f(x)=10D + \sum_{i=1}^{D}\left[x_i^2 - 10\cos(2\pi x_i)\right]\\
\text{Ackley: } & f(x)=-20\,e^{-0.2\sqrt{\frac1D\sum x_i^2}} - e^{\frac1D\sum \cos(2\pi x_i)} + 20 + e
\end{aligned}
$$

**Diseño de resorte.** Se minimiza el peso

$$
f(d,D,N) = (N+2)\,D\,d^2
$$

sujeto a cuatro restricciones:

$$
\begin{aligned}
g_1 &= 1 - \frac{D^3 N}{71785\, d^4} \le 0\\
g_2 &= \frac{4D^2 - dD}{12566\,(D d^3 - d^4)} + \frac{1}{5108\, d^2} - 1 \le 0\\
g_3 &= 1 - \frac{140.45\, d}{D^2 N} \le 0\\
g_4 &= \frac{d + D}{1.5} - 1 \le 0
\end{aligned}
$$

**Control PID.** Se minimiza

$$
\text{ITAE} = \int_0^{T} t\,\lvert e(t)\rvert\,dt
$$

sobre una planta de segundo orden con $\omega_n = 5$ y $\zeta = 0.3$, discretizada con el método de Euler y $\Delta t = 0.01$.

### 4.11 Modelos de los problemas aplicados

**Agente viajero (Actividad 2).** La matriz de costos es $D = (A + A^\top)/2$ con $A_{ij} \sim U[0,100)$; es simétrica, pero no garantiza la desigualdad triangular. El costo de un recorrido es

$$
L(\pi) = \sum_{j=0}^{d-1} D_{\pi_j,\ \pi_{(j+1) \bmod d}} .
$$

**Cartera de proyectos (Actividad 3)**, con $x \in \{0,1\}^d$:

$$
\max\ Z = \sum_j v_j x_j
$$

sujeto a:

| Restricción | Expresión |
|---|---|
| Presupuesto | $\sum_j c_j x_j \le B$ |
| Horas | $\sum_j h_j x_j \le H$ |
| Dependencias | $x_j \le x_i$ |
| Exclusiones | $x_a + x_b \le 1$ |

Con $d = 50$ hay $2^{50} \approx 10^{15}$ carteras posibles.

**Precios de temporada (Actividad 4).** Se maximiza la utilidad con demanda lineal $q = a - Bp$:

$$
\max_{p}\ \pi(p) = \sum_j (p_j - c_j)\, q_j(p) = -p^\top B p + (a + Bc)^\top p - c^\top a
$$

donde

$$
B_{jj} = \varepsilon_j \frac{q^0_j}{P_j}, \qquad B_{jk} = -\kappa_{jk}\sqrt{b_j b_k}, \qquad a = q^0 + BP .
$$

La hessiana es $-2B \prec 0$, así que el problema es **cóncavo** con reglas lineales $Gp \le h$: existencias, demanda no negativa, escalera de marcas e índice de precios. Esto permite **certificar el óptimo global** con las condiciones KKT y medir la **brecha**:

$$
\text{brecha} = 100\,\frac{\pi^* - \pi(p)}{\pi^*}\ \%,
\qquad
\text{mejora capturada} = \frac{\pi(p) - \pi(p^0)}{\pi^* - \pi(p^0)}.
$$

**Fusión de puntuaciones (previa 07 y Actividad 5).** La puntuación fusionada es

$$
s_w(i) = \sum_{k=1}^{K} w_k z_{ik},
\qquad
w \in \Delta = \Big\{w \in [0,1]^K : \textstyle\sum_k w_k = 1\Big\}.
$$

La aptitud es la precisión media (PR-AUC):

$$
\text{AP} = \sum_n (R_n - R_{n-1})\,P_n .
$$

La AP depende solo del **orden** de las puntuaciones, así que $\text{AP}(y, c\,s) = \text{AP}(y, s)$ para todo $c > 0$. Por eso la normalización al símplex elimina una redundancia de escala sin cambiar el criterio.

**Equidad (previa 07, escenario 2).** Con $s \in \{0,1\}^{Z}$, $b_i = N_i\, e_{g(i)}$ y $c_g = \sum_{i\in g} s_i b_i / \sum_{i \in g} N_i$:

$$
F(s;\lambda) = \frac{\sum_i s_i b_i}{\sum_i N_i}
- \lambda\Big(\max_g c_g - \min_g c_g\Big)
- \mu\,\Big\lvert \textstyle\sum_i s_i - K \Big\rvert .
$$

Con $Z = 20$ y $K = 6$ hay $\binom{20}{6} = 38\,760$ asignaciones factibles, así que el óptimo exacto se obtiene por enumeración.

**Alineamiento de trayectorias (previa 07, escenario 3).** Se usa Needleman–Wunsch:

$$
F(i,j) = \max\big\{F(i-1,j-1) + s(T_i,R_j),\ F(i-1,j) + g,\ F(i,j-1) + g\big\},
\qquad F(i,0) = i\,g,\quad F(0,j) = j\,g .
$$

Multiplicar $(m, x, g)$ por $c > 0$ no cambia el alineamiento óptimo. Solo importan los cocientes $x/m$ y $g/m$, así que el problema tiene **dos grados de libertad** efectivos.

### 4.12 Comparación estadística

Las configuraciones se comparan con tres pruebas:

- la **prueba de Friedman** sobre los rangos por bloque (instancia × semilla o pliegue × semilla);
- la **prueba de Wilcoxon** de rangos con signo para comparaciones pareadas;
- la **corrección de Holm** para comparaciones múltiples.

Los resultados se presentan en diagramas de diferencia crítica de Demšar (2006).

---

## 5. Desarrollo por actividad

### 5.1 Actividad previa 04 — Representación y operadores genéticos (Corte 1)

Esta actividad define los conceptos de cromosoma, gen, aptitud y generación, y los esquemas de codificación binaria, real, de permutación y entera. También explica la selección por ruleta y por torneo, el cruce de un punto y la mutación bit-flip.

Incluye un AG mínimo en Python que maximiza $f(x) = x^2$ en $[0, 31]$ con 5 bits. Sus parámetros son $N = 20$, 50 generaciones, $p_c = 0.8$ y $p_m = 0.05$. Cierra con ejercicios resueltos:

- $[1,1,0,1,0] \mapsto 26$, con $f = 676$;
- un cruce manual;
- probabilidades de ruleta.

### 5.2 Actividad previa 05 — Aptitud, parámetros y convergencia (Corte 1)

Esta actividad estudia el paisaje de aptitud, la presión selectiva y la convergencia, tanto la sana como la prematura. Presenta las técnicas de escalamiento de la sección 4.4 y los rangos típicos de los parámetros:

| Parámetro | Rango típico |
|---|---|
| $N$ | 20–200 |
| $p_c$ | 0.6–0.9 |
| $p_m$ | 0.001–0.05 |
| Generaciones | 50–500 |

Implementa un AG con torneo $k = 3$ que maximiza $f(x) = x\sin(x/10)$ con 8 bits, $N = 30$, 80 generaciones, $p_c = 0.8$ y $p_m = 0.03$. Registra la curva de convergencia.

### 5.3 Actividad 2 — Variante de AG para el agente viajero (Corte 1)

**Planteamiento.** Se usa una instancia sintética de $d = 20$ ciudades. La semilla de los datos es 20260917 y las del algoritmo van de 20260927 a 20260956: 30 corridas por configuración, 60 en total. La representación es de **claves aleatorias**.

| Etapa | Referencia | Variante |
|---|---|---|
| Inicialización | M01 uniforme | M01 uniforme |
| Evaluación | M05 directa | M05 directa |
| Selección | M12 torneo $k=3$ | **M11 ranking lineal $s=1.7$** |
| Recombinación | M14 uniforme $q=0.5$ | **M15 aritmética** |
| Mutación | M18 reinicio uniforme | **M19 gaussiana $\sigma=0.15$** |
| Reemplazo | M22 elitista $e=2$ | **M23 $(\mu+\lambda)$, $\lambda=100$** |
| Paro | M25: 500 generaciones | **M26: presupuesto** |

Ambas configuraciones comparten $N = 100$, $p_c = 0.9$ y $p_m = 1/d = 0.05$. Las dos consumen **49 100 evaluaciones**: la referencia en 500 generaciones de 98 hijos y la variante en 490 generaciones de 100 hijos.

**Implementación.** El notebook `Actividad_02_Agente_Viajero.ipynb` usa NumPy y pandas. Antes del experimento ejecuta **10 pruebas automáticas** con `unittest`, que verifican cuatro cosas:

- la reproducibilidad por semilla;
- que el presupuesto sea idéntico;
- que no haya inversión de signo en los costos;
- que los ejemplos de la literatura den lo esperado.

**Comprobaciones previas.**

*Calibración de σ*, con 20 000 mutaciones por valor:

| σ | 0.01 | 0.02 | 0.05 | 0.10 | **0.15** | 0.20 | 0.30 |
|---|---|---|---|---|---|---|---|
| Tasa bruta de cambio de ruta | 0.128 | 0.218 | 0.372 | 0.478 | **0.522** | 0.547 | 0.573 |
| Tasa condicional (≥1 gen tocado) | 0.199 | 0.340 | 0.579 | 0.744 | **0.814** | 0.852 | 0.892 |

Se eligió $\sigma = 0.15$ porque es el menor valor que supera el criterio de una tasa condicional de al menos 0.75.

*M01 frente a M03:*

| Método | Rutas distintas (de 100) | Hamming media | Dispersión de claves |
|---|---|---|---|
| M01 | 100 | 18.006 | 0.2866 |
| M03 | 100 | 18.005 | 0.2887 |

El hipercubo latino ordena mejor las claves, pero **no aumenta la diversidad de rutas**, porque `argsort` descarta la magnitud de las claves.

**Resultados** (30 semillas por configuración):

| Métrica | Referencia | Variante |
|---|---|---|
| Costo medio | **478.95** | 497.40 |
| Desviación estándar | 17.93 | 18.48 |
| Mínimo / mediana / máximo | 440.96 / 476.57 / 521.26 | 467.52 / 495.11 / 538.68 |
| Mejora relativa media | 40.90 % | 38.62 % |
| Dispersión final de la población | 0.1152 | 0.0105 |
| Tasa de acierto (≤1 % del mejor conocido) | 3.33 % | 0 % |
| Evaluaciones al 95 % de la mejora propia (mediana) | **8 087** | 11 050 |
| Tiempo por corrida (mediana) | 0.106 s | 0.185 s |

La referencia es mejor en **+18.45 unidades (+3.85 %)** en el costo medio. El mejor costo conocido es 440.957. No se aplicó prueba estadística.

### 5.4 Actividad 3 — Selección de una cartera de proyectos (Corte 1)

**Planteamiento.** Es un problema de mochila binaria con presupuesto $B$, horas $H$, dependencias y exclusiones (sección 4.11). Las instancias tienen 20 o 50 proyectos, en una versión fácil (F) y otra difícil (D).

**AG:**

| Etapa | Método |
|---|---|
| Inicialización | M02 Bernoulli |
| Evaluación | M08 lexicográfica |
| Selección | M12 torneo |
| Recombinación | M14 uniforme |
| Mutación | M17 bit-flip |
| Reemplazo | M22 con $e = 2$ |
| Paro | M28 estancamiento |

Se fijaron $N = 100$ y $p_c = 0.9$. La calibración en C50-F dio:

| Parámetro | Valores probados | Elegido |
|---|---|---|
| $p_m$ | $0.5/d$, $1/d$, $2/d$ | $1/d$ |
| $k$ | 2, 3, 5 | 3 |
| $W$ | 25, 50, 100, 200 | 50 |

**Resultados** (30 semillas, frente a búsqueda aleatoria):

| Instancia | Óptimo | Brecha AG | Brecha búsqueda aleatoria | Aciertos AG |
|---|---|---|---|---|
| E20-F | 223.45 M\$ | 0.00 % | 8.72 % | 30/30 |
| E20-D | 198.30 M\$ | 0.00 % | 12.41 % | 28/30 |
| E50-F | 512.80 M\$ | 0.87 % | 18.35 % | 8/30 |
| E50-D | 478.60 M\$ | 1.24 % | 22.10 % | 4/30 |

La prueba de Wilcoxon da $p < 0.001$ en las cuatro instancias. En el caso principal, con semilla 20261011, el AG encontró el óptimo en 4 900 evaluaciones: 13 de 20 proyectos y 223.45 M\$, con un presupuesto del 40 % de la inversión total. La diversidad bajó de cerca de 0.40 a 0.05 con $d = 20$, y la fracción de soluciones factibles subió de cerca del 30 % a más del 95 %.

**Ablación M08 frente a M07:**

| Configuración | Brecha mediana | Aciertos | Evaluaciones |
|---|---|---|---|
| **M08** | **0.87 %** | **8** | 5 800 |
| M07, ρ = 0.01 | 4.52 % | 0 | 7 200 |
| M07, ρ = 0.1 | 2.31 % | 1 | 6 400 |
| M07, ρ = 1 | 1.65 % | 3 | 5 900 |
| M07, ρ = 10 | 3.18 % | 0 | 8 100 |

La prioridad de factibilidad supera a la penalización con cualquiera de los valores de ρ probados.

### 5.5 Actividad previa 06 y `Recursos/evolucion_diferencial.py` — Evolución Diferencial (Corte 2)

**Teoría (PDF).** El reporte cubre la formulación DE/x/y/z, las variantes `rand/1`, `best/1`, `rand/2` y `current-to-best/1`, el papel de los parámetros (NP entre 5D y 10D, F entre 0.4 y 1.0, CR entre 0.1 y 0.9), los criterios de paro y las variantes autoadaptativas jDE, JADE y SHADE/L-SHADE.

**Implementación.** La clase `EvolucionDiferencial` admite las cuatro estrategias, usa cruzamiento binomial y repara los componentes fuera de rango con reinicio aleatorio dentro de los límites. Se detiene por máximo de generaciones o cuando el mejor valor cambia menos que `tol` en 20 generaciones. Trae cuatro ejemplos:

| Ejemplo | Configuración en el código |
|---|---|
| Funciones de prueba Rosenbrock, Rastrigin y Ackley ($D = 10$) | NP = 50, F = 0.8, CR = 0.9, 1 000 generaciones, semilla 42 |
| Resorte de tensión/compresión | NP = 30, F = 0.7, CR = 0.9, 500 generaciones, 30 semillas por estrategia |
| Red neuronal 2→5→1 (21 pesos, XOR extendido con 200 muestras y 5 % de ruido) | `current-to-best/1`, NP = 80, F = 0.6, CR = 0.9, 300 generaciones |
| PID ($\omega_n = 5$, $\zeta = 0.3$), minimizando el ITAE | `rand/1`, NP = 40, F = 0.7, CR = 0.8, 200 generaciones |

**Salida del script** (`python3 evolucion_diferencial.py`, semillas fijas, unos 25 s):

| Problema | `rand/1` | `best/1` | `current-to-best/1` |
|---|---|---|---|
| Rosenbrock, $f^* = 0$ | 1.87e+03 (2 350 eval.) | 5.85e−09 (19 950) | 2.04e−09 (17 950) |
| Rastrigin, $f^* = 0$ | 69.43 (1 050) | 61.40 (2 050) | 62.70 (2 750) |
| Ackley, $f^* = 0$ | 2.80 (3 150) | 5.23e−09 (16 250) | 2.25e−09 (16 050) |
| Resorte, media de 30 corridas ± d.e. | 0.012668 ± 0.000006 | 0.012665 ± 0.000001 | 0.012670 ± 0.000009 |

- **Resorte:** la mejor solución es $d = 0.05163$, $D = 0.35536$ y $N = 11.369$, con peso 0.012665, que coincide con el valor de la literatura $f^* \approx 0.012665$.
- **Red neuronal:** MSE de 0.0816 y **precisión del 94.0 %** en 24 080 evaluaciones.
- **PID:** $K_p = 49.9995$, $K_i = 26.2288$ y $K_d = 1.8072$, con ITAE de $9.46\times10^{-4}$ en 5 600 evaluaciones. El valor de $K_p$ queda en el límite superior del dominio.

En Rosenbrock, Rastrigin y Ackley, `rand/1` se detiene antes por el criterio de 20 generaciones sin mejora. En Rastrigin ninguna estrategia escapa de los óptimos locales con esta configuración.

### 5.6 Actividad 4 — Recombinación × mutación en precios de temporada (Corte 2)

**Objetivo del estudio.** Determinar qué combinación de un operador de recombinación y uno de mutación acerca más las soluciones al óptimo global con el mismo presupuesto, y justificar la elección.

**Instancias sintéticas.** Las familias de productos tienen tres niveles de marca: propia, estándar y premium.

| Elemento | Valor |
|---|---|
| Costo base | $U(8, 30)$ |
| Multiplicadores de costo por nivel | 0.7, 1.0 y 1.4 |
| Precio actual | $P = 1.45c$ |
| Piso de precio | $\ell = 1.10c$ |
| Techo de precio | $u = 1.20P$ |
| Elasticidades por nivel | $[3, 5.5]$, $[2, 3.8]$ y $[1.3, 2.4]$ |
| Existencias | $S = q^0 \cdot U(1.05, 1.5)$ |
| Sustitución débil (D) | κ = 0.10 / 0.05 |
| Sustitución fuerte (F) | κ = 0.30 / 0.12 |

La referencia exacta se obtiene partiendo de SLSQP y certificando el óptimo KKT por conjunto activo (`certificar_kkt`).

**Diseño experimental.**

- **Etapas fijas:**

  | Etapa | Método |
  |---|---|
  | Inicialización | M03 LHS con $N = 100$ |
  | Evaluación | M08 |
  | Selección | M12 con $k = 2$ |
  | Reemplazo | M22 con $e = 2$ |
  | Paro | M26 con $E_{\max} = 20\,000$ (19 994 evaluaciones) |

  Además, $p_c = 0.9$ y $p_m = 1/d$.
- **Factorial completo 3 × 3:** {M14, M15, M16} × {M18, M19, M20}.
- **Instancias:** E12-D, E12-F, E36-D y E36-F.
- **Semillas:** 30 (de 20261017 a 20261046), lo que da **1 080 corridas** más 120 de búsqueda aleatoria.
- **Calibración previa** en C24-F, con 10 semillas: BLX con $\alpha = 0.3$, gaussiana con $\sigma = 0.05$ (0.10 con M15) y polinomial con $\eta = 20$.

**Ranking global.** La prueba de Friedman da $\chi^2 = 735.13$ y $p \approx 2\times10^{-153}$.

| Configuración | Rango medio | Brecha mediana (%) | Brecha máxima (%) |
|---|---|---|---|
| M16+M19 | 2.592 | 0.078 | 0.630 |
| M14+M20 | 2.700 | 0.073 | 1.804 |
| **M16+M20** | 2.758 | 0.091 | **0.575** |
| M14+M19 | 2.992 | 0.084 | 1.718 |
| M15+M20 | 5.050 | 0.225 | 1.020 |
| M15+M19 | 5.167 | 0.202 | 0.757 |
| M16+M18 | 7.600 | 0.773 | 1.326 |
| M14+M18 | 7.800 | 0.691 | 3.915 |
| M15+M18 | 8.342 | 0.897 | 1.445 |

Se elige **M16+M20** (α = 0.3, η = 20) porque tiene la **menor brecha máxima** entre las configuraciones estadísticamente equivalentes.

**Caso principal E12-F:**

| Alternativa | Utilidad | Brecha |
|---|---|---|
| Precios actuales | 55 502.96 | 7.980 % |
| Ajuste parejo (+3 %) | 59 131.05 | 1.965 % |
| AG M15+M18 | 59 791.44 | 0.870 % |
| **AG M16+M20** | **60 263.46** | **0.088 %** (captura el 98.9 % de la mejora) |
| Óptimo certificado | 60 316.30 | 0 |

El script ejecuta **16 pruebas automáticas** antes del experimento.

### 5.7 Actividad previa 07 — Un mismo AG en tres escenarios (Corte 2)

**Núcleo común** (`ga_comun.py`):

| Parámetro | Valor |
|---|---|
| Población $N$ | 40 |
| Generaciones | 50 |
| Elitismo | 2 |
| Torneo | binario |
| $p_c$ | 0.9 |
| BLX con $\alpha$ | 0.3 |
| Mutación gaussiana con $\sigma$ | 0.05 |
| Evaluaciones por corrida | 1 940 |
| Semillas | 11, 23, 37, 53 y 71 |

Los escenarios 1 y 3 usan codificación real y el escenario 2 usa codificación binaria (cruce de un punto y mutación bit a bit). La búsqueda aleatoria de control usa el mismo número de evaluaciones.

**Escenario 1 — Fusión de riesgo clínico.** Es una cohorte sintética de 4 000 pacientes con una prevalencia del 12.2 % y 5 modelos ruidosos. Resultados en prueba:

| Método | AP en prueba (media ± d.e.) | ROC-AUC |
|---|---|---|
| AG | 0.7737 ± 0.0308 | 0.9522 |
| Búsqueda aleatoria | 0.7727 ± 0.0304 | 0.9521 |
| Pesos uniformes | 0.7413 ± 0.0290 | 0.9453 |
| Mejor modelo individual | 0.7503 ± 0.0304 | 0.9452 |
| Oráculo (z verdadero) | 0.8054 ± 0.0246 | 0.9592 |

**Escenario 2 — Equidad en un programa preventivo.** Hay $Z = 20$ zonas en 3 grupos con eficiencias de 0.80, 0.60 y 0.40, y se eligen $K = 6$ zonas con $\mu = 1$.

| λ | F exacta | Cobertura total | Brecha entre grupos | AG alcanza el óptimo |
|---|---|---|---|---|
| 0.0 | 0.2903 | 0.2903 | 0.3336 | 3/5 |
| 0.25 | 0.2713 | 0.2754 | 0.0164 | 1/5 |
| 0.5 | 0.2672 | 0.2754 | 0.0164 | 0/5 |
| 1.0 | 0.2590 | 0.2754 | 0.0164 | 2/5 |
| 2.0 | 0.2479 | 0.2573 | 0.0047 | 1/5 |

Con λ = 1, el criterio de equidad reduce la cobertura total un **5.1 %** y baja la brecha entre grupos de 0.334 a 0.016. El AG encuentra el óptimo exacto en 7 de las 25 corridas.

**Escenario 3 — Alineamiento de trayectorias clínicas.** Hay 800 trayectorias de longitud 8, con un 25 % de desviadas.

| Método | AP en prueba | ROC-AUC | Diferencia AG − método | p de Wilcoxon |
|---|---|---|---|---|
| AG | 0.8140 ± 0.0254 | 0.9133 | — | — |
| Búsqueda aleatoria | 0.8082 ± 0.0345 | 0.9134 | +0.0058 | 0.438 |
| Parámetros por defecto (1, −1, −1) | 0.8009 ± 0.0223 | 0.9140 | +0.0131 | 0.062 |

### 5.8 Actividad 5 — Codificación del cromosoma para detectar fraude médico (Corte 2)

**Objetivo general.** Diseñar, implementar y evaluar un AG que aprenda los pesos de fusión de las seis señales de un detector de fraude, comparando tres codificaciones del cromosoma, con validación por proveedor y un presupuesto de revisión fijo.

**Datos y señales.** `synpuf_generator_v2.py` genera un conjunto sintético tipo SynPUF con 800 proveedores, 50 000 reclamaciones, 314 878 eventos y un **5.58 % de fraude**, con semilla 2024. Las seis señales son:

- `s_struct`: estructura de procesos de Petri por *token replay*;
- `s_sem`;
- `s_temp`;
- `s_xgb_static`;
- `s_xgb_cas`;
- `s_if`.

Se calculan fuera de pliegue (OOF) en 5 pliegues por proveedor.

**Codificaciones comparadas:**

| Codificación | Genotipo | Cruce | Mutación |
|---|---|---|---|
| G1 binaria (Gray) | 6 × 8 = 48 bits | un punto | bit-flip con $p = 1/48$ |
| G2 real | 6 reales en $[0,1]$ | BLX-α con α = 0.3 | gaussiana con σ = 0.05 y $p = 1/6$ |
| G3 entera | 6 enteros en $\{0,\dots,20\}$ | uniforme | paso ±1 |

Las tres comparten la aptitud (PR-AUC en los pliegues de entrenamiento), M03, torneo con $k = 2$, $N = 60$, $e = 2$ y $E_{\max} = 5\,000$ (4 990 evaluaciones). Hay 5 semillas por cada uno de los 5 pliegues, es decir, 25 bloques.

**Resultados en prueba** (dificultad 1.0):

| Método | PR-AUC (media ± d.e.) |
|---|---|
| G1 Gray | 0.9712 ± 0.0078 |
| G2 real | 0.9712 ± 0.0078 |
| G3 entera | **0.9713 ± 0.0077** |
| Búsqueda aleatoria | 0.9687 |
| Mejor señal individual | 0.9703 |
| Pesos uniformes | 0.9534 |

La prueba de Friedman da $\chi^2 = 88.2$. Estos son los valores $p$ ajustados con Holm para G2 frente a cada alternativa:

| Comparación | p ajustado |
|---|---|
| G2 vs G1 | 0.346 |
| G2 vs G3 | 0.563 |
| G2 vs búsqueda aleatoria | < 10⁻⁴ |
| G2 vs pesos uniformes | < 10⁻⁴ |
| G2 vs mejor señal | 0.0029 |

- **Diversidad final:** G1 0.946, G2 0.981 y G3 0.474.
- **Pesos aprendidos:** casi toda la masa va a las señales XGBoost: cerca de 0.43 para `s_xgb_cas` y entre 0.56 y 0.57 para `s_xgb_static`.
- **Datos más difíciles (dificultad 0.25):** G1 y G2 obtienen 0.9243 y G3 0.9239, que es significativamente peor ($\chi^2 = 58.4$). La ventaja de la fusión sobre la mejor señal es de +0.0008, frente a +0.0009 con dificultad 1.0, así que la hipótesis H5 (que la ventaja crece al bajar la dificultad) no se confirma.

---

## 6. Resumen de resultados

| Actividad | Problema | Hallazgo principal |
|---|---|---|
| 2 | TSP con 20 ciudades y claves aleatorias | La referencia (torneo, cruce uniforme, reinicio y elitismo) supera a la variante en un 3.85 % y converge antes (8 087 frente a 11 050 evaluaciones). La variante pierde diversidad (0.010 frente a 0.115). |
| 3 | Cartera de proyectos binaria | Brecha de 0 % con $d = 20$ y de 0.87–1.24 % con $d = 50$, frente al 8.7–22.1 % de la búsqueda aleatoria. M08 supera a M07 con cualquier ρ probado. |
| Previa 06 | Evolución Diferencial | `best/1` y `current-to-best/1` llegan a cerca de $10^{-9}$ en Rosenbrock y Ackley. El resorte alcanza $f = 0.012665$, igual al valor de la literatura. |
| 4 | Precios de temporada (QP cóncavo) | M16+M20 queda a una brecha mediana de 0.09 % del óptimo certificado. La mutación por reinicio (M18) es la peor en todas las instancias. |
| Previa 07 | Tres escenarios con datos sintéticos | El AG empata con la búsqueda aleatoria en los escenarios 1 y 3 y supera los valores por defecto o uniformes. En el escenario 2 se acerca al óptimo exacto y aventaja mucho a la búsqueda aleatoria. |
| 5 | Fusión de señales para fraude | Las tres codificaciones son equivalentes (PR-AUC ≈ 0.971) y superan a los pesos uniformes y a la búsqueda aleatoria. La entera pierde diversidad y es peor con datos difíciles. |

---

## 7. Conclusiones

1. **La representación condiciona a los operadores.** Con claves aleatorias (Actividad 2), `argsort` descarta la magnitud de las claves. Por eso el hipercubo latino no aumenta la diversidad de rutas y $\sigma$ tuvo que calibrarse según su **efecto en el fenotipo**: la tasa condicional fue de 0.81 con $\sigma = 0.15$.
2. **Los operadores que contraen la población aceleran la pérdida de diversidad.** El cruce aritmético (M15) interpola sin extrapolar. En la Actividad 2 la variante terminó con una dispersión diez veces menor y fue más lenta. En la Actividad 4, M15 produce hijos casi siempre factibles, pero no llega a la cima.
3. **La mutación pesa más que la recombinación** en el problema de precios. El reinicio uniforme (M18) es la peor opción en las cuatro instancias, mientras que las perturbaciones locales M19 y M20 capturan la mayor parte de la mejora posible. Se eligió M16+M20 por su **menor brecha máxima**.
4. **Las restricciones se manejan mejor con prioridad de factibilidad.** En la cartera de proyectos, M08 obtuvo una brecha mediana de 0.87 % frente a entre 1.65 % y 4.52 % de la penalización, y esta última es muy sensible a ρ.
5. **El AG aporta más cuando el paisaje es discreto o tiene aristas.** En la cartera (Actividad 3) y en el escenario de equidad (previa 07) el AG supera claramente a la búsqueda aleatoria. En paisajes suaves de pocas dimensiones, como la fusión en un símplex (previa 07, escenarios 1 y 3, y Actividad 5), la búsqueda aleatoria con el mismo presupuesto obtiene resultados casi equivalentes.
6. **La codificación importa poco cuando el fenotipo y la aptitud son los mismos** (Actividad 5). Gray, real y entera empatan en PR-AUC, aunque la entera reduce la diversidad (0.474) y se degrada con datos difíciles.
7. **En Evolución Diferencial la estrategia define el equilibrio entre exploración y explotación.** `best/1` y `current-to-best/1` convergen con precisión en funciones unimodales o suaves. En Rastrigin ninguna escapa de los óptimos locales con la configuración usada, y el criterio de 20 generaciones sin mejora detiene `rand/1` antes de tiempo.
8. **El rigor experimental es parte del resultado.** Esto incluye igualar el presupuesto, fijar semillas, comparar con el óptimo exacto o certificado cuando existe, ejecutar pruebas automáticas y aplicar pruebas no paramétricas. Una variante que no mejora a la referencia también es un resultado válido si el diseño es riguroso.
9. **El AG es objeto de estudio, no siempre la herramienta de producción.** En la Actividad 4 un solucionador cuadrático resuelve cada instancia en menos de 0.11 s. El AG cobra sentido práctico con demanda no cóncava o reglas no lineales.

**Limitaciones generales.**

- Todos los datos son sintéticos.
- En varias actividades se usan pocas semillas (5 en la previa 07).
- En la Actividad 2 no se aplicó prueba estadística.
- Ningún modelo de los escenarios clínicos o sociales debe usarse para tomar decisiones reales sin validarlo con datos reales y auditarlo.

---

## 8. Requisitos y ejecución

### Requisitos

Se recomienda **Python 3.10 o superior**. Las Actividades 2 y 4 se probaron con Python 3.11 y la Actividad 5 con Python 3.13.

| Paquete | Se usa en |
|---|---|
| `numpy` | todos los scripts |
| `pandas` | Actividad 2, Actividad 4, Actividad 5 |
| `matplotlib` | Actividad 2, Actividad 4, Actividad 5, previa 07 |
| `scipy` | Actividad 4 (`minimize`, `friedmanchisquare`, `wilcoxon`), Actividad 5, previa 07 (`construir_escenarios.py`) |
| `scikit-learn` | Actividad 5 y previa 07 (`average_precision_score`, `roc_auc_score`, `train_test_split`) |
| `python-docx` | previa 07 (`construir_escenarios.py`, que genera el reporte .docx) |
| `jupyter` / `ipython` | Actividad 2 (notebook) |

```bash
pip install numpy pandas matplotlib scipy scikit-learn python-docx jupyter
```

### Ejecución

```bash
# Actividad 2 — Agente viajero (notebook)
jupyter notebook "Actividad 2/Actividad_02_Agente_Viajero.ipynb"

# Previa 06 — Evolución Diferencial (≈25 s)
python3 Recursos/evolucion_diferencial.py

# Actividad 4 — Precios de temporada (≈2 min; --sin-figuras genera solo las tablas)
cd "Actividad 4/ACT-04-ROMERO-LORA-JESSICA-MELANI"
python ae_actividad04.py --salida resultados_actividad04

# Previa 07 — Tres escenarios
cd "Actividades Previas/previa07"
python esc1_medicina.py
python esc2_grupos_sociales.py
python esc3_trayectorias.py

# Actividad 5 — Generar los datos sintéticos
cd "Actividad 5/ACT-05-ALG-EVOL-CORTE-02-ROMERO-LORA-JESSICA-MELANI"
python synpuf_generator_v2.py --out-dir data --seed 2024
# Después, ejecutar el AG sobre las señales OOF (oof_predictions.csv)
python ga_codificacion.py --resultados <carpeta_con_oof_predictions.csv> --salida <carpeta_salida>
```

---

## 9. Autoría

| | |
|---|---|
| **Alumna** | Jessica Melani Romero Lora (matrícula 25322116) |
| **Coautor (Actividades 2 y 3)** | Víctor Manuel Santos Martínez (253220020) · [github.com/VManuelSM](https://github.com/VManuelSM) |
| **Profesor** | Dr. Jaime Aguilar Ortíz |
| **Materia** | Algoritmos Evolutivos |
| **Programa** | Maestría en Inteligencia Artificial, UPMH |
| **Línea de investigación** | Automatización de Procesos Aplicados a la Ingeniería |
| **Periodo** | Septiembre–diciembre de 2026 |

---

## 10. Referencias

- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
- Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata. *Foundations of Genetic Algorithms*, 2, 187–202.
- Storn, R., & Price, K. (1997). Differential Evolution – A simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359.
- Brest, J. et al. (2006). Self-adapting control parameters in differential evolution. *IEEE Trans. Evol. Comput.*, 10(6), 646–657.
- Zhang, J., & Sanderson, A. (2009). JADE: Adaptive differential evolution with optional external archive. *IEEE Trans. Evol. Comput.*, 13(5), 945–958.
- Das, S., & Suganthan, P. N. (2011). Differential evolution: A survey of the state-of-the-art. *IEEE Trans. Evol. Comput.*, 15(1), 4–31.
- Tanaka, R., & Fukunaga, A. (2013). Success-history based parameter adaptation for differential evolution. *IEEE CEC 2013*, 71–78.
- Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2.ª ed.). Springer.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7, 1–30.
- Needleman, S. B., & Wunsch, C. D. (1970). A general method applicable to the search for similarities in the amino acid sequence of two proteins. *J. Mol. Biol.*, 48(3), 443–453.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3).
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*, 29.
- van der Aalst, W. M. P. (2016). *Process Mining: Data Science in Action* (2.ª ed.). Springer.
