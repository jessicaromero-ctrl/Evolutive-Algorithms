<div align="center">

# UNIVERSIDAD POLITÉCNICA METROPOLITANA DE HIDALGO

**Maestría en Inteligencia Artificial**

---

## Algoritmos Evolutivos · Actividad 04 · Corte 2

# Comparación de tres formas de recombinación y tres de mutación en un algoritmo genético para fijar precios de temporada

| | |
|---|---|
| **Alumna** | Jessica Melani Romero Lora |
| **Matrícula** | 25322116 |
| **Catedrático** | Dr. Jaime Aguilar Ortiz |
| **Asignatura** | Algoritmos Evolutivos |
| **Periodo** | Septiembre–diciembre de 2026 |
| **Fecha de entrega** | 18 de octubre de 2026 |

</div>

---

## Tabla de contenido

1. [Resumen](#1-resumen)
2. [Planteamiento del problema](#2-planteamiento-del-problema)
3. [Modelo de demanda y utilidad](#3-modelo-de-demanda-y-utilidad)
4. [Reglas comerciales y factibilidad](#4-reglas-comerciales-y-factibilidad)
5. [Generación de instancias sintéticas](#5-generación-de-instancias-sintéticas)
6. [Referencia exacta: óptimo global certificado](#6-referencia-exacta-óptimo-global-certificado)
7. [Líneas base](#7-líneas-base)
8. [Algoritmo genético: las siete etapas](#8-algoritmo-genético-las-siete-etapas)
9. [Propiedades teóricas de los operadores](#9-propiedades-teóricas-de-los-operadores)
10. [Diseño experimental e hipótesis](#10-diseño-experimental-e-hipótesis)
11. [Calibración de parámetros](#11-calibración-de-parámetros)
12. [Análisis estadístico](#12-análisis-estadístico)
13. [Resultados](#13-resultados)
14. [Discusión](#14-discusión)
15. [Limitaciones](#15-limitaciones)
16. [Implementación y reproducibilidad](#16-implementación-y-reproducibilidad)
17. [Ejemplos de uso](#17-ejemplos-de-uso)
18. [Estructura del repositorio](#18-estructura-del-repositorio)
19. [Procedencia de la base de código](#19-procedencia-de-la-base-de-código)
20. [Referencias](#20-referencias)

---

## 1. Resumen

Se formula la fijación de precios de temporada de una categoría de productos como un problema de optimización con objetivo cuadrático cóncavo y restricciones lineales. Como la función objetivo es estrictamente cóncava y el conjunto factible es un poliedro convexo, el óptimo global es único y puede certificarse exactamente mediante las condiciones de Karush-Kuhn-Tucker (KKT). Esa referencia exacta permite medir la brecha de cada método respecto al óptimo y no sólo respecto a otros métodos.

Sobre este problema se estudia un algoritmo genético (AG) de siete etapas en el que dos etapas se realizan de tres formas cada una:

- **Recombinación:** uniforme por máscara (M14), aritmética (M15) y BLX-α (M16).
- **Mutación:** reinicio uniforme (M18), gaussiana acotada (M19) y polinomial acotada (M20).

Las nueve combinaciones se comparan en un diseño factorial completo 3 × 3 con cuatro instancias de evaluación, treinta semillas emparejadas por instancia y un presupuesto fijo de 19 994 evaluaciones. Los parámetros propios de cada operador se calibran en una instancia separada.

**Principales hallazgos.**

- La mutación M18 es la peor de las tres en las cuatro instancias.
- La mejor recombinación depende del problema: M14 gana o empata en tres instancias, y M16 gana sola en la instancia grande con sustitución fuerte.
- Cuatro configuraciones forman un grupo superior estadísticamente equivalente en el ranking global de 120 bloques. La regla de elección fijada de antemano selecciona **M16+M20** con α = 0.3 y η = 20, cuya peor corrida queda a 0.575 % del óptimo.
- Con el mismo presupuesto, el AG captura entre 95.8 % y 99.2 % de la mejora posible (mediana de la configuración elegida), mientras que la búsqueda aleatoria no encuentra soluciones factibles en las instancias de 36 productos.

---

## 2. Planteamiento del problema

Una cadena regional de tiendas de conveniencia fija el precio de temporada de $d$ productos. Los productos se agrupan en familias (por ejemplo, agua natural de 1.5 L), y cada familia tiene tres niveles: marca propia, marca estándar y marca premium. El precio vigente se obtiene con la regla *costo más 45 %*, igual para todos los productos, que ignora la sensibilidad de la demanda a cada precio.

**Variable de decisión.** El vector de precios $p = (p_1,\dots,p_d)^\top \in \mathbb{R}^d$, donde $p_j$ es el precio en pesos del producto $j$. Cada componente es un gen del cromosoma del AG.

**Problema.**

$$
\max_{p\in\mathcal{F}}\ \pi(p)=\sum_{j=1}^{d}(p_j-c_j)\,q_j(p)
\qquad\text{s.a.}\qquad
\mathcal{F}=\{p\in\mathbb{R}^d:\ \ell\le p\le u,\ \ Gp\le h\}
$$

donde $c_j$ es el costo unitario, $q_j(p)$ la demanda de la temporada, $[\ell_j,u_j]$ los límites de precio (piso de margen y techo de alza) y $Gp\le h$ las reglas comerciales de la sección 4.

---

## 3. Modelo de demanda y utilidad

**Demanda lineal con sustitutos.** La demanda de la temporada es

$$
q(p)=a-Bp,\qquad B\in\mathbb{R}^{d\times d},\qquad B^\top=B,
$$

con las entradas

$$
B_{jj}=b_j=\frac{\varepsilon_j\,q^0_j}{P_j},\qquad
B_{jk}=-\kappa_{jk}\sqrt{b_j\,b_k}\ \ (j\neq k),\qquad \kappa_{jk}\ge 0.
$$

Aquí $\varepsilon_j$ es la elasticidad propia medida al precio actual $P_j$, $q^0_j=q_j(P)$ es la demanda actual y $\kappa_{jk}$ mide la intensidad de sustitución. El intercepto se fija para que la demanda reproduzca la situación actual:

$$
a = q^0 + BP.
$$

Un descuento en un producto reduce la demanda de sus sustitutos. Con $\kappa_{jk}$ mayor dentro de una familia que entre familias, el efecto cruzado intra-familia domina.

**Definición positiva.** Si $B$ es simétrica y definida positiva, la utilidad es estrictamente cóncava. Expandiendo,

$$
\pi(p)=\sum_j (p_j-c_j)\,(a_j-(Bp)_j)
=-p^\top Bp+(a+Bc)^\top p-c^\top a .
$$

El gradiente y el hessiano son

$$
\nabla\pi(p)=a+Bc-2Bp,\qquad \nabla^2\pi(p)=-2B\prec 0 .
$$

Como $\nabla^2\pi\prec 0$, $\pi$ es estrictamente cóncava y su máximo sobre un conjunto convexo es único. La propiedad se verifica numéricamente con `validar_instancia`, que exige $\lambda_{\min}(B)>0$.

---

## 4. Reglas comerciales y factibilidad

Las reglas se escriben como $Gp\le h$, con cada fila adimensionalizada para que los multiplicadores tengan unidades comparables:

| Regla | Desigualdad | Fila de $G$ | $h$ |
|---|---|---|---|
| Existencias | $q_j(p)\le S_j$ | $-B_j/S_j$ | $(S_j-a_j)/S_j$ |
| Demanda no negativa | $q_j(p)\ge 0$ | $B_j/q^0_j$ | $a_j/q^0_j$ |
| Escalera de precios | $p_{\text{sup}}\ge(1+g)\,p_{\text{inf}}$ | $((1+g)e_{\text{inf}}-e_{\text{sup}})/P_{\text{sup}}$ | $0$ |
| Índice de precios | $\sum_j w_j\,p_j/P_j\le 1+\eta_I$ | $w_j/P_j$ | $1+\eta_I$ |

donde $B_j$ es la fila $j$ de $B$, $e_j$ el vector canónico, $w_j = q^0_j P_j/\sum_k q^0_k P_k$ la participación en el ingreso actual y $g$, $\eta_I$ las holguras de escalera e índice.

**Violación agregada.** Para medir la factibilidad de cualquier vector se define

$$
V(p)=\sum_i \max\{0,\,(Gp-h)_i\}\ \ge 0,
$$

que vale cero si y sólo si $p$ satisface todas las reglas. Como las reglas son lineales, $\mathcal{F}$ es un poliedro convexo.

**Tolerancia.** La factibilidad se decide con $V(p)\le 10^{-12}$ en unidades relativas. Esta tolerancia sólo absorbe error de punto flotante; equivale a exceder unas existencias de mil unidades en una milmillonésima de unidad.

---

## 5. Generación de instancias sintéticas

Cada instancia se genera a partir de una semilla $s$ con `numpy.random.default_rng(s)`. Los parámetros de generación son:

| Parámetro | Valor | Significado |
|---|---|---|
| Costo base de familia | $\mathcal{U}(8,30)$ | Escala de costo por familia |
| Multiplicador de costo | $0.70,\ 1.00,\ 1.40$ | Propia, estándar, premium |
| Precio actual | $P_j=1.45\,c_j$ | Regla vigente |
| Piso | $\ell_j=1.10\,c_j$ | Margen mínimo de 10 % |
| Techo | $u_j=1.20\,P_j$ | Alza máxima de 20 % |
| Elasticidad propia | $[3.0,5.5]$, $[2.0,3.8]$, $[1.3,2.4]$ | Propia, estándar, premium |
| Existencias | $S_j=q^0_j\,\mathcal{U}(1.05,1.50)$ | Surtido del proveedor |
| Intensidad débil (D) | $\kappa_{\text{intra}}=0.10$, $\sum\kappa_{\text{inter}}=0.05$ | Sustitución débil |
| Intensidad fuerte (F) | $\kappa_{\text{intra}}=0.30$, $\sum\kappa_{\text{inter}}=0.12$ | Sustitución fuerte |

La intensidad fuerte produce un valle alargado en el espacio de los genes: moverse en la dirección en que los tres niveles de una familia se mueven juntos apenas cambia la utilidad, mientras que separarlos la reduce con rapidez. Ese valle es donde la forma de recombinar debería importar más.

**Validez.** `validar_instancia` rechaza instancias con $B$ asimétrica, $B$ no definida positiva, $\ell_j\ge P_j$ o precios actuales que incumplen las reglas.

---

## 6. Referencia exacta: óptimo global certificado

El problema $\max\pi$ se escribe como $\min f(p)=\tfrac12p^\top Hp-g^\top p$ con $H=2B\succ 0$ y $g=a+Bc$, sujeto a $Ap\le b$, donde $A$ y $b$ apilan $G,h$ y las cotas de caja $p\le u$, $-p\le -\ell$.

**Condiciones KKT.** Un punto $p^\ast$ es óptimo si existe $\lambda^\ast\ge 0$ tal que

$$
\begin{aligned}
&Hp^\ast+A^\top\lambda^\ast=g &&\text{(estacionariedad)}\\
&Ap^\ast\le b &&\text{(factibilidad primal)}\\
&\lambda^\ast\ge 0 &&\text{(factibilidad dual)}\\
&\lambda^\ast_i\,(b_i-(Ap^\ast)_i)=0 &&\text{(holguras complementarias)}
\end{aligned}
$$

Como $f$ es estrictamente convexa sobre un conjunto convexo, las condiciones KKT son suficientes y el óptimo es único.

**Certificado por conjunto activo.** Dado un conjunto activo $\mathcal{A}$, las condiciones de estacionariedad y de igualdad activa forman el sistema lineal

$$
\begin{pmatrix}H & A_{\mathcal{A}}^\top\\ A_{\mathcal{A}} & 0\end{pmatrix}
\begin{pmatrix}p\\ \lambda_{\mathcal{A}}\end{pmatrix}
=
\begin{pmatrix}g\\ b_{\mathcal{A}}\end{pmatrix}.
$$

El algoritmo `certificar_kkt` resuelve este sistema y ajusta $\mathcal{A}$: quita una restricción si su multiplicador es negativo y agrega la más violada si $p$ es infactible. El certificado se acepta cuando $\max_i V_i\le 10^{-9}$ y el residuo de estacionariedad es menor que $10^{-6}\max(1,\|a\|_\infty)$.

**Interpretación comercial.** El multiplicador $\lambda_i$ es la utilidad que se ganaría por relajar la regla $i$ en una unidad de su escala. En el caso principal, el multiplicador del índice de precios vale $128\,640$ pesos por unidad: cada punto porcentual adicional de holgura del índice valdría unos $1\,286$ pesos de utilidad por temporada.

**Cálculo inicial.** Se usa SLSQP desde cuatro arranques (precio actual, piso, techo y punto medio), con `ftol = 1e-15`. El mejor punto se pasa al certificado. El AG **no** usa esta referencia para inicializarse, guiarse ni detenerse.

---

## 7. Líneas base

Tres referencias que no son algoritmos genéticos:

1. **Precios actuales.** $p=P$, factible por construcción.
2. **Ajuste parejo.** $p_j(t)=\operatorname{clip}\big(P_j(1+t),\ \ell_j,\ u_j\big)$, con $t\in[-0.25,\,0.20]$ en una rejilla de paso $0.0005$. Se elige el $t$ factible de mayor utilidad. Es la decisión que se tomaría con una hoja de cálculo.
3. **Búsqueda aleatoria.** Muestreo uniforme $p\sim\mathcal{U}(\ell,u)$ en bloques de 100 y luego de 98 evaluaciones, con el mismo presupuesto de 19 994 evaluaciones que el AG y el mismo criterio M08 para elegir el mejor.

---

## 8. Algoritmo genético: las siete etapas

Las etapas siguen la guía de la materia. Cinco son fijas y dos se comparan en tres formas cada una. El orden efectivo de una generación es: 1 → 2 → control 7 → 3 → 4 → 5 → repetir 2 sobre los hijos → 6 → control 7.

| Etapa | Método | Tipo |
|---|---|---|
| 1 · Inicialización | M03 hipercubo latino | Fijo |
| 2 · Evaluación | M08 prioridad de factibilidad | Fijo |
| 3 · Selección | M12 torneo con reemplazo, $k=2$ | Fijo |
| 4 · Recombinación | M14 · M15 · M16 | **Comparado** |
| 5 · Mutación | M18 · M19 · M20 | **Comparado** |
| 6 · Reemplazo | M22 generacional con elitismo, $e=2$ | Fijo |
| 7 · Paro | M26 presupuesto de evaluaciones, $E_{\max}=20\,000$ | Fijo |

El cromosoma es el vector $p\in\mathbb{R}^d$ sin decodificador: el gen es el precio.

### 8.1 Etapa 1 · M03 hipercubo latino

Cada precio se divide en $N=100$ estratos de igual ancho. Cada estrato recibe exactamente un individuo:

$$
x_{ij}=\ell_j+\frac{\pi_j(i)+U_{ij}}{N}\,(u_j-\ell_j),\qquad
\pi_j\ \text{permutación aleatoria de}\ \{0,\dots,N-1\},\quad U_{ij}\sim\mathcal U(0,1).
$$

Garantiza que la población inicial recorra el rango completo de cada precio (McKay, Beckman y Conover, 1979). Como es la primera operación con el generador, **las nueve configuraciones parten de la misma población** cuando comparten semilla.

**Alternativas descartadas.** M01 (uniforme) deja huecos al azar; M02 (Bernoulli) produce bits; M04 (Dirichlet) impone suma unitaria.

### 8.2 Etapa 2 · M08 prioridad de factibilidad

Cada individuo recibe la clave $\kappa(p)=\big(V(p),\,-\pi(p)\big)$, comparada en orden lexicográfico:

$$
p \prec q \iff V(p)<V(q)\ \ \text{o}\ \ \big(V(p)=V(q)\ \text{y}\ \pi(p)>\pi(q)\big).
$$

Un factible siempre supera a un inviable; entre inviables gana el de menor violación; entre factibles, el de mayor utilidad (Deb, 2000). A diferencia de la penalización M07, $\phi(p)=-\pi(p)+\rho V(p)$, no requiere elegir $\rho$. Con $\rho$ pequeño, la penalización elige inviables; la prueba 07 lo muestra.

### 8.3 Etapa 3 · M12 torneo con reemplazo

Para cada torneo se sortean $k=2$ participantes con reemplazo y gana el de menor rango. El rango denso $r(p)\in\{0,1,\dots\}$ se deriva de $\kappa$; claves idénticas comparten rango. La ruleta (M09) y el muestreo universal (M10) requieren aptitudes positivas y proporcionales, que la clave lexicográfica no tiene. En empate gana el primero sorteado.

### 8.4 Etapa 4 · Recombinación (comparada)

Para una pareja $(a,b)$ de padres se generan dos hijos $h^{(1)},h^{(2)}$.

**M14 · uniforme por máscara.** Con máscara $m_j\sim\operatorname{Bernoulli}(0.5)$:

$$
h^{(1)}_j=m_j a_j+(1-m_j)\,b_j,\qquad h^{(2)}_j=(1-m_j)\,a_j+m_j\,b_j .
$$

No crea valores nuevos. Conserva la suma: $h^{(1)}+h^{(2)}=a+b$.

**M15 · aritmética.** Con $\alpha\sim\mathcal U(0,1)$ por pareja:

$$
h^{(1)}=\alpha a+(1-\alpha)\,b,\qquad h^{(2)}=(1-\alpha)\,a+\alpha\,b .
$$

Conserva la suma y cada hijo cae en el segmento entre los padres.

**M16 · BLX-α (Eshelman y Schaffer, 1993).** Con $I_j=|a_j-b_j|$:

$$
h_j\sim\mathcal U\Big[\min(a_j,b_j)-\alpha I_j,\ \max(a_j,b_j)+\alpha I_j\Big]\cap[\ell_j,u_j].
$$

Extrapola fuera del segmento y no conserva la suma. Se usa $\alpha=0.3$ en la configuración elegida.

**Cruce.** La pareja se cruza con probabilidad $p_c=0.9$; si no, los hijos son copias de los padres.

### 8.5 Etapa 5 · Mutación (comparada)

Cada gen muta con probabilidad $p_m=1/d$, común a las nueve configuraciones.

**M18 · reinicio uniforme.** Con probabilidad $p_m$, el gen se sortea de nuevo en su caja:

$$
y_j\sim\mathcal U(\ell_j,u_j).
$$

**M19 · gaussiana acotada.** Con probabilidad $p_m$:

$$
y_j=\operatorname{clip}\big(x_j+\sigma\,(u_j-\ell_j)\,z,\ \ell_j,\ u_j\big),\qquad z\sim\mathcal N(0,1).
$$

Si $x_j=u_j$, la mitad de las mutaciones queda exactamente en el techo: $\Pr(y_j=u_j\mid x_j=u_j)=\Pr(z\ge 0)=\tfrac12$. La prueba 13 verifica esta propiedad.

**M20 · polinomial acotada (Deb y Goyal, 1996; Deb y Deb, 2014).** Sean $\delta_1=\dfrac{x-\ell}{u-\ell}$, $\delta_2=\dfrac{u-x}{u-\ell}$, $r\sim\mathcal U(0,1)$ y $\eta=\eta_m$:

$$
\delta_q=
\begin{cases}
\big[2r+(1-2r)(1-\delta_1)^{\eta+1}\big]^{\frac{1}{\eta+1}}-1, & r\le 0.5,\\[4pt]
1-\big[2(1-r)+2(r-0.5)(1-\delta_2)^{\eta+1}\big]^{\frac{1}{\eta+1}}, & r>0.5,
\end{cases}
\qquad y=x+\delta_q\,(u-\ell).
$$

La mutación nunca rebasa la caja: si $r\to 0$, $y\to\ell$; si $r\to 1$, $y\to u$. Un $\eta$ grande concentra la perturbación alrededor de $x$.

### 8.6 Etapa 6 · Reemplazo elitista M22

$$
P^{t+1}=\operatorname{Élite}_e(P^t)\ \Vert\ H^t,\qquad |H^t|=N-e,\qquad e=2 .
$$

Como las claves de M08 son monótonas, **la mejor clave nunca empeora**. El código verifica este contrato en cada generación.

### 8.7 Etapa 7 · Paro por presupuesto M26

$$
\text{parar en } t\iff E_t+(N-e)>E_{\max},\qquad E_t=N+t\,(N-e).
$$

Con $N=100$, $e=2$ y $E_{\max}=20\,000$, cada corrida realiza $G=203$ generaciones y $E=100+203\cdot 98=19\,994$ evaluaciones. Como todas las configuraciones gastan lo mismo, las diferencias de calidad se atribuyen a los operadores, no al esfuerzo. El estancamiento (M28), usado en la Actividad 03, detendría cada corrida en un momento distinto.

### 8.8 Medidas de dispersión y de cota

- **Dispersión de la población:** $D(P)=\dfrac1d\displaystyle\sum_j\frac{\operatorname{sd}(x_{\cdot j})}{u_j-\ell_j}$.
- **Fracción de genes en cota:** proporción de entradas con $x_{ij}\in\{\ell_j,u_j\}$.
- **Brecha:** $\text{brecha}=100\,\dfrac{\pi^\ast-\pi}{\pi^\ast}$.
- **Mejora capturada:** $\text{mejora}=100\,\dfrac{\pi-\pi_{\text{actual}}}{\pi^\ast-\pi_{\text{actual}}}$.

---

## 9. Propiedades teóricas de los operadores

**Lema 1 (factibilidad de M15).** Sean $a,b\in\mathcal F$ y $\alpha\in[0,1]$. Entonces $h=\alpha a+(1-\alpha)b\in\mathcal F$.

*Demostración.* $\mathcal F$ es intersección de semiespacios cerrados y de una caja, luego es convexo. Como $h$ es combinación convexa de $a$ y $b$, pertenece a $\mathcal F$. $\square$

**Consecuencia.** M15 garantiza hijos factibles a partir de padres factibles. M14 y M16 no: M14 combina genes de padres distintos, y como las reglas acoplan varios genes (índice, escalera), el resultado puede violar una regla; M16 extrapola fuera del segmento. El experimento lo confirma: en E36-F, con mutación M19 y en la generación 10, la fracción de hijos factibles es 0.93 con M15 y 0.00 con M14.

**Lema 2 (contracción de M15).** Para $\alpha\sim\mathcal U(0,1)$, la distancia entre hijos es $|2\alpha-1|$ veces la distancia entre padres, cuyo valor esperado es $\mathbb E|2\alpha-1|=\tfrac12$. La varianza disminuye en cada generación, lo que explica la rápida contracción observada.

**Lema 3 (conservación de la suma).** M14 y M15 satisfacen $h^{(1)}+h^{(2)}=a+b$. M16 no. Esta propiedad es la que hace que M14 y M15 no alteren el centro de la población, mientras M16 lo puede desplazar.

**Lema 4 (acotación de M20).** Para todo $r\in[0,1]$ y $\eta\ge 0$, el valor $y$ de M20 pertenece a $[\ell,u]$. *Demostración.* Sea $\beta=(1-\delta_1)^{\eta+1}\in[0,1]$ y $r\le 0.5$. La base de la raíz es $\beta_r=2r+(1-2r)\beta$. Como $\beta_r-\beta=2r(1-\beta)\ge 0$, se cumple $\beta_r\ge\beta$, así que $\delta_q\ge(1-\delta_1)-1=-\delta_1$. Además $\beta_r\le 1$ porque $r\le 0.5$, así que $\delta_q\le 0$. Entonces $y=x+\delta_q(u-\ell)\ge x-\delta_1(u-\ell)=\ell$ y $y\le x\le u$. El caso $r>0.5$ es simétrico, con $\delta_2$ en lugar de $\delta_1$. $\square$

La implementación aplica además un recorte de error de punto flotante, verificado por la prueba 14.

---

## 10. Diseño experimental e hipótesis

**Factores.** Recombinación $\{M14,M15,M16\}$ × mutación $\{M18,M19,M20\}$: nueve configuraciones en un factorial completo. Un diseño de un factor a la vez exigiría sólo cinco configuraciones, pero no detectaría una interacción entre operadores.

**Bloques.** Cada semilla $s$ define un bloque: las nueve configuraciones que comparten $s$ parten de la misma población inicial. Así las comparaciones son emparejadas.

**Fijo en las nueve configuraciones.** $N=100$, $k=2$, $e=2$, $p_c=0.9$, $p_m=1/d$, $E_{\max}=20\,000$.

**Replicación.** Cuatro instancias de evaluación (E12-D, E12-F, E36-D, E36-F) × nueve configuraciones × treinta semillas $\{20261017,\dots,20261046\}$: 1 080 corridas. Más 120 búsquedas aleatorias con presupuesto idéntico.

**Hipótesis (fijadas tras un piloto con parámetros por defecto, en instancias preliminares).**

- **H1 · Mutación.** M18 será la peor mutación en las cuatro instancias.
- **H2 · Recombinación.** M15 avanzará más rápido al inicio por producir hijos factibles, pero contraerá la población y quedará atrás al final.
- **H3 · Interacción.** El orden de las recombinaciones depende de la mutación que las acompaña.
- **H4 · Dificultad.** Las diferencias entre configuraciones crecerán con el tamaño y con la sustitución fuerte.

**Veredicto.** H1 se confirma. H2 se confirma en el mecanismo (factibilidad y contracción) y se refuta en la velocidad de convergencia. H3 se sostiene en parte: hay interacción, pero menor que los efectos principales. H4 se confirma en magnitud; cambia la forma, porque la recombinación ganadora cambia con la instancia.

**Transparencia.** Las hipótesis se escribieron después de un piloto con parámetros por defecto ($\alpha=0.5$, $\sigma=0.05$, $\eta_m=20$), diez semillas e instancias preliminares. El experimento formal usa instancias regeneradas con otras semillas, parámetros calibrados en una instancia aparte y semillas nuevas.

---

## 11. Calibración de parámetros

Cada configuración calibra **sólo sus propios parámetros** en la instancia C24-F (8 familias, 24 productos, sustitución fuerte), con diez semillas $\{20261200,\dots,20261209\}$:

$$
\alpha\in\{0.1,0.3,0.5\}\ \ (\text{M16}),\qquad
\sigma\in\{0.02,0.05,0.10\}\ \ (\text{M19}),\qquad
\eta\in\{5,20,50\}\ \ (\text{M20}).
$$

Hay 35 celdas y 350 corridas. **Regla fijada antes de ejecutar:** se elige la celda de menor brecha mediana; en empate, la de menor brecha media.

| Configuración | α | σ | η | Brecha mediana (%) |
|---|---|---|---|---|
| M14+M18 | — | — | — | 1.2073 |
| M14+M19 | — | 0.05 | — | 0.2780 |
| M14+M20 | — | — | 20 | 0.4315 |
| M15+M18 | — | — | — | 0.9975 |
| M15+M19 | — | 0.10 | — | 0.4415 |
| M15+M20 | — | — | 20 | 0.4129 |
| M16+M18 | 0.3 | — | — | 0.9511 |
| M16+M19 | 0.3 | 0.05 | — | 0.2438 |
| M16+M20 | 0.3 | — | 20 | 0.2669 |

Parámetros en el interior de la rejilla, salvo σ = 0.10 en M15+M19, que queda en el extremo: la recombinación que contrae la población necesita la mutación más amplia.

---

## 12. Análisis estadístico

Sean $b$ bloques y $k$ tratamientos. Para cada instancia se compara la distribución de la brecha con los siguientes estadísticos.

**Prueba de Friedman (Friedman, 1937).** Con $R_j$ la suma de rangos del tratamiento $j$ dentro de cada bloque:

$$
\chi^2_F=\frac{12}{b\,k(k+1)}\sum_{j=1}^{k}R_j^2-3b(k+1).
$$

Bajo $H_0$, $\chi^2_F$ se aproxima a $\chi^2_{k-1}$.

**Prueba de rangos con signo de Wilcoxon (1945).** Para cada par de tratamientos, sobre las diferencias emparejadas $d_i$, se usa el estadístico $W=\min\big(\sum_{d_i>0}R_i,\sum_{d_i<0}R_i\big)$, con rangos de $|d_i|$ y los empates tratados con el procedimiento estándar de SciPy.

**Ajuste de Holm (1979).** Para $m$ comparaciones con $p_{(1)}\le\dots\le p_{(m)}$:

$$
\tilde p_{(i)}=\max_{j\le i}\,\min\{1,\ (m-j+1)\,p_{(j)}\}.
$$

**Efectos principales y simples.** Los principales se prueban con Friedman sobre las tres formas de un factor, con bloques formados por la semilla y el nivel del otro factor. Los simples se prueban dentro de cada nivel del otro factor.

**Ranking global y prueba post-hoc de Demšar (2006).** Se combinan $n=120$ bloques (cuatro instancias × treinta semillas) y se comparan rangos medios $R_j$ con la configuración de menor rango:

$$
z_j=\frac{R_j-R_{\text{mejor}}}{\sqrt{\dfrac{k(k+1)}{6n}}},\qquad p_j=2\big(1-\Phi(|z_j|)\big),
$$

con $k=9$, y los $p_j$ ajustados por Holm. Nivel de significancia $\alpha=0.05$.

**Regla de elección final, fijada antes de ejecutar.** Se elige la configuración de menor rango medio. Entre las que no difieren de ella según la prueba post-hoc con Holm, se elige la de **menor brecha máxima** en las 120 corridas.

---

## 13. Resultados

### 13.1 Brecha mediana por instancia (%)

| Instancia | Recombinación | M18 | M19 | M20 |
|---|---|---:|---:|---:|
| E12-D | M14 | 0.322 | 0.033 | 0.032 |
| E12-D | M15 | 0.911 | 0.144 | 0.058 |
| E12-D | M16 | 0.740 | 0.042 | 0.041 |
| E12-F | M14 | 0.621 | 0.088 | 0.074 |
| E12-F | M15 | 0.878 | 0.190 | 0.111 |
| E12-F | M16 | 0.727 | 0.073 | 0.079 |
| E36-D | M14 | 0.769 | 0.079 | 0.073 |
| E36-D | M15 | 0.890 | 0.210 | 0.300 |
| E36-D | M16 | 0.756 | 0.088 | 0.098 |
| E36-F | M14 | 1.408 | 0.548 | 0.474 |
| E36-F | M15 | 0.931 | 0.461 | 0.698 |
| E36-F | M16 | 0.841 | 0.365 | 0.331 |

### 13.2 Ranking global (120 bloques)

Friedman global: $\chi^2 = 735.13$, $p\approx 2\times10^{-153}$.

| Configuración | Rango medio | $p$ (Holm) | Brecha mediana (%) | Brecha máxima (%) |
|---|---:|---:|---:|---:|
| M16+M19 | 2.592 | — | 0.078 | 0.630 |
| M14+M20 | 2.700 | 1.000 | 0.073 | 1.804 |
| M16+M20 | 2.758 | 1.000 | 0.091 | **0.575** |
| M14+M19 | 2.992 | 0.774 | 0.084 | 1.718 |
| M15+M20 | 5.050 | < 0.001 | 0.225 | 1.020 |
| M15+M19 | 5.167 | < 0.001 | 0.202 | 0.757 |
| M16+M18 | 7.600 | < 0.001 | 0.773 | 1.326 |
| M14+M18 | 7.800 | < 0.001 | 0.691 | 3.915 |
| M15+M18 | 8.342 | < 0.001 | 0.897 | 1.445 |

**Configuración elegida:** M16+M20, con α = 0.3 y η = 20, porque su brecha máxima (0.575 %) es la menor del grupo superior.

### 13.3 Caso principal: E12-F (doce bebidas)

| Alternativa | Utilidad (pesos) | Brecha (%) | Mejora capturada (%) | Factible |
|---|---:|---:|---:|:---:|
| Precios actuales | 55 502.96 | 7.980 | 0.0 | Sí |
| Ajuste parejo (+3.00 %) | 59 131.05 | 1.965 | 75.4 | Sí |
| AG M16+M20, semilla 20261017 | 60 263.46 | 0.088 | 98.9 | Sí |
| AG M15+M18, semilla 20261017 | 59 791.44 | 0.870 | 89.1 | Sí |
| Óptimo redondeado a 10 centavos | 60 312.01 | 0.007 | 99.9 | Sí |
| **Óptimo certificado** | **60 316.30** | **0.000** | **100.0** | Sí |

El óptimo eleva los precios premium entre 4.3 % y 12.8 % y compensa el índice de precios con bajas de hasta 2.9 % en productos de marca propia y estándar. Redondear a 50 centavos rompe el índice o las existencias.

### 13.4 Verificación contra la referencia de la plantilla

La ejecución se compara automáticamente con las cifras de la referencia de la plantilla (`resultados/verificacion_readme.csv`). De 13 cifras, 12 coinciden dentro de la tolerancia de redondeo. La única discrepancia es la brecha del ajuste parejo en E12-D: la referencia publicada dice 3.84 % y la ejecución calcula 3.8297 %, que redondea a 3.83 %.

### 13.5 Figuras

| Archivo | Contenido |
|---|---|
| `figuras/ae_a04_01_diseno_factorial.png` | Etapas del AG y diseño factorial 3 × 3 |
| `figuras/ae_a04_02_operadores.png` | Distribución de hijos de cada recombinación y de cada mutación |
| `figuras/ae_a04_03_calibracion.png` | Superficies y perfiles de la calibración en C24-F |
| `figuras/ae_a04_04_brechas_3x3.png` | Mapas de brecha mediana por instancia |
| `figuras/ae_a04_05_interaccion.png` | Gráfica de interacción recombinación × mutación |
| `figuras/ae_a04_06_convergencia.png` | Brecha frente a evaluaciones, con búsqueda aleatoria |
| `figuras/ae_a04_07_mecanismo.png` | Dispersión, hijos factibles y brecha a 5 000 y 19 994 evaluaciones (E36-F) |
| `figuras/ae_a04_08_caso_principal.png` | Cambio porcentual por producto en E12-F |

---

## 14. Discusión

**La mutación determina la mayor parte del desempeño.** M18 reinicia cada gen en todo su intervalo y no puede afinar cerca del óptimo. Las perturbaciones locales (M19, M20) capturan entre 91 % y 99 % de la mejora posible en la mayoría de las configuraciones.

**La recombinación interactúa con la mutación y con el tamaño.** M15 produce hijos factibles, por el Lema 1, pero contrae la población por el Lema 2. Esa contracción es ventajosa al inicio y perjudicial al final. En E36-F, con mutación M19, la dispersión de M15 en la generación 10 es cercana a 0.03, frente a cerca de 0.19 de M14. La rapidez de M15 no se convierte en calidad final porque la población pierde diversidad antes de explorar el valle.

**Por qué M16 gana en la instancia grande con sustitución fuerte.** BLX-α extrapola fuera del segmento entre padres y mantiene diversidad en valles estrechos. Esa diversidad es la que falta a M15 en el valle alargado de la sustitución fuerte.

**Relevancia del ajuste parejo.** La decisión que se tomaría con una hoja de cálculo captura entre 75 % y 76 % de la mejora posible en E12-F, y deja entre 1.9 % y 3.8 % de utilidad sobre la mesa según la instancia. El AG captura más de 95 % con el mismo tipo de decisión, pero a costo computacional mucho mayor.

---

## 15. Limitaciones

- **Datos sintéticos.** Los costos, elasticidades, existencias y volúmenes son una escala de trabajo, no estimaciones de una empresa real.
- **Demanda lineal y utilidad cóncava.** Con estas propiedades, un solucionador de programación cuadrática resuelve cada instancia en menos de 0.11 s. El AG es aquí el objeto de estudio, no la herramienta de producción. Tendría sentido práctico con demanda no cóncava, demanda estimada por simulación o reglas no lineales.
- **Alcance de la inferencia.** Los resultados describen cuatro instancias de evaluación y una calibración en una quinta. No se generalizan a otras categorías sin nuevas instancias.
- **Brecha en utilidad, no en precios.** La utilidad es plana cerca del óptimo: soluciones con precios bastante distintos pueden tener utilidades casi iguales. Por eso la brecha mide utilidad y no cercanía de precios. En la instancia E36-F, un precio de la corrida mediana de M16+M20 puede diferir hasta 10.5 puntos porcentuales del óptimo aun con brecha de 0.33 %.
- **Piloto de diseño.** Las hipótesis se fijaron tras un piloto preliminar, aunque el experimento formal usa instancias y semillas nuevas.

---

## 16. Implementación y reproducibilidad

**Requisitos.** Python ≥ 3.10. Dependencias en `requirements.txt`: NumPy, pandas, SciPy, Matplotlib y python-docx (sólo para el informe).

```bash
pip install -r requirements.txt
```

**Ejecución completa** (pruebas, instancias, calibración, experimento, estadística, figuras y verificación; cerca de dos minutos):

```bash
python ae_actividad04.py --salida resultados_actividad04
```

**Sólo tablas** (sin figuras):

```bash
python ae_actividad04.py --salida resultados_actividad04 --sin-figuras
```

**Informe Word** a partir de los resultados:

```bash
python construir_docx.py --carpeta resultados_actividad04 --docx Actividad04_Romero_Lora.docx
```

**Verificación.** Las 16 pruebas se ejecutan antes de cualquier experimento; si una falla, el script se detiene con `AssertionError`. Las semillas están fijadas: dos ejecuciones producen las mismas cifras, salvo las columnas de tiempo.

**Trazabilidad.** `manifiesto.json` registra versiones de Python, NumPy, pandas, SciPy y Matplotlib; la plataforma; la configuración; las semillas; los parámetros calibrados; los óptimos; y las huellas SHA-256 de cada CSV y figura.

**Entorno comprobado.** Python 3.11, NumPy 2.5, pandas 3.0, SciPy 1.18, Matplotlib 3.11.

---

## 17. Ejemplos de uso

Todos los ejemplos se ejecutan desde la carpeta del repositorio, después de `import ae_actividad04 as ae`.

### 17.1 Generar una instancia y calcular su óptimo certificado

```python
import ae_actividad04 as ae

inst = ae.generar_instancia("demo", n_familias=4, intensidad="F", semilla=20261092)
ex = ae.resolver_exacto(inst)

print("Utilidad óptima:", round(ex["utilidad"], 2))
print("Residuo KKT:", ex["residuo"])
print("Reglas activas:", ex["activas"])
```

### 17.2 Verificar la factibilidad y la utilidad de un vector de precios

```python
p = inst.precio_actual                  # regla vigente: costo × 1.45
print("Utilidad actual:", ae.utilidad(inst, p)[0])
print("Violación V(p):", ae.violacion(inst, p)[0])   # 0 en el conjunto factible

aud = ae.evaluar_escalar(inst, p)       # auditoría con bucles, sin G ni h
print("Coincide:", abs(aud["utilidad"] - ae.utilidad(inst, p)[0]) < 1e-9)
```

### 17.3 Ejecutar una corrida del AG con la configuración elegida

```python
r = ae.algoritmo_genetico(
    inst, cruce="M16", mutacion="M20", semilla=20261017,
    parametros={"alfa": 0.3, "eta": 20}, E_max=20_000,
)
print("Mejor utilidad factible:", round(r["utilidad"], 2))
print("Brecha (%):", 100 * (ex["utilidad"] - r["utilidad"]) / ex["utilidad"])
print("Evaluaciones:", r["evaluaciones"], "· generaciones:", r["generaciones"])
```

### 17.4 Comparar contra las dos líneas base

```python
parejo = ae.ajuste_parejo(inst)
ba = ae.busqueda_aleatoria(inst, semilla=31, evaluaciones=r["evaluaciones"])

print("Ajuste parejo t =", parejo["t"], "· utilidad =", round(parejo["utilidad"], 2))
print("Búsqueda aleatoria factible:", ba["factible"], "· utilidad =", ba["utilidad"])
```

### 17.5 Probar un operador de forma aislada

```python
import numpy as np

rng = np.random.default_rng(0)
A = inst.piso + rng.random((5, inst.d)) * (inst.techo - inst.piso)
B = inst.piso + rng.random((5, inst.d)) * (inst.techo - inst.piso)

h1, h2 = ae.etapa_4_recombinar_M15(A, B, rng)
print("Suma conservada:", np.allclose(h1 + h2, A + B))

# Lema 1: si ambos padres son factibles, los hijos de M15 también lo son.
factibles = ae.violacion(inst, np.vstack([A, B])) == 0
if factibles.all():
    print("Hijos factibles:", bool(np.all(ae.violacion(inst, np.vstack([h1, h2])) <= ae.TOL_FACTIBLE)))
else:
    print("Algún padre es inviable; el lema no aplica a esta muestra.")
```

### 17.6 Ejecutar el experimento completo

El experimento completo (nueve configuraciones, cuatro instancias, treinta semillas) se ejecuta desde la línea de comandos, como en la sección 16. Desde Python, el flujo es el mismo que sigue `main()`:

```python
from pathlib import Path

res = Path("resultados_demo")
res.mkdir(exist_ok=True)
instancias, exactos, parejo, tabla = ae.preparar_instancias(res)
parametros, tabla_cal, calibracion = ae.calibrar(
    instancias["C24-F"], exactos["C24-F"]["utilidad"], res)
corridas, historial, aleatoria, historial_ba = ae.experimento(
    instancias, exactos, parametros, res)
analisis = ae.analizar(corridas, historial, historial_ba, exactos, res)
print("Configuración elegida:", analisis["elegida"])
```

> La calibración más el experimento tardan cerca de dos minutos en una computadora de escritorio.

---

## 18. Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `ae_actividad04.py` | Análisis completo: modelo, óptimo certificado, siete etapas, 16 pruebas, calibración, experimento, estadística, figuras y verificación |
| `construir_docx.py` | Genera el informe Word a partir de los CSV de `resultados/` |
| `Actividad04_Romero_Lora.docx` | Informe final de 8 páginas |
| `README.md` | Este documento |
| `requirements.txt` | Dependencias |
| `resultados/instancias.csv` | Óptimos, líneas base, reglas activas y medidas de dificultad |
| `resultados/calibracion.csv`, `calibracion_elegidos.csv` | 350 corridas de calibración y parámetros elegidos |
| `resultados/corridas.csv` | Una fila por corrida de evaluación (1 080) |
| `resultados/historiales.csv` | Trayectoria generación a generación |
| `resultados/busqueda_aleatoria.csv`, `historial_busqueda_aleatoria.csv` | 120 búsquedas aleatorias con presupuesto idéntico |
| `resultados/resumen.csv`, `pruebas_estadisticas.csv`, `ranking_global.csv` | Agregados, pruebas de Friedman y Wilcoxon-Holm, ranking global |
| `resultados/mecanismo_medianas.csv` | Dispersión, hijos factibles y genes en cota por generación |
| `resultados/caso_principal_*.csv` | Precios de E12-F y alternativas comparadas |
| `resultados/verificacion_readme.csv` | Comparación con la referencia de la plantilla |
| `resultados/manifiesto.json` | Versiones, configuración, semillas y huellas SHA-256 |
| `figuras/` | Ocho figuras PNG a 200 ppp |

---
## 19. Referencias

- Blickle, T., y Thiele, L. (1996). A comparison of selection schemes used in evolutionary algorithms. *Evolutionary Computation*, 4(4), 361–394.
- Boyd, S., y Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Deb, K. (2000). An efficient constraint handling method for genetic algorithms. *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.
- Deb, K., y Deb, D. (2014). Analysing mutation schemes for real-parameter genetic algorithms. *International Journal of Artificial Intelligence and Soft Computing*, 4(1), 1–28.
- Deb, K., y Goyal, M. (1996). A combined genetic adaptive search (GeneAS) for engineering design. *Computer Science and Informatics*, 26(4), 30–45.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1–30.
- Eshelman, L. J., y Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata. En *Foundations of Genetic Algorithms 2* (pp. 187–202). Morgan Kaufmann.
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. *Journal of the American Statistical Association*, 32(200), 675–701.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- McKay, M. D., Beckman, R. J., y Conover, W. J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics*, 21(2), 239–245.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83.
