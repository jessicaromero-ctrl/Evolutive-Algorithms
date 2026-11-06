# Actividad previa número 9 · Cartera estatal de inversión para la productividad de Hidalgo

**Evolución diferencial con reglas de factibilidad: comparación de tres mutaciones y tres cruces**

| | |
|---|---|
| **Alumna** | Jessica Melani Romero Lora — matrícula 25322116 |
| **Materia** | Algoritmos Evolutivos — Maestría en Inteligencia Artificial |
| **Docente** | Dr. Jaime Aguilar Ortiz |
| **Institución** | Universidad Politécnica Metropolitana de Hidalgo |
| **Periodo** | Septiembre–diciembre de 2026 · Corte 2 |

## Descripción

Un algoritmo evolutivo reparte un presupuesto de inversión de 20 000 millones de pesos entre **cuatro regiones** de Hidalgo (Zapotlán–Tizayuca, Pachuca–Mineral de la Reforma, Tula–Tepeji y Valle del Mezquital) y **cinco mercados** (logística, manufactura ligera, agroindustria, turismo, y agua y energía industrial), para maximizar el valor agregado anual.

Restricciones: presupuesto total, agua disponible por región, equidad mínima de 15 % para la Valle del Mezquital, empleo mínimo, y tope de 40 % por región. El valor agregado tiene umbrales de infraestructura, por lo que la función no es cóncava.

Los datos son **sintéticos e ilustrativos**. El contexto de la entidad proviene de fuentes públicas citadas en el documento.

## Algoritmo y comparación

Se usa **evolución diferencial** con la regla de factibilidad de Deb (2000). Se comparan tres estrategias de mutación (rand/1, best/1, current-to-best/1) y tres de cruce (binomial, exponencial, aritmético), en un diseño completo de nueve configuraciones. Cada configuración calibra sus parámetros F y CR en una instancia distinta de las de evaluación.

## Resultados principales

Presupuesto de 20 000 evaluaciones de la función, 30 semillas por instancia. Brecha mediana frente al mejor valor conocido.

| Instancia | Reparto proporcional | AG de la actividad 04 | Configuración elegida (rand/1 + binomial) |
|---|---:|---:|---:|
| E-base | 54.8 % | 11.3 % | **3.3 %** |
| E-escasez | 54.0 % | 14.3 % | **2.7 %** |
| E-amplia | 41.0 % | 11.6 % | **2.8 %** |

- **Ranking de las nueve configuraciones** (Friedman χ² = 482.2, p < 0.001, 90 bloques): rand/1 + binomial tiene el menor rango medio (2.23), y best/1 + binomial sigue con 3.01.
- **El cruce binomial** es el factor más influyente. El cruce aritmético queda en los últimos lugares.
- **La cartera propuesta** llega a 13 412 millones de pesos de valor agregado anual, a 0.4 % del mejor conocido (13 466). Prioriza logística y manufactura ligera en Zapotlán–Tizayuca y Pachuca–Mineral de la Reforma, y cumple el mínimo de equidad para la Valle del Mezquital.

## Contenido del repositorio

| Ruta | Contenido |
|---|---|
| `documento/Actividad_previa_09_Romero_Lora.docx` | Informe de la actividad: introducción, desarrollo, resultados, conclusiones y futuras aplicaciones |
| `ag_inversion.py` | Modelo de la cartera, referencia SLSQP, líneas base, algoritmo genético de la actividad 04 y evolución diferencial con sus variantes |
| `ag_precios.py` | Operadores y estadística compartidos (Friedman, Wilcoxon-Holm, ranking de Demšar, bootstrap, tamaño de efecto A12) |
| `experimento.py` | Calibración, experimento factorial 3 × 3, líneas base y pruebas estadísticas; escribe `resultados/` |
| `figuras.py` | Genera las seis figuras a partir de los resultados |
| `tests/test_ag_inversion.py` | 22 pruebas: modelo, no concavidad, restricciones, reglas de factibilidad, variantes y reproducibilidad |
| `resultados/` | CSV de referencias, calibración, corridas, búsqueda aleatoria, ranking, pruebas estadísticas, cartera y manifiesto con huellas SHA-256 |
| `figuras/` | Seis figuras PNG usadas en el documento |

## Reproducir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
python experimento.py      # ≈ 30 minutos en un núcleo
python figuras.py
```

## Alcance y límites

- **Datos sintéticos.** Rendimientos, umbrales, empleo, uso de agua y cotas son ilustrativos; no son estimaciones oficiales.
- **Referencia no certificada.** El mejor valor conocido es el mejor resultado de 200 arranques de SLSQP, no un óptimo global demostrado.
- **Método gradiente competitivo.** En este modelo suave, el multiarranque con gradiente encuentra mejores valores que los algoritmos evolutivos probados. El algoritmo evolutivo se justifica mejor con variables discretas o con varios objetivos, que se proponen como futuras aplicaciones en el documento.

## Procedencia

Modelo, datos sintéticos, calibración, experimento y análisis: decisiones de este desarrollo. La evolución diferencial sigue a Storn y Price (1997), y la regla de factibilidad a Deb (2000). Los operadores y la estadística compartidos provienen del trabajo previo de la actividad 04.

Se empleó asistencia de inteligencia artificial para la programación y la redacción. Las cifras se obtienen al ejecutar `experimento.py`.
