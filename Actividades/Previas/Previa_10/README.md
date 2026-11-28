# Optimización multiobjetivo de hiperparámetros de un modelo de IA con MOEA

**Conceptos básicos de optimización multiobjetivo y algoritmos evolutivos multiobjetivo (MOEA)**

| | |
|---|---|
| **Alumna** | Jessica Melani Romero Lora — matrícula 25322116 |
| **Materia** | Algoritmos Evolutivos — Maestría en Inteligencia Artificial |
| **Docente** | Dr. Jaime Aguilar Ortiz |
| **Institución** | Universidad Politécnica Metropolitana de Hidalgo |

## Descripción

Se ajustan seis hiperparámetros de un perceptrón multicapa (MLP) sobre el conjunto Breast Cancer Wisconsin de scikit-learn, con tres objetivos en conflicto:

- **f₁** = 1 − exactitud, medida con validación cruzada estratificada de 3 pliegues;
- **f₂** = costo de entrenamiento en MFLOPs (estimación analítica, reproducible);
- **f₃** = número de parámetros del modelo.

El espacio de decisión tiene 960 configuraciones. Se evaluaron todas para obtener el frente de Pareto exacto, y luego se comparan **NSGA-II**, **MOEA/D** y **búsqueda aleatoria** con presupuestos idénticos de 100, 200 y 300 evaluaciones y 30 semillas emparejadas.

## Resultados principales

- El frente exacto tiene **13 genotipos no dominados**, pero sólo **4 vectores de objetivos distintos**: α no cambia los objetivos de esas configuraciones.
- A 300 evaluaciones, la búsqueda aleatoria recupera en promedio 2.73 de los 4 vectores; NSGA-II, 2.57; MOEA/D, 2.43.
- Los puntos de alto rendimiento (16 y 8 unidades, tasa 0.01, lote 256) se encuentran de forma consistente. Los puntos intermedios (8-16 y 8-8, con tasa 0.001 y lote 32) separan a los métodos: la aleatoria encuentra 8-16 en 70 % de las semillas; NSGA-II, en 30 %, y MOEA/D, en 17 %.
- En IGD, MOEA/D es significativamente peor que la aleatoria a 300 evaluaciones (Holm p = 0.015). NSGA-II no difiere significativamente de la aleatoria tras la corrección (Holm p = 0.081).
- El hipervolumen es poco sensible en este problema: todos los métodos alcanzan entre 1.3301 y 1.3305, frente a 1.3305 del frente exacto.
- Solución recomendada (punto de compromiso): 8 unidades ocultas, tasa 0.01, lote 256; exactitud 97.72 %, 35.1 MFLOPs y 257 parámetros.

## Contenido del repositorio

| Ruta | Contenido |
|---|---|
| `documento/Optimizacion_MOEA_Hiperparametros_Romero_Lora.docx` | Informe académico: introducción, conceptos básicos, desarrollo, resultados, conclusiones y futuras aplicaciones |
| `moea_hiper.py` | Modelo, codificación, evaluación con validación cruzada, dominancia, hipervolumen, IGD, NSGA-II, MOEA/D y búsqueda aleatoria |
| `exhaustivo.py` | Entrena y evalúa las 960 configuraciones; genera `resultados/tabla_exhaustiva.csv` |
| `experimento.py` | Comparación de algoritmos con tres presupuestos, pruebas estadísticas y frente exacto |
| `figuras.py` | Genera las figuras y las tablas de frecuencia de hallazgo y recuperación |
| `tests/test_moea_hiper.py` | 12 pruebas: dominancia, nds, crowding, hipervolumen 2D y 3D, IGD, codificación, conteo de parámetros frente a scikit-learn, reproducibilidad y presupuesto |
| `resultados/` | Tabla exhaustiva, frente exacto (completo y de vectores distintos), recomendada, corridas, convergencia, recuperación, frecuencia de hallazgo, estadística y manifiesto |
| `figuras/` | Figuras PNG usadas en el informe |

## Reproducir

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
python exhaustivo.py   # ≈ 5 minutos: entrena las 960 configuraciones
python experimento.py  # ≈ 30 segundos
python figuras.py
```

Si ya existe `resultados/tabla_exhaustiva.csv`, `exhaustivo.py` puede omitirse.

## Alcance y límites

- **Un solo conjunto de datos** (Breast Cancer Wisconsin, 569 muestras). Los resultados describen este problema, no los MLP en general.
- **Espacio discreto y pequeño.** La base de verdad es exhaustiva, pero con 960 configuraciones la búsqueda aleatoria es muy competitiva. Con espacios grandes, los MOEA deberían tener ventaja; esa comparación queda como trabajo futuro.
- **Costo estimado, no medido.** f₂ usa 6 · p · n_train · épocas en lugar de tiempo de reloj, para que el resultado sea reproducible.
- **Semilla fija en el entrenamiento.** Cada configuración tiene una sola evaluación determinista; no se mide la variabilidad del entrenamiento.
- **Pocas generaciones.** Con 300 evaluaciones y población de 30, NSGA-II y MOEA/D completan unas 10 generaciones.

## Procedencia

Modelo, codificación, objetivos, tabla exhaustiva, experimento y análisis: decisiones de este desarrollo. NSGA-II sigue a Deb et al. (2002); MOEA/D a Zhang y Li (2007); la distancia generacional invertida y el hipervolumen siguen a la literatura de indicadores de Zitzler y Thiele (1999).

Se empleó asistencia de inteligencia artificial para la programación y la redacción. Las cifras se obtienen al ejecutar los scripts del repositorio.
