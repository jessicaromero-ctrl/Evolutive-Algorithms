# Actividad 04 · Algoritmo genético para fijar precios de temporada

Comparación de tres formas de recombinar (M14, M15, M16) y tres formas de mutar (M18, M19, M20) en un algoritmo genético, en un diseño factorial completo 3 × 3, para fijar precios de temporada con reglas comerciales reales.

| | |
|---|---|
| **Alumna** | Jessica Melani Romero Lora — matrícula 25322116 |
| **Materia** | Algoritmos Evolutivos — Maestría en Inteligencia Artificial |
| **Catedrático** | Dr. Jaime Aguilar Ortiz |
| **Institución** | Universidad Politécnica Metropolitana de Hidalgo |
| **Periodo** | Septiembre–diciembre de 2026 · Corte 2 |

## Contenido

| Ruta | Contenido |
|---|---|
| `ae_actividad04.py` | Análisis completo: modelo, óptimo certificado, siete etapas, 16 pruebas, calibración, experimento, estadística, figuras y verificación |
| `construir_docx.py` | Genera el informe Word a partir de los CSV de `resultados/` |
| `Actividad04_Romero_Lora.docx` | Informe de 8 páginas (introducción, desarrollo, continuación, figuras y tablas) |
| `resultados/` | Tablas CSV de cada etapa del análisis y `manifiesto.json` con versiones y huellas SHA-256 |
| `figuras/` | Ocho figuras PNG a 200 ppp generadas por el script |
| `requirements.txt` | Dependencias |

## Reproducir

```bash
pip install -r requirements.txt
python ae_actividad04.py                           # análisis, resultados y figuras (~2 min)
python ae_actividad04.py --sin-figuras             # sólo tablas y pruebas (~2 min)
python construir_docx.py --carpeta . --docx Actividad04_Romero_Lora.docx
```

Las 16 pruebas se ejecutan antes de cualquier experimento; si una falla, el script se detiene. Todas las semillas están fijadas, de modo que dos ejecuciones producen las mismas cifras salvo las columnas de tiempo.

## Diseño

- **Etapas fijas:** M03 hipercubo latino (inicialización), M08 prioridad de factibilidad (evaluación), M12 torneo con k = 2 (selección), M22 reemplazo elitista con e = 2, M26 presupuesto de 20 000 evaluaciones.
- **Etapas comparadas:** recombinación M14 / M15 / M16 y mutación M18 / M19 / M20.
- **Calibración:** α de M16, σ de M19 y η de M20 se calibran por configuración en la instancia C24-F, con diez semillas (350 corridas). La regla se fijó antes de ejecutar.
- **Evaluación:** cuatro instancias (12 y 36 productos, con sustitución débil y fuerte), treinta semillas emparejadas, 1 080 corridas y 120 búsquedas aleatorias con presupuesto idéntico.
- **Referencia exacta:** la utilidad es estrictamente cóncava y las reglas son lineales, así que el óptimo global se certifica con las condiciones KKT. El AG nunca lo usa.

## Resultados principales

| Instancia | Mejora posible sobre la regla actual | Brecha del ajuste parejo | Mejor configuración (brecha mediana) | M16+M20 (brecha mediana) |
|---|---:|---:|---|---:|
| E12-D | 5.68 % | 3.83 % | M14+M20 (0.032 %) | 0.041 % |
| E12-F | 8.67 % | 1.97 % | M16+M19 (0.073 %) | 0.079 % |
| E36-D | 5.30 % | 2.64 % | M14+M19 (0.079 %) | 0.098 % |
| E36-F | 8.62 % | 1.86 % | M16+M20 (0.331 %) | 0.331 % |

- M18 es la peor mutación en las cuatro instancias.
- La mejor recombinación depende del problema: M14 gana o empata en tres instancias y M16 gana sola en la instancia grande con sustitución fuerte.
- Cuatro configuraciones forman el grupo superior del ranking global (120 bloques): M16+M19, M14+M20, M16+M20 y M14+M19.
- Regla de elección fijada de antemano: entre las configuraciones estadísticamente equivalentes, la de menor brecha máxima. Resultado: **M16+M20 con α = 0.3 y η = 20**, cuya peor corrida queda a 0.575 % del óptimo.

## Mejoras respecto al cuadernillo original

- Código reorganizado en funciones independientes; sin celdas que dependan del orden de ejecución.
- Eliminados código muerto, parámetros sin uso y una leyenda no utilizada.
- Pruebas declaradas en una lista explícita, con reporte de resultado.
- Verificación automática: 12 de 13 cifras publicadas en el README original coinciden. La discrepancia es un error de redondeo del README: la brecha del ajuste parejo en E12-D se publicó como 3.84 % y la ejecución da 3.83 % (`resultados/verificacion_readme.csv`).
- Línea de comandos con carpeta de salida y opción para omitir figuras.
- Manifiesto con versiones y huellas SHA-256 de cada archivo.

## Alcance y límites

- **Datos sintéticos.** Precios, costos, elasticidades y existencias son una escala de trabajo, no estimaciones de una empresa real.
- **El AG no es la herramienta práctica para este modelo.** Un solucionador de programación cuadrática resuelve cada instancia en menos de 0.11 s. El AG sería útil con demanda no cóncava o reglas no lineales.
- **Resultados acotados.** Describen cuatro instancias de evaluación y una calibración en una quinta.
- **Utilidad plana cerca del óptimo.** Soluciones casi igual de buenas pueden tener precios bastante distintos; la brecha mide utilidad, no cercanía de precios.

## Procedencia

El modelo y el cuadernillo base provienen de la Actividad 04 tal como se entregó originalmente; su README indica como autor a Víctor Manuel Santos Martínez. Las siete etapas y la nomenclatura M01–M30 provienen del material de la materia. La reorganización del código, la verificación de cifras y el informe corresponden a esta entrega.
