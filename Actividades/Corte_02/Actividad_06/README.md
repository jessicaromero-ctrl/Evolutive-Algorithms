# Optimización de condiciones de corte en torneado mediante evolución diferencial

Aplicación industrial con datos sintéticos. Alumna: Jessica Melani Romero Lora (matrícula 25322116).
Maestría en Inteligencia Artificial · Algoritmos Evolutivos · UPMH.

## Problema

Minimizar el costo de producción por pieza (modelo de Gilbert) en torneado, con variables velocidad de corte,
avance y profundidad, sujeto a rugosidad superficial, potencia de corte y vida mínima de herramienta
(ecuación de Taylor generalizada).

## Método

- **DE clásica** (DE/rand/1/bin) y **JADE** (DE/current-to-pbest/1 con archivo y adaptación de F y CR).
- Manejo de restricciones por reglas de factibilidad de Deb (2000).
- Protocolo: 30 semillas emparejadas, presupuestos de 300, 600 y 1 000 evaluaciones, Wilcoxon con corrección de Holm.
- Análisis de robustez: 5 000 perturbaciones de coeficientes y barrido de márgenes en rugosidad y vida mínima.

## Estructura

| Archivo | Función |
|---|---|
| `modelo_mecanizado.py` | Parámetros, costo, vida, rugosidad, potencia, restricciones, referencia y perturbaciones |
| `evolucion_diferencial.py` | DE clásica, JADE, regla de factibilidad y corrección de límites |
| `experimento.py` | Convergencia, estadística y robustez; escribe `resultados/` |
| `figuras.py` | Genera las figuras a partir de `resultados/` |
| `tests/test_de_mecanizado.py` | 10 pruebas automatizadas |
| `resultados/` | CSV de convergencia, historial, estadística, referencia, robustez y manifiesto |
| `figuras/` | Figuras del informe |
| `documento/` | Informe en Word |

## Reproducir

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
python experimento.py
python figuras.py
```

El experimento tarda menos de un minuto. Los resultados son deterministas: las semillas están fijadas.

## Resultados principales

- Referencia: v = 144.3 m/min, f = 0.2263 mm/rev, a = 3.0 mm; costo 3.2835 €/pieza.
- JADE supera a la DE clásica en los tres presupuestos (p de Holm = 0.011).
- Bajo incertidumbre, el óptimo nominal incumple alguna restricción en 70.5 % de los escenarios; la vida de herramienta es la restricción crítica.
- Recomendación (margen 15 % en Ra y vida mínima de 30 min): v = 129.5 m/min, f = 0.209 mm/rev, a = 3.0 mm; costo 3.506 €/pieza; incumplimiento 4.1 %.

## Límites

- Los coeficientes son supuestos representativos, no datos de un fabricante.
- La referencia es el mejor resultado de malla y SLSQP, no un óptimo certificado.
- El punto recomendado queda en el borde del barrido de vida mínima (30 min).

## Procedencia

Modelo de costo de Gilbert (1950), ecuación de Taylor (1907), DE (Storn y Price, 1997), JADE (Zhang y Sanderson, 2009) y reglas de factibilidad (Deb, 2000). Se empleó asistencia de inteligencia artificial para la programación y la redacción; las cifras se obtienen al ejecutar el código.
