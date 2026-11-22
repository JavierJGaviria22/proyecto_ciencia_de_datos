# Problema de ML — Supervivencia en el Titanic

Cifras tomadas de `notebooks/04_modelado.ipynb`. Semilla 42, validación cruzada de 5 pliegues estratificados sobre el conjunto de entrenamiento (712 pasajeros); prueba con 179 pasajeros.

## Definición del problema

| Decisión | Respuesta |
|---|---|
| Tipo de problema | Clasificación binaria supervisada. |
| Variable objetivo | `Survived` (1 = sobrevivió). Es lo que la aseguradora quiere estimar. |
| Unidad de análisis y momento de predicción | Un pasajero, *al embarcar*: solo se usan datos que existen antes del naufragio. |
| Métrica principal | **ROC-AUC**: no depende del umbral, compara bien modelos y no engaña con clases 38 % / 62 %, como sí pasa con la exactitud. |
| Métricas secundarias | Precisión, recall, F1 y exactitud. |
| Costo de errores | El taller no define costos, así que se asume un costo parecido para ambos errores (por eso se usa F1 para el umbral). Para la aseguradora, un falso positivo (se predice que sobrevive y murió) subestima el riesgo; un falso negativo (se predice que muere y sobrevivió) lo sobreestima. |
| Criterio de éxito | Superar la regla de género por un margen mayor que la variabilidad entre pliegues (≈ 0.02–0.03). |

## Baselines y criterio de éxito (validación cruzada, entrenamiento)

| Baseline | Exactitud | F1 | ROC-AUC |
|---|---|---|---|
| Dummy (siempre "no sobrevive") | 0.617 | 0.000 | 0.500 |
| Regla de género (mujer = sobrevive) | 0.789 | 0.714 | 0.770 |

**Criterio de éxito:** exactitud mayor a 0.79 y ROC-AUC mayor a 0.77 con un margen defendible. El modelo elegido tiene 0.826 de exactitud (+0.037) y 0.891 de ROC-AUC en validación cruzada, así que cumple el criterio. En prueba la ventaja en exactitud es de solo +0.016 (0.793 vs. 0.777), dentro de la incertidumbre.

## Selección de modelos

Resultados de validación cruzada con los parámetros por defecto (ROC-AUC media ± desviación):

| Modelo | ROC-AUC | Brecha entrenamiento − validación | Interpretabilidad | Costo (5 pliegues) |
|---|---|---|---|---|
| `hgb` | 0.887 ± 0.019 | 0.107 | Media (importancia por permutación) | ≈ 2 s |
| `forest` | 0.866 ± 0.016 | 0.133 | Media | ≈ 6 s |
| `logreg` | 0.858 ± 0.024 | 0.006 | Alta (coeficientes y razones de momios) | < 1 s |
| `tree` | 0.847 ± 0.026 | 0.046 | Alta | < 1 s |
| `knn` | 0.841 ± 0.017 | 0.087 | Baja | ≈ 1 s |

**Finalistas: `hgb`, `forest` y `logreg`.**

- **`hgb`:** el mejor desempeño con estabilidad similar a los demás y costo bajo. Sobreajusta (brecha de 0.107), pero valida mejor que el resto. Es el elegido.
- **`forest`:** segundo mejor y el más estable (desviación 0.016), pero con la mayor brecha y más costo computacional.
- **`logreg`:** un poco peor, pero casi sin sobreajuste y totalmente explicable; sirve como modelo de referencia y para interpretar coeficientes.
- **Descartados:** `tree` y `knn` rinden menos y no aportan ventajas en interpretabilidad (el árbol) ni en costo. `dummy` es solo un baseline.

## Decisiones congeladas

| Decisión | Valor | Justificación |
|---|---|---|
| Modelo | `hgb` | Mejor ROC-AUC en validación cruzada (Paso 49). |
| Hiperparámetros | `learning_rate=0.1`, `max_depth=2`, `max_iter=200`, `l2_regularization=1.0` | La superficie de búsqueda es plana (los 5 mejores quedan entre 0.8899 y 0.8925), así que se eligió la profundidad más simple (0.8910). El ajuste mejoró +0.0057 ± 0.0103, menos que la variabilidad entre pliegues. |
| Imputación de edad | Por `Sex` y `Pclass`, aprendida en entrenamiento | Igual que solo por `Sex` (0.8585 vs. 0.8584); es la más específica. |
| Variables de cabina | No se incluyen | No mejoran (−0.0038 de ROC-AUC) y pueden ser un artefacto del registro: la cabina pudo conservarse más para quienes sobrevivieron. |
| Umbral | 0.43 | Maximiza F1 fuera de muestra (0.782 vs. 0.758 con 0.5). La mejora es modesta. |

## Resultado en prueba (una sola vez)

| Métrica | Valor | IC 95 % (bootstrap, 1000 remuestreos) |
|---|---|---|
| Exactitud | 0.793 | 0.732 – 0.849 |
| F1 | 0.726 | 0.627 – 0.803 |
| ROC-AUC | 0.815 | 0.740 – 0.880 |

El ROC-AUC de prueba (0.815) es menor que el de validación cruzada (≈ 0.89). Con 179 pasajeros el intervalo es ancho, y la validación cruzada es algo optimista porque con ella se eligieron el modelo, los hiperparámetros y el umbral. No se volvió a ajustar nada mirando el conjunto de prueba.

## Reflexión ética

El modelo usa `Sex` y `Pclass`, que dominaron las decisiones de evacuación de 1912 (`Sex` es la variable más importante en la prueba por permutación). Predecir bien lo que ocurrió no significa que deba ocurrir así.

- **Se podría usar** para estudiar patrones históricos de supervivencia, como material académico, o para comparar modelos de riesgo en un entorno de práctica.
- **No se debería usar** para decidir quién recibe un servicio, una prima o una atención hoy, porque usaría sexo y clase social para discriminar.

## Limitaciones

- **Tamaño de muestra:** 891 pasajeros etiquetados y solo 179 en prueba; los intervalos de confianza son anchos (≈ 0.12 en exactitud y 0.14 en ROC-AUC).
- **Un solo suceso histórico:** los patrones de un naufragio de 1912 no se generalizan a otros desastres ni a la actualidad.
- **Datos faltantes:** 20 % de edades y 77 % de cabinas. La edad se imputa con estadísticas del entrenamiento, lo cual introduce sesgo hacia el promedio.
- **Incertidumbre:** el desempeño en prueba (ROC-AUC 0.815, IC 0.74–0.88) es menor que el de validación cruzada; la diferencia con la regla de género en prueba no es concluyente.
- **Variables sensibles:** `Sex` y `Pclass` son muy influyentes (ver reflexión ética).
- **Qué decisiones NO debería respaldar este modelo:** decisiones sobre personas reales (precios de seguros, acceso a servicios, evacuaciones) y conclusiones causales ("ser mujer protege"): el modelo mide asociación, no causalidad.
