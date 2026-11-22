# Ficha del modelo — `titanic_model_v1`

## Propósito y uso previsto

Estimar la probabilidad de que un pasajero del Titanic haya sobrevivido, a partir de datos conocidos al embarcar. Es un ejercicio académico (Taller #2, Ciencia de Datos y Machine Learning) para estudiar patrones históricos de supervivencia. **No** es un modelo para tomar decisiones sobre personas reales.

## Datos de entrenamiento

- **Fuente:** el CSV del Titanic definido en `.env` (`TITANIC_URL`); se guarda en Bronze con su huella en `data/bronze/_ingestion_metadata.json`.
- **Tamaño:** 891 pasajeros etiquetados (los 418 restantes no tienen etiqueta y solo se usan en la prueba de humo).
- **Periodo:** abril de 1912 (un solo naufragio).
- **Huella SHA-256 de Silver** (`data/silver/titanic_silver.parquet`): `1169b3fa24d4b86ae51e4b7402d10c24d464c234da3560e691dd56bf02073a0e`.
- **Modelo final:** entrenado con los 891 pasajeros. La huella del `.joblib` queda en `models/titanic_model_v1.metadata.json` (cambia cada vez que se reentrena).

## Modelo

`HistGradientBoostingClassifier` dentro de un `Pipeline` (imputación de edad por `Sex` y `Pclass` → imputación y escalado de numéricas, *one-hot* de categóricas → modelo). Hiperparámetros: `learning_rate=0.1`, `max_depth=2`, `max_iter=200`, `l2_regularization=1.0`. Semilla 42. Umbral de decisión: 0.43.

## Variables

- **Usadas:** `Age`, `SibSp`, `Parch`, `Fare`, `Familia`, `Pclass`, `Sex`, `Embarked`, `TTitulo`, `Alone`.
- **Excluidas:** `Survived` y `Survivor` (objetivo), `PassengerId`, `Ticket`, `Name` (identificadores), `NSex`, `NEmbarked`, `NCabin`, `NTitulo` (codificaciones con orden arbitrario), `Person` (redundante) y las variables de cabina (`Cabin`, `TCabin`, `HasCabin`): no mejoran el desempeño y pueden ser un artefacto del registro.

## Métricas

| Evaluación | Exactitud | F1 | ROC-AUC |
|---|---|---|---|
| Validación cruzada (5 pliegues, entrenamiento, n = 712) | 0.826 | 0.758 | 0.891 (± 0.026) |
| Prueba (n = 179) | 0.793 (IC 95 %: 0.732–0.849) | 0.726 (0.627–0.803) | 0.815 (0.740–0.880) |
| Regla de género, prueba | 0.777 | 0.692 | 0.753 |

La ventaja sobre la regla de género en prueba (+0.016 de exactitud) está dentro de la incertidumbre. El desempeño en prueba es menor que el de validación cruzada, que es algo optimista porque con ella se eligieron el modelo, los hiperparámetros y el umbral.

## Desempeño por subgrupos

Calculado con predicciones fuera de muestra sobre el entrenamiento (Paso 54, `hgb` con parámetros por defecto y umbral 0.5):

| Subgrupo | n | Exactitud | Recall |
|---|---|---|---|
| Mujeres | 253 | 0.806 | 0.888 |
| Hombres | 459 | 0.828 | 0.353 |
| Primera clase | 171 | 0.789 | 0.820 |
| Segunda clase | 150 | 0.913 | 0.881 |
| Tercera clase | 391 | 0.798 | 0.495 |
| Niños (< 16) | 66 | 0.879 | 0.897 |

El modelo detecta mal a los hombres y a los pasajeros de tercera clase que sobrevivieron.

## Limitaciones y usos no recomendados

- Pocos datos (891; prueba de 179) y un solo suceso histórico.
- Edades faltantes (20 %) y cabinas faltantes (77 %).
- Usa `Sex` y `Pclass`, variables sensibles que dominaron la evacuación de 1912.
- **No usar** para fijar precios, aceptar o rechazar personas, ni para asignar servicios o prioridades hoy.
- **No interpretar** la importancia de las variables como causalidad.
- Los archivos `.joblib` pueden ejecutar código al cargarse: solo cargar modelos de una fuente confiable.

## Versiones y reproducción

- Python 3.13.3, scikit-learn 1.9.1, pandas 3.1.1 (versiones exactas en `requirements.lock.txt` y en los metadatos del modelo).
- Para reproducir: `make install` y luego `make pipeline` (ingesta → Silver → Gold → modelo guardado en `models/`); las pruebas con `make test`.
