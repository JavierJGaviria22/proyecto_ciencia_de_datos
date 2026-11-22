# Conclusiones del EDA — Titanic

Informe para la gerencia de la aseguradora. Los datos son los 891 pasajeros con etiqueta (capa Silver).
Cifras tomadas de `notebooks/02_exploracion.ipynb` y de las tablas de `data/gold/`.

## 1. Hallazgos

1. **El sexo es el factor más asociado con sobrevivir.** Sobrevivió el 74.2 % de las mujeres (233 de 314) y el 18.9 % de los hombres (109 de 577). La prueba chi-cuadrado da p ≈ 1.2e-58, por lo que la asociación no parece azar.
2. **La clase también influye.** La supervivencia fue de 63.0 % en primera clase (n = 216), 47.3 % en segunda (n = 184) y 24.2 % en tercera (n = 491).
3. **Sexo y clase interactúan.** En primera clase sobrevivió el 96.8 % de las mujeres y el 36.9 % de los hombres; en tercera, el 50.0 % de las mujeres y el 13.5 % de los hombres (tabla `survival_by_class_sex`).
4. **Los niños sobrevivieron más que los adultos hombres.** Los 83 niños (menores de 16) tuvieron 59.0 % de supervivencia frente a 16.4 % de los 537 hombres. En tercera clase los niños (n = 58) sobrevivieron solo el 43 %.
5. **Viajar con familia se asocia con mayor supervivencia.** 50.6 % (n = 354) frente a 30.4 % de quienes viajaban solos (n = 537). Con 4 o más familiares la supervivencia cae, pero con muy pocos pasajeros por grupo.

## 2. Decisiones de limpieza y su sesgo

- **Edad imputada (177 faltantes, 19.9 %).** Se rellenó con el promedio por sexo. La desviación estándar bajó de 14.53 a 13.01 porque todas las edades imputadas toman el mismo valor, lo que aplana la distribución y agrega un pico en el histograma. Por eso Silver conserva `AgeOriginal`, y el modelo imputa por su cuenta.
- **`Person` sin edad (Reto 8.1).** Un pasajero sin edad nunca es `child`. Entre los 714 con edad hay 83 niños (11.6 %); si los 177 sin edad tuvieran la misma proporción, habría unos 21 niños más. `Person` subestima entonces a los niños en unos 21 casos.
- **`Master` → `Mr`.** Se fusionan 40 `Master` (edad promedio 4.6 años y máximo 12) con los adultos `Mr`. Mezclar niños con adultos no es del todo defendible. `Person` compensa en parte, porque marca como `child` a los menores con edad (36 de los 40 `Master` tienen edad); los otros 4 quedan como `male`.
- **`X` en cabina.** El 77 % de las cabinas faltan (687 de 891) y se convierten en la categoría `X` ("desconocida"). Esto trata "no sé" como una categoría, y esa categoría tiene supervivencia de 30 % frente a 67 % de quienes tienen cabina registrada. Puede ser un artefacto del registro, no un efecto real de la cabina.
- **`Embarked` (2 faltantes).** Se rellenó con la moda (`S`); el impacto es mínimo.

## 3. Confusiones detectadas

- **Puerto de embarque vs. clase.** Cherbourg parece tener más supervivencia (55.4 %, n = 168) que Southampton (33.9 %, n = 646), pero Cherbourg tiene más pasajeros de primera clase (50.6 %). Al estratificar por clase la diferencia se reduce (primera: 69 % vs. 59 %; tercera: 38 % vs. 19 %). Parte del efecto del puerto es en realidad el de la clase.
- **Tarifa vs. clase.** Pagar más se asocia con más supervivencia dentro de primera y segunda clase (en primera, de 46 % a 74 % entre el primer y el cuarto cuartil), pero en tercera casi no cambia (23 % a 28 %).
- **Sexo, clase y edad.** Los hombres mayores de 10 años sobreviven poco en todos los rangos de edad (entre 11 % y 19 %), así que el sexo domina sobre la edad.

## 4. Qué NO podemos afirmar

- **Asociación no es causalidad.** El chi-cuadrado dice que sexo y supervivencia no son independientes, pero no que el sexo sea la causa ni cuánto pesa por sí solo.
- **Un solo naufragio.** Son datos de un único evento de 1912, con protocolos de evacuación propios; no se pueden generalizar directamente a otros desastres.
- **Datos incompletos.** Faltan edades (20 %) y cabinas (77 %); los resultados por edad y cubierta dependen de cómo se rellenaron.
- **Grupos pequeños.** Con pocos pasajeros (familias de 6 o más, la cubierta `T` con un solo registro, `Otro` con 21) los porcentajes son poco confiables.
- **Atípicos.** Los 116 atípicos de `Fare` (supervivencia 68.1 % vs. 33.9 % del resto) son pasajeros de primera clase, es decir casos reales, no errores; no se eliminaron.
