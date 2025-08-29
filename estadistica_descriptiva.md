
-----

* **1. Población**: Corresponde al conjunto completo de individuos de interés. En este caso, son **todos los alumnos de Enseñanza Media de establecimientos municipales de la Región Metropolitana** durante octubre de 2020.

* **2. Muestra**: Es el subconjunto de la población que fue efectivamente estudiado. Aquí, la muestra es **el grupo de alumnos seleccionados aleatoriamente, correspondiente al 10% de la matrícula de cada establecimiento**.

* **3. Variables**: Son las características que se miden en la muestra.
    * *¿En qué curso va actualmente?*: **Variable Cualitativa Ordinal** (porque existe un orden jerárquico: 1°, 2°, 3°, 4° Medio).
    * *De reanudarse las clases presenciales, ¿ud. se incorporaría?*: **Variable Cualitativa Nominal** (las respuestas, "Sí" o "No", no tienen un orden intrínseco).
    * *¿Cuántas horas a la semana tiene actualmente de clases online?*: **Variable Cuantitativa Continua** (puede tomar cualquier valor en un rango, como 3.5 horas, aunque a menudo se reporte como discreta).

* **4. Datos**: Son los valores específicos recolectados para cada variable. Por ejemplo: "3° Medio", "Sí", "10 horas" serían los datos de un alumno encuestado.

### Toda variable se puede clasificar en dos grandes grupos:

#### 1. Variables Cualitativas (o Categóricas)
Describen una cualidad o característica. No se miden con números, sino que se agrupan en categorías.

* **Nominales**: Son categorías que no tienen un orden o jerarquía.
    * *Ejemplos*: Estado civil (`'Soltero'`, `'Casado'`), color de ojos (`'Café'`, `'Azul'`), tipo de contrato (`'Indefinido'`, `'Plazo Fijo'`).
* **Ordinales**: Son categorías que sí siguen un orden o jerarquía.
    * *Ejemplos*: Nivel educativo (`'Básica'`, `'Media'`, `'Superior'`), nivel socioeconómico (`'ABC1'`, `'C2'`, `'C3'`), calificación (`'Malo'`, `'Regular'`, `'Bueno'`).

#### 2. Variables Cuantitativas (o Numéricas)
Representan cantidades y se expresan mediante números, por lo que se pueden realizar operaciones aritméticas con ellas.

* **Discretas**: Toman un número finito o contable de valores. Generalmente son números enteros y provienen de un conteo.
    * *Ejemplos*: Número de hijos (`0`, `1`, `2`), cantidad de asignaturas inscritas (`5`, `6`), número de errores de compilación (`10`, `11`).
* **Continuas**: Pueden tomar un número infinito de valores dentro de un intervalo. Provienen de mediciones.
    * *Ejemplos*: Altura (`1.75m`), peso (`68.7kg`), temperatura (`21.5°C`), tiempo de ejecución de un programa (`0.012s`).

# Conceptos clave del Análisis Exploratorio de Datos (EDA) 

## 1. Conceptos Fundamentales

* **Población**: El universo total de individuos o elementos de interés para un estudio.
* **Muestra**: Un subconjunto representativo y aleatorio de la población, sobre el cual se realizan las mediciones.
* **Parámetro**: Una medida numérica que describe una característica de la **población** (e.g., la media poblacional $\mu$).
* **Estadístico**: Una medida numérica que describe una característica de la **muestra** (e.g., la media muestral $\bar{X}$). Se utiliza para estimar el parámetro poblacional.
* **Error Muestral**: La diferencia entre un parámetro poblacional y el estadístico muestral correspondiente. Un buen muestreo (aleatorio y de tamaño adecuado) minimiza este error.

## 2. Tipos de Variables

Las variables son las características que se miden y se clasifican en:

* **Cualitativas (Categóricas)**: Describen cualidades.
    * **Nominales**: Categorías sin orden (Ej: `Sí/No`, colores).
    * **Ordinales**: Categorías con un orden jerárquico (Ej: nivel educativo, tallas de ropa).
* **Cuantitativas (Numéricas)**: Describen cantidades medibles.
    * **Discretas**: Valores enteros, contables (Ej: número de hijos, páginas de un libro).
    * **Continuas**: Cualquier valor en un rango (Ej: altura, temperatura, tiempo).

## 3. Medidas Estadísticas

### Medidas de Tendencia Central
Indican el valor "típico" o central de los datos.

* **Media ($\bar{X}$)**: El promedio aritmético. Sensible a valores extremos (outliers).
* **Mediana (Me)**: El valor que ocupa la posición central en un conjunto de datos ordenado. Robusta frente a outliers.
* **Moda**: El valor que aparece con mayor frecuencia. Puede ser unimodal, bimodal, multimodal o no existir (amodal).

### Medidas de Dispersión
Miden qué tan esparcidos o variados están los datos.

* **Rango**: Diferencia entre el valor máximo y el mínimo (`$X_{max} - X_{min}$`). Muy simple y sensible a extremos.
* **Rango Intercuartílico (IQR)**: Diferencia entre el tercer y el primer cuartil (`$Q_3 - Q_1$`). Mide la dispersión del 50% central de los datos, siendo robusto a outliers.
* **Varianza ($S^2$) y Desviación Estándar ($S$)**: Miden la dispersión promedio de los datos respecto a la media. La desviación estándar es la raíz cuadrada de la varianza y se expresa en las mismas unidades que los datos originales.
* **Coeficiente de Variación (CV)**: `$(S / \bar{X})$`. Es una medida de dispersión *relativa* (expresada como porcentaje) que permite comparar la variabilidad de conjuntos de datos con diferentes unidades o escalas. Un CV bajo (≤ 30%) sugiere datos homogéneos donde la media es representativa.

### Medidas de Posición
Dividen un conjunto de datos ordenado en partes iguales.

* **Cuartiles ($Q_1, Q_2, Q_3$)**: Dividen los datos en cuatro partes iguales (25% cada una). $Q_2$ es la mediana.
* **Percentiles ($P_k$)**: Dividen los datos en 100 partes iguales. El percentil $k$ es el valor que deja por debajo al $k\%$ de los datos.

## 4. Forma de la Distribución

* **Asimetría (Skewness)**: Mide el grado de simetría de la distribución.
    * **Positiva (a la derecha)**: La cola derecha es más larga. Media > Mediana.
    * **Negativa (a la izquierda)**: La cola izquierda es más larga. Media < Mediana.
    * **Simétrica**: Las colas son iguales. Media ≈ Mediana.
* **Curtosis**: Mide qué tan "puntiaguda" o "achatada" es una distribución en comparación con la distribución normal.
    * **Leptocúrtica**: Más puntiaguda, con colas más pesadas (más outliers).
    * **Mesocúrtica**: Similar a la distribución normal.
    * **Platicúrtica**: Más aplanada, con colas más ligeras.
```
