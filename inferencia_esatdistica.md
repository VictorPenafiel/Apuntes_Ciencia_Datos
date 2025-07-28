### 1\. **IE01\_Conceptos\_Básicos-Parte1.ipynb**

Este notebook introduce los fundamentos de la inferencia estadística, diferenciándola de la estadística descriptiva.

  * **Conceptos Clave**:

      * **Estadística**: Disciplina para procesar y organizar datos.
      * **Población**: Conjunto total de elementos con una característica común.
      * **Muestra**: Subconjunto representativo de una población.
      * **Parámetro**: Medida numérica que describe una característica de la **población** (e.g., media poblacional μ).
      * **Estadístico**: Medida numérica que describe una característica de la **muestra** (e.g., media muestral x̄).
      * **Sesgo (Bias)**: Errores sistemáticos en la medición o muestreo que no se deben al azar.

  * **Técnicas de Muestreo Aleatorio**:

      * **Muestreo Aleatorio Simple**: Cada individuo tiene la misma probabilidad de ser seleccionado.
      * **Muestreo Estratificado**: La población se divide en estratos homogéneos y se muestrea de cada uno.
      * **Muestreo por Conglomerados**: Se seleccionan grupos completos (conglomerados) al azar.
      * **Muestreo de Varias Etapas**: Combinación de las técnicas anteriores.

  * **Teoría de Probabilidad**:

      * Introduce los conceptos de **espacio muestral**, **evento** y la **Regla de Laplace** para calcular probabilidades en experimentos con resultados equiprobables.
      * `P(A) = Casos Favorables / Casos Totales`

-----

### 2\. **IE02\_Distribuciones\_de\_probabilidades-2.ipynb**

Este notebook profundiza en las variables aleatorias y sus distribuciones de probabilidad.

  * **Variables Aleatorias**:

      * **Discretas**: Toman un número contable de valores (e.g., número de caras en un lanzamiento de moneda).
      * **Continuas**: Toman un número incontable de valores en un rango (e.g., la altura de una persona).

  * **Funciones de Probabilidad**:

      * **Función de Masa de Probabilidad (FMP/PMF)**: Para variables discretas, asigna una probabilidad a cada valor posible.
      * **Función de Densidad de Probabilidad (FDP/PDF)**: Para variables continuas, el área bajo la curva representa la probabilidad.
      * **Función de Distribución Acumulada (FDA/CDF)**: Devuelve la probabilidad de que una variable aleatoria sea menor o igual a un valor `x`.

-----

### 3\. **IE03\_Graficando\_distribuciones\_de\_probabilidades.ipynb**

Aquí se explica cómo visualizar las distribuciones de probabilidad utilizando Python.

  * **Visualizaciones Clave**:

      * **Histogramas**: Representan la frecuencia de los valores de un conjunto de datos. Se utiliza `matplotlib.pyplot.hist()`.
      * **Función de Masa de Probabilidad (FMP)**: Gráfica que relaciona cada valor con su probabilidad. Ideal para distribuciones discretas.

  * **Ejemplo de Código (Histograma de una Normal)**:

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    mu, sigma = 0, 0.2 # media y desviación estándar
    datos = np.random.normal(mu, sigma, 1000)
    plt.hist(datos, bins=30, density=True)
    plt.show()
    ```

-----

### 4\. **IE04\_Distribuciones\_discretas.ipynb**

Este notebook se centra en las distribuciones de probabilidad para variables aleatorias discretas.

  * **Distribuciones Notables**:
      * **Uniforme Discreta**: Todos los resultados tienen la misma probabilidad (e.g., lanzar un dado).
      * **Bernoulli**: Experimento con dos resultados (éxito/fracaso) con probabilidad `p` y `1-p`.
      * **Binomial**: Número de éxitos en `n` ensayos de Bernoulli independientes. `X ~ B(n, p)`.
      * **Poisson**: Modela el número de eventos que ocurren en un intervalo de tiempo o espacio, con una tasa media conocida (λ).

-----

### 5\. **IE05\_Distribuciones\_continuas.ipynb**

Se presentan las distribuciones de probabilidad para variables aleatorias continuas.

  * **Distribuciones Notables**:
      * **Uniforme Continua**: La probabilidad es constante sobre un intervalo (a, b).
      * **Normal (Gaussiana)**: La distribución más importante en estadística, caracterizada por su media (μ) y desviación estándar (σ). `X ~ N(μ, σ)`.
      * **Exponencial**: Modela el tiempo entre eventos en un proceso de Poisson.
      * **Log-Normal, Beta, Pareto**: Otras distribuciones para modelar fenómenos específicos.

-----

### 6\. **IE06\_Conceptos\_Básicos-Parte2.ipynb**

Este notebook introduce dos de los teoremas más importantes de la estadística.

  * **Ley de los Grandes Números**:

      * Afirma que, si un experimento se repite un gran número de veces, la frecuencia relativa de un evento convergerá a su probabilidad teórica. Es la razón por la que el muestreo funciona.

  * **Teorema del Límite Central (TLC)**:

      * Establece que la distribución de las **medias muestrales** de una población (con cualquier distribución) se aproximará a una **distribución normal** a medida que el tamaño de la muestra (`n`) aumenta (generalmente `n ≥ 30`).
      * La media de esta distribución de medias muestrales será igual a la media poblacional (μ).
      * La desviación estándar de esta distribución (llamada **error estándar**) será `σ / sqrt(n)`.

-----

### 7\. **IE07\_Caso\_Practico\_de\_TLC-Simulacion\_Montecarlo.ipynb**

Se aplica el Teorema del Límite Central en un caso práctico utilizando simulación de Montecarlo.

  * **Simulación de Montecarlo**:

      * Técnica que utiliza muestreo aleatorio para obtener resultados numéricos. Es útil para entender el impacto de la aleatoriedad.
      * El notebook simula la producción semanal de una fábrica para predecir la producción futura.

  * **Pasos del Ejemplo**:

    1.  Se parte de datos históricos de producción (una muestra).
    2.  Se realizan miles de simulaciones de producción para un período futuro (e.g., 5 semanas), muestreando aleatoriamente de los datos históricos.
    3.  Se calcula la suma de la producción para cada simulación.
    4.  El histograma de estas sumas (gracias al TLC) se asemeja a una distribución normal, lo que permite calcular probabilidades sobre la producción futura.

-----

### 8\. **IE08\_Intervalos\_de\_Confianza.ipynb**

Este notebook explica cómo construir e interpretar intervalos de confianza, una herramienta clave de la inferencia.

  * **Intervalo de Confianza (IC)**:

      * Es un rango de valores calculado a partir de una muestra, dentro del cual se espera que se encuentre el verdadero **parámetro poblacional** (e.g., la media μ) con un cierto nivel de confianza (e.g., 95%).
      * Un IC del 95% **no** significa que hay un 95% de probabilidad de que el parámetro esté en el intervalo. Significa que si repitiéramos el muestreo muchas veces, el 95% de los intervalos calculados contendrían el parámetro.

  * **Fórmula (para la media con σ conocida o n grande)**:
    `IC = x̄ ± Z * (σ / sqrt(n))`

      * `x̄`: Media muestral.
      * `Z`: Valor crítico de la distribución normal estándar (e.g., 1.96 para un 95% de confianza).
      * `σ / sqrt(n)`: Error estándar.

  * **Distribución t de Student**:

      * Se utiliza cuando la desviación estándar de la población (σ) es **desconocida** y el tamaño de la muestra (`n`) es **pequeño** (típicamente n \< 30).

-----

### 9\. **Pruebas\_Hipotesis\_1y2\_Muestras-2.ipynb**

Introduce el concepto y procedimiento de las pruebas de hipótesis.

  * **Prueba de Hipótesis**:

      * Procedimiento estadístico para tomar una decisión sobre una afirmación acerca de un parámetro poblacional, basándose en la evidencia de una muestra.

  * **Componentes Clave**:

      * **Hipótesis Nula (H₀)**: La afirmación que se asume como cierta inicialmente. Generalmente representa "no hay efecto" o "no hay diferencia".
      * **Hipótesis Alternativa (H₁)**: La afirmación que se quiere probar.
      * **Nivel de Significancia (α)**: La probabilidad de cometer un **Error de Tipo I** (rechazar H₀ cuando es verdadera). Comúnmente se usa α = 0.05.
      * **Valor p (p-value)**: La probabilidad de obtener un resultado tan extremo (o más) como el observado en la muestra, asumiendo que H₀ es verdadera.
          * Si `p-value < α`, se **rechaza** la hipótesis nula H₀.
          * Si `p-value ≥ α`, **no se puede rechazar** la hipótesis nula H₀.

  * **Tipos de Pruebas**:

      * **Prueba Z**: Para la media, cuando σ poblacional es conocida o `n` es grande.
      * **Prueba T**: Para la media, cuando σ poblacional es desconocida y `n` es pequeña.
      * **Pruebas para proporciones** y para **comparar dos muestras**.
