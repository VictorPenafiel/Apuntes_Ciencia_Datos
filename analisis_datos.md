### Data analisis
Es la cleccion, transformacion y organizacion de los datos para poder hacer predicciones, sacar conclusiones e impulsar la toma de decisiones informadas.

Un dato es la representación simbólica de un hecho. No son exactamente lo mismo, pero en el mundo de la información, tratamos al dato como si fuera el hecho mismo.

### Principales habilidades: 
la curiosidad, la comprensión del contexto, la mentalidad técnica, el diseño de datos y la estrategia de datos.

Ser concientes de las necesidades empresariales

Lo principal es descubrir tendencias, patrones y relaciones

Los cinco porqués es una técnica sencilla pero eficaz para identificar una causa raíz. Consiste en preguntar "¿Por qué?" repetidamente hasta que se revela la respuesta. Esto suele ocurrir al quinto "por qué", pero a veces tendrá que seguir preguntando más veces, otras menos.

los analistas de datos desempeñan un papel fundamental en el éxito de sus empresas, pero es importante tener en cuenta que, independientemente de lo valiosa que sea la toma de decisiones basada en los datos, los datos por sí solos nunca serán tan poderosos como los datos combinados con la experiencia humana, la observación y, a veces, incluso la intuición.  

### Pensamiento analitico, aspectos claves:
la visualización, 
la estrategia, 
la orientación a los problemas, 
la correlación
El pensamiento general y orientado a los detalles.

Las fases del proceso de análisis de datos son preguntar, preparar, procesar, analizar, compartir y actuar.

### Proceso de análisis de datos:

    Pregunte a

    Prepare

    Procesar

    Analizar

    Compartir

    Actuar
    
Los datos prácticamente en todas partes. Cada vez que observa y evalúa algo en el mundo, está recopilando y analizando datos. Su análisis le ayuda a encontrar formas más sencillas de hacer las cosas, a identificar patrones que le ahorran tiempo y a descubrir nuevas perspectivas sorprendentes que pueden cambiar por completo su forma de experimentar las cosas.

¿Cómo hacen los analistas de datos para dar vida a los datos?
Pues bien, todo empieza con la herramienta de análisis de datos adecuada.
Entre ellas se incluyen las hojas de cálculo, las bases de datos, los lenguajes de consulta y el software de visualización

---

### El ciclo de vida de los datos es:

    Planificar: Decidir qué tipo de datos se necesitan, cómo se gestionarán y quién será responsable de ellos.

    Captar: Recopilar o traer datos de distintas fuentes.

    Gestionar: Cuidar y mantener los Datos. Esto incluye determinar cómo y dónde se almacenan y las herramientas utilizadas para ello.

    Analizar: Utilizar los Datos para resolver problemas, tomar decisiones y respaldar los objetivos empresariales.

    Archivar: Mantener los Datos relevantes almacenados para su consulta a largo plazo y en el futuro.

    Destruir: Sacar los datos del almacén y eliminar cualquier copia compartida de los mismos.


### Escuela de Negocios de Harvard (HBS)

Ciclo de vida de los datos basado en la investigación de la Universidad de Harvard consta de ocho etapas:

    Generación

    Recopilación

    Procesamiento

    Almacenamiento 

    Gestionar

    Análisis

    Visualización de datos

    Interpretación


---

### Ciclo de vida de analisis de datos


    Formular preguntas y definir el problema.

    Preparar los Datos recopilando y almacenando la Información.

    Procese los datos limpiando y comprobando la información.

    Analice los Datos para encontrar Patrones, Relaciones y Tendencias.

    Comparta los Datos con su público.

    Actúe sobre los Datos y utilice los resultados del análisis.

---

## <font color='blue'>**1D-CNN aplicado a series de tiempo univariadas.**</font>
<p style='text-align: justify;'>

### **Etapa 1: Análisis Exploratorio y Limpieza de Datos (EDA) 📊**

Antes de construir cualquier modelo, es fundamental entender y preparar los datos. En esta primera etapa, realizaremos un **Análisis Exploratorio de Datos (EDA)** para familiarizarnos con la distribución, tendencias y características de la serie temporal de PM2.5. Las librerías principales para esta fase son `pandas`, `matplotlib`, `numpy` y `seaborn`.

---

#### **Manejo de Datos Faltantes y Orden Cronológico 🧼**

El primer paso de la preparación es la limpieza y estructuración de los datos.

* **Datos Faltantes:** En una serie temporal, los valores ausentes (NAs o `NaN`) pueden romper la continuidad. Existen dos estrategias principales:
    1.  **Eliminación:** Descartar los registros (filas) que contienen valores faltantes.
    2.  **Imputación:** 
        1. Usando la media
        2. Usando la mediana
        3. Usando la moda
        4. Calcular una medición apropiada y reemplazar los NAs.
        5. Utilizar modelos estadísticos y de Machine Leaning.


* **Orden Cronológico:** A continuación, se asegura la integridad cronológica de la serie. Dado que en las series de tiempo **el orden es primordial**, se crea una columna `datetime` unificada y se ordena el DataFrame para garantizar que las observaciones estén en la secuencia correcta.

---

#### **Visualización de la Distribución y Tendencias**

Una vez que los datos están limpios y ordenados, la visualización nos ayuda a descubrir patrones.

* **Diagrama de Caja (Box Plot):** Se utiliza para analizar la distribución de los datos de PM2.5. Este gráfico es excelente para visualizar rápidamente el resumen de cinco números (mínimo, cuartiles y máximo), identificar la dispersión de los datos y detectar la presencia de **valores atípicos** (outliers). 

* **Gráfico de Línea:** Complementariamente, un gráfico de línea muestra la evolución de la variable PM2.5 a lo largo del tiempo, permitiendo identificar patrones generales, **tendencias y estacionalidad**.

---

### **Etapa 2: Preprocesamiento y Transformación de Datos ⚙️**

<p style='text-align: justify;'>
En esta segunda etapa preparamos los datos con el objetivo de realizar un entrenamiento robusto de nuestra red neuronal. Usualmente los datos como primera etapa se normalizan. Empiricamente se ha observando que los datos normalizados (No siempre) generan modelos de clasificiación y de regresión con mejores metricas que los no normalizadas. Por otra parte, Los algoritmos de descenso de gradiente funcionan mejor (por ejemplo, convergen más rápido) si las variables están dentro del rango [-1, 1]. Muchas fuentes relajan el límite incluso [-3, 3].</p>
<p style='text-align: justify;'>
Posteriormente los datos debens ser separados en tres conjuntos: Entrenamiento, validación y test. Usualmente el ultimo de test se utiliza con una prueba nueva de datos. Finalmente debemos construir el conjunto de vectores que serán utilizados para entrenar la red neuronal perceptron multicapa. Las librerias. Adicionalmente en esta sección utilizaremos la librería sklearn para realizar la normalización.
</p>

---

#### **Normalización de Datos**

El primer paso es la **normalización**. Se escalan los valores de `pm2.5` a un rango entre 0 y 1. Esta técnica es casi obligatoria en redes neuronales, ya que ayuda a que los algoritmos de optimización (como el descenso de gradiente) converjan más rápido y de forma más estable. Un modelo entrenado con datos normalizados generalmente produce mejores resultados.

---

#### **División del Conjunto de Datos: Entrenamiento y Validación**

El modelo necesita datos para aprender y datos para ser evaluado. Por ello, el conjunto de datos se divide cronológicamente en:

* **Conjunto de Entrenamiento (Training Set):** Se utiliza para entrenar el modelo. Aquí, la red ajusta sus pesos mediante el cálculo de la función de pérdida y la retropropagación (*backpropagation*).
* **Conjunto de Validación (Validation Set):** Se usa para evaluar el rendimiento del modelo en datos que no ha visto durante el entrenamiento. Es crucial para detectar el **sobreajuste** (*overfitting*) y para ajustar hiperparámetros como el número de épocas.

---

#### **Transformación para Aprendizaje Supervisado*

Una serie temporal, por sí misma, no tiene un formato de entrada (`X`) y salida (`y`). Para que la red neuronal pueda aprender, debemos transformarla en un problema de **aprendizaje supervisado**. 

En este caso, se define que para predecir el valor de PM2.5 en una hora determinada (`y`), el modelo utilizará como información las mediciones de las **7 horas anteriores** (`X`). Se crea una función que recorre la serie y genera estos pares `(X, y)`, preparando así los vectores de entrada para el entrenamiento del modelo. La elección de una ventana de 7 horas se basa en la observación de patrones en los datos.