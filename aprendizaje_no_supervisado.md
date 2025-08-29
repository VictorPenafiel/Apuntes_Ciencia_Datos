[Overfitting y Underfitting](https://www.youtube.com/watch?v=7-6X3DTt3R8)

[Cómo identificar el OVERFITTING en tu RED NEURONAL](https://www.youtube.com/watch?v=ZmLKqZYlYUI)


## Selección de Características 🎯

La **Selección de Características** (`Feature Selection`), un proceso crucial en el aprendizaje automático para mejorar el rendimiento de los modelos, reducir el sobreajuste (`overfitting`) y disminuir el tiempo de entrenamiento.

### **1. Métodos de Filtrado (Filter Methods)** 📊

Estos métodos seleccionan características basándose en sus propiedades estadísticas, **independientemente de cualquier algoritmo de machine learning**. Son rápidos y computacionalmente eficientes, pero su principal debilidad es que no consideran las interacciones entre las distintas características.

#### **a) Valor F de ANOVA (`ANOVA F-value`)**
* **Concepto**: Estima la relación lineal entre cada característica de entrada (predictor) y la variable de salida. Un valor F alto indica una mayor linealidad y, por lo tanto, una mayor importancia de la característica.
* **Implementación**: Se utiliza la función `f_classif` de `scikit-learn` para calcular los valores F de cada característica del dataset Iris. Los resultados muestran que `petal length` y `petal width` son las características con la relación lineal más fuerte con la clase de la flor.

#### **b) Umbral de Varianza (`Variance Threshold`)**
* **Concepto**: Este método elimina las características que tienen una varianza por debajo de un umbral predefinido. La idea es que las características con poca o ninguna variación aportan poco poder predictivo.
* **Implementación**: Es fundamental **estandarizar los datos** antes de aplicar este filtro. Sin estandarizar, las varianzas pueden ser engañosas debido a las diferentes escalas. Usando `VarianceThreshold` de `scikit-learn` con un umbral, se elimina la característica `sepal width`, que es la que menos varía en el conjunto de datos estandarizado.

#### **c) Información Mutua (`Mutual Information`)**
* **Concepto**: Mide la dependencia entre dos variables, capturando tanto relaciones lineales como no lineales. Cuantifica cuánta información sobre la variable de salida se obtiene al conocer una característica de entrada.
* **Implementación**: Se emplea `mutual_info_classif`. Al igual que con ANOVA, los resultados indican que `petal length` y `petal width` son las características que comparten más información con la variable objetivo, haciéndolas las más relevantes.

---

### **2. Métodos de Envoltura (Wrapper Methods)** 📦

Estos métodos utilizan un **modelo de aprendizaje automático específico** para evaluar subconjuntos de características. Buscan la combinación de características que ofrece el mejor rendimiento para ese modelo en particular. Son más potentes que los métodos de filtro porque detectan interacciones entre variables, pero son computacionalmente mucho más costosos y tienen un mayor riesgo de sobreajuste.

#### **a) Selección Secuencial de Características (`Sequential Feature Selection` - SFS)**
* **Concepto**: SFS es un método iterativo que agrega características una por una al conjunto de características seleccionadas. En cada paso, se añade la característica que más mejora el rendimiento del modelo.
* **Implementación**: Se utiliza `SequentialFeatureSelector` de la librería `mlxtend` con un modelo de Regresión Logística. Se configura para que busque el mejor subconjunto de entre 1 y 4 características. El resultado final muestra que el mejor rendimiento se obtiene utilizando **todas las 4 características** del dataset Iris.

#### **b) Selección Secuencial hacia Atrás (`Sequential Backward Selection` - SBS)**
* **Concepto**: Es el enfoque opuesto a SFS. Comienza con todas las características y, en cada iteración, elimina la característica que menos afecta negativamente al rendimiento del modelo, hasta alcanzar el número deseado de características.
* **Implementación**: Al igual que con SFS, se usa `SequentialFeatureSelector` pero con el parámetro `forward=False`. De nuevo, para el dataset Iris y el modelo de Regresión Logística, el mejor rendimiento se logra conservando **todas las características**.

---

### **3. Métodos Integrados (Embedded Methods)** 🧩

Estos métodos realizan la selección de características **como parte del propio proceso de entrenamiento del modelo**. Son un punto intermedio entre los métodos de filtro y los de envoltura en términos de costo computacional y rendimiento.

#### **Selección de Características usando Random Forest**
* **Concepto**: Los modelos basados en árboles, como Random Forest, calculan de forma natural la importancia de cada característica. Esta importancia se mide por cuánto contribuye cada característica a reducir la impureza (índice de Gini) en los nodos del árbol.
* **Implementación**:
    1.  Se entrena un `RandomForestClassifier` con el conjunto de datos.
    2.  Se accede al atributo `feature_importances_` del modelo entrenado para obtener la puntuación de importancia de cada característica. Los resultados confirman una vez más que `petal length` y `petal width` son, con diferencia, las más importantes.
    3.  Se utiliza `SelectFromModel`, una meta-transformador que selecciona características basándose en un umbral de importancia proporcionado por el estimador (en este caso, Random Forest).
    4.  Finalmente, se compara la precisión (`accuracy`) de un modelo Random Forest entrenado con todas las características frente a uno entrenado solo con las características seleccionadas por `SelectFromModel`. El notebook demuestra que **se puede lograr un rendimiento casi idéntico utilizando un subconjunto más pequeño de características**, lo que valida la eficacia del método.

---

# Transfer Learning

## ¿Qué es? 🧠
El Aprendizaje por Transferencia es una técnica de Machine Learning que consiste en reutilizar un modelo pre-entrenado en una tarea origen como punto de partida para una segunda tarea objetivo.

En Deep Learning, esto es especialmente poderoso porque las primeras capas de una red neuronal convolucional (CNN) tienden a aprender características muy generales y reutilizables (bordes, texturas, colores), mientras que las capas más profundas aprenden características más específicas de la tarea original (ojos, ruedas de coche, etc.). Aprovechamos esas capas iniciales ya entrenadas, lo que nos da una ventaja considerable.


## Beneficios Principales 🏆
* **Ahorro de Tiempo y Recursos:** Reduce drásticamente el tiempo de entrenamiento y la necesidad de potentes GPUs.
* **Mejor Rendimiento con Pocos Datos:** Es la solución ideal cuando no dispones de un dataset masivo para obtener alta precisión con un dataset pequeño.
* **Acceso a Arquitecturas de Vanguardia:** Te permite utilizar arquitecturas de red extremadamente potentes y probadas (como ResNet, InceptionV3, EfficientNet), diseñadas por equipos de investigación de primer nivel, sin tener que implementarlas desde cero.

## Técnicas Comunes 🛠️
1.  **Extracción de Características (Feature Extraction):**
    * **Qué hace:** Se "congela" el modelo base y solo se entrena un nuevo clasificador final.
    * **Cuándo usarlo:** Ideal para datasets pequeños. Es la aproximación más rápida y segura.

2.  **Ajuste Fino (Fine-Tuning):**
    * **Qué hace:** Se re-entrena una pequeña parte de las capas finales del modelo base junto con el nuevo clasificador, usando una tasa de aprendizaje muy baja.
    * **Cuándo usarlo:** Ideal para datasets más grandes para "especializar" aún más el modelo a tus datos.


