[Qué es el Aprendizaje Supervisado y No Supervisado](https://www.youtube.com/watch?v=oT3arRRB2Cw&t=3s)
---

#### Fundamentos y Flujo de Trabajo

* **Conceptos Fundamentales**:
    * **Objetivo**: Introducir la taxonomía del aprendizaje supervisado.
    * **Contenido**:
        * **Regresión vs. Clasificación**: La diferencia fundamental es que la regresión predice valores continuos (ej. precio de una casa), mientras que la clasificación predice etiquetas discretas (ej. spam o no spam).
        * **División de Datos**: Se aplica el protocolo esencial de dividir los datos usando `train_test_split` de Scikit-Learn en conjuntos de **entrenamiento** (para ajustar el modelo), **validación** (para ajustar hiperparámetros) y **prueba** (para una evaluación final e imparcial).

* **Preprocesamiento y Selección de Modelos**:
    * **Objetivo**: Demostrar los pasos prácticos de preparación de datos y optimización de modelos.
    * **Contenido Técnico**:
        * **Pipelines**: Se ocupa el objeto `Pipeline` de Scikit-Learn para encapsular y secuenciar transformaciones de datos y el ajuste del modelo, garantizando un flujo de trabajo limpio y sin fuga de datos.
        * **Codificación de Categóricas**: Se utiliza `OneHotEncoder` para convertir variables categóricas en una representación numérica binaria.
        * **Escalado de Características**: Se aplica `StandardScaler` para estandarizar las características numéricas (media 0, desviación estándar 1), un requisito para muchos algoritmos como SVM y Regresión Logística.
        * **Optimización de Hiperparámetros**: Se implementa `GridSearchCV`, una técnica de búsqueda exhaustiva combinada con validación cruzada (`cross-validation`) para encontrar la mejor combinación de hiperparámetros para un modelo de forma automatizada y robusta.

[Modelos para entender una realidad caótica](https://www.youtube.com/watch?v=Sb8XVheowVQ&t=306shttps://www.youtube.com/watch?v=Sb8XVheowVQ&t=306s)

[Principales Algoritmos usados en Machine Learning](https://www.aprendemachinelearning.com/principales-algoritmos-usados-en-machine-learning/)


#### Conceptos Teóricos Fundamentales

* **Objetivo**: Explicar la teoría detrás del error del modelo.
* **Contenido**:
    * **Descomposición del Error**: El error de generalización de un modelo se descompone en **sesgo (bias)**, **varianza (variance)** y **error irreducible**.
    * **Sesgo (Bias)**: Error por suposiciones erróneas. Un alto sesgo causa **subajuste (underfitting)**.
    * **Varianza**: Error por sensibilidad a los datos de entrenamiento. Una alta varianza causa **sobreajuste (overfitting)**.
    * **El Trade-Off**: El núcleo del aprendizaje automático. Aumentar la complejidad de un modelo generalmente reduce el sesgo pero aumenta la varianza. El objetivo es encontrar el punto óptimo de complejidad que minimice el error total.

#### Descomposición del error cuadrático medio (MSE)

    Error Total=(Sesgo)2+Varianza+Error Irreducible
    E[(y−f^​(x))2]=(E[f^​(x)]−f(x))2+E[(f^​(x)−E[f^​(x)])2]+σϵ2​


#### Modelos de Regresión

* **Objetivo**: Profundizar en el modelo de regresión más fundamental.
* **Contenido Técnico**:
    * **Regresión Lineal Simple**: Se modela la relación entre variables usando la ecuación $y = \beta_0 + \beta_1 x$. Se implementa con la clase `LinearRegression` de Scikit-Learn.
    * **Regresión Polinomial**: Se demuestra cómo los modelos lineales pueden capturar relaciones no lineales. Esto se logra creando características polinómicas (ej. $x^2, x^3$) a partir de las características originales usando `PolynomialFeatures`. Técnicamente, el modelo sigue siendo lineal en sus coeficientes, pero la curva resultante se ajusta a los datos de forma no lineal.

#### Modelos de Clasificación

* **Regresión Logística**:
    * **Concepto**: A pesar de su nombre, es un modelo de clasificación. Utiliza la función logística (sigmoide) para mapear cualquier entrada de valor real a una probabilidad entre 0 y 1.
    * **Hiperparámetros Clave**: Se detalla la optimización de `LogisticRegression` a través de:
        * `penalty` ('l1', 'l2'): Controla el tipo de regularización para prevenir el sobreajuste.
        * `C`: Parámetro que controla la inversa de la fuerza de regularización (valores pequeños indican una regularización más fuerte).
        * `solver`: Algoritmo a utilizar en el problema de optimización (ej. 'liblinear', 'lbfgs').

* **Naive Bayes**
[Cómo Escapar de la Trampa Bayesiana](https://www.youtube.com/watch?v=D7KKlC0LOyw)

    * **Concepto**: Se basa en el **Teorema de Bayes** con la suposición "ingenua" (`naive`) de independencia condicional entre las características.
    * **Implementación**: Se utiliza `GaussianNB`, una variante que asume que la probabilidad de las características para cada clase sigue una distribución gaussiana. Es extremadamente rápido y funciona bien como un buen modelo de referencia inicial.

* **k-Nearest Neighbors (k-NN)**:
[¿Qué es el algoritmo de k vecinos más cercanos (KNN)? ](https://www.ibm.com/mx-es/think/topics/knn)

    * **Concepto**: Es un algoritmo "perezoso" o basado en instancias. No "aprende" un modelo, sino que memoriza todo el conjunto de entrenamiento.
    * **Funcionamiento**: Para clasificar un nuevo punto, encuentra los `k` puntos más cercanos (vecinos) en los datos de entrenamiento y asigna la clase que es más común entre esos vecinos. La elección de `k` es crucial.

* **Máquinas de Soporte Vectorial (SVM)**:
    * **Concepto**: Un clasificador discriminativo que encuentra un **hiperplano** óptimo que separa las clases en el espacio de características.
    * **Optimización**: El hiperplano se elige para maximizar el **margen** (la distancia entre el hiperplano y los puntos de datos más cercanos de cualquier clase, llamados **vectores de soporte**).
    * **El Truco del Kernel (`Kernel Trick`)**: La característica más potente de las SVM. Permite realizar clasificaciones no lineales proyectando los datos a un espacio de características de mayor dimensión donde un separador lineal es suficiente. Se exploran kernels como `'linear'`, `'poly'` y `'rbf'` (base radial).

* **Árboles de Decisión y Random Forest**:
    * **Árboles de Decisión**: Modelos no paramétricos que aprenden reglas de decisión simples inferidas de los datos. Son muy interpretables pero propensos al sobreajuste.
    * **Random Forest**: Un método de **ensamble** que corrige la tendencia al sobreajuste de los árboles de decisión. Construye múltiples árboles de decisión durante el entrenamiento y emite la clase que es el modo de las clases de los árboles individuales. Introduce aleatoriedad mediante **bagging** (muestreo con reemplazo de los datos) y selección aleatoria de subconjuntos de características en cada división.


