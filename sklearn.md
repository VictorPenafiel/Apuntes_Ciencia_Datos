# Escalamiento 
Asegura que las variables tengan un impacto equilibrado en los modelos.

##  Normalización (Min-Max Scaling)
Transforma los datos a un rango específico (por defecto, [0, 1]).
Usos: Algoritmos como KNN y redes neuronales donde los datos deben estar en un rango fijo.

    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    # Datos de ejemplo
    data = np.array([[10, 2], [5, 8], [3, 6]])

    # Crear y aplicar scaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    print("Datos originales:\n", data)
    print("Datos normalizados (0-1):\n", scaled_data)

## Estandarización (Z-Score Scaling)
Transforma los datos para que tengan media (μ = 0) y desviación estándar (σ = 1).
Usos: SVM, Regresión Lineal, PCA y algoritmos que asumen distribución normal.
    
    from sklearn.preprocessing import StandardScaler

    # Datos de ejemplo
    data = np.array([[10, 2], [5, 8], [3, 6]])

    # Crear y aplicar scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    print("Datos originales:\n", data)
    print("Datos estandarizados (μ=0, σ=1):\n", scaled_data)

## Robust Scaling (Escalado Robusto)
Utiliza los percentiles 25 y 75 (rango intercuartílico) para reducir el impacto de outliers.
Usos: Datos con outliers significativos.

    from sklearn.preprocessing import RobustScaler

    # Datos con outlier
    data = np.array([[10, 2], [5, 8], [3, 6], [100, 1]])

    # Crear y aplicar scaler
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)

    print("Datos originales:\n", data)
    print("Datos escalados robustos:\n", scaled_data)

## Normalización L1 y L2
L1 (Manhattan): Escala los datos para que la suma de los valores absolutos sea 1.
L2 (Euclidiana): Escala para que la suma de los cuadrados sea 1.
Usos: NLP, K-Means. Text mining (TF-IDF) o cuando se requiere vectores unitarios.

    from sklearn.preprocessing import normalize

    # Normalización L1 (suma de valores absolutos = 1)
    l1_scaled = normalize(data, norm='l1')

    # Normalización L2 (suma de cuadrados = 1)
    l2_scaled = normalize(data, norm='l2')
----------------------------------------------------------------------------------------------------------------------

### Gradient Boosting 
## GradientBoostingClassifie
Algoritmo de machine learning supervisado para problemas de clasificación (predecir categorías/binarias). Es parte de la familia de métodos de ensamble por boosting, donde modelos débiles (generalmente árboles de decisión simples) se combinan secuencialmente para crear un predictor fuerte.
Es un algoritmo de ensamble que construye múltiples árboles de decisión secuencialmente, donde cada árbol corrige los errores del anterior.

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

````
from sklearn.datasets import make_hastie_10_2

from sklearn.ensemble import GradientBoostingClassifier

X, y = make_hastie_10_2(random_state=0)

X_train, X_test = X[:2000], X[2000:]

y_train, y_test = y[:2000], y[2000:]

clf = GradientBoostingClassifier(
    n_estimators=100, 
    learning_rate=1.0,
    max_depth=1, 
    random_state=0)
.fit(X_train, y_train)

clf.score(X_test, y_test)
````

## GradientBoostingRegressor
Algoritmo de aprendizaje automático supervisado utilizado para problemas de regresión (predicción de valores continuos). Pertenece a la familia de métodos de ensamble por boosting, donde múltiples modelos débiles (normalmente árboles de decisión simples) se combinan secuencialmente para formar un modelo fuerte.

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

````
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
X, y = make_regression(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
reg.predict(X_test[1:2])
reg.score(X_test, y_test)
````