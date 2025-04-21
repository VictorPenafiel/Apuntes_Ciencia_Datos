# # Dendrograma (Agrupamiento Jerárquico)

Es un método de Machine Learning generalmente empleado para la organización y clasificación de datos, con el fin de detectar patrones y agrupar elementos, permitiendo así diferenciar unos de otros.

````
# Tratamiento de datos

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# Gráficos

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score

# Configuración warnings

import warnings
warnings.filterwarnings('ignore')

def plot_dendrogram(model, **kwargs):
    '''
    Esta f unción extrae la información de un modelo AgglomerativeClustering
    y representa su dendograma con la función dendogram de scipy.cluster.hierarchy
    '''
    
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot
    dendrogram(linkage_matrix, **kwargs)

# Simulación de datos

X, y = make_blobs(
        n_samples    = 200, 
        n_features   = 2, 
        centers      = 4, 
        cluster_std  = 0.60, 
        shuffle      = True, 
        random_state = 0
       )

fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
for i in np.unique(y):
    ax.scatter(
        x = X[y == i, 0],
        y = X[y == i, 1], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        marker    = 'o',
        edgecolor = 'black', 
        label= f"Grupo {i}"
    )
ax.set_title('Datos simulados')
ax.legend();

# Simulación de datos

X, y = make_blobs(
        n_samples    = 200, 
        n_features   = 2, 
        centers      = 4, 
        cluster_std  = 0.60, 
        shuffle      = True, 
        random_state = 0
       )

fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
for i in np.unique(y):
    ax.scatter(
        x = X[y == i, 0],
        y = X[y == i, 1], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        marker    = 'o',
        edgecolor = 'black', 
        label= f"Grupo {i}"
    )
ax.set_title('Datos simulados')
ax.legend();

# Escalado de datos

X_scaled = scale(X)

# Modelos

modelo_hclust_complete = AgglomerativeClustering(
                            metric='euclidean', # Changed 'affinity' to 'metric'
                            linkage  = 'complete',
                            distance_threshold = 0,
                            n_clusters         = None
                        )
modelo_hclust_complete.fit(X=X_scaled)

modelo_hclust_average = AgglomerativeClustering(
                            metric='euclidean', # Changed 'affinity' to 'metric'
                            linkage  = 'average',
                            distance_threshold = 0,
                            n_clusters         = None
                        )
modelo_hclust_average.fit(X=X_scaled)

modelo_hclust_ward = AgglomerativeClustering(
                            metric='euclidean',  # Changed 'affinity' to 'metric'
                            linkage  = 'ward',
                            distance_threshold = 0,
                            n_clusters         = None
                     )
modelo_hclust_ward.fit(X=X_scaled)

# Dendrogramas

fig, axs = plt.subplots(3, 1, figsize=(8, 8))
plot_dendrogram(modelo_hclust_average, color_threshold=0, ax=axs[0])
axs[0].set_title("Distancia euclídea, Linkage average")
plot_dendrogram(modelo_hclust_complete, color_threshold=0, ax=axs[1])
axs[1].set_title("Distancia euclídea, Linkage complete")
plot_dendrogram(modelo_hclust_ward, color_threshold=0, ax=axs[2])
axs[2].set_title("Distancia euclídea, Linkage ward")
plt.tight_layout();

````


# K-means

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

x1 = np.random.standard_normal((100,2))*0.6+np.ones((100,2))
x2 = np.random.standard_normal((100,2))*0.5-np.ones((100,2))
x3 = np.random.standard_normal((100,2))*0.4-2*np.ones((100,2))+5
X = np.concatenate((x1,x2,x3),axis=0)

plt.plot(X[:,0],X[:,1],'k.')
plt.show()

from sklearn.cluster import KMeans

n = 3
k_means = KMeans(n_clusters=n)
k_means.fit(X)

centroides = k_means.cluster_centers_
etiquetas = k_means.labels_

plt.plot(X[etiquetas==0,0],X[etiquetas==0,1],'r.', label='cluster 1')
plt.plot(X[etiquetas==1,0],X[etiquetas==1,1],'b.', label='cluster 2')
plt.plot(X[etiquetas==2,0],X[etiquetas==2,1],'g.', label='cluster 3')

plt.plot(centroides[:,0],centroides[:,1],'mo',markersize=8, label='centroides')

plt.legend(loc='best')
plt.show()
------------------------------------------------------------------------------------------------------------

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