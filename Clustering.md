DBSCAN vs. K-Means: Se destaca que, a diferencia de K-Means, DBSCAN no requiere que se especifique el número de clusters y es capaz de manejar clusters de formas no esféricas y con ruido.

# K-means
La agrupación (clustering) de medias K es un algoritmo de aprendizaje no supervisado utilizado para la agrupación en clústeres de datos, que agrupa puntos de datos no etiquetados en grupos o clústeres , donde cada punto pertenece al cluster con el centroide (media) más cercano.

Evaluación de los Clusters: Se aplican métricas para evaluar la calidad de los clusters, como el coeficiente de silueta, la homogeneidad y la completitud.

## <font color='blue'>**Algoritmo**</font>

1. Seleccionar de forma aleatoria $k$ centroides $C = \{c_1, c_2, \dots, c_k\}$ de los puntos de datos $X = \{x_1, x_2, \dots, x_n\} \in \mathbb{R}^D $.
2. Para cada observación $x_i$, se calcula la suma de errores al cuadrado de esa observación respecto a cada uno de los $k$ centroides, $ D(x_i, c_j) = \displaystyle\sum_{i=1}^{n}{\| x_i - c_j \|^2}$.
3. A cada observación se le asigna el centroide que menos error tenga.
4. Calcular la diferencia entre el antiguo y el nuevo centroide y repetir los pasos 2 y 3 si la diferencia es menor que un umbral (convergente).

[K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

[K-means_traduccion](https://qu4nt.github.io/sklearn-doc-es/modules/clustering.html#k-means)

[k-means-clustering](https://www.ibm.com/mx-es/think/topics/k-means-clustering)

--------------------------------------------------------------------------------------
````
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
kmeans.labels_
kmeans.predict([[0, 0], [12, 3]])
kmeans.cluster_centers_
````

------------------------------------------------------------------------------------
````
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
````
----------------------------------------------------------------------------------------------


````
import numpy as np

from sklearn.datasets import load_digits

data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")


from time import time

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

pca = PCA(n_components=n_digits).fit(data)
kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

print(82 * "_")

import matplotlib.pyplot as plt

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on the digits dataset (PCA-reduced data)\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

````
--------------------------------------------------------------------------------------------

# DBSCAN
El algoritmo de clustering DBSCAN (Density-Based Spatial Clustering of Applications with Noise), Utiliza un enfoque de clustering espacial basado en la densidad para crear clústeres con una densidad pasada por el usuario que se centra en un centroide espacial. El área inmediatamente alrededor del centroide se denomina vecindad y DBSCAN intenta definir las vecindades de los clústeres que tienen la densidad especificada.

El concepto en el que se basa DBSCAN es el de **core points**, o puntos base, que son muestras situadas en áreas de alta densidad. Junto a éstas encontramos también las **border points**, o puntos de borde, que se encuentran próximas a un core point (sin ser una de ellas). Por este motivo, un cluster va a ser una agrupación de core points y de border points situadas a una distancia máxima de alguna core point (distancia medida según algún criterio).

### Parámetros

* **Eps**: máxima distancia entre dos muestras para poder ser consideradas pertenecientes al mismo "vecindario", y
* **MinPts**, número de muestras en un vecindario para que un punto pueda ser considerado core point.

Elección de eps: Se puede utilizar para encontrar un valor óptimo para el hiperparámetro eps un gráfico de k-distance.

## <font color='blue'>**Algoritmo**</font>

1. Elegir aleatoriamente un punto $p$.
2. Identificar todos los puntos alcanzables por densidad desde $p$ con respecto a $Eps$ y $MinPts$.
3. Si $p$ es un core point, un cluster es formado.
4. Si $p$ es un border point, ningún punto es de densidad alcanzable desde $p$, luego continuar con un siguiente punto.
5. Repetir el proceso hasta que se hayan procesado todos los puntos.

Una buena forma de entender el algoritmo es visualizar el proceso. En el siguiente link podrán ver una visualización del funcionamiento de DBSCAN para distintos conjuntos de datos:

[visualización](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

[DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
[DBSCAN_traduccion](https://qu4nt.github.io/sklearn-doc-es/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN)

````
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

````

---


## <font color='blue'>**Clustering Jerárquico (Hierarchical clustering)**</font>🌳

El clustering jerárquico es una alternativa a los métodos de partitioning clustering que no requiere que se pre-especifique el número de clusters. 

* Aglomerativo (de abajo hacia arriba)

* Divisivo ((de arriba hacia abajo))

En ambos casos, los resultados pueden representarse de forma muy intuitiva en una estructura de árbol llamada dendrograma.



 

    Métodos de Enlace (Linkage): Para medir la distancia entre clusters, como "complete", "average" y "ward".

* **Complete or Maximum**: se calcula la distancia entre todos los posibles pares formados por una observación del cluster A y una del cluster B. La mayor de todas ellas se selecciona como la distancia entre los dos clusters. Se trata de la medida más conservadora (*maximal intercluster dissimilarity*).

* **Single or Minimum**: se calcula la distancia entre todos los posibles pares formados por una observación del cluster A y una del cluster B. La menor de todas ellas se selecciona como la distancia entre los dos clusters. Se trata de la medida menos conservadora (*minimal intercluster dissimilarity*).

* **Average**: Se calcula la distancia entre todos los posibles pares formados por una observación del cluster A y una del cluster B. El valor promedio de todas ellas se selecciona como la distancia entre los dos clusters (*mean intercluster dissimilarity*).

* **Centroid**: Se calcula el centroide de cada uno de los clusters y se selecciona la distancia entre ellos como la distancia entre los dos clusters.

* **Ward**: Se trata de un método general. La selección del par de clusters que se combinan en cada paso del agglomerative hierarchical clustering se basa en el valor óptimo de una función objetivo, pudiendo ser esta última cualquier función definida por el analista. El método Ward's minimum variance es un caso particular en el que el objetivo es minimizar la suma total de varianza intra-cluster. En cada paso, se identifican aquellos 2 clusters cuya fusión conlleva menor incremento de la varianza total intra-cluster. Esta es la misma métrica que se minimiza en K-means.

Dendrogramas: Se puede visualizar los resultados del clustering jerárquico a través de dendrogramas y se pueden interpretar para decidir el número de clusters.
