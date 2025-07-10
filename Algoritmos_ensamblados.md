
## BaggingRegressor

Un regresor Bagging es un metaestimador de ensemble que ajusta regresores base cada uno en subconjuntos aleatorios del conjunto de datos original y luego agrega sus predicciones individuales (ya sea por votación o por promedio) para formar una predicción final.

[BaggingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)

````

from sklearn.svm import SVR

from sklearn.ensemble import BaggingRegressor

from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=4,

                       n_informative=2, n_targets=1,

                       random_state=0, shuffle=False)

regr = BaggingRegressor(estimator=SVR(),

                        n_estimators=10, random_state=0).fit(X, y)

regr.predict([[0, 0, 0, 0]])
array([-2.8720...])

````

## BaggingClassifier
Un clasificador Bagging es un metaestimador de ensemble que ajusta a los clasificadores base cada uno en subconjuntos aleatorios del conjunto de datos original y luego agrega sus predicciones individuales (ya sea por votación o por promedio) para formar una predicción final.

````
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = BaggingClassifier(estimator=SVC(),
                        n_estimators=10, random_state=0).fit(X, y)
clf.predict([[0, 0, 0, 0]])
array([1])
````

### Gradient Boosting 
## GradientBoostingClassifie
Algoritmo de machine learning supervisado para problemas de clasificación (predecir categorías/binarias). Es parte de la familia de métodos de ensamble por boosting, donde modelos débiles (generalmente árboles de decisión simples) se combinan secuencialmente para crear un predictor fuerte.
Es un algoritmo de ensamble que construye múltiples árboles de decisión secuencialmente, donde cada árbol corrige los errores del anterior.

[Gradient Boosting ](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier)

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

[GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

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
