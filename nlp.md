[Natural Language Processing (NLP) #1](https://www.youtube.com/watch?v=Tg1MjMIVArc&t=97s)
[Natural Language Processing (NLP) #2](https://www.youtube.com/watch?v=RkYuH_K7Fx4)


## 1\. Preparación de Datos

El primer paso consiste en crear un conjunto de datos de ejemplo utilizando la librería `pandas`. El `DataFrame` contiene tres columnas: `id`, `frase` (el texto a clasificar) y `categoria` (la etiqueta, "bueno" o "malo").

```python
import pandas as pd

# Dataset de prueba
data = {
    'id': list(range(1, 31)),
    'frase': [
        # Frases buenas
        "Hola, ¿cómo estás?",
        "Buenos días, espero que tengas un gran día",
        # ... (más frases) ...
        # Frases malas
        "Hace mucho calor hoy",
        "Va a llover en la tarde",
        # ... (más frases) ...
    ],
    'categoria': [
        "bueno", "bueno", ..., "malo", "malo", ...
    ]
}

df = pd.DataFrame(data)
```

## 2\. Método 1: Vectorización con `CountVectorizer`

Esta sección introduce el método de **Bolsa de Palabras (Bag of Words)**, que representa el texto contando la frecuencia de cada palabra.

### ¿Qué es `CountVectorizer`?

Es una herramienta de `Scikit-learn` que convierte una colección de textos en una matriz numérica. Cada fila de la matriz representa un documento (una frase) y cada columna representa una palabra única del vocabulario. El valor de cada celda es el número de veces que la palabra aparece en la frase.

### Implementación

1.  **Importar y aplicar `CountVectorizer`**: Se instancia el vectorizador y se ajusta al corpus de frases para construir el vocabulario.
2.  **Transformar el texto**: Se convierten las frases en una matriz de conteo de tokens.

<!-- end list -->

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

# Ajustar y transformar en un solo paso
X = vectorizer.fit_transform(df['frase'])

# Convertir a un DataFrame para visualización (opcional)
df_vectorizado = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
```

### Entrenamiento del Modelo

Con los datos ya vectorizados, se entrena un clasificador de **Máquinas de Vectores de Soporte (SVC)**.

1.  **División de datos**: Se separan los datos en conjuntos de entrenamiento y prueba.
2.  **Entrenamiento del modelo**: Se ajusta un `SVC` con un kernel lineal a los datos de entrenamiento.
3.  **Evaluación**: Se mide la precisión (`accuracy`) del modelo en el conjunto de prueba, obteniendo un **62.5%**.

<!-- end list -->

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Variables: X = df_vectorizado, y = df['categoria']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, df['categoria'])

model = SVC(kernel='linear')
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print(accuracy_score(ytest, ypred)) # Output: 0.625
```

### Predicción con Nuevas Frases

Para predecir la categoría de una nueva frase, esta debe ser transformada con el **mismo vectorizador** que se usó para el entrenamiento.

```python
frase = ['no para de llover']
vec_frase = vectorizer.transform(frase)
model.predict(vec_frase) # Output: array(['malo'], dtype=object)
```

## 3\. Introducción a Word Vectors (Embeddings)

El notebook explica por qué `CountVectorizer` es limitado: no captura el **significado semántico** de las palabras. Para resolver esto, se introducen los **Word Vectors** o *Embeddings*.

  - **Concepto**: Representaciones numéricas de palabras en un espacio multidimensional donde palabras con significados similares tienen vectores cercanos.
  - **Limitaciones de One-Hot Encoding**: Se menciona que este método simple sufre de alta dimensionalidad y no captura relaciones semánticas.
  - **Vectores Densos**: A diferencia de los métodos de conteo, los vectores densos (generados por técnicas como Word2Vec, GloVe o FastText) capturan relaciones complejas. Permiten operaciones aritméticas como:
    `"rey" - "hombre" + "mujer" ≈ "reina"`

## 4\. Método 2: Word Vectors con `spaCy`

Esta sección implementa un enfoque más avanzado utilizando los **vectores de palabras pre-entrenados** de la librería `spaCy`.

### Implementación

1.  **Instalación y carga del modelo**: Se descarga y carga el modelo `en_core_web_md`, que incluye vectores de 300 dimensiones para un amplio vocabulario.

<!-- end list -->

```python
!python -m spacy download en_core_web_md
import spacy
nlp = spacy.load('en_core_web_md')
```

2.  **Vectorización de frases**: `spaCy` procesa cada frase y calcula un vector promedio para todo el documento (frase). Este vector único representa el significado semántico de la frase completa.

<!-- end list -->

```python
# spaCy procesa cada texto y genera un objeto Doc
docs = [nlp(text) for text in df['frase']]

# Se extrae el vector de cada Doc
train_word_vector = [x.vector for x in docs]
```

### Entrenamiento del Modelo con Embeddings

Se entrena un nuevo clasificador `SVC` usando los vectores de `spaCy`.

1.  **Entrenamiento**: Se ajusta el modelo directamente con los vectores de `spaCy` y las etiquetas originales.
2.  **Predicción**: Para una nueva frase, primero se obtiene su vector con `spaCy` y luego se pasa al modelo para la predicción.

<!-- end list -->

```python
from sklearn.svm import SVC

# No se necesita train_test_split en este ejemplo, se entrena con todo el dataset
model_spacy = SVC()
model_spacy.fit(train_word_vector, df['categoria'])

# Predecir una nueva frase
frase_nueva = nlp('estoy viendo television').vector
model_spacy.predict([frase_nueva]) # Output: array(['bueno'], dtype=object)
```