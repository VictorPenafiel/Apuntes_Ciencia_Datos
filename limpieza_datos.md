
# 💎 Guía Práctica: Limpieza de Datos Categóricos con RapidFuzz

En el mundo real, los datos rara vez son perfectos. Una de las tareas más comunes en ciencia de datos es la **limpieza de datos**, especialmente cuando trabajamos con columnas categóricas que provienen de entradas manuales.

**El Problema:**
Imagina que tienes una columna "Ciudad" con valores como: `["Santiago", "sntiago", "Stgo.", "Santyago"]`. Aunque para un humano es obvio que todos se refieren a la misma ciudad, para una máquina son textos completamente diferentes.

**La Solución:**
Utilizaremos **Fuzzy String Matching** (coincidencia de cadenas difusas), una técnica que nos permite encontrar cadenas que son "aproximadamente iguales". Para ello, usaremos **[RapidFuzz](https://github.com/maxbachmann/RapidFuzz)**, una librería de Python increíblemente rápida y eficiente para esta tarea.

Este notebook te guiará paso a paso para transformar datos categóricos "sucios" en una columna limpia y estandarizada.

### 1\. Instalación

Primero, asegurémonos de tener la librería instalada.

```python
!pip install rapidfuzz
```

### 2\. El Corazón de RapidFuzz: Calculando la Similitud

Antes de aplicarlo a un conjunto de datos, entendamos el concepto clave: la **puntuación de similitud**. RapidFuzz puede comparar dos cadenas y devolver una puntuación de 0 a 100, donde 100 es una coincidencia exacta.

La función más común es `fuzz.ratio`, que mide la similitud basándose en la [distancia de Levenshtein](https://es.wikipedia.org/wiki/Distancia_de_Levenshtein) (el número de ediciones necesarias para convertir una cadena en otra).

```python
from rapidfuzz import fuzz, process

# Ejemplo 1: Coincidencia perfecta
score_perfecto = fuzz.ratio("Santiago", "Santiago")
print(f"'Santiago' vs 'Santiago': {score_perfecto}% de similitud")

# Ejemplo 2: Cadenas muy similares
score_similar = fuzz.ratio("Santiago", "sntiago")
print(f"'Santiago' vs 'sntiago': {score_similar}% de similitud")

# Ejemplo 3: Cadenas diferentes
score_diferente = fuzz.ratio("Santiago", "Valparaiso")
print(f"'Santiago' vs 'Valparaiso': {score_diferente}% de similitud")
```

Como puedes ver, `fuzz.ratio` cuantifica de manera muy efectiva qué tan parecidas son dos cadenas.

### 3\. Escenario Práctico: Limpiando un DataFrame

Ahora, apliquemos este concepto a un problema real. Crearemos un DataFrame de `pandas` con una columna `Ciudad` que necesita limpieza.

```python
import pandas as pd

data = {
    "Ciudad": [
        "Santiago", "sntiago", "Stgo", "Santyago", "Valparaiso",
        "Valpo", "Concepcion", "Conce", "Concepción", "Concepn",
        "La Serena", "la serena", "Serena"
    ]
}
df = pd.DataFrame(data)
print("DataFrame Original (Datos Sucios):")
df
```

### 4\. La Estrategia de Normalización

Nuestro plan para limpiar la columna es el siguiente:

1.  **Definir una lista de categorías "maestras"** o correctas. Estos son los valores que queremos tener al final.
2.  **Crear una función** que tome un valor "sucio" de nuestra columna.
3.  Dentro de la función, usar RapidFuzz para **encontrar la mejor coincidencia** entre el valor sucio y nuestra lista de categorías maestras.
4.  **Establecer un umbral de confianza**. Si la mejor coincidencia tiene una puntuación de similitud por encima de este umbral (ej. 60%), reemplazamos el valor sucio por el valor maestro. Si no, lo dejamos como está para evitar cambios incorrectos.

### 5\. Manos a la Obra: Implementando la Limpieza

Primero, definimos nuestra lista de categorías correctas y la función de normalización.

```python
# Paso 1: Definir las categorías oficiales
categorias_oficiales = ["Santiago", "Valparaiso", "Concepcion", "La Serena"]

# Paso 2: Crear la función de normalización
def normalizar_texto(valor_sucio, categorias_maestras, umbral=60):
    """
    Compara un string con una lista de categorías y devuelve la mejor coincidencia
    si la similitud supera un umbral.
    """
    # process.extractOne encuentra la mejor coincidencia y su puntuación
    mejor_match = process.extractOne(valor_sucio, categorias_maestras, scorer=fuzz.ratio)

    # Paso 4: Aplicar el umbral
    if mejor_match and mejor_match[1] >= umbral:
        return mejor_match[0]  # Devolvemos el nombre de la categoría maestra
    return valor_sucio # Si no supera el umbral, devolvemos el original
```

Ahora, simplemente aplicamos esta función a nuestra columna `Ciudad` para crear una nueva columna normalizada.

```python
# Aplicamos la función a cada fila de la columna "Ciudad"
df['Ciudad_Normalizada'] = df['Ciudad'].apply(lambda x: normalizar_texto(x, categorias_oficiales))

print("DataFrame con la Columna Normalizada:")
df
```

