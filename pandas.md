import pandas as pd

# pd.Dataframe(data, index, columns, dtype)

data = {
    "Nombre": ["Ana", "Juan", "Luisa"],
    "Edad": [25, 30, 22],
    "Ciudad": ["Lima", None, "Bogotá"]  # Permite strings y nulos
}

df = pd.DataFrame(data)
print(df)


# DataFrame a Matriz (solo funciona si todas las columnas son numéricas)
matriz_from_df = df.values  # o df.to_numpy()

# Matriz a DataFrame
df_from_matriz = pd.DataFrame(matriz, columns=["A", "B", "C"])



https://pandas.pydata.org/docs/reference/api/pandas.Series.html
nums = [1, 2, 3, 4, 5]

s = pd.Series(nums, index = ['aa', 'bb', 'cc', 'dd', 'ee'])
print(s)


list(s.index)

values = list(s.values)
values


dct = {
    'name':'Asabeneh',
    'country':'Finland',
    'city':'Helsinki'
}
s = pd.Series(dct)
print(s)


https://interactivechaos.com/es/manual/tutorial-de-numpy/las-funciones-linspace-y-logspace

s = pd.Series(np.linspace(5, 20, 100)) # linspace(starting, end, items)
print(s)
----------------------------------------------------------------------------------------------------------------------

Leer archivo .csv

df =pd.read_csv("/content/surveys.csv", encoding="ISO-8859-1", delimiter=";")


----------------------------------------------------------------------------------------------------------------------
```pd explorando Datos.
  Serie o estructura unidimensional.
  Dataframes o estructura bidimensional
  Los métodos pandas.
  pd.Series.metodo 
  pd.DataFrame.metodo
    read 
        Leer archivos csv, excel, json, sql, HTML, etc.
    head 
        Leer los primeros 5 elementos de la estructura
    tail 
        Leer los ultimos 5 elementos de la estructura
    dtype
        Devuelve el tipo de datos de cada columna
    colums 
        Devuelve una lista con todos los nombres de las columnas de un DataFrame
    shape 
        Devuelve una tupla con el número de filas y columnas de un DataFrame
    unique
        pd.unique(df['Nombre_columna'])
        Devuelve una matriz (array) con los valores únicos de una serie o columna.
    nunique
        pd.nunique(df['Nombre_columna'])
        Recorre una serie o columna y cuenta cuántos valores diferentes hay
```


```pd calculando Datos.

    describe
   df['nombre_columna'].describe() 
        Devuelve estadísticas descriptivas incluyendo: media, meadiana, máx, mín, std y percentiles para una columna en particular de los datos.
        std(standard deviation). Es una medida de la dispersión de los datos respecto a la media, indicando qué tan alejados están los valores individuales de la media
    min 
        Leer los primeros 5 elementos de la estructura
    max 
        Leer los ultimos 5 elementos de la estructura
    mean
        Devuelve el tipo de datos de cada columna
    colums 
        Devuelve una lista con todos los nombres de las columnas de un DataFrame
    std 
        Devuelve una tupla con el número de filas y columnas de un DataFrame
    count
        pd.unique(xxx['Nombre_columna'])
        Devuelve una matriz (array) con los valores únicos de una serie o columna.
```
# Operaciones comunes con Pandas

## 1. Transformación de datos

- `pd.melt(df)`  
  Convierte columnas en filas, reuniendo la información.

- `pd.pivot(df)`  
  Convierte filas en columnas, "esparciendo" la información.

- `pd.concat([df1, df2])`  
  Apila dos datasets (juntando por filas) y los convierte en uno. Ambos datasets deben tener las mismas columnas.

- `pd.concat([df1, df2], axis=1)`  
  Junta dos datasets anexando columnas. Ambos datasets deben tener las mismas filas.

- `df.sort_values("variable")`  
  Ordena las filas del dataset en base a los valores de la "variable" (por defecto de menor a mayor).

- `df.rename(columns={"nombre_antiguo": "nombre_nuevo"})`  
  Renombra las columnas de un dataset.

- `df.sort_index()`  
  Ordena el índice de un DataFrame.

- `df.reset_index()`  
  Redefine el índice de un DataFrame asignando el número de fila correspondiente, convirtiendo el índice previo en una columna.

- `df.drop(columns=["var1", "var2"])`  
  Elimina del DataFrame la lista de columnas especificadas.

---

## 2. Subconjuntos de filas


- `df.info() `
    Imprime cantidad de no nulos de cada atributo y el tipo de dato que es

- `df[df.columna2 < 10]`  
  Extrae las filas que cumplen el criterio de la columna definida (en este caso, valor < 10 en columna2).

- `df.drop_duplicates()`  
  Remueve las filas duplicadas (solo considera columnas).

- `df.sample(frac=0.5)`  
  Selecciona una fracción de datos (en este caso 50%) de manera aleatoria.

- `df.sample(n=10)`  
  Selecciona `n` filas de manera aleatoria.

- `df.head(n)`  
  Muestra las primeras `n` filas del DataFrame.

-  `df.columns`
    Muestra nombre de las columnas

-   `df_nations["nombre_columna"].head(198)`
    Para mostrar una columna, indicando el numero de datos

- `df.tail(n)`  
  Muestra las últimas `n` filas del DataFrame.

---

## 3. Subconjuntos de columnas

- `df[["columna1", "columna2", "columna4"]]`  
  Selecciona múltiples columnas por nombre.

- `df["columna2"]`  
  Selecciona una columna específica.

- `df.filter(regex="texto")`  
  Selecciona columnas que contengan el "texto" definido.

---

## 4. Tablas de resumen (Summarize Data)

- `df.value_counts()`  
  Cuenta el número de filas con valores únicos en el DataFrame.

- `df["columna3"].value_counts()`  
  Cuenta valores únicos para una columna específica.

- `len(df)`  
  Devuelve el número de filas en el DataFrame.

- `df.shape`  
  Devuelve una tupla con (número de filas, número de columnas).

- `df["columna1"].nunique()`  
  Cuenta valores únicos en una columna específica.

- `df.describe()`  
  Devuelve estadísticos descriptivos básicos (media, percentiles, etc.) para cada columna numérica.


- `df.["nombre_variable"].mean()`
    Promedio  de la variable

- `df.["nombre_variable"].count()`
    Cuenta el numero de datos existentes

- `df.["nombre_variable"].min()`
    Dato mas bajo de una variable

- `df.["nombre_variable"].max()`
    Dato mas alto de una variable

---


## 5. Agrupación

- `df.groupby(by="col")`  
  Agrupa datos según los valores de la columna "col".

- `df.groupby(level="index")`  
  Agrupa datos por los valores del índice del DataFrame.

# Estadísticas para todas las columnas numéricas por sexo
grouped_data.describe()
# Regresa la media de cada columna numérica por sexo
grouped_data.mean()
---

# Para observar un atributo categórico usaremos una tabla de frecuencias
df_nations["region"].value_counts()

# Otra forma de hacer este conteo, puede ser a través de una agrupación
df.groupby(["region"])[["country"]].count()


# Usando where(), crearemos una variable nueva de co2, en dónde identifiquemos qué países están por debajo de la media, y qué países por sobre la media.
df_nations["co2_recodificada"] = np.where(df_nations["co2"]> df_nations["co2"].mean(), 1, 0)