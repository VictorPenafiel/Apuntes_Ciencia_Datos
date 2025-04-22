````
Serie o estructura unidimensional.
Dataframes o estructura bidimensional
Los métodos pandas.
  pd.Series.metodo 
  pd.DataFrame.metodo

import pandas as pd
````

### pd.Dataframe(data, index, columns, dtype)

````
data = {
    "Nombre": ["Ana", "Juan", "Luisa"],
    "Edad": [25, 30, 22],
    "Ciudad": ["Lima", None, "Bogotá"]  # Permite strings y nulos
}

df = pd.DataFrame(data)
print(df)
````

### DataFrame a Matriz (solo funciona si todas las columnas son numéricas)
    matriz_from_df = df.values  # o df.to_numpy()

### Matriz a DataFrame
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

------------------------------------------------------------------------------------------------------------

### Leer archivo .csv

    df =pd.read_csv("/content/surveys.csv", encoding="ISO-8859-1", delimiter=";")

### Convertir archivo CSV a XLSX

    pd.read_csv('data.csv').to_excel('data.xlsx', index=False)

### Convertir archivo XLSX a CSV
    pd.read_excel('data.xlsx').to_csv('data.csv', index=False)

------------------------------------------------------------------------------------------------------------



## Explorando Datos.
df.metodo() 

    read 
        Leer archivos csv, excel, json, sql, HTML, etc.
    head 
        Leer los primeros 5 elementos de la estructura
    tail 
        Leer los ultimos 5 elementos de la estructura
    info
        Muestra el tipo de datos de las columnas, el número de valores no nulos (non-null) y la memoria utilizada por la estructura de datos.
    describe
        Devuelve estadísticas descriptivas incluyendo: media, meadiana, máx, mín, std y percentiles para una columna en particular de los datos.

df.metodo

    colums 
        Devuelve una lista con todos los nombres de las columnas de un DataFrame
    shape 
        Devuelve una tupla con el número de filas y columnas de un DataFrame


## calculando Datos.

## 

df['nombre_columna'].metodo() 

    min 
        Leer los primeros 5 elementos de la estructura
    max 
        Leer los ultimos 5 elementos de la estructura
    unique
        pd.unique(df['Nombre_columna'])
        Devuelve una matriz (array) con los valores únicos de una serie o columna.
    nunique
        pd.nunique(df['Nombre_columna'])
        Recorre una serie o columna y cuenta cuántos valores diferentes hay
    isnull
        Identificar valores nulos (faltantes) en un DataFrame o Series
    dropna()
        Elimina las filas que contienen datos nulos,
    mean (Media)
        Calcular los valores promedio  en un DataFrame o Serie.
    var (Varianza)
        Calcula la varianza de los datos en un DataFrame o Serie.
    cov 
        Calcula la matriz de covarianza de las columnas en el DataFrame.
    std (Desviación estándar)
        Calcular la desviación estándar de los valores en un DataFrame o Series.  
    corr
        Calcula la correlación entre las columnas de un DataFrame.
    count
        Devuelve el número de valores no nulos (no vacíos) en un DataFrame o una Serie.
    dtype
        Devuelve el tipo de datos de cada columna


##  Transformación de datos
df['nombre_columna'].metodo() 

    melt 
        Convierte columnas en filas, reuniendo la información.
    pivot 
        
        pivot_df = reservas.pivot_table( columns='Noches', 
                                 values='Total Ganancias', 
                                 aggfunc='sum')  # or other appropriate aggregation function like 'mean', 'first' etc.
    Nos permite reorganizar y transformar los datos de un DataFrame creando una nueva tabla con un formato diferente.
        
    concat([df1, df2])
        Apila dos datasets (juntando por filas) y los convierte en uno. Ambos datasets deben tener las mismas columnas.

        pd.concat([df1, df2], axis=1)
        Junta dos datasets anexando columnas. Ambos datasets deben tener las mismas filas.

    sort_values
      Ordena las filas del dataset en base a los valores de la "variable" (por defecto de menor a mayor).

        df.sort_values("variable", ascending=False)
        Ordena las filas del dataset en base a los valores de la "variable" (por defecto de mayor a menor).

        df.sort_index()
        Ordena el índice de un DataFrame.

    rename(columns={"nombre_antiguo": "nombre_nuevo"})
        Renombra las columnas de un dataset.
    drop
        Elimina del DataFrame la lista de columnas especificadas.
    reset_index
        Redefine el índice de un DataFrame asignando el número de fila correspondiente, convirtiendo el índice previo en una columna.



## Subconjuntos de filas


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