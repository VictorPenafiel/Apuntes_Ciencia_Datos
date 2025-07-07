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

[Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html)

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




