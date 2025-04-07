import pandas as pd

# pd.Dataframe(data, index, columns, dtype)

diccionario = 

Leer archivo .csv

df =pd.read_csv(r'C:\Users\victor\Desktop\Data Science con Python - Numpy _ Pandas [2023]\1.3\Info_pais.csv', encoding="ISO-8859-1", delimiter=";")

df_nations = pd.read_csv("https://raw.githubusercontent.com/DireccionAcademicaADL/Nations-DB/main/nations.csv", encoding="ISO-8859-1")

----------------------------------------------------------------------------------------------------------------------


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

---

## 5. Agrupación

- `df.groupby(by="col")`  
  Agrupa datos según los valores de la columna "col".

- `df.groupby(level="index")`  
  Agrupa datos por los valores del índice del DataFrame.

---

Fuente: [www.desafiolatam.com](https://www.desafiolatam.com)