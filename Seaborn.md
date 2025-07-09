[seaborn](https://seaborn.pydata.org/)

## Importar Seaborn
    import seaborn as sns

## Configurar el tema
    sns.set_theme()

## Cargar datos de ejemplo
    tips = sns.load_dataset("tips")

## Crear visualización
    sns.relplot(
        data=tips,
        x="total_bill", y="tip", col="time",
        hue="smoker", style="smoker", size="size",
    )

```
import seaborn as sns

# Configuración
sns.set_theme(style="whitegrid")

# Datos
tips = sns.load_dataset("tips")

# Gráfico combinado
g = sns.FacetGrid(tips, col="time", hue="smoker")
g.map(sns.scatterplot, "total_bill", "tip").add_legend()
```

# Principales Métodos de Seaborn

## Gráficos Relacionales
Muestran relaciones entre variables numéricas.

* **scatterplot**

    Gráfico de dispersión.
    * ```python
        sns.scatterplot(x="col1", y="col2", data=df)
        ```

* **lineplot**

    Gráfico de líneas (series temporales).
    * ```python
        sns.lineplot(x="fecha", y="valor", data=df)
        ```

* **relplot**

    Versión flexible (combina scatter/line).
    * ```python
        sns.relplot(x="col1", y="col2", hue="categoría", data=df)
        ```

##  Gráficos de Distribución
Visualizan distribuciones de datos.

  * **histplot**

    Histograma.
      * ```python
          sns.histplot(x="edad", data=df, bins=20)
        ```
  * **kdeplot**

    Densidad de kernel (suavizado).
      * ```python
          sns.kdeplot(x="edad", data=df, hue="sexo")
        ```
  * **displot**

    Versión flexible (histograma/KDE).
      * ```python
          sns.displot(x="edad", kind="kde", data=df)
        ```
  * **ecdfplot**

    Función de distribución acumulada.
      * ```python
          sns.ecdfplot(x="edad", data=df)
        ```

## Gráficos Categóricos
Comparan variables categóricas vs. numéricas.

  * **barplot**

    Barras con intervalos de confianza.
      * ```python
          sns.barplot(x="categoría", y="valor", data=df)
        ```
  * **boxplot**

    Diagrama de cajas (percentiles).
      * ```python
          sns.boxplot(x="grupo", y="valor", data=df)
        ```
  * **violinplot**

    Mezcla de boxplot + KDE.
      * ```python
          sns.violinplot(x="grupo", y="valor", data=df)
        ```
  * **catplot**

    Versión flexible (combina métodos categóricos).
      * ```python
          sns.catplot(x="grupo", y="valor", kind="box", data=df)
        ```

## Gráficos Matriciales
Útiles para datos estructurados (ej: matrices de correlación).

  * **heatmap**
    Mapa de calor (ej: correlaciones).
      * ```python
          sns.heatmap(corr_matrix, annot=True)
        ```
  * **clustermap**
    Heatmap con clustering jerárquico.
      * ```python
          sns.clustermap(corr_matrix)
        ```

## Gráficos de Regresión
Muestran relaciones estadísticas con ajustes.

  * **regplot**

    Gráfico de regresión lineal.
      * ```python
          sns.regplot(x="col1", y="col2", data=df)
        ```
  * **lmplot**

    Versión flexible (hue/col/row).
      * ```python
          sns.lmplot(x="col1", y="col2", hue="grupo", data=df)
        ```

## Personalización y Estilo

  * **set\_theme**

    Configura el estilo global.
      * ```python
          sns.set_theme(style="darkgrid")
        ```
  * **set\_palette()**

    Cambia la paleta de colores.
      * ```python
          sns.set_palette("husl")
        ```
  * **despine()**

    Elimina los spines "líneas de los ejes" no deseadas de los gráficos.

  * **FacetGrid**

    Para crear grillas de gráficos.
      * ```python
          g = sns.FacetGrid(df, col="categoría")
        ```

Recuerda que Seaborn está construido sobre Matplotlib. Esto significa que puedes usar las funciones de Matplotlib (como plt.title(), plt.xlabel(), plt.ylabel(), plt.xlim(), plt.ylim(), plt.figure(figsize=(x,y)) etc.) para un control aún mayor sobre tus gráficos.