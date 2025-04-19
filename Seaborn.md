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


# Principales Métodos de Seaborn

## Gráficos Relacionales
Muestran relaciones entre variables numéricas.
```
    scatterplot
        Gráfico de dispersión.
            sns.scatterplot(x="col1", y="col2", data=df)
    lineplot
        Gráfico de líneas (series temporales).
            sns.lineplot(x="fecha", y="valor", data=df)
    relplot
        Versión flexible (combina scatter/line).
            sns.relplot(x="col1", y="col2", hue="categoría", data=df)
```

# Principales Métodos de Seaborn

Seaborn es una biblioteca de visualización de datos estadísticos en Python basada en Matplotlib. Ofrece una interfaz de alto nivel para crear gráficos atractivos e informativos.

## 📊 Gráficos Relacionales
Muestran relaciones entre variables numéricas.

### `scatterplot()`
- **Descripción**: Gráfico de dispersión
- **Uso**: 
  ```python
  sns.scatterplot(x="col1", y="col2", data=df)

lineplot()

    Descripción: Gráfico de líneas para series temporales

    Uso:
    python

    sns.lineplot(x="fecha", y="valor", data=df)

relplot()

    Descripción: Versión flexible que combina scatter y line plots

    Uso:
    python

    sns.relplot(x="col1", y="col2", hue="categoría", data=df)

📈 Gráficos de Distribución

Visualizan distribuciones de datos.
histplot()

    Descripción: Histograma

    Uso:
    python

    sns.histplot(x="edad", data=df, bins=20)

kdeplot()

    Descripción: Estimación de densidad kernel

    Uso:
    python

    sns.kdeplot(x="edad", data=df, hue="sexo")

📉 Gráficos Categóricos

Comparan variables categóricas vs numéricas.
barplot()

    Descripción: Diagrama de barras

    Uso:
    python

    sns.barplot(x="categoría", y="valor", data=df)

boxplot()

    Descripción: Diagrama de cajas

    Uso:
    python

    sns.boxplot(x="grupo", y="valor", data=df)

🔥 Gráficos Matriciales

Para datos estructurados.
heatmap()

    Descripción: Mapa de calor

    Uso:
    python

    sns.heatmap(corr_matrix, annot=True)

🔄 Gráficos de Regresión

Muestran relaciones estadísticas.
regplot()

    Descripción: Gráfico de regresión lineal

    Uso:
    python

    sns.regplot(x="col1", y="col2", data=df)

🎨 Personalización
set_theme()

    Descripción: Configura el estilo global

    Uso:
    python

    sns.set_theme(style="darkgrid")

Ejemplo Integrado
python

import seaborn as sns

# Configuración
sns.set_theme(style="whitegrid")

# Datos
tips = sns.load_dataset("tips")

# Gráfico combinado
g = sns.FacetGrid(tips, col="time", hue="smoker")
g.map(sns.scatterplot, "total_bill", "tip").add_legend()

Nota: Seaborn es ideal para análisis exploratorios y visualizaciones estadísticas, mientras que Matplotlib ofrece más control para personalizaciones avanzadas.


Este resumen en Markdown:
1. Organiza los métodos por categorías
2. Incluye descripciones breves
3. Muestra ejemplos de código formateados
4. Usa emojis para mejorar la legibilidad
5. Destaca las secciones principales con encabezados
6. Incluye un ejemplo integrado al final
7. Menciona la relación con Matplotlib