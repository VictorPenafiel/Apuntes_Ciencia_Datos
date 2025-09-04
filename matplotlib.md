Utilizando el gestor de paquetes de Python (pip)
```bash
pip install matplotlib
```

[matplotlib](https://matplotlib.org/)

# Dendrograma (Agrupamiento Jerárquico)

Es un método de Machine Learning generalmente empleado para la organización y clasificación de datos, con el fin de detectar patrones y agrupar elementos, permitiendo así diferenciar unos de otros.

````
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/content/Mall_Customers.csv')

datapoints = dataset.iloc[:, [3, 4]].values

import scipy.cluster.hierarchy as hc
from sklearn.cluster import AgglomerativeClustering

dend= hc.dendrogram(hc.linkage(datapoints, method = 'ward'))

plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distancia Euclidiana')
plt.show()

````

# Creación de gráficos

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(x = [7, 2, 5], y = [6, 1, 6])
    plt.savefig('diagrama.png')
    plt.show()

----------------------------------------------------------------------------------------------------------------------

## Histogramas
Usalo para analizar la distribucion de una variable continua.

hist(x, bins)

[histograma](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html#matplotlib.pyplot.hist)

````
N_points = 100000
n_bins = 20

# Generate two normal distributions
dist1 = rng.standard_normal(N_points)
dist2 = 0.4 * rng.standard_normal(N_points) + 5

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(dist1, bins=n_bins)
axs[1].hist(dist2, bins=n_bins)

plt.show()

````

## Diagramas de caja y bigotes (boxplot)
Perfecto para detectar outliers y analizar la dispersión

boxplot(x)

[caja](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html#matplotlib.pyplot.boxplot)

````
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
fruit_weights = [
    np.random.normal(130, 10, size=100),
    np.random.normal(125, 20, size=100),
    np.random.normal(120, 30, size=100),
]
labels = ['peaches', 'oranges', 'tomatoes']
colors = ['peachpuff', 'orange', 'tomato']

fig, ax = plt.subplots()
ax.set_ylabel('fruit weight (g)')

bplot = ax.boxplot(fruit_weights,
                   patch_artist=True,  # fill with color
                   tick_labels=labels)  # will be used to label x-ticks

# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.show()
````

## Diagramas de dispersión o puntos
Ideal para ver las relaciones entre dos variables numéricas.

scatter(x, y)

[puntos](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter)

````
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis((0, 6, 0, 20))
plt.show()
````

## Diagramas de líneas
Usalo para tendencias en el tiempo.

plot(x, y)

[líneas](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot)

````
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2 * np.pi * t)

upper = 0.77
lower = -0.77

supper = np.ma.masked_where(s < upper, s)
slower = np.ma.masked_where(s > lower, s)
smiddle = np.ma.masked_where((s < lower) | (s > upper), s)

fig, ax = plt.subplots()
ax.plot(t, smiddle, t, slower, t, supper)
plt.show()

````

## Diagramas de areas
Utilizada para mostrar la evolución de una o más series de datos a lo largo de un eje continuo, generalmente el tiempo. 

fill_between(x, y)

[area](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html#matplotlib.pyplot.fill_between)

````
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.fill_between([1, 2, 3, 4], [1, 2, 0, 0.5])
plt.show()
````

## Diagramas de barras verticales
Mejor para comparar categorías

bar(x, y)

[barras_verticales](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html#matplotlib.pyplot.bar)

````
fruit_names = ['Coffee', 'Salted Caramel', 'Pistachio']
fruit_counts = [4000, 2000, 7000]

fig, ax = plt.subplots()
bar_container = ax.bar(fruit_names, fruit_counts)
ax.set(ylabel='pints sold', title='Gelato sales by flavor', ylim=(0, 8000))
ax.bar_label(bar_container, fmt='{:,.0f}')
````

## Diagramas de barras horizontales
Se utiliza para comparar magnitudes entre diferentes categorías.

barh(x, y)

[barras_horizontales](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html#matplotlib.pyplot.barh)

````
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.barh([1, 2, 3], [3, 2, 1])
plt.show()
````

## Diagramas de sectores
Se usa para representar la proporción de cada categoría en relación con un todo.

pie(x)

[sectores](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html#matplotlib.pyplot.pie)

````
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.pie([5, 4, 3, 2, 1])
plt.show()
````

## Diagramas de violín
Se utiliza para visualizar la distribución de datos numéricos y comparar esas distribuciones entre diferentes categorías. Es una combinación de un diagrama de caja y un gráfico de densidad de kernel

violinplot(x)

[violín](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.violinplot.html#matplotlib.pyplot.violinplot)

````
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.violinplot([1, 2, 1, 2, 3, 4, 3, 3, 5, 7])
plt.show()
````

## Diagramas de contorno
Se utiliza para representar tres variables en un espacio bidimensional.

contourf([X, Y,] Z, /, [levels], **kwargs)

[contorno](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html#matplotlib.pyplot.contourf)

````
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-3.0, 3.0, 100)
x, y = np.meshgrid(x, y)
z = np.sqrt(x**2 + 2*y**2)
ax.contourf(x, y, z)
plt.show()
````

## Mapas de color
Se utiliza para representar la magnitud de los valores en una matriz bidimensional mediante un degradado de colores.

imshow(x)

[mapas_color](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html#matplotlib.pyplot.imshow)

````
import matplotlib.pyplot as plt
import numpy as np

def func3(x, y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2 + y**2))
dx, dy = 0.05, 0.05
x = np.arange(-3.0, 3.0, dx)
y = np.arange(-3.0, 3.0, dy)
X, Y = np.meshgrid(x, y)
extent = np.min(x), np.max(x), np.min(y), np.max(y)
fig = plt.figure(frameon=False)
Z1 = np.add.outer(range(8), range(8)) % 2  # chessboard
im1 = plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest',
                 extent=extent)
Z2 = func3(X, Y)
im2 = plt.imshow(Z2, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear',
                 extent=extent)

plt.show()
````
### mapas de calor (heatmap)

z = contingency.values # Obtenemos una representación matricial de la matriz de contingencia
heatmap = plt.pcolormesh(z) # Generamos el mapa de calor y guardamos el artista en la variable heatmap

#### añadir una barra de color

heatmap = plt.pcolormesh(z) #Generamos el heatmap
cbar = plt.colorbar(heatmap) # Añadimos la barra de color
----------------------------------------------------------------------------------------------------------------------

# Cambiar el aspecto de los gráficos

## Títulos
Para añadir un título principal al gráfico se utiliza el siguiente método:

ax.set_title

````
import matplotlib.pyplot as plt

dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

temperaturas_chile = {
    'Viña del Mar': [15.0, 17.5, 18.0, 17.2, 19.0, 16.5, 15.8],  # Temperaturas para Viña del Mar
    'La Serena': [18.0, 20.0, 21.5, 20.8, 22.0, 19.5, 18.2]    # Temperaturas para La Serena
}
fig, ax = plt.subplots(figsize=(10, 6)) # Ajustamos el tamaño para mejor legibilidad
ax.plot(dias, temperaturas_chile['Viña del Mar'],
        label='Viña del Mar',     # Etiqueta para la leyenda
        color='tab:purple',       # Color de línea personalizado
        marker='o',               # Marcador circular en cada punto
        linewidth=2)              # Grosor de la línea
ax.plot(dias, temperaturas_chile['La Serena'],
        label='La Serena',        # Etiqueta para la leyenda
        color='tab:green',        # Color de línea personalizado
        marker='s',               # Marcador cuadrado en cada punto
        linestyle='--',           # Línea discontinua para diferenciar
        linewidth=2)              # Grosor de la línea
ax.set_title('Evolución de la Temperatura Diaria',
             loc='left', # Ubicación del título a la izquierda
             fontdict={'fontsize': 16, 'fontweight': 'bold', 'color': 'darkblue'}) # Ajustado el fontsize y color para visibilidad
ax.set_xlabel('Día de la Semana', fontsize=12)
ax.set_ylabel('Temperatura (°C)', fontsize=12)
ax.set_ylim([14, 23])
ax.set_yticks(range(14, 24, 1)) # Ticks cada 1 grado para mayor detalle
ax.legend(loc='upper left', title='Ciudad', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
````

## Rejilla

ax.grid

````
import matplotlib.pyplot as plt

dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

temperaturas_chile = {
    'Puerto Montt': [9.0, 11.5, 12.0, 10.8, 13.0, 10.0, 8.5], # Temperaturas para Puerto Montt
    'Castro': [8.0, 10.5, 11.0, 9.8, 12.0, 9.0, 7.5]        # Temperaturas para Castro
}
fig, ax = plt.subplots(figsize=(10, 6)) # Ajustamos el tamaño para mejor legibilidad
ax.plot(dias, temperaturas_chile['Puerto Montt'],
        label='Puerto Montt',     # Etiqueta para la leyenda
        color='tab:blue',         # Color de la línea
        marker='o',               # Marcador circular en cada punto
        linewidth=2)              # Grosor de la línea
ax.plot(dias, temperaturas_chile['Castro'],
        label='Castro',           # Etiqueta para la leyenda
        color='tab:green',        # Color de la línea
        marker='s',               # Marcador cuadrado en cada punto
        linestyle='--',           # Línea discontinua para diferenciar
        linewidth=2)              # Grosor de la línea
ax.set_title('Temperaturas Semanales: Puerto Montt vs Castro', fontsize=16, fontweight='bold')
ax.set_xlabel('Día de la Semana', fontsize=12)
ax.set_ylabel('Temperatura (°C)', fontsize=12)
ax.set_ylim([5, 15])
ax.set_yticks(range(5, 16, 1)) # Ticks cada 1 grado para mayor detalle en este rango
ax.legend(loc='upper right', title='Ciudad', fontsize=10)
ax.grid(axis='y', color='gray', linestyle='dashed', alpha=0.7) # Añadí alpha para suavizar
plt.tight_layout()
plt.show()
````

## Colores
Para cambiar el color de los objetos se utiliza el parámetro color = nombre-color, donde nombre-color es una cadena con el nombre del color de entre los colores disponibles.

````
import matplotlib.pyplot as plt

dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

temperaturas_chile = {
    'Osorno': [12.0, 14.5, 15.0, 13.8, 16.0, 13.0, 11.5],   # Temperaturas para Osorno
    'Temuco': [10.5, 13.0, 14.0, 12.5, 15.0, 11.8, 10.0]    # Temperaturas para Temuco
}
fig, ax = plt.subplots(figsize=(10, 6)) # Aumentamos el tamaño para mejor legibilidad
ax.plot(dias, temperaturas_chile['Osorno'],
        label='Osorno',           # Etiqueta para la leyenda
        color='tab:purple',       # Color de línea personalizado
        marker='o',               # Marcador circular en cada punto
        linewidth=2)              # Grosor de la línea
ax.plot(dias, temperaturas_chile['Temuco'],
        label='Temuco',           # Etiqueta para la leyenda
        color='tab:green',        # Color de línea personalizado
        marker='s',               # Marcador cuadrado en cada punto
        linestyle='--',           # Línea discontinua para diferenciar
        linewidth=2)              # Grosor de la línea
ax.set_title('Temperaturas Semanales: Osorno vs Temuco', fontsize=16, fontweight='bold')
ax.set_xlabel('Día de la Semana', fontsize=12)
ax.set_ylabel('Temperatura (°C)', fontsize=12)
ax.set_ylim([8, 18])
ax.set_yticks(range(8, 19, 2)) # Ticks cada 2 grados
ax.legend(loc='upper right', title='Ciudad', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
````

## Marcadores

Para cambiar la forma de los puntos marcadores se utiliza el parámetro marker = nombre-marcador donde nombre-marcador es una cadena con el nombre del marcador de entre los marcadores disponibles

````
import matplotlib.pyplot as plt


dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

# Nuevos datos de temperatura para ciudades chilenas (ejemplo)
# Consideremos las temperaturas en Punta Arenas (sur) y Arica (norte).
temperaturas_chile = {
    'Punta Arenas': [6.0, 8.5, 9.0, 7.8, 10.0, 7.0, 5.5],   # Temperaturas para Punta Arenas
    'Arica': [22.0, 24.5, 25.0, 23.8, 26.0, 21.5, 20.9]     # Temperaturas para Arica
}
fig, ax = plt.subplots(figsize=(10, 6)) # Ajustamos el tamaño para mejor legibilidad
ax.plot(dias, temperaturas_chile['Punta Arenas'],
        label='Punta Arenas',     # Etiqueta para la leyenda
        color='tab:blue',         # Color de la línea
        marker='^',               # Marcador de triángulo hacia arriba (como solicitaste)
        linestyle='-',            # Línea sólida
        linewidth=2)              # Grosor de la línea
ax.plot(dias, temperaturas_chile['Arica'],
        label='Arica',            # Etiqueta para la leyenda
        color='tab:red',          # Color de la línea
        marker='o',               # Marcador circular (como solicitaste)
        linestyle='--',           # Línea discontinua para diferenciar
        linewidth=2)              # Grosor de la línea
ax.set_title('Temperaturas Semanales: Punta Arenas vs Arica', fontsize=16, fontweight='bold')
ax.set_xlabel('Día de la Semana', fontsize=12)
ax.set_ylabel('Temperatura (°C)', fontsize=12)
ax.set_ylim([0, 30])
ax.set_yticks(range(0, 31, 5)) # Ticks cada 5 grados
ax.legend(loc='upper left', title='Ciudad', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
````

## Líneas

Para cambiar el estilo de las líneas se utiliza el parámetro linestyle = nombre-estilo donde nombre-estilo es una cadena con el nombre del estilo de entre los estilos disponibles

````
import matplotlib.pyplot as plt

dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

temperaturas_chile = {
    'Pucón': [14.0, 16.5, 17.0, 15.8, 18.0, 15.0, 13.5],   # Temperaturas para Pucón
    'Valparaíso': [16.5, 18.0, 19.2, 18.8, 20.1, 17.5, 16.0] # Temperaturas para Valparaíso
}
fig, ax = plt.subplots(figsize=(10, 6)) # Ajustamos el tamaño para mejor legibilidad
ax.plot(dias, temperaturas_chile['Pucón'],
        label='Pucón',          # Etiqueta para la leyenda
        color='tab:blue',       # Color de la línea
        linestyle='--',         # Línea discontinua (dashed)
        marker='o',             # Marcador circular en cada punto
        linewidth=2)            # Grosor de la línea
ax.plot(dias, temperaturas_chile['Valparaíso'],
        label='Valparaíso',     # Etiqueta para la leyenda
        color='tab:orange',     # Color de la línea
        linestyle=':',          # Línea punteada (dotted)
        marker='s',             # Marcador cuadrado en cada punto
        linewidth=2)            # Grosor de la línea
ax.set_title('Temperaturas Semanales: Pucón vs Valparaíso', fontsize=16, fontweight='bold')
ax.set_xlabel('Día de la Semana', fontsize=12)
ax.set_ylabel('Temperatura (°C)', fontsize=12)
ax.legend(loc='upper right', title='Ciudad', fontsize=10)
ax.set_ylim([10, 25])
ax.set_yticks(range(10, 26, 2))
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
````

## Ejes

````
import matplotlib.pyplot as plt

dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

temperaturas_chile = {
    'Pucón': [14.0, 16.5, 17.0, 15.8, 18.0, 15.0, 13.5],   # Temperaturas para Pucón
    'Santiago': [20.5, 23.0, 24.5, 22.8, 25.0, 21.0, 19.5] # Temperaturas para Santiago
}

fig, ax = plt.subplots(figsize=(10, 6)) # Aumentamos el tamaño para mejor legibilidad
ax.plot(dias, temperaturas_chile['Pucón'],
        label='Pucón',          # Etiqueta para la leyenda
        color='tab:blue',       # Color de la línea
        marker='o',             # Marcador en cada punto
        linewidth=2)            # Grosor de la línea
ax.plot(dias, temperaturas_chile['Santiago'],
        label='Santiago',       # Etiqueta para la leyenda
        color='tab:red',        # Color de la línea
        marker='s',             # Marcador diferente
        linestyle='--',         # Estilo de línea diferente
        linewidth=2)            # Grosor de la línea
ax.set_xlabel("Días", fontdict={'fontsize': 14, 'fontweight': 'bold', 'color': 'darkgreen'})
ax.set_ylabel("Temperatura ºC", fontsize=12)
ax.set_ylim([10, 30]) # Rango ajustado para Pucón y Santiago
ax.set_yticks(range(10, 31, 2)) # Ticks cada 2 grados dentro del nuevo rango
ax.set_title('Temperaturas Semanales: Pucón vs Santiago', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', title='Ciudad', fontsize=10) # Leyenda arriba a la izquierda
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
````

## Leyenda

````
import matplotlib.pyplot as plt

dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

temperaturas_chile = {
    'Concepción': [13.0, 15.5, 16.0, 14.8, 17.0, 14.2, 12.5], # Temperaturas para Concepción
    'La Serena': [18.5, 20.0, 21.5, 20.8, 22.0, 19.5, 18.0]   # Temperaturas para La Serena
}
fig, ax = plt.subplots(figsize=(10, 6)) # Aumentamos el tamaño para mejor legibilidad
ax.plot(dias, temperaturas_chile['Concepción'],
        label='Concepción',       # Etiqueta para la leyenda
        marker='o',               # Marcador circular en cada punto
        linestyle='-',            # Línea sólida
        color='tab:blue',         # Color azul
        linewidth=2)              # Ancho de la línea
ax.plot(dias, temperaturas_chile['La Serena'],
        label='La Serena',        # Etiqueta para la leyenda
        marker='s',               # Marcador cuadrado en cada punto
        linestyle='--',           # Línea discontinua
        color='tab:red',          # Color rojo
        linewidth=2)              # Ancho de la línea
ax.set_title('Temperaturas Semanales: Concepción vs La Serena', fontsize=16, fontweight='bold')
# Añadir etiquetas a los ejes
ax.set_xlabel('Día de la Semana', fontsize=12)
ax.set_ylabel('Temperatura (°C)', fontsize=12)
ax.legend(loc='upper right', title='Ciudad', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
````

## Múltiples gráficos

Es posible dibujar varios gráficos en distintos ejes en una misma figura organizados en forma de tabla. Para ello, cuando se inicializa la figura y los ejes, hay que pasarle a la función subplots el número de filas y columnas de la tabla que contendrá los gráficos. Con esto los distintos ejes se organizan en un array y se puede acceder a cada uno de ellos a través de sus índices. Si se quiere que los distintos ejes compartan los mismos límites para los ejes se pueden pasar los parámetros sharex = True para el eje x o sharey = True para el eje y.

````
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, sharey=True, figsize=(10, 8)) # Ajustado figsize para mejor visibilidad


dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
temperaturas_chile = {
    'Pucón': [15.2, 17.5, 18.0, 16.5, 19.0, 16.0, 14.8], # Temperaturas para Pucón
    'Santiago': [22.1, 24.5, 25.0, 23.8, 26.0, 21.5, 20.9] # Temperaturas para Santiago
}
ax[0, 0].plot(dias, temperaturas_chile['Pucón'], marker='o', linestyle='-', color='tab:blue')
ax[0, 0].set_title('Temperaturas Semanales en Pucón', fontsize=12)
ax[0, 0].set_ylabel('Temperatura (°C)', fontsize=10) # Etiqueta del eje Y
ax[0, 1].plot(dias, temperaturas_chile['Santiago'], marker='o', linestyle='-', color='tab:red')
ax[0, 1].set_title('Temperaturas Semanales en Santiago', fontsize=12)
ax[1, 0].bar(dias, temperaturas_chile['Pucón'], color='skyblue')
ax[1, 0].set_title('Temperaturas Semanales en Pucón', fontsize=12)
ax[1, 0].set_xlabel('Día de la Semana', fontsize=10) # Etiqueta del eje X
ax[1, 0].set_ylabel('Temperatura (°C)', fontsize=10) # Etiqueta del eje Y
ax[1, 1].bar(dias, temperaturas_chile['Santiago'], color='lightcoral')
ax[1, 1].set_title('Temperaturas Semanales en Santiago', fontsize=12)
ax[1, 1].set_xlabel('Día de la Semana', fontsize=10) # Etiqueta del eje X
plt.tight_layout()
plt.suptitle('Comparación de Temperaturas Semanales (Pucón vs Santiago)', fontsize=16, fontweight='bold', y=1.02) # Título general
plt.show()
````

## Integración con Pandas

Matplotlib se integra a la perfección con la librería Pandas, permitiendo dibujar gráficos a partir de los datos de las series y DataFrames de Pandas.

````
import pandas as pd
import matplotlib.pyplot as plt

data_temperaturas_chile = {
    'Días': ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
    'Valparaíso': [16.5, 18.0, 19.2, 18.8, 20.1, 17.5, 16.0], # Temperaturas para Valparaíso
    'Concepción': [12.0, 14.5, 15.0, 13.8, 16.0, 13.0, 11.5]  # Temperaturas para Concepción
}

df = pd.DataFrame(data_temperaturas_chile)

fig, ax = plt.subplots(figsize=(10, 6)) # Ajuste del tamaño para una mejor visualización
df.plot(x='Días', y='Valparaíso', ax=ax, label='Valparaíso',
        marker='o', linestyle='-', color='tab:blue', linewidth=2)
df.plot(x='Días', y='Concepción', ax=ax, label='Concepción',
        marker='s', linestyle='--', color='tab:green', linewidth=2)
ax.set_title('Temperaturas Semanales: Valparaíso vs Concepción', fontsize=16, fontweight='bold')
ax.set_xlabel('Día de la Semana', fontsize=12)
ax.set_ylabel('Temperatura (°C)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6) # Añadir una cuadrícula para mejor lectura
ax.legend(title='Ciudad', fontsize=10) # Mostrar leyenda para identificar las líneas
plt.tight_layout()
plt.show()
````



