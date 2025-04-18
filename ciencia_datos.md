### cuadernos Jupyter
https://jupyter.org/

### Gestor que te instala todo lo que puedes necesitar
https://anaconda.org/anaconda/conda

### Otra forma de ocupar los cuadernos es por medio de Google el cual te proporciona tarjetas gráficas de última generación.
https://colab.research.google.com/ 

# Google Colab - Guía Completa de Funciones Básicas

# --------------------------------------------------
# 1. MANEJO DE CELDAS
# --------------------------------------------------
"""
Colab ofrece dos tipos de celdas:
- Code cells: Para ejecutar código Python
- Text cells: Para documentación en Markdown

Atajos útiles:
- Añadir celda de código: Ctrl + M + B
- Añadir celda de texto: Click en +Texto
- Ejecutar celda: Shift + Enter
"""

# --------------------------------------------------
# 2. CAMBIO DE DIRECTORIO
# --------------------------------------------------
import os

# Directorio actual
print(f"Directorio actual: {os.getcwd()}")

# Cambiar directorio (ejemplo)
nuevo_dir = "/content/sample_data"
os.chdir(nuevo_dir)
print(f"Nuevo directorio: {os.getcwd()}")

# --------------------------------------------------
# 3. MANEJO DE ARCHIVOS COMPRIMIDOS
# --------------------------------------------------
# Descomprimir archivo zip
!unzip "/content/archivo_comprimido.zip" -d "/content/destino"

# --------------------------------------------------
# 4. DESCARGA DE DATASETS
# --------------------------------------------------
# Descargar desde URL
!wget "https://ejemplo.com/dataset.csv"

# --------------------------------------------------
# 5. EXPORTACIÓN DE ARCHIVOS
# --------------------------------------------------
from google.colab import files

# Crear archivo de ejemplo
with open('ejemplo.txt', 'w') as f:
    f.write("Contenido de ejemplo")

# Descargar archivo
files.download('ejemplo.txt')

# --------------------------------------------------
# 6. MONTAJE DE GOOGLE DRIVE (OPCIONAL)
# --------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

# Acceder a archivos en Drive
!ls "/content/drive/MyDrive"

# --------------------------------------------------
# 7. INSTALACIÓN DE LIBRERÍAS ADICIONALES
# --------------------------------------------------
!pip install nombre_libreria

# --------------------------------------------------
# 8. EJECUCIÓN DE SCRIPTS EXTERNOS
# --------------------------------------------------
!python "ruta/al/script.py"

# --------------------------------------------------
# 9. LIMPIEZA DE MEMORIA
# --------------------------------------------------
import gc
gc.collect()

print("¡Todas las operaciones básicas de Colab fueron ejecutadas!")

----------------------------------------------------------------------------------------------------------------------

# Aprendizaje supervisado
## Regresión- Variable numerica, predecir un valor numerico (Continua)

from sklearn.linear_model import LinearRegression

# Datos: [[kilómetros, año]], precio
X = [[100000, 2010], [50000, 2015], [20000, 2020]]
y = [5000, 15000, 25000]  # Precios en USD

modelo = LinearRegression()
modelo.fit(X, y)
prediccion = modelo.predict([[80000, 2012]])
print(prediccion)  # Ejemplo: [12000.50]


#Clasificación Etiquetas, predecir etiqueta o clase (Discretas)

from sklearn.ensemble import RandomForestClassifier

# Datos: [[núm_palabras, contiene_emoji]], etiqueta
X = [[50, 0], [20, 1], [100, 0]]
y = [0, 1, 0]  # 0: No spam, 1: Spam

modelo = RandomForestClassifier()
modelo.fit(X, y)
prediccion = modelo.predict([[80, 1]])
print(prediccion)  # Ejemplo: [1] (spam)
----------------------------------------------------------------------------------------------------------------------


El 1 es teoría, el 2 es proyecto.

## Descenso del Gradiente
1
https://www.youtube.com/watch?v=A6FiCDoz8_4&t=322s&ab_channel=DotCSV


2
https://www.youtube.com/watch?v=-_A_AAxqzCg&t=624s&ab_channel=DotCSV


----------------------------------------------------------------------------------------------------------------------

## Regresión lineal y mínimos cuadrados ordinarios
1
https://www.youtube.com/watch?v=k964_uNn3l0&ab_channel=DotCSV

2
https://www.youtube.com/watch?v=w2RJ1D6kz-o&t=215s&ab_channel=DotCSV


----------------------------------------------------------------------------------------------------------------------

## Red neuronal
1
https://www.youtube.com/watch?v=MRIv2IwFTPg&t=453s&ab_channel=DotCSV
https://www.youtube.com/watch?v=uwbHOpp9xkc&ab_channel=DotCSV
https://www.youtube.com/watch?v=eNIqz_noix8&ab_channel=DotCSV
https://www.youtube.com/watch?v=M5QHwkkHgAA&ab_channel=DotCSV

2
https://www.youtube.com/watch?v=W8AeOXa_FqU&ab_channel=DotCSV

----------------------------------------------------------------------------------------------------------------------

### Otros proyectos

## Ataque adversario
https://www.youtube.com/watch?v=JoQx39CoXW8&t=314s&ab_channel=DotCSV

## Generando flores realistas con IA
https://www.youtube.com/watch?v=YsrMGcgfETY&t=4423s&ab_channel=DotCSV

## Programa el juego de la vida
https://www.youtube.com/watch?v=qPtKv9fSHZY&ab_channel=DotCSV
