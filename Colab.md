Gestor que te instala todo lo que puedes necesitar

[Anaconda](https://anaconda.org/anaconda/conda)

[Cuadernos Jupyter](https://jupyter.org/)

Otra forma de ocupar los cuadernos es por medio de Google el cual te proporciona tarjetas gráficas de última generación. 

[Colab](https://colab.research.google.com/) 

### Obtener bases de datos

[Kaggle](https://www.kaggle.com/datasets)

[TensorFlow](https://www.tensorflow.org/datasets?hl=es)

[Hugging Face](https://huggingface.co/datasets)

[UC_Irvine](https://archive.ics.uci.edu/datasets)

# Google Colab - Guía Completa de Funciones Básicas


### MANEJO DE CELDAS

````
Colab ofrece dos tipos de celdas:
- Code cells: Para ejecutar código Python
- Text cells: Para documentación en Markdown

Atajos útiles:
- Añadir celda de código: Ctrl + M + B
- Añadir celda de texto: Click en +Texto
- Ejecutar celda: Shift + Enter
````

### CAMBIO DE DIRECTORIO

    import os

Directorio actual

    print(f"Directorio actual: {os.getcwd()}")

Cambiar directorio (ejemplo)

    nuevo_dir = "/content/sample_data"
    os.chdir(nuevo_dir)
    print(f"Nuevo directorio: {os.getcwd()}")


### DESCARGA DE DATASETS

Descargar desde URL

    !wget "https://ejemplo.com/dataset.csv"


### EXPORTACIÓN DE ARCHIVOS

Crear archivo de ejemplo

    from google.colab import files

    with open('ejemplo.txt', 'w') as f:
        f.write("Contenido de ejemplo")

Descargar archivo

    files.download('ejemplo.txt')


### MONTAJE DE GOOGLE DRIVE (OPCIONAL)


    from google.colab import drive
    drive.mount('/content/drive')

Acceder a archivos en Drive
    !ls "/content/drive/MyDrive"

### INSTALACIÓN DE LIBRERÍAS ADICIONALES

    !pip install nombre_libreria

### EJECUCIÓN DE SCRIPTS EXTERNOS

    !python "ruta/al/script.py"

### LIMPIEZA DE MEMORIA

    import gc
    gc.collect()

    print("¡Todas las operaciones básicas de Colab fueron ejecutadas!")

------------------------------------------------------------------------------------------------------------