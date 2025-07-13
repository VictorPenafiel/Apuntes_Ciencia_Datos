[numpy](https://numpy.org/)

    import numpy as np

### np.metodo()

`array()` 
        Crea una array NumPy a partir de una lista, tupla o cualquier secuencia. 

`.info()`

`arange()`
        Crea un array con valores espaciados uniformemente dentro de un intervalo dado. 

`np.linspace()`
        Genera un array con valores espaciados linealmente en un intervalo especificado

`zeros()` 
        Genera un array de NumPy con las dimensiones especificadas, donde todos los elementos son ceros. Útil para inicializar arrays.

`ones()`
        Similar a `zeros()`, pero crea un array donde todos los elementos son unos.

`concatenate()`
    Se utiliza para unir arrays a lo largo de un eje existente.

`reshape`
    Redimensionamiento de matrices

`unique`
    Valores unicos de un array
    
### Forma y Tamaño de una matriz

`ndarray.ndim` entrega el número de ejes, o dimensiones, de la matriz.

`ndarray.size` entrega el número total de elementos de la matriz.

`ndarray.shape` entrega una tupla de números enteros que indican el número de elementos almacenados a lo largo de cada dimensión de la matriz. Si, por ejemplo, tiene una matriz 2D con 2 filas y 3 columnas, la forma de su matriz es (2, 3).

---












`np.nonzero()`
        Devuelve los **índices de los elementos que no son cero** en un array. Es útil para encontrar la ubicación de valores específicos.

`np.eye()`
        Crea una **matriz identidad** cuadrada, donde la diagonal principal está compuesta por unos y el resto son ceros.

`np.random` (módulo)
        No es un método, sino un **módulo de NumPy** que contiene funciones para generar números aleatorios. Puedes generar enteros, flotantes, muestrear distribuciones, etc.

`np.nan`
        Representa **"Not a Number"** (No es un número). Es un valor flotante especial en NumPy para indicar datos perdidos o indefinidos. Pandas lo usa extensamente para representar valores faltantes.

`np.diag()`
        Puede hacer dos cosas:

1.  **Extraer la diagonal** de una matriz cuadrada.
2.  **Crear una matriz diagonal** a partir de un array unidimensional.

