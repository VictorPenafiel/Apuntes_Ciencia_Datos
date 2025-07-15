[numpy](https://numpy.org/)

    import numpy as np

### np.metodo()

`array()` 
        Crea una array NumPy a partir de una lista, tupla o cualquier secuencia. 

`info()`

`arange()`
        Crea un array con valores espaciados uniformemente dentro de un intervalo dado. 

`linspace()`
        Genera un array con valores espaciados linealmente en un intervalo especificado

`zeros()` 
        Genera un array de NumPy con las dimensiones especificadas, donde todos los elementos son ceros. Útil para inicializar arrays.
`nonzero()`
        Devuelve los índices de los elementos que no son cero.

`ones()`
        Similar a `zeros()`, pero crea un array donde todos los elementos son unos.

`eye()`
        Crea una matriz identidad cuadrada, donde la diagonal principal está compuesta por unos y el resto son ceros.

`reshape`
        Redimensionamiento de matrices

`concatenate()`
        Se utiliza para unir arrays a lo largo de un eje existente.

`unique`
    Valores unicos de un array

`matmul`
        Multiplicación de matrices

`nan`
        Not a Number(No es un número). Es un valor flotante especial en NumPy para indicar datos perdidos o indefinidos

`diag()`
        Puede hacer dos cosas:

`std`
        Calcular la desviación estándar de los elementos en un array

1.  Extraer la diagonal de una matriz.
2.  Crear una matriz diagonal a partir de un array unidimensional.
    

### Forma y Tamaño de una matriz
### ndarray.metodo()

`ndim` 
        entrega el número de ejes, o dimensiones, de la matriz.

`size` 
        entrega el número total de elementos de la matriz.

`shape` 
        entrega una tupla de números enteros que indican el número de elementos almacenados a lo largo de cada dimensión de la matriz. Si, por ejemplo, tiene una matriz 2D con 2 filas y 3 columnas, la forma de su matriz es (2, 3).

---

 (módulo)
`random`
        No es un método, sino un módulo de NumPy que contiene funciones para generar números aleatorios. Puedes generar enteros, flotantes, muestrear distribuciones, etc.


`random.seed` 
        Función que se utiliza para inicializar el generador de números pseudoaleatorios de NumPy.

`random.normal` 
        Función que se utiliza para generar muestras aleatorias de una distribución normal (o Gaussiana)

`random.expovariate`
        Función que se utiliza para generar un número aleatorio a partir de una distribución exponencial.

`random.randint(a, b)`
        Genera un número entero aleatorio dentro de un rango especificado, incluyendo ambos extremos (tanto a como b).