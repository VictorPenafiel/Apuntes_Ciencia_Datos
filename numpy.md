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
        Redimensionamiento de matrices, sin cambiar datos

`concatenate()`
        Se utiliza para unir arrays a lo largo de un eje existente.

`unique`
        Valores unicos de un array

`dot`
        Se utiliza para calcular el producto escalar de dos arrays

`matmul`
        Multiplicación de matrices

`sin`
        Calcula el seno trigonométrico de uno o varios ángulos. La principal característica, y la más importante a recordar, es que opera elemento por elemento sobre los arrays de NumPy.

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
        Entrega el número de ejes, o dimensiones, de la matriz.

`size` 
        Entrega el número total de elementos de la matriz.

`shape` 
        Entrega una tupla de números enteros que indican el número de elementos almacenados a lo largo de cada dimensión de la matriz. Si, por ejemplo, tiene una matriz 2D con 2 filas y 3 columnas, la forma de su matriz es (2, 3).

`newaxis`
        Se utiliza para aumentar la dimensionalidad de un array existente insertando un nuevo eje

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

`random.RandomState`
        Es una clase que proporciona un generador de números pseudoaleatorios

¡Excelente pregunta! Parece que te refieres a numpy.random.permutation. Como programador experto, puedo decirte que esta es una función muy útil dentro del módulo numpy.random para trabajar con aleatoriedad y reordenamiento de datos.

`random.permutation(x)`

Dos comportamientos principales dependiendo del tipo de argumento x que le pases:

    Si x es un entero (int):

        numpy.random.permutation(n) generará un array ndarray con los números enteros desde 0 hasta n-1 (es decir, np.arange(n)) y luego permutará aleatoriamente esos números.

        Es útil para obtener un orden aleatorio de índices, por ejemplo, para seleccionar filas aleatorias en un dataset.