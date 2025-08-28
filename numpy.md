[numpy](https://numpy.org/)

    import numpy as np

## Creación de Arrays
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
`full(shape, fill_value)`
        Crea un array de una forma (shape) determinada, rellenándolo completamente con un valor específico (fill_value).
`empty(shape)`
        Crea un array con la forma especificada sin inicializar sus entradas a ningún valor en particular. Es ligeramente más rápido que zeros o ones cuando no necesitas que los valores iniciales sean específicos.

`copy(array)`
        Crea una copia explícita de un array en memoria. Esto es crucial para evitar que los cambios en la copia afecten al array original. 


## Manipulación de Arrays

`reshape`
        Redimensionamiento de matrices, sin cambiar datos

`concatenate()`
        Se utiliza para unir arrays a lo largo de un eje existente.

`flatten() / ravel()`
        Ambos convierten un array multidimensional en uno de una sola dimensión. La diferencia principal es que flatten() siempre devuelve una copia, mientras que ravel() intenta devolver una vista del array original si es posible, lo que es más eficiente en memoria.

`transpose() o .T`
        Permuta las dimensiones de un array. Es la forma más común de obtener la transpuesta de una matriz.

`split() / hsplit() / vsplit()` 
        La contraparte de concatenate. Permiten dividir un array en varios subarrays, ya sea horizontal, verticalmente o a lo largo de un eje específico.

`stack() / hstack() / vstack()`
        Sirven para apilar arrays. vstack apila arrays en secuencia vertical (fila por fila), hstack los apila en secuencia horizontal (columna por columna), y stack une una secuencia de arrays a lo largo de un nuevo eje.

`append()`
        Añade valores al final de un array.

`insert()`
        Inserta valores en un array antes de un índice dado.

`delete()`
        Elimina elementos de un array a lo largo de un eje especificado.

`sort()`
        Ordena los elementos de un array. Puede hacerse "in-place" (modificando el array original) o devolviendo una copia ordenada.



## Funciones Matemáticas y Estadísticas
Operaciones básicas: add(), subtract(), multiply(), divide().

`sum()` 
        Calcula la suma de los elementos del array.

`min() / argmin()`
        Encuentra el valor mínimo y el índice de ese valor.

`max() / argmax()`
        Encuentra el valor máximo y el índice de ese valor.

`mean()`
        Calcula la media aritmética.

`median()`
        Calcula la mediana.

`corrcoef()`
        Calcula el coeficiente de correlación de Pearson.

`cumsum()`
        Calcula la suma acumulada de los elementos.

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
`std`
        Calcular la desviación estándar de los elementos en un array



`diag()`
        Puede hacer dos cosas:
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

## (módulo)
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

`random.permutation(x)`

Dos comportamientos principales dependiendo del tipo de argumento x que le pases:

    Si x es un entero (int):

        numpy.random.permutation(n) generará un array ndarray con los números enteros desde 0 hasta n-1 (es decir, np.arange(n)) y luego permutará aleatoriamente esos números.

        Es útil para obtener un orden aleatorio de índices, por ejemplo, para seleccionar filas aleatorias en un dataset.

## Álgebra Lineal (numpy.linalg)

Este es un submódulo importante:

`inv()`
        Calcula la inversa de una matriz.

` det()`
        Calcula el determinante de una matriz.

`eig()`
        Calcula los autovalores y autovectores de una matriz cuadrada.

`solve()`
        Resuelve un sistema de ecuaciones lineales.