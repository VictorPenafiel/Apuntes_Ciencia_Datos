 # Spark
Computación paralela sobre grandes colecciones de datos en un contexto de procesamiento distribuido. 
[Spark](https://aitor-medrano.github.io/bigdata2122/apuntes/spark01rdd.html)
[API_Spark](https://spark.apache.org/docs/latest/api/python/reference/index.html)
[SQL_Spark](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/index.html)
[Colaboratory_Spark](https://josemtech.com/2022/11/02/instalando-apache-spark-en-google-colaboratory/)
 
 ### Instalamos la librería pyspark, Java 8 y seteamos las variables de entorno para que no devuelva error:

	!pip install pyspark
    !apt-get install openjdk-8-jdk-headless -qq > /dev/null

    import os
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

### Luego, inicializamos el Spark Context seteando el master y el nombre de la aplicación:

    from pyspark import SparkConf, SparkContext

    conf = SparkConf().setMaster("URL_cluster").setAppName("Nombre_aplicacion")

    # Inicializo el Spark Context
    sc = SparkContext(conf = conf)

    sc

###  Creación de RDD, (datos distribuidos resistentes). Es posbile crear RDD de dos maneras: cargando un conjunto de datos externo o distribuyendo una colección de objetos (por ejemplo, una lista o conjunto). 

## En este caso lo haremos a través de cargar un conjunto de datos externos con el método sc.textFile():

    # Leo el archivo de texto
    text_file = sc.textFile("mr_text-file.txt")

https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.SparkContext.textFile.html

### Para apagar Spark, debemos llamaar al método stop () del SparkContext, en nuestro caso la variable sc:

    sc.stop()


# listado de las acciones más comunes y su utilización:


```python
# RDD de ejemplo, inicializado en tu Spark Context (sc)
rdd = sc.parallelize([1, 2, 3, 3])
```

-----

### **Acciones más Comunes en PySpark RDDs**

Las **acciones** son operaciones que devuelven un valor al *driver program* o escriben datos en un sistema de almacenamiento externo. Son las que, en última instancia, disparan la ejecución de las transformaciones perezosas (`lazy transformations`) que hayas definido previamente.

| Comando (Acción) | Descripción | Ejemplo de Código | Resultado |
| :--- | :--- | :--- | :--- |
| **`collect()`** | Retorna todos los elementos del RDD como una lista al driver. **¡Cuidado\!** Úsalo solo con RDDs pequeños, ya que puede agotar la memoria del nodo principal. | `rdd.collect()` | `[1, 2, 3, 3]` |
| **`count()`** | Devuelve la cantidad total de elementos en el RDD. Es una operación muy eficiente. | `rdd.count()` | `4` |
| **`take(num)`** | Retorna una lista con los primeros `num` elementos del RDD. No garantiza ningún orden específico si el RDD no ha sido ordenado previamente. | `rdd.take(2)` | `[1, 2]` |
| **`first()`** | Retorna el primer elemento del RDD. Es un atajo para `take(1)[0]`. | `rdd.first()` | `1` |
| **`top(num)`** | Retorna una lista con los `num` elementos más grandes del RDD, ordenados de forma **descendente** (de mayor a menor). | `rdd.top(3)` | `[3, 3, 2]` |
| **`takeOrdered(num)`** | Retorna una lista con los `num` elementos más pequeños del RDD, ordenados de forma **ascendente** (de menor a mayor). | `rdd.takeOrdered(3)` | `[1, 2, 3]` |
| **`takeSample(withReplacement, num, [seed])`** | Retorna una muestra aleatoria de `num` elementos. `withReplacement` (booleano) indica si un mismo elemento puede ser seleccionado varias veces. El `seed` es opcional para reproducibilidad. | `rdd.takeSample(False, 2)` | `[1, 3]` *(el resultado puede variar)* |
| **`reduce(func)`** | Agrega los elementos del RDD usando una función que toma dos argumentos y retorna uno. La función debe ser conmutativa y asociativa. | `rdd.reduce(lambda a, b: a + b)` | `9` *(1+2+3+3)* |
| **`countByValue()`** | Agrupa los valores únicos del RDD y cuenta la frecuencia de cada uno. Devuelve un objeto tipo diccionario. | `rdd.countByValue()` | `{1: 1, 2: 1, 3: 2}` |
| **`foreach(func)`** | Aplica una función a cada elemento del RDD. Generalmente se usa para interactuar con sistemas externos (ej. escribir en una base de datos) ya que no devuelve ningún valor al driver. | `rdd.foreach(lambda x: print(x))` | *Imprime 1, 2, 3, 3 en la salida de los workers* |

-----

### **Puntos Clave a Recordar**

  * **Transformaciones vs. Acciones**: Las transformaciones (como `map()`, `filter()`, `flatMap()`) son perezosas y solo construyen el plan de ejecución (DAG). Las **acciones** (como `collect()`, `count()`) son las que obligan a Spark a ejecutar ese plan y producir un resultado.
  * **Memoria del Driver**: Ten siempre presente qué comandos devuelven datos al driver (`collect`, `take`, `top`, etc.) y su potencial impacto en la memoria si el RDD es masivo.

---

**transformaciones**.

A diferencia de las acciones, las **transformaciones** en Spark son **perezosas** (*lazy*). Esto significa que cuando aplicas una transformación a un RDD, Spark no ejecuta la operación inmediatamente. En su lugar, construye un plan de ejecución (un DAG o Grafo Acíclico Dirigido). La computación real solo se dispara cuando se invoca una *acción* sobre el RDD resultante.

Las transformaciones se dividen en dos categorías clave según cómo manejan las particiones de datos.

-----

### **Transformaciones Estrechas (Narrow Transformations)**

Estas transformaciones son muy eficientes porque cada partición de entrada se utiliza para calcular, como máximo, una partición de salida. **No requieren mover datos entre particiones** (un proceso costoso conocido como *shuffle*).

Para los ejemplos, supongamos que tenemos este RDD inicial:
`rdd = sc.parallelize([1, 2, 3, 4])`

| Transformación | Descripción | Ejemplo de Código | Resultado (conceptual) |
| :--- | :--- | :--- | :--- |
| **`map(func)`** | Aplica una función a cada elemento del RDD para producir un nuevo RDD con la misma cantidad de elementos. | `rdd.map(lambda x: x * 2)` | `[2, 4, 6, 8]` |
| **`filter(func)`** | Retorna un nuevo RDD que contiene solo los elementos que cumplen con una condición (la función debe devolver `True`). | `rdd.filter(lambda x: x % 2 == 0)` | `[2, 4]` |
| **`flatMap(func)`** | Similar a `map`, pero cada elemento de entrada puede ser mapeado a cero o más elementos de salida. Es muy útil para "desplegar" listas. | `text_rdd = sc.parallelize(["hola mundo", "apache spark"])`\<br\>`text_rdd.flatMap(lambda s: s.split(' '))` | `['hola', 'mundo', 'apache', 'spark']` |
| **`union(otherRDD)`** | Retorna un nuevo RDD que contiene la unión de los elementos de ambos RDDs (incluyendo duplicados). | `rdd2 = sc.parallelize([3, 5])`\<br\>`rdd.union(rdd2)` | `[1, 2, 3, 4, 3, 5]` |

-----

### **Transformaciones Amplias (Wide Transformations)**

Estas transformaciones **requieren un *shuffle***. Esto significa que Spark debe mover datos entre los nodos del clúster para agrupar los que tienen la misma clave o para reordenarlos. Son operaciones computacionalmente más costosas.

Para los ejemplos, usaremos un RDD de pares clave-valor:
`rdd_kv = sc.parallelize([("a", 1), ("b", 2), ("a", 3)])`

| Transformación | Descripción | Ejemplo de Código | Resultado (conceptual) |
| :--- | :--- | :--- | :--- |
| **`groupByKey()`** | Agrupa todos los valores asociados a una misma clave en una sola tupla (`(clave, [lista_de_valores])`). **Uso con precaución**, puede causar problemas de memoria si una clave tiene muchos valores. | `rdd_kv.groupByKey().mapValues(list)` | `[('b', [2]), ('a', [1, 3])]` |
| **`reduceByKey(func)`** | Agrega los valores para cada clave usando una función de reducción asociativa. Es **mucho más eficiente que `groupByKey`** porque combina valores en cada partición antes del shuffle final. | `rdd_kv.reduceByKey(lambda v1, v2: v1 + v2)` | `[('b', 2), ('a', 4)]` |
| **`sortByKey()`** | Ordena el RDD de pares clave-valor basándose en la clave. | `rdd_kv.sortByKey()` | `[('a', 1), ('a', 3), ('b', 2)]` |
| **`distinct()`** | Retorna un nuevo RDD que contiene solo los elementos únicos del RDD original. | `sc.parallelize([1, 2, 3, 3, 2, 1]).distinct()` | `[1, 2, 3]` |
| **`join(otherRDD)`** | Realiza un `INNER JOIN` entre dos RDDs de pares clave-valor basándose en sus claves. | `rdd_kv2 = sc.parallelize([("a", "x"), ("c", "y")])`\<br\>`rdd_kv.join(rdd_kv2)` | `[('a', (1, 'x')), ('a', (3, 'x'))]`|
| **`repartition(num)`** | Cambia el número de particiones del RDD. Siempre causa un shuffle completo de los datos. | `rdd.repartition(2)` | Un nuevo RDD con 2 particiones. |

### **Concepto Clave: La Pereza es Eficiencia  laziness is efficiency**

Recordar que Spark no hará ningún trabajo hasta que llames a una acción. Puedes encadenar múltiples transformaciones (`map`, `filter`, `reduceByKey`, etc.) y Spark las optimizará en un único plan de ejecución eficiente que se lanzará solo al final. ¡Ese es el poder del modelo de ejecución de Spark\! 🚀