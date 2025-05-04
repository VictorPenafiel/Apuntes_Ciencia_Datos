 # Spark
Computación paralela sobre grandes colecciones de datos en un contexto de procesamiento distribuido. 

[API_Spark](https://spark.apache.org/docs/latest/api/python/reference/index.html)

 
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
Para la columna Result se calcula con un RDD con los valores {1, 2, 3, 3}

    collect 
        Retorna todos los elementos del RDD
    take(num)
        Retorna num de elementos del RDD
    count 
        Numero de elementos del RDD
    takeSample
        Retorna una muestra aleatoria de elementos de un RDD
    top(num)
        Retorna (num) de los primeros elementos una vez ordenado el RDD
    TakeOrdered()
         Retorna (num) de registros necesarios pero ordenados ascendentemente(al contrario de top)

