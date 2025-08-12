# Machine learning
El machine learning es una rama de la inteligencia artificial (IA) centrada en entrenar a computadoras y máquinas para imitar el modo en que aprenden los humanos, realizar tareas de forma autónoma y mejorar su rendimiento y precisión a través de la experiencia y la exposición a más datos.

[¿Qué es el Machine Learning?¿Y Deep Learning? ](https://www.youtube.com/watch?v=KytW151dpqU&list=PL-Ogd76BhmcC_E2RjgIIJZd1DQdYHcVf0)

## Etiqueta (label o target) 
    Valor de salida o la variable objetivo que un modelo de aprendizaje supervisado está tratando de predecir. 

# Red neuronal
Las redes neuronales pretenden imitar aproximadamente la estructura del cerebro humano. Están compuestas de muchos nodos interconectados (o neuronas), dispuestos en capas. Las redes neuronales hacen predicciones cuando los datos de entrada originales han realizado un "paso hacia adelante" a través de toda la red.

[Representacion_red_neuronal](https://playground.tensorflow.org/#activation=tanh&batchSize=1&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=30&networkShape=4,2&seed=0.76503&showTestData=false&discretize=false&percTrainData=10&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

*Red neuronal

[Red_Neuronal_teoría](https://www.youtube.com/watch?v=MRIv2IwFTPg&t=453s&ab_channel=DotCSV)

[Red_Neuronal_teoría_2](https://www.youtube.com/watch?v=uwbHOpp9xkc&ab_channel=DotCSV)

[Backpropagation](https://www.youtube.com/watch?v=eNIqz_noix8&ab_channel=DotCSV)

[Matemáticas_Backpropagation](https://www.youtube.com/watch?v=M5QHwkkHgAA&ab_channel=DotCSV)

[Programando_Red_Neuronal](https://www.youtube.com/watch?v=W8AeOXa_FqU&ab_channel=DotCSV)

[Funciones de activación](https://developers.google.com/machine-learning/crash-course/neural-networks/activation-functions?hl=es-419)

------------------------------------------------------------------------------------------------------------
## Regresión lineal y mínimos cuadrados ordinarios
Es un algoritmo de aprendizaje supervisado que se utiliza para predecir el valor de una variable dependiente basada en el valor de una o más variables independientes

y = ax = b

[Regresion_lineal](https://www.aprendemachinelearning.com/regresion-lineal-en-espanol-con-python/)

*Regresión lineal y mínimos cuadrados ordinarios

[teoría](https://www.youtube.com/watch?v=k964_uNn3l0&ab_channel=DotCSV)

[proyecto](https://www.youtube.com/watch?v=w2RJ1D6kz-o&t=215s&ab_channel=DotCSV)

## Descenso del Gradiente
Algoritmo de optimización que entrena  modelos de machine learning mediante la minimización de errores entre los resultados previstos y los reales.

[gradient-descent](https://www.ibm.com/es-es/think/topics/gradient-descent)

[numerical-optimization](https://www.benfrederickson.com/numerical-optimization/)

*Descenso del Gradiente

[teoría](https://www.youtube.com/watch?v=A6FiCDoz8_4&t=322s&ab_channel=DotCSV)

[proyecto](https://www.youtube.com/watch?v=-_A_AAxqzCg&t=624s&ab_channel=DotCSV)

## backpropagation (retropropagación) 
Es una técnica de machine learning esencial para la optimización de las redes neuronales. Facilita el uso de algoritmos de descenso gradiente para actualizar las ponderaciones de la red. La lógica de la retropropagación es que las capas de neuronas de las redes neuronales artificiales son básicamente una serie de funciones matemáticas anidadas. Durante el entrenamiento, esas ecuaciones interconectadas se anidan en otra función más: una "función de pérdida" que mide la diferencia (o "pérdida") entre la salida deseada (o "verdad fundamental") para una entrada dada y la salida real de las redes neuronales.
Se actualizan los pesos y el bias.

[backpropagation](https://www.ibm.com/es-es/think/topics/backpropagation)

[¿Qué es la tasa de aprendizaje(learning rate) en el machine learning?](https://www.ibm.com/es-es/think/topics/learning-rate)

Época(epoch): Controla la duración total del entrenamiento.

Tamaño del Lote(batch size): Determina la frecuencia de las actualizaciones de los pesos.

Tasa de Aprendizaje(learning rate): Define la magnitud del aprendizaje en cada actualización.

Verbose: Se refiere al nivel de detalle que un algoritmo o un proceso de entrenamiento imprime en la consola.

verbose=0: El algoritmo se ejecuta sin imprimir nada en la consola, a excepción de posibles errores o advertencias críticas. Es útil cuando el código está en producción y no necesitas monitorear cada paso.

# Aprendizaje supervisado
Regresión- Variable numerica, predecir un valor numerico (Continua)

````
from sklearn.linear_model import LinearRegression

Datos: [[kilómetros, año]], precio
X = [[100000, 2010], [50000, 2015], [20000, 2020]]
y = [5000, 15000, 25000]  # Precios en USD

modelo = LinearRegression()
modelo.fit(X, y)
prediccion = modelo.predict([[80000, 2012]])
print(prediccion)  # Ejemplo: [12000.50]
````
-----------------------------------------------------------------------------------------------------

# Clasificación Etiquetas, predecir etiqueta o clase (Discretas)

    from sklearn.ensemble import RandomForestClassifier

# Datos: [[núm_palabras, contiene_emoji]], etiqueta

````
X = [[50, 0], [20, 1], [100, 0]]
y = [0, 1, 0]  # 0: No spam, 1: Spam

modelo = RandomForestClassifier()
modelo.fit(X, y)
prediccion = modelo.predict([[80, 1]])
print(prediccion)  # Ejemplo: [1] (spam)
````

------------------------------------------------------------------------------------------------------------
# Proyectos

[Ataque adversario](https://www.youtube.com/watch?v=JoQx39CoXW8&t=314s&ab_channel=DotCSV)

[Generando flores realistas con IA](https://www.youtube.com/watch?v=YsrMGcgfETY&t=4423s&ab_channel=DotCSV)

[Programa el juego de la vida](https://www.youtube.com/watch?v=qPtKv9fSHZY&ab_channel=DotCSV)

