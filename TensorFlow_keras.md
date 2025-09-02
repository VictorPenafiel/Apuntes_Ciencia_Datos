# Tensor
Un tensor es un vector o matriz de n dimensiones que representa todo tipo de datos. Todos los valores de un tensor contienen tipos de datos idénticos con una forma conocida (o parcialmente conocida). La forma de los datos es la dimensionalidad de la matriz o matriz.
[Tensor](https://www.guru99.com/es/tensor-tensorflow.html)
[Tensorflow](https://www.tensorflow.org/?hl=es)
[Introduccion](https://www.tensorflow.org/guide/basics?hl=es-419)


# Keras
Keras es una API de alto nivel para Deep Learning integrada en TensorFlow, convirtiéndose en su API oficial para el desarrollo de redes neuronales.. Su diseño modular y enfocado en la usabilidad simplifica la creación de modelos, reduciendo significativamente el código y acelerando el proceso de desarrollo.

[keras](https://keras.io/)
[Introduccion](https://www.tensorflow.org/guide/keras?hl=es-419)
[Introduccion_API](https://www.tensorflow.org/guide/keras/functional?hl=es-419)

## TensorFlow y tf.keras

```Python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
````

##  base para clasificación de imágenes

```Python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```
---------------------------------------------------------------------------------------------
## base para clasificación de imágenes 2

```Python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
```

## base para clasificación de imágenes 3

```Python
# -----------------------------------------------------------------------------
# 1. IMPORTACIÓN Y CONFIGURACIÓN INICIAL
# -----------------------------------------------------------------------------
import tensorflow as tf

# Es una buena práctica verificar la versión de TensorFlow que estás utilizando.
# Ayuda a asegurar la compatibilidad y a depurar posibles problemas.
print("TensorFlow version:", tf.__version__)

# Importamos las capas específicas que usaremos para construir nuestro modelo.
# - Conv2D: Capa convolucional para extraer características de las imágenes.
# - Flatten: "Aplana" los datos multidimensionales (como una imagen) en un vector unidimensional.
# - Dense: La clásica capa de neuronas totalmente conectada.
from tensorflow.keras.layers import Dense, Flatten, Conv2D

# Importamos la clase base 'Model' de Keras. La usaremos para crear nuestro propio
# modelo personalizado mediante la técnica de "subclassing". Esto nos da máxima flexibilidad.
from tensorflow.keras import Model


# -----------------------------------------------------------------------------
# 2. PREPARACIÓN DE LOS DATOS (DATASET MNIST)
# -----------------------------------------------------------------------------
# Cargamos el famoso dataset MNIST, que contiene imágenes de dígitos escritos a mano (0-9).
# Keras lo proporciona de forma muy conveniente.
mnist = tf.keras.datasets.mnist

# El dataset ya viene dividido en conjuntos de entrenamiento y de prueba.
# x_train/x_test: Contienen las imágenes (los pixeles).
# y_train/y_test: Contienen las etiquetas (el número correcto para cada imagen).
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizamos los valores de los pixeles. Las imágenes originales tienen valores de 0 a 255.
# Al dividirlos por 255.0, los transformamos a un rango de 0.0 a 1.0.
# Esto ayuda a que el modelo entrene de forma más estable y rápida.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Añadimos una dimensión para los "canales" de color.
# Las capas convolucionales (Conv2D) esperan tensores con formato: (batch, altura, ancho, canales).
# Nuestras imágenes son en escala de grises, por lo que solo tienen 1 canal.
# Pasamos de (60000, 28, 28) a (60000, 28, 28, 1).
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

# Creamos pipelines de datos eficientes usando tf.data.Dataset.
# Es la forma recomendada de alimentar datos a un modelo en TensorFlow.
# .from_tensor_slices(): Crea un dataset a partir de nuestros tensores en memoria.
# .shuffle(10000): Mezcla aleatoriamente los datos para evitar que el modelo aprenda el orden.
# .batch(32): Agrupa los datos en lotes (batches) de 32. El modelo se actualizará después de procesar cada lote.
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

# Hacemos lo mismo para el conjunto de prueba, pero sin mezclar (.shuffle).
# Queremos evaluar el modelo de forma consistente en cada época.
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# -----------------------------------------------------------------------------
# 3. DEFINICIÓN DEL MODELO (KERAS SUBCLASSING API)
# -----------------------------------------------------------------------------
# Creamos nuestra propia clase de modelo heredando de tf.keras.Model.
class MyModel(Model):
  # El constructor '__init__' es donde definimos todas las capas que usaremos.
  def __init__(self):
    super().__init__()
    # Capa convolucional con 32 filtros, un kernel de 3x3 y activación ReLU.
    self.conv1 = Conv2D(32, 3, activation='relu')
    # Capa para aplanar la salida de la capa convolucional a un solo vector.
    self.flatten = Flatten()
    # Capa densa con 128 neuronas y activación ReLU.
    self.d1 = Dense(128, activation='relu')
    # Capa de salida con 10 neuronas (una para cada dígito, 0-9).
    # No tiene activación, ya que la función de pérdida que usaremos es más eficiente si la aplica internamente.
    self.d2 = Dense(10)

  # El método 'call' define el "forward pass" (la pasada hacia adelante).
  # Aquí especificamos cómo los datos fluyen a través de las capas que definimos antes.
  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Instanciamos el modelo. A partir de este objeto 'model' podremos entrenar y evaluar.
model = MyModel()


# -----------------------------------------------------------------------------
# 4. CONFIGURACIÓN DEL ENTRENAMIENTO
# -----------------------------------------------------------------------------
# Definimos la función de pérdida (loss function).
# SparseCategoricalCrossentropy es ideal para problemas de clasificación multiclase
# donde las etiquetas son números enteros (como 0, 1, 2...).
# from_logits=True le indica que la salida de nuestro modelo no ha pasado por una
# función de activación (como softmax), y que la función de pérdida debe encargarse de ello.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Elegimos el optimizador. Adam es un algoritmo de optimización muy popular y efectivo
# que ajustará los pesos del modelo para minimizar la función de pérdida.
optimizer = tf.keras.optimizers.Adam()

# Definimos las métricas para monitorear el rendimiento del modelo durante el entrenamiento.
# Usamos 'Mean' para llevar un promedio de la pérdida.
train_loss = tf.keras.metrics.Mean(name='train_loss')
# Usamos 'SparseCategoricalAccuracy' para calcular el porcentaje de aciertos.
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# Hacemos lo mismo para las métricas de evaluación en el conjunto de prueba.
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# -----------------------------------------------------------------------------
# 5. LÓGICA DE ENTRENAMIENTO Y EVALUACIÓN
# -----------------------------------------------------------------------------

# El decorador @tf.function es clave para el rendimiento. Convierte la función de Python
# en un grafo de TensorFlow pre-compilado, lo que acelera enormemente la ejecución.
@tf.function
def train_step(images, labels):
  # tf.GradientTape "graba" las operaciones para poder calcular los gradientes automáticamente.
  with tf.GradientTape() as tape:
    # 1. Hacemos una pasada hacia adelante (forward pass) para obtener las predicciones.
    # 'training=True' es importante si el modelo tiene capas que se comportan
    # diferente en entrenamiento y en inferencia (ej. Dropout, BatchNorm).
    predictions = model(images, training=True)
    # 2. Calculamos la pérdida comparando las predicciones con las etiquetas reales.
    loss = loss_object(labels, predictions)
  # 3. Calculamos los gradientes de la pérdida con respecto a los pesos del modelo.
  gradients = tape.gradient(loss, model.trainable_variables)
  # 4. El optimizador aplica estos gradientes para actualizar los pesos. ¡Aquí es donde el modelo aprende!
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  # Actualizamos nuestras métricas de entrenamiento con los resultados de este lote.
  train_loss(loss)
  train_accuracy(labels, predictions)

# También compilamos la función de prueba para que sea rápida.
@tf.function
def test_step(images, labels):
  # Para la evaluación no necesitamos calcular gradientes ni actualizar pesos,
  # por lo que no usamos tf.GradientTape.
  # 'training=False' asegura que las capas se comporten en modo de inferencia.
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  # Actualizamos las métricas de prueba.
  test_loss(t_loss)
  test_accuracy(labels, predictions)


# -----------------------------------------------------------------------------
# 6. BUCLE PRINCIPAL DE ENTRENAMIENTO
# -----------------------------------------------------------------------------
# Definimos el número de épocas. Una época es una pasada completa por todo el dataset de entrenamiento.
EPOCHS = 5

for epoch in range(EPOCHS):
  # Es crucial reiniciar el estado de nuestras métricas al comienzo de cada época.
  # Si no lo hiciéramos, los resultados se promediarían entre todas las épocas.
  train_loss.reset_state()
  train_accuracy.reset_state()
  test_loss.reset_state()
  test_accuracy.reset_state()

  # Bucle de entrenamiento: Itera sobre cada lote del dataset de entrenamiento.
  for images, labels in train_ds:
    train_step(images, labels)

  # Bucle de validación: Itera sobre cada lote del dataset de prueba.
  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  # Al final de cada época, imprimimos el resumen del rendimiento.
  # .result() calcula el valor final de la métrica acumulada durante la época.
  print(
    f'Época {epoch + 1}, '
    f'Pérdida (Loss): {train_loss.result():.4f}, '
    f'Precisión (Accuracy): {train_accuracy.result() * 100:.2f}%, '
    f'Pérdida de Prueba (Test Loss): {test_loss.result():.4f}, '
    f'Precisión de Prueba (Test Accuracy): {test_accuracy.result() * 100:.2f}%'
  )
```
-------------------------------------------------------------------------------------------

## Ejemplo_clasificacion_imagenes, Predecir una imagen de moda

## TensorFlow and tf.keras
    import tensorflow as tf

## Helper libraries
```Python
  import numpy as np
  import matplotlib.pyplot as plt

  print(tf.__version__)

  fashion_mnist = tf.keras.datasets.fashion_mnist

  (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

## Explorando datos
    train_images.shape
    len(train_labels)
    train_labels
    test_images.shape
    len(test_labels)

## Preprocesar los datos
El set de datos debe ser pre-procesada antes de entrenar la red. Si usted inspecciona la primera imagen en el set de entrenamiento, va a encontrar que los valores de los pixeles estan entre 0 y 255:
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

Escale estos valores en un rango de 0 a 1 antes de alimentarlos al modelo de la red neuronal. Para hacero, divida los valores por 255. Es importante que el training set y el testing set se pre-procesen de la misma forma:
    train_images = train_images / 255.0
    test_images = test_images / 255.0

Para verificar que el set de datos esta en el formato adecuado y que estan listos para construir y entrenar la red, vamos a desplegar las primeras 25 imagenes de el training set y despleguemos el nombre de cada clase debajo de cada imagen.

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

## Generar modelo
```Python
Configurar las Capas
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

Compile el modelo
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

Para comenzar a entrenar, llame el metodo model.fit, es llamado asi por que fit (ajusta) el modelo a el set de datos de entrenamiento:
    model.fit(train_images, train_labels, epochs=10)

## Evaluar precision

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

## Hacer predicciones

    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    predictions[0]
    np.argmax(predictions[0])
    test_labels[0]


Grafique esto para poder ver todo el set de la prediccion de las 10 clases.
    def plot_image(i, predictions_array, true_label, img):
      true_label, img = true_label[i], img[i]
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])

      plt.imshow(img, cmap=plt.cm.binary)
      
      predicted_label = np.argmax(predictions_array)
      if predicted_label == true_label:
        color = 'blue'
     else:
        color = 'red'

      plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

    def plot_value_array(i, predictions_array, true_label):
      true_label = true_label[i]
      plt.grid(False)
      lt.xticks(range(10))
      plt.yticks([])
      thisplot = plt.bar(range(10), predictions_array, color="#777777")
      plt.ylim([0, 1])
      predicted_label = np.argmax(predictions_array)

      thisplot[predicted_label].set_color('red')
      thisplot[true_label].set_color('blue')

## Verificar predicciones

    i = 0
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()

    i = 12
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()


Vamos a graficar multiples imagenes con sus predicciones. Notese que el modelo puede estar equivocado aun cuando tiene mucha confianza.
## Plot the first X test images, their predicted labels, and the true labels.
## Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      plot_image(i, predictions[i], test_labels, test_images)
      plt.subplot(num_rows, 2*num_cols, 2*i+2)
      plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()

## Usar modelo entrenado

## Grab an image from the test dataset.
    img = test_images[1]

    print(img.shape)

## Add the image to a batch where it's the only member.
Los modelos de tf.keras son optimizados sobre batch o bloques, o coleciones de ejemplos por vez. De acuerdo a esto, aunque use una unica imagen toca agregarla a una lista:
    img = (np.expand_dims(img,0))

## Ahora prediga la etiqueta correcta para esta imagen:
    print(img.shape)

    predictions_single = probability_model.predict(img)

    print(predictions_single)

    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()

    np.argmax(predictions_single[0])
```

