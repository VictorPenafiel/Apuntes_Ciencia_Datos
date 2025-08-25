
-----

## Fundamentos de Redes Neuronales Convolucionales (CNNs)

### ¿Qué es una Red Neuronal Convolucional (CNN)?

Una **Red Neuronal Convolucional (CNN o ConvNet)** es un tipo especializado de red neuronal diseñada para procesar datos que tienen una estructura de rejilla, como las imágenes. A diferencia de las redes neuronales tradicionales (MLP), que tratan a los datos de entrada como un vector plano, las CNNs asumen explícitamente que las entradas son imágenes, lo que les permite identificar patrones espaciales.

La intuición es similar a cómo los humanos reconocemos objetos: no vemos un rostro como un conjunto de píxeles, sino que identificamos características como ojos, nariz y boca. A su vez, estas características están compuestas por patrones más simples como líneas, curvas y texturas. Las CNNs emulan este proceso a través de sus capas, aprendiendo jerarquías de características cada vez más complejas.

### Componentes Clave de una CNN

Las CNNs se construyen principalmente con dos tipos de capas especiales: convolucionales y de agrupamiento (pooling).

#### 1\. Capa Convolucional (`Conv2D`)

Es el bloque de construcción principal. Su objetivo es detectar características locales en la imagen.

  * **El Kernel o Filtro:** Es una pequeña matriz de pesos (por ejemplo, 3x3 o 5x5) que actúa como un "detector de características". Este filtro se desliza (o *convoluciona*) sobre toda la imagen de entrada. En cada posición, realiza un producto punto entre sus valores y los píxeles de la imagen que cubre. El resultado es un único valor en un nuevo mapa de características de salida.
      * *Ejemplo práctico:* Un filtro como el **operador de Sobel** está diseñado para detectar bordes verticales u horizontales.
  * **Parámetros Clave:**
      * **Filters (Filtros):** El número de kernels que la capa aprenderá. Cada filtro se especializa en detectar un patrón diferente (un filtro para bordes verticales, otro para una textura específica, etc.).
      * **Kernel Size (Tamaño del Kernel):** Las dimensiones del filtro (ej. `(5, 5)`).
      * **Stride (Paso):** El número de píxeles que el filtro se desplaza en cada paso. Un stride de 1 es lo más común.
      * **Padding (Relleno):** Añadir ceros alrededor de la imagen de entrada para controlar el tamaño del mapa de características de salida y permitir que el kernel procese los bordes de la imagen adecuadamente.
  * **Ventajas:**
      * **Parameter Sharing (Parámetros Compartidos):** El mismo filtro se usa en toda la imagen, reduciendo drásticamente la cantidad de parámetros a aprender.
      * **Jerarquía Espacial:** Las primeras capas aprenden características simples (líneas), y las capas más profundas combinan estas para aprender patrones complejos (formas, objetos).

#### 2\. Capa de Agrupamiento (`MaxPooling2D`)

Su función principal es reducir la dimensionalidad espacial (el alto y ancho) de los mapas de características.

  * **Operación:** La más común es **Max Pooling**. Funciona deslizando una ventana (ej. 2x2) sobre el mapa de características y seleccionando solo el valor máximo de esa ventana. Esto reduce el tamaño de la representación.
  * **Ventajas:**
      * **Reducción Computacional:** Al reducir el tamaño, disminuye la cantidad de cálculos y parámetros en las capas posteriores.
      * **Invarianza a la Traslación:** Hace que el modelo sea más robusto a pequeñas variaciones en la posición de las características en la imagen. Si un borde se mueve un píxel, es probable que el max pooling siga produciendo la misma salida.

### Construyendo una CNN con Keras: Ejemplo Práctico

Un modelo de CNN típico para clasificación de imágenes sigue esta estructura:

1.  Se aplican uno o más bloques de `Conv2D` y `MaxPooling2D` para extraer características.
2.  La salida del último bloque, que es un tensor 3D (alto, ancho, canales), se "aplana" con una capa `Flatten` para convertirla en un vector 1D.
3.  Este vector se pasa a una o más capas `Dense` (totalmente conectadas) para realizar la clasificación final.

Modelo clásico para el dataset Fashion MNIST:

```python
model = models.Sequential([
    # Primer bloque: extrae características iniciales
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Segundo bloque: aprende patrones más complejos
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Aplanado para pasar a la red de clasificación
    layers.Flatten(),
    
    # Capa Densa final con 10 neuronas (una para cada clase) y activación softmax
    layers.Dense(10, activation='softmax')
])
```

Este modelo se compila con un optimizador y una función de pérdida, se entrena con las imágenes y sus etiquetas, y finalmente se utiliza para hacer predicciones.

-----


### Kernels (Fundamentos)


```python

# --- Otros filtros comunes ---

# Kernel de realce/enfoque (sharpen)
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

# Kernel de desenfoque (blur)
blur_kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]) / 9.0

# Aplicamos los filtros
sharpened_image = cv2.filter2D(gray, -1, sharpen_kernel)
blurred_image = cv2.filter2D(gray, -1, blur_kernel)

# Visualizamos los resultados
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title("Imagen Original en Gris")

plt.subplot(1, 3, 2)
plt.imshow(sharpened_image, cmap='gray')
plt.title("Filtro de Enfoque (Sharpen)")

plt.subplot(1, 3, 3)
plt.imshow(blurred_image, cmap='gray')
plt.title("Filtro de Desenfoque (Blur)")

plt.show()
```

### Capa_Maxpool (Visualización de Capas)


```python
# Celda sugerida para añadir en DL10-Capa_Maxpool.ipynb

# Preparar la imagen para el modelo PyTorch
# Se necesita en formato (batch_size, channels, height, width)
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0)

# Pasar la imagen por el modelo
conv_layer, activated_layer, pooled_layer = model(gray_img_tensor)

# --- Visualización de las capas ---

# 1. Capa Convolucional (antes de la activación)
# Muestra cómo cada filtro detecta diferentes características.
# Los valores pueden ser positivos (blanco) o negativos (negro).
print("Salida de la Capa Convolucional (4 filtros)")
viz_layer(conv_layer)

# 2. Capa de Activación (ReLU)
# La función ReLU elimina todos los valores negativos (los pone a cero).
# Nota cómo las áreas negras se expanden.
print("Salida después de la Activación ReLU")
viz_layer(activated_layer)

# 3. Capa de Max Pooling
# La imagen se reduce a la mitad de su tamaño (mira los ejes).
# Se conservan las características más intensas (píxeles más brillantes).
print("Salida después de Max Pooling (tamaño reducido)")
viz_layer(pooled_layer)
```


#### **Actividad 1: Modelo más grande**

Duplicamos las neuronas y añadimos una capa densa.

```python
# Actividad 1: Modelo más grande
model_2 = models.Sequential([
    layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)), # Neuronas duplicadas
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (5, 5), activation='relu'), # Neuronas duplicadas
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'), # Capa densa añadida
    layers.Dense(10, activation='softmax')
])

model_2.compile(optimizer='sgd',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

print("--- Modelo 2 (Más Grande) ---")
model_2.summary()

# Entrenamos el modelo
history_2 = model_2.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
```

#### **Actividad 2: Cambiar Optimizador a `adam`**

Usamos el modelo base con el optimizador `adam`.

```python
# Actividad 2: Optimizador Adam
# Re-creamos el modelo base para no usar el ya entrenado
model_base = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model_base.compile(optimizer='adam', # Cambiamos a Adam
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

print("\n--- Modelo Base con Optimizador Adam ---")
model_base.summary()

# Entrenamos el modelo
history_base_adam = model_base.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
```

**Observación didáctica:** El optimizador `adam` generalmente converge más rápido y a mejores resultados que `sgd` en muchos problemas, lo que se debería notar en la precisión final.

####  Modelo Complejo con `BatchNormalization` y `Dropout`**

Estos tres puntos construyen un modelo más robusto y lo entrenan por más épocas.

```python
# Actividades 3, 4 y 5: Modelo Complejo
from tensorflow.keras.layers import BatchNormalization, Dropout

model_complex = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(), # Estabiliza y acelera el entrenamiento
    layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2), # Regularización para evitar sobreajuste

    layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Para la actividad 6, podrías ajustar los hiperparámetros de Adam así:
# optimizer_custom = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

model_complex.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

print("\n--- Modelo Complejo con Batch Normalization y Dropout ---")
model_complex.summary()

# Entrenamos por más épocas (Actividad 4 y 5)
history_complex = model_complex.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
```

#### Visualizar el Historial de Entrenamiento**

 Permite diagnosticar el sobreajuste (overfitting) y entender el rendimiento del modelo.

```python
# Celda sugerida para añadir al final de DL11-CNN.ipynb

def plot_training_history(history, title):
    """Función para visualizar las curvas de precisión y pérdida."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Precisión de Entrenamiento')
    plt.plot(epochs, val_acc, 'ro-', label='Precisión de Validación')
    plt.title('Precisión de Entrenamiento y Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Pérdida de Entrenamiento')
    plt.plot(epochs, val_loss, 'ro-', label='Pérdida de Validación')
    plt.title('Pérdida de Entrenamiento y Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.show()

# Visualizamos las curvas para el último modelo entrenado
plot_training_history(history_complex, "Curvas de Entrenamiento del Modelo Complejo")
```
