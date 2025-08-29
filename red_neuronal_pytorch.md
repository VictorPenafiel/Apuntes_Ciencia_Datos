
-----

# 🎓 Creación de Redes Neuronales en PyTorch 


### 🗺️ Hoja de Ruta

1.  **Preparar el Entorno**: Importaremos las librerías necesarias y configuraremos nuestro dispositivo (CPU o GPU).
2.  **Conceptos Clave**: Repasaremos los pilares teóricos antes de escribir una sola línea de código del modelo.
3.  **Construir el Modelo**: Definiremos nuestra primera red neuronal usando la clase `nn.Module`, el estándar en PyTorch.
4.  **El Viaje de los Datos (Forward Pass)**: Pasaremos datos de ejemplo a través de la red para ver qué sucede.
5.  **Interpretar los Resultados**: Transformaremos la salida cruda del modelo en algo útil, como probabilidades.

-----

## 1\. Preparar el Entorno 🛠️


```python
# La librería principal de PyTorch
import torch
# nn es el módulo que contiene todas las herramientas para construir redes neuronales (capas, funciones de activación, etc.)
import torch.nn as nn
# init nos da control sobre cómo se inicializan los pesos del modelo
import torch.nn.init as init

# --- Configuración del Dispositivo ---
# El Deep Learning es computacionalmente intensivo. Las GPUs (Graphics Processing Units)
# pueden acelerar los cálculos masivamente gracias a su arquitectura paralela.
# Este código comprueba si tienes una GPU compatible con CUDA disponible.
# Si la tienes, la usará. Si no, usará la CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dispositivo seleccionado: {device}")
```

-----

## 2\. Conceptos Clave Antes de Construir 🧠


### 2.1 La Importancia de la Forma de los Tensores (Shapes)

Imagina los datos como una hoja de cálculo. Un **tensor** en PyTorch es como esa hoja: tiene filas y columnas.

  * **Filas (`n_muestras`)**: Cada fila es una muestra o ejemplo de tus datos (ej. una imagen, un usuario, una frase).
  * **Columnas (`in_features`)**: Cada columna es una característica o "feature" de esa muestra (ej. el valor de un píxel, la edad de un usuario, el embedding de una palabra).

Las capas de PyTorch, como `nn.Linear(in_features, out_features)`, son muy estrictas con las dimensiones. La `in_features` de una capa **debe coincidir** con el número de características de los datos que recibe.


### 2.2 Inicialización de Pesos

Los **pesos** de una red son los parámetros que el modelo aprende durante el entrenamiento. Son, en esencia, el "conocimiento" de la red. Antes de empezar a entrenar, estos pesos deben tener un valor inicial.

PyTorch lo hace automáticamente con métodos probados, pero es importante saber que una buena inicialización puede ayudar al modelo a aprender más rápido y mejor. El módulo `torch.nn.init` permite personalizar este proceso.

### 2.3 Funciones de Activación: El Interruptor de la Neurona

Las funciones de activación son el componente que introduce la **no-linealidad**. Sin ellas, una red neuronal de 100 capas sería matemáticamente equivalente a una sola capa lineal, ¡incapaz de aprender patrones complejos\!

> **Analogía 💡**: Piensa en ellas como un "regulador de intensidad" (dimmer) para cada neurona. Después de que la neurona suma todas sus entradas ponderadas, la función de activación decide qué tan fuerte es la señal que pasará a la siguiente capa.
>
>   * `torch.relu`: La más popular. Es como un interruptor de "todo o nada" (si la entrada es negativa, se apaga; si es positiva, la deja pasar). Es muy eficiente.
>   * `torch.sigmoid` y `torch.tanh`: Curvas más suaves que "atenúan" la señal. Sigmoid la comprime entre 0 y 1, ideal para probabilidades.

### 2.4 Arquitectura: Capas y Neuronas

  * **Capas**: Más capas (una red "profunda") le permiten aprender jerarquías de características más complejas. Por ejemplo, en imágenes, las primeras capas aprenden bordes, las siguientes aprenden formas y las últimas aprenden objetos.
  * **Neuronas**: Más neuronas por capa le dan al modelo más "capacidad" para aprender en ese nivel de abstracción.

⚠️ **Cuidado**: Un modelo muy grande (demasiadas capas/neuronas) puede "memorizar" los datos de entrenamiento en lugar de aprender patrones generales. Esto se conoce como **sobreajuste (overfitting)**.

-----

## 3\. Construir el Modelo con `nn.Module` 👷‍♀️

En PyTorch, la forma estándar de definir una red es creando una clase que hereda de `nn.Module`. Esta clase tiene dos métodos fundamentales:

1.  `__init__(self)`: El constructor. Aquí **declaras** todas las capas que usarás, como si compraras los bloques de LEGO para tu modelo.
2.  `forward(self, x)`: Aquí **defines la arquitectura**, es decir, cómo se conectan los bloques. Describe el viaje que los datos (`x`) hacen a través de las capas y funciones de activación.

<!-- end list -->

```python
class SimpleNN(nn.Module):
    def __init__(self):
        # Llama al constructor de la clase padre (nn.Module) para que todo se configure correctamente
        super(SimpleNN, self).__init__()

        # --- Declaramos nuestras capas ---
        # "fc" es una abreviatura común para "fully connected" (totalmente conectada)
        # Es una capa de tipo nn.Linear

        # Capa 1: Recibe 2 características de entrada y las transforma en 4.
        self.fc1 = nn.Linear(in_features=2, out_features=4)

        # Capa 2: Recibe las 4 características de la capa anterior y las transforma en 3 (nuestra salida final).
        self.fc2 = nn.Linear(in_features=4, out_features=3)

    def forward(self, x):
        # --- Definimos el flujo de los datos ---

        # 1. Pasa por la primera capa
        x = self.fc1(x)

        # 2. Se aplica la función de activación ReLU
        # Esta es la no-linealidad clave
        x = torch.relu(x)

        # 3. Pasa por la segunda capa para obtener la salida final
        x = self.fc2(x)

        return x

# --- Creamos una instancia del modelo ---
# .to(device) mueve todos los parámetros del modelo (pesos y biases)
# al dispositivo que seleccionamos (CPU o GPU).
model = SimpleNN().to(device)

# Imprimimos la arquitectura para verificarla
print(model)
```

-----

## 4\. El Viaje de los Datos (Forward Pass) 🚀

Un **forward pass** (o propagación hacia adelante) es simplemente el acto de pasar datos a través de la red para obtener una predicción. Aún no estamos entrenando nada, solo observando la salida que produce el modelo con sus pesos iniciales aleatorios.

```python
# Creamos un tensor de datos de ejemplo.
# Shape: (5, 2) -> 5 muestras, cada una con 2 características.
# Esto coincide con el `in_features=2` de nuestra primera capa. ¡Perfecto!
entrada = torch.tensor([[0.5, 1.0],
                        [1.5, 2.0],
                        [2.0, -1.0],
                        [-0.5, 2.0],
                        [1.0, 1.0]], dtype=torch.float32).to(device)

# Para hacer el forward pass, simplemente llamamos al modelo como si fuera una función
# PyTorch se encarga de invocar el método .forward() internamente.
salida_logits = model(entrada)

print("--- Salida Cruda (Logits) ---")
print(salida_logits)
print(f"\nShape de la salida: {salida_logits.shape}") # Será (5, 3), 5 muestras con 3 valores de salida cada una.
```

La salida que vemos se conoce como **logits**. Son puntuaciones crudas, no normalizadas. No son fáciles de interpretar directamente.

-----

## 5\. Interpretar los Resultados: De Logits a Probabilidades 📊

Para que la salida tenga sentido, especialmente en un problema de clasificación, necesitamos convertir esos *logits* en probabilidades. La función perfecta para esto es **Softmax**.

**Softmax** toma un vector de números y lo transforma en un vector de probabilidades donde:

1.  Cada valor está entre 0 y 1.
2.  La suma de todos los valores es igual a 1.

<!-- end list -->

```python
# Aplicamos softmax a la dimensión 1 (la de las características de salida)
# para que las probabilidades de cada muestra sumen 1.
probabilidades = torch.softmax(salida_logits, dim=1)

# Para obtener una predicción final, tomamos la clase con la probabilidad más alta.
# torch.argmax() nos devuelve el índice (la clase) del valor máximo en la dimensión 1.
predicciones = torch.argmax(probabilidades, dim=1)

print("--- Probabilidades ---")
print(probabilidades)
print("\n--- Predicciones Finales (Índice de la clase con mayor probabilidad) ---")
print(predicciones)

# Verifiquemos que las probabilidades de la primera muestra suman 1
print(f"\nSuma de probabilidades de la primera muestra: {torch.sum(probabilidades[0]):.4f}")
```

-----

