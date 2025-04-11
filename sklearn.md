# Escalamiento 
Asegura que las variables tengan un impacto equilibrado en los modelos.

##  Normalización (Min-Max Scaling)
Transforma los datos a un rango específico (por defecto, [0, 1]).
Usos: Algoritmos como KNN y redes neuronales donde los datos deben estar en un rango fijo.

    from sklearn.preprocessing import MinMaxScaler
    import numpy as np

    # Datos de ejemplo
    data = np.array([[10, 2], [5, 8], [3, 6]])

    # Crear y aplicar scaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    print("Datos originales:\n", data)
    print("Datos normalizados (0-1):\n", scaled_data)

## Estandarización (Z-Score Scaling)
Transforma los datos para que tengan media (μ = 0) y desviación estándar (σ = 1).
Usos: SVM, Regresión Lineal, PCA y algoritmos que asumen distribución normal.
    
    from sklearn.preprocessing import StandardScaler

    # Datos de ejemplo
    data = np.array([[10, 2], [5, 8], [3, 6]])

    # Crear y aplicar scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    print("Datos originales:\n", data)
    print("Datos estandarizados (μ=0, σ=1):\n", scaled_data)

## Robust Scaling (Escalado Robusto)
Utiliza los percentiles 25 y 75 (rango intercuartílico) para reducir el impacto de outliers.
Usos: Datos con outliers significativos.

    from sklearn.preprocessing import RobustScaler

    # Datos con outlier
    data = np.array([[10, 2], [5, 8], [3, 6], [100, 1]])

    # Crear y aplicar scaler
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)

    print("Datos originales:\n", data)
    print("Datos escalados robustos:\n", scaled_data)

## Normalización L1 y L2
L1 (Manhattan): Escala los datos para que la suma de los valores absolutos sea 1.
L2 (Euclidiana): Escala para que la suma de los cuadrados sea 1.
Usos: NLP, K-Means. Text mining (TF-IDF) o cuando se requiere vectores unitarios.

    from sklearn.preprocessing import normalize

    # Normalización L1 (suma de valores absolutos = 1)
    l1_scaled = normalize(data, norm='l1')

    # Normalización L2 (suma de cuadrados = 1)
    l2_scaled = normalize(data, norm='l2')



----------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------------------------