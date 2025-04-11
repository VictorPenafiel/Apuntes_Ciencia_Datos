### Escalamiento asegura que las variables tengan un impacto equilibrado en los modelos.

##  Normalización (Min-Max Scaling)
Transforma los datos a un rango específico (por defecto, [0, 1]).
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
    
    from sklearn.preprocessing import StandardScaler

    # Datos de ejemplo
    data = np.array([[10, 2], [5, 8], [3, 6]])

    # Crear y aplicar scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    print("Datos originales:\n", data)
    print("Datos estandarizados (μ=0, σ=1):\n", scaled_data)