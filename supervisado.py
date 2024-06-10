import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Crear el dataset
data = {
    'Punto Origen': ['A', 'A', 'B', 'B', 'C'],
    'Punto Destino': ['B', 'C', 'C', 'D', 'D'],
    'Distancia': [4, 2, 5, 10, 3]
}
df = pd.DataFrame(data)
print("Dataset inicial:")
print(df)

# Codificar las variables categóricas
label_encoder = LabelEncoder()
df['Punto Origen'] = label_encoder.fit_transform(df['Punto Origen'])
df['Punto Destino'] = label_encoder.fit_transform(df['Punto Destino'])
print("\nDataset después de la codificación:")
print(df)

# Dividir los datos en características (X) y la variable objetivo (y)
X = df[['Punto Origen', 'Punto Destino']]
y = df['Distancia']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de árbol de decisión
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print("\nPredicciones del modelo:", y_pred)
print("Error cuadrático medio:", mse)
