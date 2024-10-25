import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pickle

# Suponiendo que 'data' es tu DataFrame con los datos proporcionados
data = pd.read_csv('Formato.csv')  # Ajusta el nombre del archivo si es necesario

# Crear un diccionario para los meses
meses = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}

# Convertir los nombres de los meses en números
data['MESES_NUM'] = data['MESES'].map(meses)

# Seleccionar las características y el objetivo
X = data[['ANIO', 'MESES_NUM', 'TOTAL_VENTAS']]
y = data['TOTAL_VENTAS_EN_SOLES']

# Escalar las características
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Crear y entrenar el modelo XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}\n\n")

# Guardar el modelo y el escalador como archivos .sav
with open("modelo_eficiencia_xgboost.sav", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler_eficiencia_xgboost.sav", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)


# Función para hacer predicciones
def predict_total_productos(anio, total_ventas_por_mes):
    # Cargar el modelo y el escalador
    model = pickle.load(open("modelo_eficiencia_xgboost.sav", "rb"))
    scaler = pickle.load(open("scaler_eficiencia_xgboost.sav", "rb"))

    predictions = []
    for mes, total_venta in enumerate(total_ventas_por_mes, start=1):
        input_data = pd.DataFrame([[anio, mes, total_venta]],  # Usar 'mes' en lugar de 'MES_SIN' y 'MES_COS'
                                  columns=['ANIO', 'MESES_NUM', 'TOTAL_VENTAS'])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        predictions.append(prediction)

    return predictions


# Ejemplo de uso
anio_prediccion = 2024
total_ventas_por_mes = [476, 634, 613, 519, 620, 502, 816, 850, 500, 950, 820, 1500]

predicciones = predict_total_productos(anio_prediccion, total_ventas_por_mes)

# Crear un DataFrame con las predicciones
predictions_df = pd.DataFrame({
    'MES': range(1, 13),
    'TOTAL_VENTAS': total_ventas_por_mes,
    'Predicción VENTAS TOTALES EN CANTIDAD': predicciones
})

# Mostrar el DataFrame con las predicciones
print(predictions_df)
