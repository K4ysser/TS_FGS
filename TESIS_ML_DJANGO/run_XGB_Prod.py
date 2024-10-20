import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle

# Cargar datos
data = pd.read_csv('Formato.csv')  # Ajusta el nombre del archivo si es necesario

# Crear un diccionario para los meses
meses = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}

# Convertir los nombres de los meses en números
data['MESES_NUM'] = data['MESES'].map(meses)

# Agregar características cíclicas
data['MES_SIN'] = np.sin(2 * np.pi * data['MESES_NUM'] / 12)
data['MES_COS'] = np.cos(2 * np.pi * data['MESES_NUM'] / 12)

# Seleccionar las características y el objetivo
X = data[['ANIO', 'MES_SIN', 'MES_COS', 'TOTAL_VENTAS_EN_SOLES']]
y = data['TOTAL_VENTAS']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir los parámetros para la búsqueda de hiperparámetros
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3, 5]
}

# Crear el modelo base
base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Realizar búsqueda de hiperparámetros
grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

# Evaluar el modelo
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"MAE: {mae:.2f}\n")

# Guardar el modelo y el escalador como archivos .sav
pickle.dump(best_model, open("modelo_productividad_xgboost.sav", "wb"))
pickle.dump(scaler, open("scaler_productividad_xgboost.sav", "wb"))

# Función para hacer predicciones
def predict_total_ventas(anio, avena_por_mes):
    model = pickle.load(open("modelo_productividad_xgboost.sav", "rb"))
    scaler = pickle.load(open("scaler_productividad_xgboost.sav", "rb"))
    
    predictions = []
    for mes, total_venta_en_soles in enumerate(avena_por_mes, start=1):
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)
        
        input_data = pd.DataFrame([[anio, mes_sin, mes_cos, total_venta_en_soles]], 
                                  columns=['ANIO', 'MES_SIN', 'MES_COS', 'TOTAL_VENTAS_EN_SOLES'])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        predictions.append(prediction)
    
    return predictions

# Ejemplo de uso
anio_prediccion = 2024
avena_por_mes = [26791, 29417, 32550, 33372, 31989, 26213, 39217, 37939, 32550, 40300, 38565, 60320]

predicciones = predict_total_ventas(anio_prediccion, avena_por_mes)

# Crear un DataFrame con las predicciones
predictions_df = pd.DataFrame({
    'MES': range(1, 13),
    'TOTAL_VENTAS_EN_SOLES': avena_por_mes,
    'Predicción VENTAS TOTALES EN CANTIDAD': predicciones
})

#predictions_df['Predicción VENTAS TOTALES EN CANTIDAD'] = predictions_df['Predicción VENTAS TOTALES EN CANTIDAD'].map('{:.2f}'.format)
predictions_df['Predicción VENTAS TOTALES EN CANTIDAD'] = predictions_df['Predicción VENTAS TOTALES EN CANTIDAD'].round(0).astype(int)


print(predictions_df)

# Graficar las predicciones
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['MES'], predictions_df['Predicción VENTAS TOTALES EN CANTIDAD'].astype(float), marker='o')
plt.title(f'Predicciones de Ventas por Mes ({anio_prediccion})')
plt.xlabel('Mes')
plt.ylabel('Ventas Predichas')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

# Imprimir importancia de características
importances = best_model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance}")