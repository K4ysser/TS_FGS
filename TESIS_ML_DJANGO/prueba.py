import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle

# Cargar datos
df = pd.read_csv("Formato.csv")

# Diccionario para mapear los meses a valores numéricos
meses_a_numeros = {
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
    'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}

# Crear una nueva columna con los valores numéricos de los meses
df['MESES_NUM'] = df['MESES'].map(meses_a_numeros)

# Agregar características cíclicas
df['MES_SIN'] = np.sin(2 * np.pi * df['MESES_NUM'] / 12)
df['MES_COS'] = np.cos(2 * np.pi * df['MESES_NUM'] / 12)

# Seleccionar las características para el modelo
X = df[['ANIO', 'MES_SIN', 'MES_COS', 'TOTAL_VENTAS_SOLES']]
y = df['TOTAL_VENTAS']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Escalar las características
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir los parámetros para la búsqueda de hiperparámetros
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5]
}

# Crear el modelo base
base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Realizar búsqueda de hiperparámetros
grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Obtener el mejor modelo
best_regressor = grid_search.best_estimator_

# Evaluar el modelo
y_pred = best_regressor.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")

# Guardar el modelo y el escalador como archivos .sav
pickle.dump(best_regressor, open("modelo_ventas_xgboost.sav", "wb"))
pickle.dump(scaler, open("scaler_ventas_xgboost.sav", "wb"))

# Función para hacer predicciones
def predict_total_ventas(anio, ventas_por_mes):
    model = pickle.load(open("modelo_ventas_xgboost.sav", "rb"))
    scaler = pickle.load(open("scaler_ventas_xgboost.sav", "rb"))
    
    predictions = []
    for mes, total_venta_en_soles in enumerate(ventas_por_mes, start=1):
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)
        
        input_data = pd.DataFrame([[anio, mes_sin, mes_cos, total_venta_en_soles]], 
                                  columns=['ANIO', 'MES_SIN', 'MES_COS', 'TOTAL_VENTAS_SOLES'])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        predictions.append(prediction)
    
    return predictions

# Ejemplo de uso
anio_prediccion = 2024
ventas_por_mes = [26791, 29417, 32550, 33372, 31989, 26213, 39217, 37939, 32550, 40300, 38565, 60320]

predicciones = predict_total_ventas(anio_prediccion, ventas_por_mes)

# Crear un DataFrame con las predicciones
predictions_df = pd.DataFrame({
    'MES': range(1, 13),
    'TOTAL_VENTAS_SOLES': ventas_por_mes,
    'Predicción Total de Ventas': predicciones
})

print(predictions_df)

# Graficar las predicciones
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['MES'], predictions_df['Predicción Total de Ventas'], marker='o')
plt.title(f'Predicciones de Ventas por Mes ({anio_prediccion})')
plt.xlabel('Mes')
plt.ylabel('Ventas Predichas')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()

# Imprimir importancia de características
importances = best_regressor.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"{feature}: {importance}")