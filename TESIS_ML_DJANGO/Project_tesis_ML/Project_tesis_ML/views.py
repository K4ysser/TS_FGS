from django.shortcuts import render
import pickle
import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import MinMaxScaler
from django.http import JsonResponse


# our home page view

def inicio(request):
    return render(request, 'inicio.html')


# Función para obtener predicciones - Efciecia 
def getPredictions(anio, avena_por_mes):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "avena_sales_prediction_model.sav")
    scaler_path = os.path.join(base_dir, "avena_sales_scaler.sav")
    
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    
    predictions = []
    for mes, total_avena in enumerate(avena_por_mes, start=1):
        mes_sin = np.sin(2 * np.pi * mes / 12)
        mes_cos = np.cos(2 * np.pi * mes / 12)
        
        input_data = pd.DataFrame([[anio, mes_sin, mes_cos, total_avena]], 
                                  columns=['ANIO', 'MES_SIN', 'MES_COS', 'TOTAL_AVENA'])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        predictions.append(prediction)
    
    return predictions


def prediccion_data(request):
    months = range(1, 13)
    return render(request, 'prediccion_data.html', {'months': months})

def result(request):
    if request.method == 'POST':
        anio = int(request.POST['anio'])
        avena_por_mes = [int(request.POST[f'avena_mes_{i}']) for i in range(1, 13)]
        
        predictions = getPredictions(anio, avena_por_mes)
        
        result = {
            'anio': anio,
            'avena_por_mes': avena_por_mes,
            'predictions': predictions,
        }
        
        return JsonResponse(result)
    else:
        return render(request, 'prediccion_data.html')


##VENTAS PRODUCTOS EFICIENCIA -

def ventas_productos_soles_2019(request):
    data = [
        {"ANIO": 2019, "MESES": "Enero",   "TOTAL_VENTAS_EN_SOLES" : 208329.61},
        {"ANIO": 2019, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 173151.12},
        {"ANIO": 2019, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 277780.46},
        {"ANIO": 2019, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 239057.74},
        {"ANIO": 2019, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 218411.74},
        {"ANIO": 2019, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 181765.09},
        {"ANIO": 2019, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 199520.65},
        {"ANIO": 2019, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 237784.57},
        {"ANIO": 2019, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 230730.52},
        {"ANIO": 2019, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 261194.84},
        {"ANIO": 2019, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 168058.44},
        {"ANIO": 2019, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 149064.12},
        ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_soles_2019.html', {'json_data': json_data})

def ventas_productos_soles_2020(request):
    data = [
        {"ANIO": 2020, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 36349.70},
        {"ANIO": 2020, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 31670.58},
        {"ANIO": 2020, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 44626.70},
        {"ANIO": 2020, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 35555.00},
        {"ANIO": 2020, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 35423.00},
        {"ANIO": 2020, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 30506.00},
        {"ANIO": 2020, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 40202.00},
        {"ANIO": 2020, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 45550.00},
        {"ANIO": 2020, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 35000.00},
        {"ANIO": 2020, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 35552.00},
        {"ANIO": 2020, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 35666.00},
        {"ANIO": 2020, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 36050.00},
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_soles_2020.html', {'json_data': json_data})

def ventas_productos_soles_2021(request):
    data = [
        {"ANIO": 2021, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 41709.00},
        {"ANIO": 2021, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 45550.00},
        {"ANIO": 2021, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 39954.00},
        {"ANIO": 2021, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 40979.00},
        {"ANIO": 2021, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 47833.00},
        {"ANIO": 2021, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 49761.00},
        {"ANIO": 2021, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 40720.00},
        {"ANIO": 2021, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 39599.00},
        {"ANIO": 2021, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 35455.00},
        {"ANIO": 2021, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 32000.00},
        {"ANIO": 2021, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 29293.00},
        {"ANIO": 2021, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 60000.00},    
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_soles_2021.html', {'json_data': json_data})

def ventas_productos_soles_2022(request):
    data = [
        {"ANIO": 2022, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 26463.00},
        {"ANIO": 2022, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 27633.00},
        {"ANIO": 2022, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 17143.00},
        {"ANIO": 2022, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 17330.00},
        {"ANIO": 2022, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 44182.00},
        {"ANIO": 2022, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 25567.00},
        {"ANIO": 2022, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 26973.00},
        {"ANIO": 2022, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 31246.00},
        {"ANIO": 2022, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 30446.00},
        {"ANIO": 2022, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 30417.00},
        {"ANIO": 2022, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 36886.00},
        {"ANIO": 2022, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 70000.00},        
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_soles_2022.html', {'json_data': json_data})


def ventas_productos_soles_2023(request):
    data = [
        {"ANIO": 2023, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES":29869.00},
        {"ANIO": 2023, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES":25525.00},
        {"ANIO": 2023, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES":23260.00},
        {"ANIO": 2023, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES":30349.00},
        {"ANIO": 2023, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES":32734.00},
        {"ANIO": 2023, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES":32567.00},
        {"ANIO": 2023, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES":30446.00},
        {"ANIO": 2023, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES":47502.00},
        {"ANIO": 2023, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES":43583.00},
        {"ANIO": 2023, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES":42350.00},
        {"ANIO": 2023, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES":32411.00},
        {"ANIO": 2023, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES":80250.00},        
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_soles_2023.html', {'json_data': json_data}) 



# Datos para 209,2020,2021,2022,2023

def presentacion_graficos(request, year):
    # Aquí deberías tener lógica para obtener los datos del año específico
    # Por ejemplo:
    data = {
           '2019': [  # Datos para 2019         
                  {"ANIO": 2019, "MESES": "Enero",   "TOTAL_VENTAS_EN_SOLES" : 208329.61},
                  {"ANIO": 2019, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 173151.12},
                  {"ANIO": 2019, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 277780.46},
                  {"ANIO": 2019, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 239057.74},
                  {"ANIO": 2019, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 218411.74},
                  {"ANIO": 2019, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 181765.09},
                  {"ANIO": 2019, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 199520.65},
                  {"ANIO": 2019, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 237784.57},
                  {"ANIO": 2019, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 230730.52},
                  {"ANIO": 2019, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 261194.84},
                  {"ANIO": 2019, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 168058.44},
                  {"ANIO": 2019, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 149064.12},     
                ],  
        '2020': [ # Datos para 2020
                  {"ANIO": 2020, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 36349.70},
                  {"ANIO": 2020, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 31670.58},
                  {"ANIO": 2020, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 44626.70},
                  {"ANIO": 2020, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 35555.00},
                  {"ANIO": 2020, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 35423.00},
                  {"ANIO": 2020, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 30506.00},
                  {"ANIO": 2020, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 40202.00},
                  {"ANIO": 2020, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 45550.00},
                  {"ANIO": 2020, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 35000.00},
                  {"ANIO": 2020, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 35552.00},
                  {"ANIO": 2020, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 35666.00},
                  {"ANIO": 2020, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 36050.00},                                                            
                ],  
        '2021': [ # Datos para 2021
                {"ANIO": 2021, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 41709.00},
                {"ANIO": 2021, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 45550.00},
                {"ANIO": 2021, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 39954.00},
                {"ANIO": 2021, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 40979.00},
                {"ANIO": 2021, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 47833.00},
                {"ANIO": 2021, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 49761.00},
                {"ANIO": 2021, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 40720.00},
                {"ANIO": 2021, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 39599.00},
                {"ANIO": 2021, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 35455.00},
                {"ANIO": 2021, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 32000.00},
                {"ANIO": 2021, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 29293.00},
                {"ANIO": 2021, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 60000.00},                                
                ],  
        '2022': [ # Datos para 2022
                {"ANIO": 2022, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES": 26463.00},
                {"ANIO": 2022, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES": 27633.00},
                {"ANIO": 2022, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES": 17143.00},
                {"ANIO": 2022, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES": 17330.00},
                {"ANIO": 2022, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES": 44182.00},
                {"ANIO": 2022, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES": 25567.00},
                {"ANIO": 2022, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES": 26973.00},
                {"ANIO": 2022, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES": 31246.00},
                {"ANIO": 2022, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES": 30446.00},
                {"ANIO": 2022, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES": 30417.00},
                {"ANIO": 2022, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES": 36886.00},
                {"ANIO": 2022, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES": 70000.00},            
                ], 
        
        '2023': [ # Datos para 2023
                {"ANIO": 2023, "MESES": "Febrero",  "TOTAL_VENTAS_EN_SOLES":29869.00},
                {"ANIO": 2023, "MESES": "Enero",    "TOTAL_VENTAS_EN_SOLES":25525.00},
                {"ANIO": 2023, "MESES": "Marzo",    "TOTAL_VENTAS_EN_SOLES":23260.00},
                {"ANIO": 2023, "MESES": "Abril",    "TOTAL_VENTAS_EN_SOLES":30349.00},
                {"ANIO": 2023, "MESES": "Mayo",     "TOTAL_VENTAS_EN_SOLES":32734.00},
                {"ANIO": 2023, "MESES": "Junio",    "TOTAL_VENTAS_EN_SOLES":32567.00},
                {"ANIO": 2023, "MESES": "Julio",    "TOTAL_VENTAS_EN_SOLES":30446.00},
                {"ANIO": 2023, "MESES": "Agosto",   "TOTAL_VENTAS_EN_SOLES":47502.00},
                {"ANIO": 2023, "MESES": "Setiembre","TOTAL_VENTAS_EN_SOLES":43583.00},
                {"ANIO": 2023, "MESES": "Octubre",  "TOTAL_VENTAS_EN_SOLES":42350.00},
                {"ANIO": 2023, "MESES": "Noviembre","TOTAL_VENTAS_EN_SOLES":32411.00},
                {"ANIO": 2023, "MESES": "Diciembre","TOTAL_VENTAS_EN_SOLES":80250.00},            
                ], 
            }
    
    return JsonResponse(data.get(str(year), []), safe=False)



##VENTAS EN SOLES PRODUCTIVIDAD -
  
##Prediccion para PRODUCTIVIDAD
  
def getPredictions_produccion(anio, avena_por_mes):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "modelo_productividad.sav")
    scaler_path = os.path.join(base_dir, "scaler_productividad.sav")
    
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    
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


def prediccion_data_prod(request):
    months = range(1, 13)
    return render(request, 'prediccion_data_prod.html', {'months': months})

def result_prod(request):
    if request.method == 'POST':
        anio = int(request.POST['anio'])
        avena_por_mes = [int(request.POST[f'avena_mes_{i}']) for i in range(1, 13)]
        
        predictions = getPredictions_produccion(anio, avena_por_mes)
        
        result = {
            'anio': anio,
            'avena_por_mes': avena_por_mes,
            'predictions': predictions,
        }
        
        return JsonResponse(result)
    else:
        return render(request, 'prediccion_data_prod.html')


  
  
  
  
# Datos para 209,2020,2021,2022,2023

def presentacion_graficos_productividad(request, year):
    # Aquí deberías tener lógica para obtener los datos del año específico
    # Por ejemplo:
    data = {
           '2019': [  # Datos para 2019         
                  {"ANIO": 2019, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 6851},
                  {"ANIO": 2019, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 7830},
                  {"ANIO": 2019, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 10391},
                  {"ANIO": 2019, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 8944},
                  {"ANIO": 2019, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 9147},
                  {"ANIO": 2019, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 8827},
                  {"ANIO": 2019, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 5333},
                  {"ANIO": 2019, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 12540},
                  {"ANIO": 2019, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 9976},
                  {"ANIO": 2019, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 11249},
                  {"ANIO": 2019, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 5179},
                  {"ANIO": 2019, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 6146},     
                ],  
        '2020': [ # Datos para 2020
                  {"ANIO": 2020, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 650},
                  {"ANIO": 2020, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 652},
                  {"ANIO": 2020, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 1100},
                  {"ANIO": 2020, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 612},
                  {"ANIO": 2020, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 655},
                  {"ANIO": 2020, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 956},
                  {"ANIO": 2020, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 1100},
                  {"ANIO": 2020, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 955},
                  {"ANIO": 2020, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 1190},
                  {"ANIO": 2020, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 938},
                  {"ANIO": 2020, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 752},
                  {"ANIO": 2020, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 950},                                                              
                ],     
        '2021': [ # Datos para 2021   
                  {"ANIO": 2021, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 704},
                  {"ANIO": 2021, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 1167},
                  {"ANIO": 2021, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 1077},
                  {"ANIO": 2021, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 850},
                  {"ANIO": 2021, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 927},
                  {"ANIO": 2021, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 1300},
                  {"ANIO": 2021, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 1181},
                  {"ANIO": 2021, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 634},
                  {"ANIO": 2021, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 850},
                  {"ANIO": 2021, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 750},
                  {"ANIO": 2021, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 867},
                  {"ANIO": 2021, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 1567},                               
                ],     
        '2022': [ # Datos para 2022   
                  {"ANIO": 2022, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 422},
                  {"ANIO": 2022, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 709},
                  {"ANIO": 2022, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 609},
                  {"ANIO": 2022, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 517},
                  {"ANIO": 2022, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 966},
                  {"ANIO": 2022, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 552},
                  {"ANIO": 2022, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 714},
                  {"ANIO": 2022, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 755},
                  {"ANIO": 2022, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 855},
                  {"ANIO": 2022, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 740},
                  {"ANIO": 2022, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 727},
                  {"ANIO": 2022, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 2000},           
                ],    
        '2023': [ # Datos para 2023   
                  {"ANIO": 2023, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 707},
                  {"ANIO": 2023, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 496},
                  {"ANIO": 2023, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 593},
                  {"ANIO": 2023, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 602},
                  {"ANIO": 2023, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 754},
                  {"ANIO": 2023, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 750},
                  {"ANIO": 2023, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 739},
                  {"ANIO": 2023, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 837},
                  {"ANIO": 2023, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 764},
                  {"ANIO": 2023, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 1100},
                  {"ANIO": 2023, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 666},
                  {"ANIO": 2023, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 2400},           
                ],    
            }    
    return JsonResponse(data.get(str(year), []), safe=False)
  
  
def ventas_productos_cantidad_2019(request):
    data = [
        {"ANIO": 2019, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 6851},
        {"ANIO": 2019, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 7830},
        {"ANIO": 2019, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 10391},
        {"ANIO": 2019, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 8944},
        {"ANIO": 2019, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 9147},
        {"ANIO": 2019, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 8827},
        {"ANIO": 2019, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 5333},
        {"ANIO": 2019, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 12540},
        {"ANIO": 2019, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 9976},
        {"ANIO": 2019, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 11249},
        {"ANIO": 2019, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 5179},
        {"ANIO": 2019, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 6146},    
    
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_cantidad_2019.html', {'json_data': json_data})
# 
def ventas_productos_cantidad_2020(request):
    data = [
         {"ANIO": 2020, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 650},
         {"ANIO": 2020, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 652},
         {"ANIO": 2020, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 1100},
         {"ANIO": 2020, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 612},
         {"ANIO": 2020, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 655},
         {"ANIO": 2020, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 956},
         {"ANIO": 2020, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 1100},
         {"ANIO": 2020, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 955},
         {"ANIO": 2020, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 1190},
         {"ANIO": 2020, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 938},
         {"ANIO": 2020, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 752},
         {"ANIO": 2020, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 950},    
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_cantidad_2020.html', {'json_data': json_data})

def ventas_productos_cantidad_2021(request):
    data = [
         {"ANIO": 2021, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 704},
         {"ANIO": 2021, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 1167},
         {"ANIO": 2021, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 1077},
         {"ANIO": 2021, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 850},
         {"ANIO": 2021, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 927},
         {"ANIO": 2021, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 1300},
         {"ANIO": 2021, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 1181},
         {"ANIO": 2021, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 634},
         {"ANIO": 2021, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 850},
         {"ANIO": 2021, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 750},
         {"ANIO": 2021, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 867},
         {"ANIO": 2021, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 1567},    
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_cantidad_2021.html', {'json_data': json_data})

  
def ventas_productos_cantidad_2022(request):
    data = [
         {"ANIO": 2022, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 422},
         {"ANIO": 2022, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 709},
         {"ANIO": 2022, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 609},
         {"ANIO": 2022, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 517},
         {"ANIO": 2022, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 966},
         {"ANIO": 2022, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 552},
         {"ANIO": 2022, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 714},
         {"ANIO": 2022, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 755},
         {"ANIO": 2022, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 855},
         {"ANIO": 2022, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 740},
         {"ANIO": 2022, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 727},
         {"ANIO": 2022, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 2000},    
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_cantidad_2022.html', {'json_data': json_data})

def ventas_productos_cantidad_2023(request):
    data = [
         {"ANIO": 2023, "MESES": "Febrero",  "TOTAL_VENTAS_CANTIDAD": 707},
         {"ANIO": 2023, "MESES": "Enero",    "TOTAL_VENTAS_CANTIDAD": 496},
         {"ANIO": 2023, "MESES": "Marzo",    "TOTAL_VENTAS_CANTIDAD": 593},
         {"ANIO": 2023, "MESES": "Abril",    "TOTAL_VENTAS_CANTIDAD": 602},
         {"ANIO": 2023, "MESES": "Mayo",     "TOTAL_VENTAS_CANTIDAD": 754},
         {"ANIO": 2023, "MESES": "Junio",    "TOTAL_VENTAS_CANTIDAD": 750},
         {"ANIO": 2023, "MESES": "Julio",    "TOTAL_VENTAS_CANTIDAD": 739},
         {"ANIO": 2023, "MESES": "Agosto",   "TOTAL_VENTAS_CANTIDAD": 837},
         {"ANIO": 2023, "MESES": "Setiembre","TOTAL_VENTAS_CANTIDAD": 764},
         {"ANIO": 2023, "MESES": "Octubre",  "TOTAL_VENTAS_CANTIDAD": 1100},
         {"ANIO": 2023, "MESES": "Noviembre","TOTAL_VENTAS_CANTIDAD": 666},
         {"ANIO": 2023, "MESES": "Diciembre","TOTAL_VENTAS_CANTIDAD": 2400},        
    ]

    # Convert data to JSON
    json_data = json.dumps(data)
    
    return render(request, 'ventas_productos_cantidad_2023.html', {'json_data': json_data})



                  











