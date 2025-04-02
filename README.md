# Documentación del Modelo de Predicción de Duración de Incendios

Este proyecto tiene como objetivo predecir la duración de incendios en México utilizando datos históricos, información climática (temperaturas y humedad) y características geográficas. Se implementa un pipeline de machine learning basado en XGBoost, en el que se realiza la limpieza, transformación e ingeniería de características antes del entrenamiento.

## Contenido del Proyecto

- **BD.csv:** Datos históricos de incendios.  
- **temperaturas-mexico.csv:** Temperaturas promedio mensuales por estado.  
- **humedad-mexico.csv:** Valores de humedad promedio mensuales por estado.

## Requisitos

El proyecto utiliza las siguientes librerías de Python:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`

Puedes instalar las dependencias necesarias ejecutando:

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

## Estructura y Flujo del Código

El archivo `model.py` se compone de los siguientes pasos:

### 1. Carga de Datos

Se cargan los datasets `BD.csv`, `temperaturas-mexico.csv` y `humedad-mexico.csv`.  
```python
df_incendios = pd.read_csv('BD.csv', encoding='ISO-8859-1', low_memory=False)
df_temp = pd.read_csv('temperaturas-mexico.csv')
df_humedad = pd.read_csv('humedad-mexico.csv')
```

### 2. Generación de Clusters Geográficos

Se utiliza KMeans para agrupar los incendios en 10 clusters basados en las coordenadas (`latitud_grados` y `longitud_grados`).  
```python
X_geo = df_incendios[['latitud_grados', 'longitud_grados']].dropna()
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df_incendios.loc[X_geo.index, 'Cluster_geo'] = kmeans.fit_predict(X_geo)
```

### 3. Procesamiento de Fechas y Extracción del Mes

Se convierte la columna `Fecha Inicio` a formato datetime y se extrae el mes, utilizando un diccionario para convertir el número del mes a su abreviatura.  
```python
df_incendios['Fecha Inicio'] = pd.to_datetime(df_incendios['Fecha Inicio'], errors='coerce')
df_incendios['Mes'] = df_incendios['Fecha Inicio'].dt.month
```

### 4. Limpieza de Datos de Texto

Se convierten los nombres de estados a minúsculas y se eliminan espacios en blanco para garantizar la consistencia entre datasets.  
```python
df_incendios['Estado'] = df_incendios['Estado'].str.lower().str.strip()
df_temp['Estado'] = df_temp['Estado'].str.lower().str.strip()
df_humedad['Estado'] = df_humedad['Estado'].str.lower().str.strip()
```

### 5. Asignación de Variables Climáticas

Se definen dos funciones, `obtener_temperatura` y `obtener_humedad`, que asignan el valor de temperatura y humedad correspondiente según el estado y mes.  
```python
def obtener_temperatura(row):
    estado = row['Estado']
    mes_abrev_local = meses_abrev.get(row['Mes'], 'Anual')
    if estado in df_temp['Estado'].values and mes_abrev_local in df_temp.columns:
        temp = df_temp.loc[df_temp['Estado'] == estado, mes_abrev_local]
        if not temp.empty:
            return temp.values[0]
    return np.nan

df_incendios['Temp_prom'] = df_incendios.apply(obtener_temperatura, axis=1)
```

La función para la humedad es similar.

### 6. Preparación de la Variable Objetivo

Se convierte la columna `Duración días` a numérico y se aplican transformaciones (logaritmo) para normalizar la variable objetivo.  
```python
df_incendios['Duración días'] = pd.to_numeric(df_incendios['Duración días'], errors='coerce')
df_incendios.dropna(subset=['Duración días', 'Temp_prom', 'Humedad_prom'], inplace=True)
y = np.log1p(df_incendios['Duración días'])
```

### 7. Selección de Características

Se seleccionan las variables consideradas relevantes para el modelo, incluyendo características climáticas, geográficas y de fecha.  
```python
features = [
    'Año',
    'latitud_grados', 'longitud_grados',
    'Estado', 'Municipio', 'Región', 'Causa',
    'Tipo de incendio', 'Tipo Vegetación', 'Régimen de fuego',
    'Tipo impacto', 'Total hectáreas', 'Tamaño', 'Detección', 'Llegada',
    'Cluster_geo', 'Temp_prom', 'Humedad_prom', 'Mes'
]
X = df_incendios[features].copy()
```

### 8. Definición de Variables Numéricas y Categóricas

Se especifican las variables numéricas y las categóricas para su preprocesamiento. Se convierten a tipo numérico y se eliminan registros con valores nulos.  
```python
numerical_features = [
    'Año', 'latitud_grados', 'longitud_grados', 'Total hectáreas',
    'Detección', 'Llegada', 'Temp_prom', 'Humedad_prom'
]
categorical_features = [
    'Estado', 'Municipio', 'Región', 'Causa',
    'Tipo de incendio', 'Tipo Vegetación', 'Régimen de fuego',
    'Tipo impacto', 'Tamaño', 'Cluster_geo', 'Mes'
]

for col in numerical_features:
    X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')

X.dropna(subset=numerical_features + categorical_features, inplace=True)
y = y.loc[X.index]
```

### 9. Preprocesamiento y Construcción del Pipeline

Se define un `ColumnTransformer` para escalar las variables numéricas y codificar las categóricas. Posteriormente, se crea un pipeline que integra el preprocesamiento y el modelo XGBoost con hiperparámetros fijos.  
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        random_state=42,
        n_estimators=700,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror'
    ))
])
```

### 10. División del Conjunto de Datos y Entrenamiento

Se divide el conjunto en entrenamiento y prueba, se entrena el modelo y se evalúa mediante métricas como el MSE y el R².  
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred)

mse = mean_squared_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)

print(f'MSE: {mse}')
print(f'R²: {r2}')
```

### 11. Guardado del Modelo

Finalmente, se guarda el modelo entrenado utilizando `joblib` para su uso futuro sin necesidad de reentrenamiento.  
```python
joblib.dump(model, 'modelo_incendios_fijo.pkl')
print("Modelo guardado en 'modelo_incendios_fijo.pkl'")
```

## Cómo Ejecutar el Código

1. Asegúrate de tener instaladas las dependencias necesarias (consultar sección de Requisitos).
2. Coloca los archivos `BD.csv`, `temperaturas-mexico.csv` y `humedad-mexico.csv` en el mismo directorio que `model.py`.
3. Ejecuta el script:
   ```bash
   python model.py
   ```
4. El script imprimirá en consola las métricas de evaluación y guardará el modelo entrenado en `modelo_incendios_fijo.pkl`.

## Consideraciones

- **Preprocesamiento:**  
  El código convierte variables de texto y fechas, y aplica escalado y codificación a las características. Esto es fundamental para el correcto funcionamiento del modelo.

- **Selección de Variables:**  
  Se incluyen variables climáticas (temperatura y humedad), geográficas (coordenadas y clusters) y temporales (mes). La selección y calidad de estas variables son clave para el desempeño del modelo.

- **Hiperparámetros Fijos:**  
  El modelo XGBoost se entrena con hiperparámetros predefinidos (por ejemplo, `n_estimators=700`, `max_depth=8`), obtenidos tras un proceso de optimización previo.

- **Métrica de Evaluación:**  
  Se utiliza el error cuadrático medio (MSE) y el coeficiente de determinación (R²) para evaluar el desempeño del modelo, aplicando una transformación logarítmica inversa para interpretar las predicciones.


