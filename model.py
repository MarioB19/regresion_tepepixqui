# %% ------------------------- IMPORTS Y CONFIGURACIÓN -------------------------
# Habilitar características experimentales
from sklearn.experimental import enable_iterative_imputer

# Resto de imports
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer  # Ahora correctamente habilitado
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.cluster import KMeans
import joblib
import optuna
from sklearn.base import clone

# Configuración global
pd.set_option('future.no_silent_downcasting', True)
np.random.seed(42)

# %% ------------------------- CARGA Y FUSIÓN DE DATOS -------------------------
def load_data():
    # Cargar datasets principales
    df = pd.read_csv('BD.csv', encoding='ISO-8859-1', parse_dates=['Fecha Inicio'])
    df_temp = pd.read_csv('temperaturas-mexico.csv')
    df_hum = pd.read_csv('humedad-mexico.csv')
    
    # Cargar datos externos (ejemplo)
    try:
        df_poblacion = pd.read_csv('poblacion_municipios.csv')
        df = df.merge(df_poblacion, on='Municipio', how='left')
    except FileNotFoundError:
        print("Advertencia: Archivo de población no encontrado. Continuando sin esos datos.")
    
    # Limpieza inicial
    df['Estado'] = df['Estado'].str.lower().str.strip()
    return df, df_temp, df_hum

df, df_temp, df_hum = load_data()

# %% ------------------------- INGENIERÍA DE VARIABLES -------------------------
def feature_engineering(df):
    # Variables temporales
    df['Mes'] = df['Fecha Inicio'].dt.month
    df['dia_year'] = df['Fecha Inicio'].dt.dayofyear
    
    # Variables cíclicas
    df['sin_mes'] = np.sin(2 * np.pi * df['Mes']/12)
    df['cos_mes'] = np.cos(2 * np.pi * df['Mes']/12)
    
    # Variables geoespaciales
    df['distancia_frontera'] = np.sqrt((df['latitud_grados']-32.6)**2 + (df['longitud_grados']+117.1)**2)
    
    # Interacciones climáticas
    df['temp_humedad_ratio'] = df['Temp_prom'] / (df['Humedad_prom'] + 1e-6)
    df['fire_risk_index'] = df['Temp_prom'] * (1 - df['Humedad_prom']/100)
    
    # Clusters geográficos
    coords = df[['latitud_grados', 'longitud_grados']].dropna()
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=20)
    df.loc[coords.index, 'geo_cluster'] = kmeans.fit_predict(coords)
    
    return df

df = feature_engineering(df)

# %% ------------------------- PREPROCESAMIENTO -------------------------
# Definir variables
numerical_features = [
    'Año', 'latitud_grados', 'longitud_grados', 'Total hectáreas',
    'Detección', 'Llegada', 'Temp_prom', 'Humedad_prom', 'distancia_frontera',
    'temp_humedad_ratio', 'fire_risk_index', 'sin_mes', 'cos_mes'
]

categorical_features = [
    'Estado', 'Municipio', 'Región', 'Causa', 'Tipo de incendio',
    'Tipo Vegetación', 'Régimen de fuego', 'geo_cluster'
]

target = 'Duración días'

# Pipeline de preprocesamiento
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', IterativeImputer(max_iter=15, random_state=42)),
        ('scaler', StandardScaler())
    ]), numerical_features),
    
    ('cat', Pipeline([
        ('imputer', IterativeImputer(max_iter=15, random_state=42, initial_strategy='most_frequent')),
        ('encoder', TargetEncoder())
    ]), categorical_features)
])

# %% ------------------------- SELECCIÓN DE CARACTERÍSTICAS -------------------------
def feature_selection(X, y):
    # Preprocesamiento temporal para cálculo MI
    X_temp = pd.DataFrame(
        preprocessor.fit_transform(X, y),
        columns=numerical_features + categorical_features
    )
    
    # Calcular importancia de características
    mi_scores = mutual_info_regression(X_temp, y)
    mi_df = pd.DataFrame({'feature': X_temp.columns, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values('mi_score', ascending=False)
    
    # Seleccionar top características
    selected_features = mi_df[mi_df['mi_score'] > 0.01]['feature'].tolist()
    return selected_features

# %% ------------------------- MODELADO Y OPTIMIZACIÓN -------------------------
def optimize_model(X, y):
    # División inicial de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Configuración de búsqueda de hiperparámetros
    param_dist = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Pipeline completo con selección de características
    full_pipe = Pipeline([
        ('preprocessor', clone(preprocessor)),
        ('feature_select', SelectKBest(mutual_info_regression, k=20)),
        ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])
    
    # Búsqueda aleatoria
    search = RandomizedSearchCV(
        full_pipe, param_dist, n_iter=50, cv=5,
        scoring='r2', n_jobs=-1, verbose=2
    )
    search.fit(X_train, np.log1p(y_train))
    
    return search.best_estimator_, X_test, y_test

# %% ------------------------- ENSEMBLE -------------------------
def create_ensemble(best_params):
    return StackingRegressor(
        estimators=[
            ('xgb', XGBRegressor(**best_params)),
            ('lgbm', LGBMRegressor(
                num_leaves=31,
                learning_rate=0.05,
                max_depth=-1,
                n_estimators=1000
            ))
        ],
        final_estimator=XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3
        ),
        n_jobs=-1
    )

# %% ------------------------- VALIDACIÓN CRUZADA ESPACIAL -------------------------
def spatial_cross_validation(model, X, y):
    groups = X['geo_cluster']
    cv = GroupKFold(n_splits=5)
    
    scores = []
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, np.log1p(y_train))
        preds = np.expm1(model.predict(X_test))
        scores.append(r2_score(y_test, preds))
    
    return np.mean(scores)

# %% ------------------------- FLUJO PRINCIPAL -------------------------
if __name__ == "__main__":
    # Preparación de datos
    X = df[numerical_features + categorical_features]
    y = df[target].dropna()
    X = X.loc[y.index]
    
    # Selección de características
    selected_features = feature_selection(X, y)
    X = X[selected_features]
    
    # Entrenamiento y optimización
    best_model, X_test, y_test = optimize_model(X, y)
    
    # Evaluación final
    y_pred = np.expm1(best_model.predict(X_test))
    final_r2 = r2_score(y_test, y_pred)
    print(f"\nR² final: {final_r2:.4f}")
    
    # Validación espacial
    spatial_score = spatial_cross_validation(best_model, X, y)
    print(f"Validación espacial R²: {spatial_score:.4f}")
    
    # Guardar modelo
    joblib.dump(best_model, 'mejor_modelo_incendios.pkl')
    print("Modelo guardado exitosamente.")

# %% ------------------------- ANÁLISIS POST-MODELADO -------------------------
def post_analysis(model, X_test, y_test):
    # Importancia de características
    feature_importances = model.named_steps['regressor'].feature_importances_
    features = model[:-1].get_feature_names_out()
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    # Gráfico de importancia
    plt.figure(figsize=(10, 6))
    importance_df.head(15).plot.barh(x='feature', y='importance')
    plt.title('Top 15 características más importantes')
    plt.tight_layout()
    plt.show()

    # Análisis de residuos
    residuals = y_test - np.expm1(model.predict(X_test))
    plt.scatter(np.expm1(model.predict(X_test)), residuals)
    plt.xlabel('Predicciones')
    plt.ylabel('Residuos')
    plt.title('Análisis de residuos')
    plt.axhline(0, color='red', linestyle='--')
    plt.show()

# Ejecutar análisis
post_analysis(best_model, X_test, y_test)