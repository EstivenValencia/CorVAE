import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)

def preprocessing(json_config_path, encoding='ordinal', random_state=0): # Se utiliza la misma semilla que 
    """
    Función para preprocesar un dataset tabular según configuración JSON.
    Lee rutas, separa columnas, divide en train/test, imputa, normaliza y codifica.

    Parámetros:
    - json_config_path: ruta al archivo .json con la configuración.
    - encoding: 'one-hot' para OneHotEncoder, cualquier otro valor para OrdinalEncoder.
    - random_state: semilla para reproducibilidad de la división train/test.

    Retorna:
    - X_num_train: array con datos numéricos de entrenamiento.
    - X_cat_train: array con datos categóricos codificados de entrenamiento.
    - X_num_test: array con datos numéricos de prueba.
    - X_cat_test: array con datos categóricos codificados de prueba.
    - y_train_enc: vector objetivo codificado de entrenamiento.
    - y_test_enc: vector objetivo codificado de prueba.
    - num_scaler: objeto scaler entrenado para normalización inversa de datos numéricos.
    - cat_encoder: objeto encoder entrenado para invertir codificación categórica.
    - label_encoder: objeto encoder entrenado para invertir transformación de y.

    Ejemplos de cómo revertir transformaciones:
    - Para datos numéricos: X_orig = num_scaler.inverse_transform(X_scaled)
    - Para y: y_orig = label_encoder.inverse_transform(y_enc)
    """
    # 1. Cargar configuración JSON
    try:
        with open(json_config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {json_config_path}")

    # 2. Validar llaves necesarias
    required_keys = [
        'normalization', 'num_col_idx', 'cat_col_idx',
        'target_col_idx', 'file', 'num_imputation', 'cat_imputation'
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Falta la clave en config: {key}")

    # 3. Construir ruta al CSV y cargar datos
    base_dir = os.path.dirname(json_config_path)
    data_file = os.path.join(base_dir, config['file'])
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo de datos no encontrado: {data_file}")

    # 4. Extraer índices y validar
    num_idx = config['num_col_idx']
    cat_idx = config['cat_col_idx']
    target_idx = config['target_col_idx']
    n_cols = df.shape[1]
    for idx_list, name in [(num_idx, 'num_col_idx'), (cat_idx, 'cat_col_idx'), (target_idx, 'target_col_idx')]:
        if not isinstance(idx_list, list):
            raise TypeError(f"{name} debe ser una lista de índices")
        for idx in idx_list:
            if not (0 <= idx < n_cols):
                raise IndexError(f"Índice {idx} en {name} fuera de rango")

    # 5. Separar características y objetivo
    target_col = df.columns[target_idx[0]]
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    # 6. Dividir columnas numéricas y categóricas
    X_num = X.iloc[:, num_idx].values
    X_cat = X.iloc[:, cat_idx].values

    # 7. Dividir en train/test
    X_num_train, X_num_test, X_cat_train, X_cat_test, y_train, y_test = \
        train_test_split(
            X_num, X_cat, y,
            test_size=0.1,
            random_state=random_state,
            stratify=y
        )

    # 8. Imputación de datos
    # Numéricos
    num_impute_strategy = config['num_imputation']
    if num_impute_strategy:
        num_imputer = SimpleImputer(strategy=num_impute_strategy)
        X_num_train = num_imputer.fit_transform(X_num_train)
        X_num_test = num_imputer.transform(X_num_test)

    # Categóricos
    cat_impute_strategy = config['cat_imputation']
    if cat_impute_strategy:
        cat_imputer = SimpleImputer(strategy=cat_impute_strategy)
        X_cat_train = cat_imputer.fit_transform(X_cat_train)
        X_cat_test = cat_imputer.transform(X_cat_test)

    # 9. Normalización de datos numéricos
    norm = config['normalization']
    num_scaler = None
    if norm == 'z-score':
        num_scaler = StandardScaler()
    elif norm == 'min-max':
        num_scaler = MinMaxScaler()
    elif norm == 'robust':
        num_scaler = RobustScaler()
    elif norm in (None, 'null'):
        num_scaler = None
    else:
        raise ValueError(f"Normalización no soportada: {norm}")

    if num_scaler:
        X_num_train = num_scaler.fit_transform(X_num_train)
        X_num_test = num_scaler.transform(X_num_test)

    # 10. Codificación de datos categóricos
    if encoding == 'one-hot':
        cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    else:
        cat_encoder = OrdinalEncoder()
    X_cat_train = cat_encoder.fit_transform(X_cat_train)
    X_cat_test = cat_encoder.transform(X_cat_test)

    # 11. Codificación de la variable objetivo y_label_encoder
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # 12. Retornar datos preprocesados y encoders
    return (
        X_num_train, X_cat_train,
        X_num_test, X_cat_test,
        y_train_enc, y_test_enc,
        num_scaler, cat_encoder, label_encoder
    )
