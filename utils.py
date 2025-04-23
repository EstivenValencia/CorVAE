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

import torch
import numpy as np
from models import DecoderModel
import pickle

def read_json_config(json_config_path):
    try:
        with open(json_config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {json_config_path}")
    return config

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
    config = read_json_config(json_config_path)

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

MAX_BETA = 1e-2
MIN_BETA = 1e-5
LAMBDA = 0.7

LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2

def reconstruct_data(
    z: str,
    y: str,
    models_paths: str,
    device: torch.device,
    json_config_path: str,
) -> tuple[np.ndarray, np.ndarray] | pd.DataFrame:
    """
    Reconstruye los datos originales a partir del espacio latente guardado utilizando
    el decoder entrenado y los objetos de preprocesamiento.

    Args:
        latent_data_path (str): Ruta al archivo .npy que contiene el espacio latente (train_z.npy).
        decoder_weights_path (str): Ruta al archivo .pt con los pesos del DecoderModel guardado.
        num_scaler: Objeto scaler (ej. StandardScaler) fitteado en los datos numéricos originales.
                    Puede ser None si no se usó normalización.
        cat_encoder: Objeto encoder (ej. OrdinalEncoder) fitteado en los datos categóricos originales.
        d_numerical (int): Número de características numéricas originales.
        categories (list): Lista de cardinalidades de las características categóricas.
        num_layers (int): Número de capas del Transformer en el decoder.
        d_token (int): Dimensión del token/embedding.
        n_head (int): Número de cabezas de atención.
        factor (int): Factor de expansión en las capas FFN del Transformer.
        device (torch.device): Dispositivo ('cuda' o 'cpu') donde ejecutar el modelo.
        original_num_columns (list, optional): Nombres de las columnas numéricas originales.
                                                Si se provee junto con original_cat_columns,
                                                se retorna un DataFrame. Defaults to None.
        original_cat_columns (list, optional): Nombres de las columnas categóricas originales.
                                                 Si se provee junto con original_num_columns,
                                                 se retorna un DataFrame. Defaults to None.


    Returns:
        tuple[np.ndarray, np.ndarray] | pd.DataFrame:
            Si no se proporcionan nombres de columnas:
                Una tupla conteniendo (datos_numéricos_reconstruidos, datos_categóricos_reconstruidos)
                como arrays NumPy.
            Si se proporcionan nombres de columnas:
                Un DataFrame de Pandas con los datos reconstruidos y los nombres de columna originales.
    """
    print("--- Iniciando reconstrucción de datos ---")
    
    config = read_json_config(json_config_path)
    original_num_columns = config.get('num_col_names')
    original_cat_columns = config.get('cat_col_names')
    original_target_column = config.get('target_col_name')
    original_target_column = original_target_column if type(original_target_column) == list else [original_target_column]

    num_cols_idx = config['num_col_idx']
    cat_cols_idx = config['cat_col_idx']
    target_cols_idx = config['target_col_idx']


    with open(os.path.join(models_paths, 'pre_encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
        num_cols = encoders['num_cols']
        categories = encoders['categories']

        num_scaler = encoders['num_scaler']
        cat_encoder = encoders['cat_encoder']
        label_encoder = encoders['label_encoder']
        print("Objetos de preprocesamiento cargados.")

    decoder_weights_path = os.path.join(models_paths, 'decoder.pt')

    # Convertir a tensor y mover al dispositivo
    latent_z_tensor = torch.tensor(z, dtype=torch.float32).to(device)

    # 2. Instanciar el modelo Decoder
    # Asegúrate de usar los mismos hiperparámetros que durante el entrenamiento
    decoder_model = DecoderModel(NUM_LAYERS, num_cols, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)

    # 3. Cargar los pesos del decoder entrenado
    try:
        decoder_model.load_state_dict(torch.load(decoder_weights_path, map_location=device))
        print(f"Pesos del decoder cargados desde: {decoder_weights_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo de pesos del decoder no encontrado: {decoder_weights_path}")
    except Exception as e:
        # Podría haber un error si la arquitectura no coincide exactamente
        raise RuntimeError(f"Error cargando los pesos del decoder: {e}. Asegúrate de que los hiperparámetros coinciden.")

    # Poner el modelo en modo evaluación (importante para desactivar dropout, etc.)
    decoder_model.eval()

    # 4. Pasar el espacio latente a través del decoder
    print("Pasando datos latentes a través del decoder...")
    with torch.no_grad(): # No necesitamos calcular gradientes
        if latent_z_tensor.shape[1] == num_cols + len(categories) + 1: # Verifica si CLS está presente
             print("Detectado token CLS en el espacio latente, eliminándolo para el decoder.")
             latent_z_tensor_for_decoder = latent_z_tensor[:, 1:, :]
        else:
             # Si ya no tiene el CLS (por ejemplo, si guardaste z[:,1:]), úsalo directamente
             print("Asumiendo que el espacio latente no contiene el token CLS.")
             latent_z_tensor_for_decoder = latent_z_tensor

        # Obtener reconstrucciones (aún escaladas/codificadas)
        recon_num_processed, recon_cat_processed_list = decoder_model(latent_z_tensor_for_decoder)

    print("Reconstrucción completada. Invirtiendo transformaciones...")

    # 5. Invertir transformaciones
    # Mover resultados a CPU y convertir a NumPy
    recon_num_processed_np = recon_num_processed.detach().cpu().numpy()

    # a) Invertir normalización numérica
    if num_scaler is not None:
        try:
            recon_num_original = num_scaler.inverse_transform(recon_num_processed_np)
            print("Normalización numérica invertida.")
        except Exception as e:
            raise ValueError("Error al invertir la normalización numérica: {e}")

    else:
        # Si no hubo scaler, los datos ya están en su 'escala original' (post-imputación)
        recon_num_original = recon_num_processed_np
        print("No se aplicó normalización numérica, usando datos reconstruidos directamente.")

    if label_encoder is not None:
        # Invertir la codificación de la variable objetivo
        try:
            y_reconstructed = label_encoder.inverse_transform(y)
            print("Codificación de la variable objetivo invertida.")
        except Exception as e:
            raise ValueError("Error al invertir la codificación de la variable objetivo: {e}")


    # b) Invertir codificación categórica
    recon_cat_encoded_list = []
    for recon_cat_logits in recon_cat_processed_list:
        # Obtener el índice de la categoría predicha (la de mayor logit)
        predicted_indices = torch.argmax(recon_cat_logits, dim=1)
        recon_cat_encoded_list.append(predicted_indices.detach().cpu().numpy())

    # Combinar las columnas categóricas predichas (aún codificadas)
    recon_cat_encoded_np = np.column_stack(recon_cat_encoded_list)

    if cat_encoder is not None:
        try:
            # Usar el cat_encoder fitteado para invertir la transformación
            recon_cat_original = cat_encoder.inverse_transform(recon_cat_encoded_np)
            print("Codificación categórica invertida.")
        except Exception as e:
            print(f"Error al invertir la codificación categórica: {e}")
            print("Devolviendo datos categóricos codificados.")
            recon_cat_original = recon_cat_encoded_np # Devolver codificado si falla la inversa
    else:
        # Si no hubo encoder (improbable para categóricos), devolver como está
        recon_cat_original = recon_cat_encoded_np
        print("No se aplicó codificación categórica (¿inesperado?), usando datos reconstruidos directamente.")

    print("--- Reconstrucción finalizada ---")

    print(original_num_columns, original_cat_columns)
    # 6. Devolver resultados
    if original_num_columns is not None and original_cat_columns is not None:
        print("Combinando en un DataFrame de Pandas.")
        # Crear DataFrames separados y concatenar
        df_num = pd.DataFrame(recon_num_original, columns=original_num_columns)
        df_cat = pd.DataFrame(recon_cat_original, columns=original_cat_columns)
        df_target = pd.DataFrame(y_reconstructed, columns=original_target_column)
        # Asegurarse de que los índices coincidan si se concatenan
        df_num.index = df_cat.index
        df_reconstructed = pd.concat([df_num, df_cat, df_target], axis=1)

        # Combinar los nombres en orden original usando los índices
        all_columns = dict(zip(num_cols_idx, original_num_columns)) | \
                    dict(zip(cat_cols_idx, original_cat_columns)) | \
                    dict(zip(target_cols_idx, original_target_column))

        # Ordenar los nombres por el índice original
        ordered_columns = [all_columns[i] for i in sorted(all_columns)]

        # Reordenar las columnas del DataFrame
        df_reconstructed = df_reconstructed[ordered_columns]

        return df_reconstructed
    else:
        print("Here")
        # Devolver como arrays NumPy separados
        return recon_num_original, recon_cat_original
    
if __name__ == '__main__':
    # Ejemplo de uso
    json_config_path = '/mnt/d/home-2/Documentos/master/practica-deusto-tech/TFM/MyTabsyn/CorVAE/data/shoppers/metadata.json'
    z = np.load('ckpt_custom_model/train_z.npy')
    y = np.load('ckpt_custom_model/train_y.npy')
    models_paths = 'ckpt_custom_model'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reconstructed_data = reconstruct_data(z, y, models_paths, device, json_config_path)
    reconstructed_data.to_csv('reconstructed_data.csv', index=False)
