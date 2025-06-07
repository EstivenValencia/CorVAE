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

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, make_scorer,
    classification_report, roc_auc_score, accuracy_score
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator
from tensorboardX import SummaryWriter
import time

# -----------------------------------------------------------------------------
# Registry of models and their hyper‑parameter spaces (lists) ------------------
# -----------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import optuna
from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors   import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

def _weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")

_METRIC_SCORERS = {
    "balanced": make_scorer(balanced_accuracy_score),
    "macro_f1": make_scorer(f1_score, average="macro"),
    "weighted_f1": make_scorer(_weighted_f1),
    "accuracy": make_scorer(accuracy_score),
}


MODELS = {
    "xgb": {
        "constructor": XGBClassifier,
        "static_args": {"objective": "binary:logistic", "tree_method": "hist"},
        "search_space": {
            "n_estimators": lambda t: t.suggest_categorical("n_estimators", [10,40,100]),
            "max_depth": lambda t: t.suggest_categorical("max_depth", [5,20,30]),
            "gamma": lambda t: t.suggest_float("gamma", 0.1,1.0),
            "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.5, 1.0),
        },
    },
    "rf": {
        "constructor": RandomForestClassifier,
        "static_args": {"class_weight": "balanced"},
        "search_space": {
            "n_estimators": lambda t: t.suggest_int("n_estimators", 100, 500),
            "max_depth": lambda t: t.suggest_int("max_depth", 3, 20),
            "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 10),
        },
    },
    "svc": {
        "constructor": SVC,
        "static_args": {"probability": True},
        "search_space": {
            "C": lambda t: t.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": lambda t: t.suggest_float("gamma", 1e-4, 1e0, log=True),
        },
    },
    "logreg": {
        "constructor": LogisticRegression,
        "static_args": {"max_iter": 200, "solver": "lbfgs"},
        "search_space": {
            "C": lambda t: t.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty": lambda t: t.suggest_categorical("penalty", ["l2"]),
        },
    },
    "mlp": {
            "constructor": MLPClassifier,
            "static_args": {},
            "search_space": {
                "hidden_layer_sizes": lambda t: t.suggest_categorical(
                    "hidden_layer_sizes", [(100,), (200,), (100, 100)]
                ),
                "max_iter": lambda t: t.suggest_categorical("max_iter", [50, 100]),
                "alpha": lambda t: t.suggest_categorical("alpha", [0.0001, 0.001]),
            },
        },
    "naive_bayes": {
        "constructor": GaussianNB,
        "static_args": {},
        "search_space": {
            # var_smoothing alrededor de 1e-9
            "var_smoothing": lambda t: t.suggest_float(
                "var_smoothing", 1e-10, 1e-8, log=True
            ),
        },
    },
    "knn": {
        "constructor": KNeighborsClassifier,
        "static_args": {},
        "search_space": {
            # n_neighbors de 3 a 10
            "n_neighbors": lambda t: t.suggest_int(
                "n_neighbors", 3, 10
            ),
            # leaf_size de 20 a 40
            "leaf_size": lambda t: t.suggest_int(
                "leaf_size", 20, 40
            ),
            # p = 1 o 2
            "p": lambda t: t.suggest_categorical(
                "p", [1, 2]
            ),
        },
    },
}

MODEL_XGB = {
    'max_depth': 20,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 1.0,
    'gamma': 1.0,
    # Para usar GPU: 'tree_method': 'gpu_hist', 'gpu_id': 0,
    'tree_method': 'hist',
    'objective': 'binary:logistic'
}

def compute_relative_regret(
    test_metrics_all: dict,
    metrics_random: list[dict],
    metrics_method: list[dict],
    method_name: str,
    ipc: int,
    metric_key: str = "balanced",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    test_metrics_all : dict
        { model_name: {metric_key: float, ...}, ... }
    metrics_random : list of dict
        Lista (len 5) con la misma estructura que test_metrics_all
        para cada semilla de random sampling.
    metrics_method : list of dict
        Igual, para el método (e.g. k-means).
    method_name : str
        Nombre del método (p.ej. "k-means").
    ipc : int
        Número de instancias por clase (para guardar la columna IPC).
    metric_key : str
        Qué métrica usar para el regret (por defecto "balanced").
    """
    records = []
    n_seeds = len(metrics_random)
    # para cada modelo (xgb, rf, …)
    for model_name, all_metrics in test_metrics_all.items():
        AF = all_metrics[metric_key]
        regrets = []
        for i in range(n_seeds):
            AR = metrics_random[i][model_name][metric_key]
            AM = metrics_method[i][model_name][metric_key]
            # evita división por cero
            denom = (AF - AR)
            if denom == 0:
                r = np.nan
            else:
                r = (AF - AM) / denom
            regrets.append(r)
        regrets = np.array(regrets, dtype=np.float64)
        records.append({
            "method":      method_name,
            "ipc":         ipc,
            "model":       model_name,
            "regret_mean": np.nanmean(regrets),
            "regret_std":  np.nanstd(regrets),
        })

    df = pd.DataFrame(records)
    return df


def read_json_config(json_config_path):
    
    try:
        with open(json_config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {json_config_path}")
    return config

def split_train_test_custom(config, base_dir, test_size=0.1, random_state=0):
    target_col = config['target_col_name']

    data_file = os.path.join(base_dir, config['file'])
    
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo de datos no encontrado: {data_file}")
    
    # Eliminacion de posibles caracteres especiales
    try:
        df[target_col] = df[target_col].str.replace('.', '', regex=False)
    except:
        print("No se encontraron caracteres extranos en la columna target")
    
    y = df[target_col].to_numpy()  
    X = df.drop(columns=[target_col]).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    np.save(f'{base_dir}/X_train.npy', X_train)
    np.save(f'{base_dir}/X_test.npy', X_test)
    np.save(f'{base_dir}/y_train.npy', y_train)
    np.save(f'{base_dir}/y_test.npy', y_test)
    print(f"Datos de train y test guardados en {base_dir}")

    return X_train, y_train, X_test, y_test

def split_train_test_custom_undersampling(config, base_dir, test_size=0.1, random_state=0):
    target_col = config['target_col_name']

    data_file = os.path.join(base_dir, config['file'])
    
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo de datos no encontrado: {data_file}")
    
    # Eliminacion de posibles caracteres especiales
    try:
        df[target_col] = df[target_col].str.replace('.', '', regex=False)
    except:
        print("No se encontraron caracteres extranos en la columna target")
    
    y = df[target_col].to_numpy()  
    X = df.drop(columns=[target_col]).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # --- UNDERSAMPLING de la clase mayoritaria en TRAIN ---
    # contar muestras por clase
    classes, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()
    rng = np.random.RandomState(random_state)

    idx_resampled = []
    for cls in classes:
        idx_cls = np.where(y_train == cls)[0]
        if len(idx_cls) > min_count:
            # muestreo aleatorio sin reemplazo
            idx_sel = rng.choice(idx_cls, size=min_count, replace=False)
        else:
            idx_sel = idx_cls
        idx_resampled.append(idx_sel)

    idx_resampled = np.concatenate(idx_resampled)
    rng.shuffle(idx_resampled)  # barajar

    # reconstruir X_train/y_train balanceados
    X_train = X_train[idx_resampled]
    y_train = y_train[idx_resampled]
    # -------------------------------------------------------

    # Guardar en .npy
    np.save(f'{base_dir}/X_train.npy', X_train)
    np.save(f'{base_dir}/X_test.npy', X_test)
    np.save(f'{base_dir}/y_train.npy', y_train)
    np.save(f'{base_dir}/y_test.npy', y_test)
    print(f"Datos de train y test (train undersampleado) guardados en {base_dir}")

    return X_train, y_train, X_test, y_test

def concat_l(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([X, y.reshape(-1, 1)], axis=1)

def preprocessing(config, X_train, y_train, X_test, y_test, encoding='ordinal', concat_label=False, concat=False, random_state=0): # Se utiliza la misma semilla que 
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

    # 4. Extraer índices y validar
    num_idx = config['num_col_idx']
    cat_idx = config['cat_col_idx']

    # 6. Dividir columnas numéricas y categóricas
    X_num_train = X_train[:, num_idx]
    X_cat_train = X_train[:, cat_idx]
    X_num_test = X_test[:, num_idx]
    X_cat_test = X_test[:, cat_idx]

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


    if concat_label:
        X_train = concat_l(X_train, y_train)

    # 10. Codificación de datos categóricos
    if encoding == 'one-hot':
        cat_encoder = OneHotEncoder(sparse_output=False)
    else:
        cat_encoder = OrdinalEncoder()
    print("Categoricas entrenamiento: ", X_cat_train.shape[1])
    if X_cat_train.shape[1] > 0:
        X_cat_train = cat_encoder.fit_transform(X_cat_train) 
        X_cat_test = cat_encoder.transform(X_cat_test)

    # 11. Codificación de la variable objetivo y_label_encoder
    print("Datos unicos de y_train: ",np.unique(y_train))

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    print("Datos unicos de y_train luego: ",np.unique(y_train_enc))
    # 12. Retornar datos preprocesados y encoders
    if concat:
        # Concatenar numéricas y categóricas
        X_train = np.concatenate((X_num_train, X_cat_train), axis=1)
        X_test = np.concatenate((X_num_test, X_cat_test), axis=1)
        return X_train,X_test, y_train_enc, y_test_enc, num_scaler, cat_encoder, label_encoder
    return (
        X_num_train, X_cat_train,
        X_num_test, X_cat_test,
        y_train_enc, y_test_enc,
        num_scaler, cat_encoder, label_encoder
    )

def load_reconstructed_data(
    df_recon: pd.DataFrame,
    json_config_path: str,
    num_scaler,
    cat_encoder,
    label_encoder,
):
    # 1) Carga config y localiza columnas
    config = read_json_config(json_config_path)
    target_col = config['target_col_name']
    num_idx = config['num_col_idx']
    cat_idx = config['cat_col_idx']
    
    # 2) Extrae y raw de tu df reconstruido
    y_train_raw = df_recon[target_col].to_numpy()
    X_train_raw = df_recon.drop(columns=[target_col]).to_numpy()

    # 3) Transforma las numéricas con el mismo scaler
    X_num_recon = X_train_raw[:, num_idx]
    if num_scaler:
        X_num_recon = num_scaler.transform(X_num_recon)

    # 4) Transforma las categóricas con el mismo OneHotEncoder
    X_cat_recon = X_train_raw[:, cat_idx]
    if cat_encoder:
        X_cat_recon = cat_encoder.transform(X_cat_recon)

    # 5) Concatena en el mismo orden (como hiciste en preprocessing concat=True)
    X_train_pre = np.concatenate([X_num_recon, X_cat_recon], axis=1)

    # 6) Codifica el target usando label_encoder
    y_train_pre = label_encoder.transform(y_train_raw)

    # 7) Haz lo mismo con el test “crudo” guardado
    #base_dir = os.path.dirname(json_config_path)
    #X_test_raw = np.load(os.path.join(base_dir, 'X_test.npy'), allow_pickle=True)
    #y_test_raw = np.load(os.path.join(base_dir, 'y_test.npy'), allow_pickle=True)

    #X_num_test = X_test_raw[:, num_idx]
    #if num_scaler:
    #    X_num_test = num_scaler.transform(X_num_test)

    #X_cat_test = X_test_raw[:, cat_idx]
    #if cat_encoder:
    #    X_cat_test = cat_encoder.transform(X_cat_test)

    #X_test_pre = np.concatenate([X_num_test, X_cat_test], axis=1)
    #y_test_pre = label_encoder.transform(y_test_raw)

    return X_train_pre, y_train_pre

def reconstruct_data(
    x: str,
    y: str,
    models_paths: str,
    device: torch.device,
    json_config_path: str,
    latent_space = False, # Si es True, se asume que z es un espacio latente, de lo contrario es una destilación del espacio original
    hyperparams = {},
    method = 'k-means',
    ipc = 0,
    seed = 0,
    concat_label=False
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

    # Datos recontruidos
    reconstructed_dir = os.path.join(models_paths, 'reconstructed_data', f'IPC_{ipc}')
    os.makedirs(reconstructed_dir, exist_ok=True)

    d_token =hyperparams.get('d_token',4)
    n_head = hyperparams.get('n_head',1)
    factor = hyperparams.get('factor',32)
    num_layers =hyperparams.get('num_layers',2)
    
    print("--- Iniciando reconstrucción de datos ---")
    
    config = read_json_config(json_config_path)
    original_num_columns = config.get('num_col_names')
    original_cat_columns = config.get('cat_col_names')
    original_target_column = config.get('target_col_name')
    original_target_column = original_target_column if type(original_target_column) == list else [original_target_column]

    num_cols_idx = config['num_col_idx']
    cat_cols_idx = config['cat_col_idx']
    target_cols_idx = config['target_col_idx']


    with open(os.path.join(models_paths, f'pre_encoders_seed_{seed}.pkl'), 'rb') as f:
        encoders = pickle.load(f)
        num_cols = encoders['num_cols']
        categories = encoders['categories']

        num_scaler = encoders['num_scaler']
        if latent_space:
            cat_encoder = encoders['cat_ordinal_encoder']
        else:
            cat_encoder = encoders['cat_onehot_encoder']
        label_encoder = encoders['label_encoder']
        print("Objetos de preprocesamiento cargados.")

    decoder_inference_time = 0
    if latent_space:
        decoder_weights_path = os.path.join(models_paths, f'decoder_seed_{seed}.pt')

        # Convertir a tensor y mover al dispositivo
        latent_z_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        #print("Categorias en la recons:", categories)
        # 2. Instanciar el modelo Decoder
        # Asegúrate de usar los mismos hiperparámetros que durante el entrenamiento
    
        decoder_model = DecoderModel(num_layers, num_cols, categories, d_token, n_head = n_head, factor = factor).to(device)

        # 3. Cargar los pesos del decoder entrenado
        try:
            decoder_model.load_state_dict(torch.load(decoder_weights_path, map_location=device))
            #print(f"Pesos del decoder cargados desde: {decoder_weights_path}")
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
            start = time.time()
            recon_num_processed, recon_cat_processed_list = decoder_model(latent_z_tensor_for_decoder)
            end = time.time()
            decoder_inference_time = end - start

            #print("Categorias 2: ---",[recon_cat_processed_list[i].shape for i in range(len(recon_cat_processed_list))])
        print("Reconstrucción completada. Invirtiendo transformaciones...")

        # 5. Invertir transformaciones
        # Mover resultados a CPU y convertir a NumPy
        recon_num_processed_np = recon_num_processed.detach().cpu().numpy()
        if concat_label:
            recon_cat_processed_list= [
                                        cat_tensor[:, :-1] # Se elimina la etiqueta
                                        for cat_tensor in recon_cat_processed_list
                                    ]
        
    else:
        recon_num_processed_np = x
    
    #print("Suma de categorias: ", categorical_offsets)
    # b) Invertir codificación categórica
    recon_cat_encoded_list = []
    for j, recon_cat_logits in enumerate(recon_cat_processed_list):
        # Obtener el índice de la categoría predicha (la de mayor logit)
        predicted_indices = torch.argmax(recon_cat_logits, dim=1)
        recon_cat_encoded_list.append(predicted_indices.cpu().numpy())

    # Combinar las columnas categóricas predichas (aún codificadas)
    if recon_cat_encoded_list:
        recon_cat_encoded_np = np.column_stack(recon_cat_encoded_list)
    #print("Categorias 3: ---",recon_cat_encoded_np)
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

    if cat_encoder is not None and recon_cat_encoded_list:
        try:
            # Usar el cat_encoder fitteado para invertir la transformación
            recon_cat_original = cat_encoder.inverse_transform(recon_cat_encoded_np)
            print("Codificación categórica invertida.")
        except Exception as e:
            raise ValueError("Error al invertir la codificación categórica: {e}")   
    else:
        # Si no hubo encoder (improbable para categóricos), devolver como está
        print("No se aplicó codificación categórica (¿inesperado?), usando datos reconstruidos directamente.")

    print("--- Reconstrucción finalizada ---")

    #print(original_num_columns, original_cat_columns)
    # 6. Devolver resultados
    if original_num_columns is not None:
        print("Combinando en un DataFrame de Pandas.")
        # Crear DataFrames separados y concatenar
        df_num = pd.DataFrame(recon_num_original, columns=original_num_columns)
        df_target = pd.DataFrame(y_reconstructed, columns=original_target_column)

        if recon_cat_encoded_list:
            df_cat = pd.DataFrame(recon_cat_original, columns=original_cat_columns)
            # Asegurarse de que los índices coincidan si se concatenan
            df_num.index = df_cat.index
            df_reconstructed = pd.concat([df_num, df_cat, df_target], axis=1)

            # Combinar los nombres en orden original usando los índices
            all_columns = dict(zip(num_cols_idx, original_num_columns)) | \
                        dict(zip(cat_cols_idx, original_cat_columns)) | \
                        dict(zip(target_cols_idx, original_target_column))
        else: 
            # Asegurarse de que los índices coincidan si se concatenan
            df_reconstructed = pd.concat([df_num, df_target], axis=1)

            # Combinar los nombres en orden original usando los índices
            all_columns = dict(zip(num_cols_idx, original_num_columns)) | \
                        dict(zip(target_cols_idx, original_target_column))

        # Ordenar los nombres por el índice original
        ordered_columns = [all_columns[i] for i in sorted(all_columns)]

        # Reordenar las columnas del DataFrame
        df_reconstructed = df_reconstructed[ordered_columns]


        reconstructed_path = os.path.join(reconstructed_dir, f'method_{method}_seed_{seed}.csv')
        df_reconstructed.to_csv(reconstructed_path, index=False)

        return df_reconstructed, decoder_inference_time
    raise ValueError("No se encontraron los nombres de columnas originales.")
    

# ------------------------------------------------------------------------
# 1. F-score ponderado «minority-friendly»
# ------------------------------------------------------------------------
def weighted_f1_custom(y_true, y_pred):
    """
    Versión que DA MÁS peso a las clases minoritarias.
    Para cada clase i:  w_i = (1 - p_i)/(k - 1)
      p_i  = soporte_i / N
      k    = nº de clases
    """
    report   = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes  = list(report.keys())[:-3]                       # elimina avg/acc
    supports = np.array([report[c]['support'] for c in classes], dtype=float)
    f1s      = np.array([report[c]['f1-score'] for c in classes], dtype=float)
    p        = supports / supports.sum()
    weights  = (1 - p) / (len(classes) - 1)
    return np.sum(f1s * weights)


# -----------------------------------------------------------------------------
# Funciones auxiliares (mínimas)
# -----------------------------------------------------------------------------
def _build_cv(n_splits, random_state):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def _objective(trial, cfg, X, y, cv):
    """Función objetivo para Optuna: devuelve balanced‑accuracy promedio."""
    params = {k: v(trial) for k, v in cfg["search_space"].items()}
    model = cfg["constructor"](**cfg["static_args"], **params)
    scores = cross_validate(
        model,
        X,
        y,
        scoring=_METRIC_SCORERS["balanced"],
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    return scores["test_score"].mean()

def _tune_single_model(tag, cfg, X, y, cv, n_trials, random_state):
    """Tunea hiper-parámetros con Optuna y mide tiempos de tuning y entrenamiento."""
    # Medir tiempo de optimización Optuna
    tic_opt = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name=tag
    )
    study.optimize(
        lambda t: _objective(t, cfg, X, y, cv),
        n_trials=n_trials,
        show_progress_bar=False
    )
    optuna_time = time.time() - tic_opt

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        # decides un fallback: aquí uso los static_args como "parámetros por defecto"
        best_params = {}
        print(f"⚠️  No hay trials completados en '{tag}'; usando argumentos estáticos.")
    else:
        best_params = study.best_params
    # ---------------------------------------------------------------------

    # Construye el modelo con best_params (o solo static_args si best_params quedó vacío)
    best_model = cfg["constructor"](**cfg["static_args"], **best_params)

    # Medir tiempo de entrenamiento del mejor modelo
    tic_train = time.time()
    best_model.fit(X, y)
    train_time = time.time() - tic_train

    # Métricas de CV para best_params (sin contar tiempos)
    cv_res = cross_validate(
        best_model,
        X,
        y,
        scoring=_METRIC_SCORERS,
        cv=cv,
        n_jobs=-1,
        return_train_score=False
    )
    means = {m: cv_res[f"test_{m}"].mean() for m in _METRIC_SCORERS}
    stds = {f"{m}_std": cv_res[f"test_{m}"].std() for m in _METRIC_SCORERS}

    return best_model, means, stds, best_params, optuna_time, train_time


def _evaluate_on_test(model, X_test, y_test):
    """Calcula métricas en el conjunto de prueba y mide tiempo de inferencia."""
    # Medir tiempo de inferencia
    tic_inf = time.time()
    y_pred = model.predict(X_test)
    inf_time_total = time.time() - tic_inf
    time_per_sample = inf_time_total / len(X_test)

    # Calcular métricas
    classes = model.classes_
    try:
        pos_idx = int(np.where(classes == True)[0])
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba[:, pos_idx], average="weighted")
    except Exception:
        auc = np.nan

    return {
        "balanced": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": _weighted_f1(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": auc,
        "inf_time_per_sample": time_per_sample
    }


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    cv_folds=5,
    n_trials=30,
    ckpt_dir=None,
    method='k-means',
    random_state=0,
    ipc=0,
    space='original'
):
    """Tunea y evalúa clasificadores con Optuna, incluyendo tiempos de tuning, entrenamiento e inferencia."""
    best_dir = os.path.join(ckpt_dir, method, f"IPC_{ipc}", f"best_models_{method}_seed_{random_state}_space_{space}")
    os.makedirs(best_dir, exist_ok=True)

    cv = _build_cv(max(cv_folds, 2), random_state)
    records = []
    test_metrics_all = {}
    best_models = {}

    for tag, cfg in MODELS.items():
        print(f"\n\n>>> Optuna tuning {tag} metodo {method} IPC {ipc} seed {random_state}\n\n")
        best_model, cv_means, cv_stds, best_hp, opt_time, train_time = _tune_single_model(
            tag, cfg, X_train, y_train, cv, n_trials, random_state
        )

        # Guardar parámetros
        if ckpt_dir is not None:
            json_path = os.path.join(best_dir, f"{tag}_best_params.json")
            with open(json_path, 'w', encoding='utf-8') as fp:
                json.dump(best_hp, fp, indent=4, ensure_ascii=False)
            print(f"Parámetros guardados en {json_path}")

        best_models[tag] = best_model
        test_metrics = _evaluate_on_test(best_model, X_test, y_test)
        test_metrics_all[tag] = test_metrics

        # Registrar resultados
        row = {'model': tag, 'method': method}
        # CV metrics
        for m, val in cv_means.items():
            row[f'cv_{m}'] = round(val, 3)
        for m_std, val in cv_stds.items():
            row[f'cv_{m_std}'] = round(val, 3)
        row['optuna_time'] = round(opt_time, 3)
        row['train_time'] = round(train_time, 3)
        # Test metrics including inference time
        for m, val in test_metrics.items():
            if m == 'inf_time_per_sample':
                row['inf_time_per_sample'] = round(val, 6)
            else:
                row[f'test_{m}'] = round(val, 3)

        records.append(row)

    results_df = pd.DataFrame(records)
    return results_df, test_metrics_all, best_models


def evaluate_one_model(
    X_train, y_train, X_test, y_test,
    cv_folds=5, random_state=0
):
    """
    Evalúa un XGBClassifier con parámetros fijos.

    Realiza validación cruzada y evaluación en test.

    Retorna:
      - cv_means: medias de métricas en CV
      - cv_stds: desviaciones estándar en CV
      - test_metrics: métricas en el conjunto de prueba
      - model: modelo entrenado sobre todo el set de entrenamiento
    """
    # Instanciar modelo
    model = XGBClassifier(**MODEL_XGB, random_state=random_state)

    # Crear CV reproducible
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Definir métricas para CV
    scoring = {
        'balanced': 'balanced_accuracy',
        'macro_f1': 'f1_macro',
        'weighted_f1': 'f1_weighted',
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc'
    }

    # Validación cruzada
    cv_res = cross_validate(
        model, X_train, y_train,
        scoring=scoring, cv=cv,
        n_jobs=-1, return_train_score=False
    )

    # Calcular estadísticas de CV
    cv_means = {m: cv_res[f'test_{m}'].mean() for m in scoring}
    cv_stds = {f'{m}_std': cv_res[f'test_{m}'].std() for m in scoring}

    # Entrenar en todo el set de entrenamiento
    model.fit(X_train, y_train, verbose=False)

    # Predicciones y probabilidades en test
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Índice clase positiva
    classes = model.classes_
    pos_idx = int(np.where(classes == True)[0]) if True in classes else 1

    print("Informacion de y_test:", y_test)
    print("Informacion de y_proba:", y_proba[:, pos_idx])

    # Cálculo de métricas en test
    test_metrics = {
        'balanced': balanced_accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba[:, pos_idx], average='weighted')
        if y_proba is not None else np.nan
    }

    return cv_means, cv_stds, test_metrics, model
    

if __name__ == '__main__':
    # Ejemplo de uso
    json_config_path = '/mnt/d/home-2/Documentos/master/practica-deusto-tech/TFM/MyTabsyn/CorVAE/data/shoppers/metadata.json'
    z = np.load('ckpt_custom_model/train_z.npy')
    y = np.load('ckpt_custom_model/train_y.npy')
    models_paths = 'ckpt_custom_model'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reconstructed_data = reconstruct_data(z, y, models_paths, device, json_config_path)
