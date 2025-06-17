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
    LabelEncoder,
)

import torch
import numpy as np
from models_vae import DecoderModel
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score,
    make_scorer,
    classification_report,
    roc_auc_score,
    accuracy_score,
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

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import optuna
from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

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
            "n_estimators": lambda t: t.suggest_categorical(
                "n_estimators", [10, 40, 100]
            ),
            "max_depth": lambda t: t.suggest_categorical("max_depth", [5, 20, 30]),
            "gamma": lambda t: t.suggest_float("gamma", 0.1, 1.0),
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
            "var_smoothing": lambda t: t.suggest_float(
                "var_smoothing", 1e-10, 1e-8, log=True
            ),
        },
    },
    "knn": {
        "constructor": KNeighborsClassifier,
        "static_args": {},
        "search_space": {
            "n_neighbors": lambda t: t.suggest_int("n_neighbors", 3, 10),
            "leaf_size": lambda t: t.suggest_int("leaf_size", 20, 40),
            "p": lambda t: t.suggest_categorical("p", [1, 2]),
        },
    },
}

MODEL_XGB = {
    "max_depth": 20,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 1.0,
    "gamma": 1.0,
    "tree_method": "hist",
    "objective": "binary:logistic",
}


def read_json_config(json_config_path):
    """Carga y lee un archivo de configuración en formato JSON.

    Args:
        json_config_path (str): La ruta completa al archivo .json.

    Returns:
        dict: Un diccionario con el contenido del archivo JSON.

    Raises:
        FileNotFoundError: Si no se encuentra ningún archivo en la ruta
            especificada.
    """
    try:
        with open(json_config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Archivo de configuración no encontrado: {json_config_path}"
        )
    return config


def split_train_test_custom(config, base_dir, test_size=0.1, random_state=0):
    """Carga datos desde un CSV, los divide en conjuntos de entrenamiento y prueba, y los guarda en disco.

    La función lee un archivo de datos CSV especificado en el diccionario de
    configuración. Realiza una limpieza básica en la columna objetivo, divide
    los datos de forma estratificada y guarda los arrays resultantes (X_train,
    y_train, X_test, y_test) como archivos .npy en el directorio base.

    Args:
        config (dict): Diccionario de configuración que debe contener las claves
            'file' (nombre del archivo de datos) y 'target_col_name' (nombre
            de la columna objetivo).
        base_dir (str): El directorio base donde se encuentra el archivo de datos
            y donde se guardarán los conjuntos de entrenamiento y prueba.
        test_size (float, optional): La proporción del dataset que se asignará al
            conjunto de prueba. Por defecto es 0.1.
        random_state (int, optional): La semilla para el generador de números
            aleatorios para garantizar la reproducibilidad. Por defecto es 0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Una tupla con los
            cuatro arrays generados por la división: (X_train, y_train, X_test, y_test).

    Raises:
        FileNotFoundError: Si el archivo de datos CSV no se encuentra en la ruta
            especificada.
    """
    target_col = config["target_col_name"]
    data_file = os.path.join(base_dir, config["file"])

    # Carga el dataset principal desde el archivo CSV.
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo de datos no encontrado: {data_file}")

    # Realiza una limpieza preliminar en la columna objetivo.
    try:
        df[target_col] = df[target_col].str.replace(".", "", regex=False)
    except:
        print("No se encontraron caracteres extranos en la columna target")

    # Separa las características (X) de la variable objetivo (y).
    y = df[target_col].to_numpy()
    X = df.drop(columns=[target_col]).to_numpy()

    # Divide los datos en conjuntos de entrenamiento y prueba de forma estratificada.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Guarda los conjuntos resultantes en disco para uso futuro.
    np.save(f"{base_dir}/X_train.npy", X_train)
    np.save(f"{base_dir}/X_test.npy", X_test)
    np.save(f"{base_dir}/y_train.npy", y_train)
    np.save(f"{base_dir}/y_test.npy", y_test)
    print(f"Datos de train y test guardados en {base_dir}")

    return X_train, y_train, X_test, y_test


def concat_l(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([X, y.reshape(-1, 1)], axis=1)


def preprocessing(
    config,
    X_train,
    y_train,
    X_test,
    y_test,
    encoding="ordinal",
    concat_label=False,
    concat=False,
    random_state=0,
):
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
    num_idx = config["num_col_idx"]
    cat_idx = config["cat_col_idx"]

    # Separa las columnas del dataset en numéricas y categóricas.
    X_num_train = X_train[:, num_idx]
    X_cat_train = X_train[:, cat_idx]
    X_num_test = X_test[:, num_idx]
    X_cat_test = X_test[:, cat_idx]

    # Aplica imputación para manejar valores faltantes según la estrategia definida.
    num_impute_strategy = config["num_imputation"]
    if num_impute_strategy:
        num_imputer = SimpleImputer(strategy=num_impute_strategy)
        X_num_train = num_imputer.fit_transform(X_num_train)
        X_num_test = num_imputer.transform(X_num_test)
    cat_impute_strategy = config["cat_imputation"]
    if cat_impute_strategy:
        cat_imputer = SimpleImputer(strategy=cat_impute_strategy)
        X_cat_train = cat_imputer.fit_transform(X_cat_train)
        X_cat_test = cat_imputer.transform(X_cat_test)

    # Normaliza las características numéricas para que tengan una escala similar.
    norm = config["normalization"]
    num_scaler = None
    if norm == "z-score":
        num_scaler = StandardScaler()
    elif norm == "min-max":
        num_scaler = MinMaxScaler()
    elif norm == "robust":
        num_scaler = RobustScaler()
    elif norm not in (None, "null"):
        raise ValueError(f"Normalización no soportada: {norm}")

    if num_scaler:
        X_num_train = num_scaler.fit_transform(X_num_train)
        X_num_test = num_scaler.transform(X_num_test)

    if concat_label:
        X_train = concat_l(X_train, y_train)

    # Codifica las características categóricas (One-Hot o Ordinal).
    if encoding == "one-hot":
        cat_encoder = OneHotEncoder(sparse_output=False)
    else:
        cat_encoder = OrdinalEncoder()
    if X_cat_train.shape[1] > 0:
        X_cat_train = cat_encoder.fit_transform(X_cat_train)
        X_cat_test = cat_encoder.transform(X_cat_test)

    # Codifica la variable objetivo a valores numéricos.
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # Si se solicita, concatena las características numéricas y categóricas procesadas.
    if concat:
        X_train = np.concatenate((X_num_train, X_cat_train), axis=1)
        X_test = np.concatenate((X_num_test, X_cat_test), axis=1)
        return (
            X_train,
            X_test,
            y_train_enc,
            y_test_enc,
            num_scaler,
            cat_encoder,
            label_encoder,
        )

    return (
        X_num_train,
        X_cat_train,
        X_num_test,
        X_cat_test,
        y_train_enc,
        y_test_enc,
        num_scaler,
        cat_encoder,
        label_encoder,
    )


def load_reconstructed_data(
    df_recon: pd.DataFrame,
    json_config_path: str,
    num_scaler,
    cat_encoder,
    label_encoder,
):
    """Procesa un DataFrame reconstruido para prepararlo para modelado.

    Esta función toma un DataFrame, aplica las mismas transformaciones de escalado
    y codificación que se usaron en los datos originales (usando los `scalers` y
    `encoders` ya entrenados) y devuelve los datos listos para ser utilizados
    por un modelo de machine learning.

    Args:
        df_recon (pd.DataFrame): El DataFrame con los datos reconstruidos.
        json_config_path (str): La ruta al archivo JSON de configuración que
            contiene los nombres de las columnas y sus índices.
        num_scaler: El objeto `scaler` (ej. StandardScaler) ya entrenado para
            transformar las variables numéricas.
        cat_encoder: El objeto `encoder` (ej. OneHotEncoder) ya entrenado para
            transformar las variables categóricas.
        label_encoder: El objeto `encoder` (ej. LabelEncoder) ya entrenado para
            codificar la variable objetivo (target).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Una tupla conteniendo:
            - X_train_pre (np.ndarray): Las variables predictoras procesadas.
            - y_train_pre (np.ndarray): La variable objetivo codificada.
    """
    # Carga la configuración para obtener la estructura de los datos.
    config = read_json_config(json_config_path)
    target_col = config["target_col_name"]
    num_idx = config["num_col_idx"]
    cat_idx = config["cat_col_idx"]

    # Separa las características y el objetivo del DataFrame reconstruido.
    y_train_raw = df_recon[target_col].to_numpy()
    X_train_raw = df_recon.drop(columns=[target_col]).to_numpy()

    # Aplica las mismas transformaciones que se usaron en los datos originales.
    X_num_recon = X_train_raw[:, num_idx]
    if num_scaler:
        X_num_recon = num_scaler.transform(X_num_recon)
    X_cat_recon = X_train_raw[:, cat_idx]
    if cat_encoder:
        X_cat_recon = cat_encoder.transform(X_cat_recon)

    # Une las características numéricas y categóricas ya procesadas.
    X_train_pre = np.concatenate([X_num_recon, X_cat_recon], axis=1)

    # Transforma la variable objetivo usando el codificador ajustado.
    y_train_pre = label_encoder.transform(y_train_raw)

    return X_train_pre, y_train_pre


def reconstruct_data(
    x: str,
    y: str,
    models_paths: str,
    device: torch.device,
    json_config_path: str,
    latent_space=False,
    hyperparams={},
    method="k-means",
    ipc=0,
    seed=0,
    concat_label=False,
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
    # Configura los directorios de salida y carga los hiperparámetros.
    reconstructed_dir = os.path.join(models_paths, "reconstructed_data", f"IPC_{ipc}")
    os.makedirs(reconstructed_dir, exist_ok=True)
    d_token = hyperparams.get("d_token", 4)
    n_head = hyperparams.get("n_head", 1)
    factor = hyperparams.get("factor", 32)
    num_layers = hyperparams.get("num_layers", 2)

    config = read_json_config(json_config_path)
    original_num_columns = config.get("num_col_names")
    original_cat_columns = config.get("cat_col_names")
    original_target_column = config.get("target_col_name")
    original_target_column = (
        original_target_column
        if type(original_target_column) == list
        else [original_target_column]
    )
    num_cols_idx = config["num_col_idx"]
    cat_cols_idx = config["cat_col_idx"]
    target_cols_idx = config["target_col_idx"]

    # Carga los objetos de preprocesamiento (scalers, encoders) guardados previamente.
    with open(os.path.join(models_paths, f"pre_encoders_seed_{seed}.pkl"), "rb") as f:
        encoders = pickle.load(f)
        num_cols = encoders["num_cols"]
        categories = encoders["categories"]
        num_scaler = encoders["num_scaler"]
        cat_encoder = (
            encoders["cat_ordinal_encoder"]
            if latent_space
            else encoders["cat_onehot_encoder"]
        )
        label_encoder = encoders["label_encoder"]

    decoder_inference_time = 0
    # Decide la ruta de reconstrucción: desde el espacio latente o usando datos ya procesados.
    if latent_space:
        # Carga la arquitectura del Decoder y sus pesos entrenados.
        decoder_weights_path = os.path.join(models_paths, f"decoder_seed_{seed}.pt")
        latent_z_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        decoder_model = DecoderModel(
            num_layers, num_cols, categories, d_token, n_head=n_head, factor=factor
        ).to(device)
        try:
            decoder_model.load_state_dict(
                torch.load(decoder_weights_path, map_location=device)
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Archivo de pesos del decoder no encontrado: {decoder_weights_path}"
            )
        decoder_model.eval()

        # Realiza la inferencia para reconstruir los datos desde el espacio latente.
        with torch.no_grad():
            if latent_z_tensor.shape[1] == num_cols + len(categories) + 1:
                latent_z_tensor_for_decoder = latent_z_tensor[:, 1:, :]
            else:
                latent_z_tensor_for_decoder = latent_z_tensor

            start = time.time()
            recon_num_processed, recon_cat_processed_list = decoder_model(
                latent_z_tensor_for_decoder
            )
            decoder_inference_time = time.time() - start

        recon_num_processed_np = recon_num_processed.detach().cpu().numpy()
        if concat_label:
            recon_cat_processed_list = [
                cat_tensor[:, :-1] for cat_tensor in recon_cat_processed_list
            ]
    else:
        recon_num_processed_np = x

    # Invierte las transformaciones para devolver los datos a su escala y formato original.
    recon_cat_encoded_list = []
    for j, recon_cat_logits in enumerate(recon_cat_processed_list):
        predicted_indices = torch.argmax(recon_cat_logits, dim=1)
        recon_cat_encoded_list.append(predicted_indices.cpu().numpy())
    if recon_cat_encoded_list:
        recon_cat_encoded_np = np.column_stack(recon_cat_encoded_list)

    if num_scaler is not None:
        recon_num_original = num_scaler.inverse_transform(recon_num_processed_np)
    else:
        recon_num_original = recon_num_processed_np
    if label_encoder is not None:
        y_reconstructed = label_encoder.inverse_transform(y)
    if cat_encoder is not None and recon_cat_encoded_list:
        recon_cat_original = cat_encoder.inverse_transform(recon_cat_encoded_np)

    # Recompone el DataFrame final, asegurando el orden original de las columnas.
    if original_num_columns is not None:
        df_num = pd.DataFrame(recon_num_original, columns=original_num_columns)
        df_target = pd.DataFrame(y_reconstructed, columns=original_target_column)

        if recon_cat_encoded_list:
            df_cat = pd.DataFrame(recon_cat_original, columns=original_cat_columns)
            df_reconstructed = pd.concat([df_num, df_cat, df_target], axis=1)
            all_columns = (
                dict(zip(num_cols_idx, original_num_columns))
                | dict(zip(cat_cols_idx, original_cat_columns))
                | dict(zip(target_cols_idx, original_target_column))
            )
        else:
            df_reconstructed = pd.concat([df_num, df_target], axis=1)
            all_columns = dict(zip(num_cols_idx, original_num_columns)) | dict(
                zip(target_cols_idx, original_target_column)
            )

        ordered_columns = [all_columns[i] for i in sorted(all_columns)]
        df_reconstructed = df_reconstructed[ordered_columns]

        # Guarda el DataFrame reconstruido en un archivo CSV.
        reconstructed_path = os.path.join(
            reconstructed_dir, f"method_{method}_seed_{seed}.csv"
        )
        df_reconstructed.to_csv(reconstructed_path, index=False)

        return df_reconstructed, decoder_inference_time
    raise ValueError("No se encontraron los nombres de columnas originales.")


def _build_cv(n_splits, random_state):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _objective(trial, cfg, X, y, cv):
    """Función objetivo para Optuna: devuelve balanced‑accuracy promedio."""
    params = {k: v(trial) for k, v in cfg["search_space"].items()}
    model = cfg["constructor"](**cfg["static_args"], **params)
    scores = cross_validate(
        model, X, y, scoring=_METRIC_SCORERS["balanced"], cv=cv, n_jobs=-1
    )
    return scores["test_score"].mean()


def _tune_single_model(tag, cfg, X, y, cv, n_trials, random_state):
    """Tunea hiper-parámetros con Optuna y mide tiempos de tuning y entrenamiento."""
    # Define y ejecuta el estudio de optimización con Optuna.
    tic_opt = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name=tag,
    )
    study.optimize(
        lambda t: _objective(t, cfg, X, y, cv),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    optuna_time = time.time() - tic_opt

    # Recupera los mejores hiperparámetros encontrados en el estudio.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        best_params = {}
        print(f"⚠️  No hay trials completados en '{tag}'; usando argumentos estáticos.")
    else:
        best_params = study.best_params

    # Entrena un modelo final usando los mejores hiperparámetros encontrados.
    best_model = cfg["constructor"](**cfg["static_args"], **best_params)
    tic_train = time.time()
    best_model.fit(X, y)
    train_time = time.time() - tic_train

    # Realiza una validación cruzada sobre el modelo final para obtener métricas de rendimiento robustas.
    cv_res = cross_validate(best_model, X, y, scoring=_METRIC_SCORERS, cv=cv, n_jobs=-1)
    means = {m: cv_res[f"test_{m}"].mean() for m in _METRIC_SCORERS}
    stds = {f"{m}_std": cv_res[f"test_{m}"].std() for m in _METRIC_SCORERS}

    return best_model, means, stds, best_params, optuna_time, train_time


def _evaluate_on_test(model, X_test, y_test):
    """Calcula métricas en el conjunto de prueba y mide tiempo de inferencia."""
    # Mide el tiempo de inferencia al predecir sobre el conjunto de prueba.
    tic_inf = time.time()
    y_pred = model.predict(X_test)
    inf_time_total = time.time() - tic_inf
    time_per_sample = inf_time_total / len(X_test)

    # Calcula un conjunto de métricas de clasificación para evaluar el rendimiento.
    try:
        classes = model.classes_
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
        "inf_time_per_sample": time_per_sample,
    }


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    cv_folds=5,
    n_trials=30,
    ckpt_dir=None,
    method="k-means",
    random_state=0,
    ipc=0,
    space="original",
):
    """Tunea y evalúa clasificadores con Optuna, incluyendo tiempos de tuning, entrenamiento e inferencia.

    Esta función itera sobre una lista predefinida de modelos (MODELS), realiza una
    búsqueda de hiperparámetros para cada uno usando Optuna y validación cruzada.
    Guarda los mejores hiperparámetros, evalúa el mejor modelo en el conjunto de
    prueba y recopila todas las métricas de rendimiento (CV, test) y tiempos
    (optimización, entrenamiento, inferencia) en un DataFrame.

    Args:
        X_train: Datos de entrenamiento para las variables predictoras.
        y_train: Variable objetivo para el entrenamiento.
        X_test: Datos de prueba para las variables predictoras.
        y_test: Variable objetivo para la prueba.
        cv_folds (int, optional): Número de pliegues (folds) para la validación
            cruzada durante el tuning. Por defecto es 5.
        n_trials (int, optional): Número de ensayos de optimización que realizará
            Optuna por cada modelo. Por defecto es 30.
        ckpt_dir (str, optional): Directorio base donde se guardarán los resultados,
            como los mejores parámetros de cada modelo. Si es None, no se guarda
            nada. Por defecto es None.
        method (str, optional): Identificador del método experimental, usado para
            organizar los resultados en carpetas. Por defecto es 'k-means'.
        random_state (int, optional): Semilla para la reproducibilidad de la
            validación cruzada y otros procesos aleatorios. Por defecto es 0.
        ipc (int, optional): Identificador para el experimento (posiblemente
            "items per class"), usado para nombrar carpetas. Por defecto es 0.
        space (str, optional): Identificador para el espacio de características
            usado (ej. 'original', 'reconstruido'), para nombrar carpetas.
            Por defecto es 'original'.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Dict], Dict[str, Any]]: Una tupla que contiene:
        - results_df (pd.DataFrame): Un DataFrame con los resultados agregados de
          todos los modelos, donde cada fila corresponde a un modelo.
        - test_metrics_all (Dict[str, Dict]): Un diccionario donde las claves son
          los nombres de los modelos y los valores son diccionarios con las
          métricas obtenidas en el conjunto de test.
        - best_models (Dict[str, Any]): Un diccionario donde las claves son los
          nombres de los modelos y los valores son los objetos de modelo ya
          entrenados con los mejores hiperparámetros.
    """
    # Prepara el directorio para guardar resultados y la estrategia de validación cruzada.
    best_dir = os.path.join(
        ckpt_dir,
        method,
        f"IPC_{ipc}",
        f"best_models_{method}_seed_{random_state}_space_{space}",
    )
    os.makedirs(best_dir, exist_ok=True)
    cv = _build_cv(max(cv_folds, 2), random_state)

    records = []
    test_metrics_all = {}
    best_models = {}

    # Itera sobre cada modelo configurado para realizar el proceso de tuning y evaluación.
    for tag, cfg in MODELS.items():
        print(
            f"\n\n>>> Optuna tuning {tag} metodo {method} IPC {ipc} seed {random_state}\n\n"
        )
        best_model, cv_means, cv_stds, best_hp, opt_time, train_time = (
            _tune_single_model(tag, cfg, X_train, y_train, cv, n_trials, random_state)
        )

        # Guarda los mejores hiperparámetros encontrados para el modelo actual.
        if ckpt_dir is not None:
            json_path = os.path.join(best_dir, f"{tag}_best_params.json")
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(best_hp, fp, indent=4, ensure_ascii=False)
            print(f"Parámetros guardados en {json_path}")

        # Almacena el modelo entrenado y sus métricas en el conjunto de prueba.
        best_models[tag] = best_model
        test_metrics = _evaluate_on_test(best_model, X_test, y_test)
        test_metrics_all[tag] = test_metrics

        # Recopila y formatea todas las métricas y tiempos en un registro.
        row = {"model": tag, "method": method}
        row.update({f"cv_{m}": round(val, 3) for m, val in cv_means.items()})
        row.update({f"cv_{m_std}": round(val, 3) for m_std, val in cv_stds.items()})
        row["optuna_time"] = round(opt_time, 3)
        row["train_time"] = round(train_time, 3)
        row.update(
            {
                f"test_{m}": round(val, 3)
                for m, val in test_metrics.items()
                if m != "inf_time_per_sample"
            }
        )
        row["inf_time_per_sample"] = round(
            test_metrics.get("inf_time_per_sample", 0), 6
        )
        records.append(row)

    # Convierte la lista de registros en un DataFrame de resultados.
    results_df = pd.DataFrame(records)
    return results_df, test_metrics_all, best_models


def evaluate_one_model(X_train, y_train, X_test, y_test, cv_folds=5, random_state=0):
    """
    Evalúa un XGBClassifier con parámetros fijos.

    Realiza validación cruzada y evaluación en test.

    Retorna:
      - cv_means: medias de métricas en CV
      - cv_stds: desviaciones estándar en CV
      - test_metrics: métricas en el conjunto de prueba
      - model: modelo entrenado sobre todo el set de entrenamiento
    """
    model = XGBClassifier(**MODEL_XGB, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "balanced": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }

    # Evalúa el modelo usando validación cruzada en los datos de entrenamiento.
    cv_res = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)
    cv_means = {m: cv_res[f"test_{m}"].mean() for m in scoring}
    cv_stds = {f"{m}_std": cv_res[f"test_{m}"].std() for m in scoring}

    # Entrena el modelo definitivo con todo el conjunto de entrenamiento.
    model.fit(X_train, y_train, verbose=False)

    # Calcula las métricas finales sobre el conjunto de prueba.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    pos_idx = int(np.where(model.classes_ == True)[0]) if True in model.classes_ else 1
    test_metrics = {
        "balanced": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": (
            roc_auc_score(y_test, y_proba[:, pos_idx], average="weighted")
            if y_proba is not None
            else np.nan
        ),
    }

    return cv_means, cv_stds, test_metrics, model
