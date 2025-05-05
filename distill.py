from utils import reconstruct_data
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import preprocessing
from models import ModelVAE, compute_loss, EncoderModel, DecoderModel

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time

import torch.nn.functional as F
import pickle


import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import pairwise_distances
from xgboost import XGBClassifier

def distill_random(X, y, n_per_class=10, random_state=None):
    """
    Toma X (array-like de forma [n_samples, ...]) y y (vector de etiquetas de longitud n_samples),
    y devuelve un sub-conjunto con n_per_class muestras por cada clase de y.

    Parámetros
    ----------
    X : array-like, shape (n_samples, n_features)
        Matriz de características.
    y : array-like, shape (n_samples,)
        Etiquetas correspondientes.
    n_per_class : int, opcional (default=10)
        Número de muestras que queremos tomar de cada clase.
    seed : int o None, opcional
        Semilla para la generación de números aleatorios. Si es None, no fija semilla.

    Devuelve
    -------
    X_sample : np.ndarray, shape (n_classes * n_per_class, n_features)
        Subconjunto de X con n_per_class muestras por clase.
    y_sample : np.ndarray, shape (n_classes * n_per_class,)
        Etiquetas correspondientes a X_sample.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.RandomState(random_state)

    classes = np.unique(y)
    sampled_indices = []

    for cls in classes:
        # índices de la clase cls
        idx = np.where(y == cls)[0]
        if len(idx) < n_per_class:
            raise ValueError(f"Clase {cls!r} sólo tiene {len(idx)} muestras, < n_per_class={n_per_class}")
        # barajar y tomar las primeras n_per_class
        rng.shuffle(idx)
        sampled_indices.extend(idx[:n_per_class])

    # opcional: barajar las filas resultantes
    sampled_indices = np.array(sampled_indices)
    rng.shuffle(sampled_indices)

    X_sample = X[sampled_indices]
    y_sample = y[sampled_indices]
    return X_sample, y_sample

def distill_with_kmeans(
    data: np.ndarray,
    labels: np.ndarray,
    num_centroids: int,
    seed: int = 0,
    get_closest: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Aplica KMeans por clase. Acepta `data` de forma [n, f] ó [n, t, d].
    Si return_indices=True, también devuelve índices de muestras originales.
    """
    # --- Detectar y preparar flatten & reconstruct ---
    if data.ndim == 3:
        n, t, d = data.shape
        data_flat = data.reshape(n, t * d)
        def reconstruct(arr: np.ndarray) -> np.ndarray:
            # arr: [num_centroids, t*d] -> [num_centroids, t, d]
            return arr.reshape(num_centroids, t, d)
    elif data.ndim == 2:
        n, f = data.shape
        data_flat = data
        def reconstruct(arr: np.ndarray) -> np.ndarray:
            # arr: [num_centroids, f] -> [num_centroids, f] (sin cambio)
            return arr
    else:
        raise ValueError(f"distill_with_kmeans: data debe ser 2D o 3D, no {data.ndim}D.")

    selected_points = []
    new_labels      = []
    selected_indices = [] if get_closest else None

    # --- Agrupar y clusterizar por etiqueta ---
    for class_id in np.unique(labels):
        mask = (labels == class_id)
        class_data_flat = data_flat[mask]
        original_idxs   = np.nonzero(mask)[0]

        kmeans = KMeans(
            n_clusters=num_centroids,
            random_state=seed,
            n_init="auto"
        )
        kmeans.fit(class_data_flat)

        centers_flat = kmeans.cluster_centers_

        if get_closest:
            # Distancia L2 de cada muestra a cada centroide
            dist = np.linalg.norm(
                class_data_flat[:, None, :] - centers_flat[None, :, :],
                axis=2
            )  # shape [n_c, num_centroids]
            nearest_pos = dist.argmin(axis=0)     # para cada centroide, índice en class_data_flat
            sel_idx = original_idxs[nearest_pos]  # índice en data original
            selected_indices.extend(sel_idx.tolist())
            selected_points.append(data[sel_idx])
        else:
            # Reconstruir forma final de los centroides
            selected_points.append(reconstruct(centers_flat))

        new_labels.extend([class_id] * num_centroids)

    # --- Output final ---
    final_data   = np.vstack(selected_points)
    final_labels = np.array(new_labels)

    return final_data, final_labels

def distill_with_agglomerative(
    data: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
    seed: int = 0,
    get_closest: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Aplica AgglomerativeClustering por clase.

    Parámetros
    ----------
    data : np.ndarray
        Array de forma [n, f] o [n, t, d].
    labels : np.ndarray
        Etiqueta (entera) de cada fila en data, de largo n.
    num_clusters : int
        Número de clusters/centroides a generar por clase.
    seed : int, opcional
        Semilla para reproducibilidad (no usada por defecto en Agglomerative).
    return_indices : bool, opcional
        Si True, devuelve además los índices de las muestras originales
        más cercanas a cada cluster; si False, devuelve los centroides.

    Devuelve
    -------
    final_data : np.ndarray
        Array de forma [num_clusters * n_classes, f] o [num_clusters * n_classes, t, d]
        con los centroides o puntos seleccionados.
    final_labels : np.ndarray
        Array de shape [num_clusters * n_classes] con la etiqueta de cada muestra.
    final_indices : np.ndarray | None
        Índices en el array original de las muestras seleccionadas (si return_indices=True),
        o None.
    """
    # — Detectar y preparar flatten & reconstruct —
    if data.ndim == 3:
        n, t, d = data.shape
        data_flat = data.reshape(n, t * d)
        def reconstruct(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(-1, t, d)
    elif data.ndim == 2:
        n, f = data.shape
        data_flat = data
        def reconstruct(arr: np.ndarray) -> np.ndarray:
            return arr
    else:
        raise ValueError(f"distill_with_agglomerative: data debe ser 2D o 3D, no {data.ndim}D.")

    selected_points = []
    new_labels      = []
    selected_indices = [] if get_closest else None
    all_idxs = np.arange(n)

    # — Agrupar y clusterizar por etiqueta —
    for class_id in np.unique(labels):
        mask = (labels == class_id)
        class_data = data_flat[mask]
        orig_idxs  = all_idxs[mask]

        # Clustering jerárquico
        clustering = AgglomerativeClustering(
            n_clusters=num_clusters
        ).fit(class_data)

        if get_closest:
            # Distancia L2 de cada muestra a cada cluster
            centers = []
            for k in range(num_clusters):
                pts_in_k = class_data[clustering.labels_ == k]
                # centroide temporal
                centers.append(pts_in_k.mean(axis=0))
            centers = np.stack(centers, axis=0)  # [num_clusters, dim]

            # Para cada centroide, elegimos la muestra más cercana
            dists = np.linalg.norm(class_data[:, None, :] - centers[None, :, :], axis=2)
            nearest = dists.argmin(axis=0)               # posición en class_data
            sel_idx = orig_idxs[nearest]                # índice en data original
            selected_indices.extend(sel_idx.tolist())
            selected_points.append(data[sel_idx])
        else:
            # Calculamos centroides “oficiales” con NearestCentroid
            nc = NearestCentroid().fit(class_data, clustering.labels_)
            centers_flat = nc.centroids_              # [num_clusters, dim]
            selected_points.append(reconstruct(centers_flat))

        new_labels.extend([class_id] * num_clusters)

    # — Preparar salida —
    final_data   = np.vstack(selected_points)
    final_labels = np.array(new_labels)

    return final_data, final_labels


def k_center_greedy(
    X: np.ndarray,
    k: int,
    metric: str = 'euclidean',
    random_state: int = None
) -> list[int]:
    """
    Greedy K-Center: selecciona k índices de X como centros.
    """
    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)

    # 1) Primer centro al azar
    first = rng.randint(0, n_samples)
    centers = [int(first)]

    # 2) Distancias iniciales al primer centro
    dist = pairwise_distances(X, X[[first]], metric=metric).reshape(-1)

    # 3) Añadir los k-1 siguientes centros
    for _ in range(1, k):
        nxt = int(np.argmax(dist))
        centers.append(nxt)
        new_dist = pairwise_distances(X, X[[nxt]], metric=metric).reshape(-1)
        dist = np.minimum(dist, new_dist)

    return centers

def distill_with_kcenters(
    data: np.ndarray,
    labels: np.ndarray,
    num_centroids: int,
    seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica K-Centers por clase. Devuelve (new_data, new_labels).

    - Si una clase tiene <= num_centroids muestras, devuelve todas.
    - Si tiene > num_centroids, clampa k y aplica el greedy.
    """

    # 1) Aplanar y preparar reconstrucción
    if data.ndim == 3:
        n, t, d = data.shape
        data_flat = data.reshape(n, t * d)
        def reconstruct(arr_flat):
            return arr_flat.reshape(-1, t, d)
    elif data.ndim == 2:
        data_flat = data
        def reconstruct(arr_flat):
            return arr_flat
    else:
        raise ValueError(f"data.ndim debe ser 2 o 3, no {data.ndim}")

    all_centers = []
    all_labels  = []

    for cls in np.unique(labels):
        # índices de la clase y sus muestras planas
        idxs = np.where(labels == cls)[0]
        Xc   = data_flat[idxs]
        n_c  = Xc.shape[0]
        if n_c == 0:
            continue

        # número real de centros a seleccionar
        k = min(num_centroids, n_c)

        if n_c <= num_centroids:
            # tomo todas las muestras
            sel_abs = idxs
        else:
            # greedy k-center: asegúrate de importar tu función
            centers_rel = k_center_greedy(
                Xc,
                k,
                metric='euclidean',
                random_state=seed
            )
            sel_abs = idxs[centers_rel]

        # extraer y reconstruir
        C_flat = data_flat[sel_abs]
        C = reconstruct(C_flat)

        all_centers.append(C)
        all_labels.extend([cls] * C.shape[0])

    # 3) concatenar y devolver
    if not all_centers:
        # caso borde: no hay datos
        return np.empty((0,) + data.shape[1:]), np.empty((0,))

    new_data   = np.vstack(all_centers)
    new_labels = np.array(all_labels)
    return new_data, new_labels

def distill_least_confidence(
    data: np.ndarray,
    labels: np.ndarray,
    num_samples: int,
    model=None,
    seed: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Para cada clase en `labels`, selecciona los `num_samples` ejemplos
    con menor confianza usando un modelo. Si `model` es None, se entrena
    automáticamente un LogisticRegression con random_state=seed.

    Parámetros
    ----------
    data : np.ndarray, shape (n, f) or (n, t, d)
    labels : np.ndarray, shape (n,)
    num_samples : int
    model : objeto con .predict_proba(X) y .classes_, o None
    seed : int, opcional
    
    Retorna
    -------
    selected_data : np.ndarray, shape (k·num_samples, f) o (k·num_samples, t, d)
    selected_labels : np.ndarray, shape (k·num_samples,)
    """
    # 1) Flatten si es 3D, y preparar reconstrucción
    if data.ndim == 3:
        n, t, d = data.shape
        data_flat = data.reshape(n, t * d)
        def reconstruct(arr_flat: np.ndarray) -> np.ndarray:
            return arr_flat.reshape(-1, t, d)
    elif data.ndim == 2:
        data_flat = data
        def reconstruct(arr_flat: np.ndarray) -> np.ndarray:
            return arr_flat
    else:
        raise ValueError(f"data.ndim debe ser 2 o 3, no {data.ndim}")

    # 2) Si no hay modelo, entrenar uno automáticamente
    if model is None:
        clf = LogisticRegression(
            random_state=seed,
            max_iter=500,
            multi_class='auto',
            solver='lbfgs'
        )
        clf.fit(data_flat, labels)
        model = clf

    # 3) Preparar mapeo de clase → posición en predict_proba
    class_indices = {
        cls: int(np.where(model.classes_ == cls)[0][0])
        for cls in model.classes_
    }

    selected_list = []
    labels_list   = []

    # 4) Para cada clase, seleccionar los menos confiados
    for cls in np.unique(labels):
        mask = (labels == cls)
        Xc   = data_flat[mask]

        print("Clases: ",mask)
        idxs = np.nonzero(mask)

        proba = model.predict_proba(Xc)
        pos_idx = class_indices[cls]
        confidences = proba[:, pos_idx]

        order = np.argsort(confidences, kind='stable')
        chosen = order[:min(num_samples, Xc.shape[0])]

        print("Forma de chosen: ",chosen,type(chosen))
        print("Forma de idxs:", idxs, type(idxs))
        try:
            abs_idx = idxs[chosen]
        except:
            abs_idx = idxs[0][chosen]
        sel_flat = data_flat[abs_idx]

        selected_list.append(reconstruct(sel_flat))
        labels_list.extend([cls] * sel_flat.shape[0])

    # 5) Concatenar resultados
    selected_data   = np.vstack(selected_list)
    selected_labels = np.array(labels_list)

    return selected_data, selected_labels

def distill_with_craig(
    data: np.ndarray,
    labels: np.ndarray,
    num_samples: int,
    batch_size: int = 32,
    device: str = None,
    metric: str = 'euclidean',
    seed: int = None,
    train_epochs: int = 5,
    lr: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica CRAIG Sampling: entrena un modelo lineal interno y selecciona "num_samples" muestras
    por clase usando un coreset de gradientes.

    Parámetros
    ----------
    data : np.ndarray, shape (n, f) o (n, t, d)
        Datos a muestrear.
    labels : np.ndarray, shape (n,)
        Etiquetas verdaderas.
    num_samples : int
        Número de ejemplos a extraer por clase.
    batch_size : int, opcional
        Tamaño de lote para DataLoader (por defecto 32).
    device : str, opcional
        Dispositivo ('cuda' o 'cpu'). Si None, se detecta automáticamente.
    metric : str
        Métrica de distancia para k_center_greedy.
    seed : int, opcional
        Semilla para reproducibilidad.
    train_epochs : int
        Número de épocas para entrenar el modelo interno.
    lr : float
        Learning rate para el optimizador.

    Retorna
    -------
    selected_data : np.ndarray
        Muestras seleccionadas, misma dimensionalidad que `data`.
    selected_labels : np.ndarray
        Etiquetas de las muestras seleccionadas.
    """
    # Preparar dispositivo
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Flatten y reconstrucción
    if data.ndim == 3:
        labels = labels.cpu().numpy()
        n, t, d = data.shape
        data_flat = data.reshape(n, t * d)
        def reconstruct(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(-1, t, d)
    elif data.ndim == 2:
        n, f = data.shape
        data_flat = data
        def reconstruct(arr: np.ndarray) -> np.ndarray:
            return arr
    else:
        raise ValueError(f"data.ndim debe ser 2 o 3, no {data.ndim}")

    # Mapear etiquetas a índices consecutivos 0..C-1
    classes = np.unique(labels)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_idx = np.array([class_to_idx[lab] for lab in labels])

    # Crear y entrenar modelo lineal interno
    input_dim = data_flat.shape[1]
    num_classes = len(classes)
    model = nn.Linear(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.from_numpy(data_flat).float()
    y_tensor = torch.from_numpy(y_idx).long()
    train_ds = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(train_epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # Calcular gradientes por muestra
    model.eval()
    X_ds = TensorDataset(X_tensor)
    loader = DataLoader(X_ds, batch_size=batch_size, shuffle=False)
    grads = []
    with torch.enable_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            for x in xb:
                x = x.unsqueeze(0)
                model.zero_grad()
                logits = model(x)
                pred = logits.argmax(dim=1)
                loss = F.cross_entropy(logits, pred)
                loss.backward()
                grad_list = [p.grad.detach().cpu().flatten() for p in model.parameters() if p.grad is not None]
                grads.append(torch.cat(grad_list).numpy())

    grads = np.stack(grads, axis=0)

    # Seleccionar índices por k-center en el espacio de gradientes
    centers_idx = k_center_greedy(grads, num_samples, metric=metric, random_state=seed)

    # Extraer datos y etiquetas seleccionados
    sel_flat = data_flat[centers_idx]
    sel_labels = labels[centers_idx]
    selected_data = reconstruct(sel_flat)

    return selected_data, sel_labels








################ Sirve para plotaer las categorias del dataset

# df = pd.read_csv('/mnt/d/home-2/Documentos/master/practica-deusto-tech/TFM/MyTabsyn/CorVAE/data/shoppers/dataset.csv')
# columns = list(df.columns)

# def changeN(list):
#     print("[", end='', sep='')
#     for i in list:
#         print('"',i,'",', end='', sep='')
#     print("]", end='', sep='')
#     print("")

# names1 = [columns[i] for i in [0,1,2,3,4,5,6,7,8,9]]
# names2 = [columns[i] for i in [10,11,12,13,14,15,16]]

# changeN(names1)
# changeN(names2)
# print(columns[17])

# def load_original_space():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     os.makedirs(ckpt_dir, exist_ok=True)
#     model_save_path = os.path.join(ckpt_dir, 'best_vae.pt')

#     (
#         X_num_train, X_cat_train,
#         X_num_test,  X_cat_test,
#         y_train_enc, y_test_enc,
#         num_scaler,  cat_encoder,
#         label_encoder
#                         ) = preprocessing(dataset_path, encoding='ordinal', random_state=random_state)

# if __name__ == '__main__':
#     encoder = 

