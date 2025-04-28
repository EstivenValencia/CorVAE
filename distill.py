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
import ray


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

