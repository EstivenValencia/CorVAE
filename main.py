import argparse
from train_vae import main as main_vae
from distill import (
    distill_with_kmeans,
    distill_random,
    distill_with_agglomerative,
    distill_with_kcenters,
    distill_least_confidence,
    distill_with_craig,
)
from utils import preprocessing
import os
import numpy as np
from utils import (
    evaluate_models,
    reconstruct_data,
    load_reconstructed_data,
    split_train_test_custom,
)
import torch
from utils import read_json_config
import pandas as pd
import json
import time

IPC_LIST = list(range(10, 110, 10))
# IPC_LIST = list(range(10,30,10)) # Solo para depuración

RANDOM_SEED_EVALUATE = range(5, 10)
# RANDOM_SEED_EVALUATE = range(1) # Solo para depuracion

# Semillas estaticas para división de datos
STATIC_SEED = 0
SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Distillation Configuration")

    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to the metadata file or dataset.",
    )

    parser.add_argument(
        "--distillation_space",
        type=str,
        choices=["latent", "original"],
        required=True,
        help="Space in which to perform distillation.",
    )

    parser.add_argument(
        "--epochs_latent",
        type=int,
        required=True,
        help="Number of epochs to train in the latent space.",
    )

    parser.add_argument(
        "--epochs_fine_tuning_latent",
        type=int,
        required=False,
        help="Number of epochs to train fine tunning in the latent space.",
    )

    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size for training."
    )

    parser.add_argument(
        "--concat_label_vae",
        type=int,
        required=False,
        default=False,
        help="Number of examples to use for distillation.",
    )

    parser.add_argument("--checkpoint_path", type=str, required=True, help="")

    parser.add_argument("--kmeans_type", type=str, required=True, help="")

    parser.add_argument("--hyperparams_vae", type=str, required=False, help="")

    parser.add_argument(
        "--architecture",
        type=str,
        choices=["transformer", "mlp"],
        default="transformer",
        help="VAE backbone architecture: 'transformer' (CorVAE) or 'mlp' (ablation).",
    )

    args = parser.parse_args()

    return args


META_COLS = [
    "MLE_space" "IPC",
    "seed",
    "vae_pretrain_time",
    "vae_fine_tunning_time",
    "encoder_inference_time",
    "decoder_inference_time",
    "destillation_time",
]


def add_meta(
    df,
    mle_space="original",
    ipc=None,
    seed=None,
    pretrain_time=None,
    finetune_time=None,
    encoder_time=None,
    decoder_time=None,
    destillation_time=None,
):
    """
    Añade las columnas META_COLS al df, con los valores
    pasados o None por defecto.
    """
    return df.assign(
        space=mle_space,
        IPC=ipc,
        seed=seed,
        vae_pretrain_time=pretrain_time,
        vae_fine_tunning_time=finetune_time,
        encoder_inference_time=encoder_time,
        decoder_inference_time=decoder_time,
        destillation_time=destillation_time,
    )


def flatten_vectors(x: np.ndarray) -> np.ndarray:
    """
    Aplana un array de forma (n, t, d) a (n, t*d).

    Parámetros
    ----------
    x : np.ndarray
        Array de entrada con forma (n, t, d).

    Devuelve
    -------
    np.ndarray
        Array de salida con forma (n, t*d).
    """
    if x.ndim != 3:
        raise ValueError(f"Se esperaba un array 3D, pero x.ndim = {x.ndim}")

    # x = x[:,1:,:] # Eliminar cls

    n, t, d = x.shape
    return x.reshape(n, t * d)


def recontructed_distillation(
    distill_z,
    distill_y,
    test_z,
    test_y,
    X_test_pre,
    y_test_pre,
    method,
    metadata_path,
    hyperparams_vae,
    checkpoint_path,
    vae_dir,
    device,
    ipc,
    num_scaler,
    cat_encoder,
    label_encoder,
    pretrain_time,
    finetune_time,
    encoder_inference_time,
    concat_label,
    seed,
    destillation_time,
    architecture="transformer",
):

    # Reconstrucción de los datos destilados al espacio original
    df_reconstruct, decoder_inference_time = reconstruct_data(
        distill_z,
        distill_y,
        models_paths=vae_dir,
        device=device,
        json_config_path=metadata_path,
        latent_space=True,
        hyperparams=hyperparams_vae,
        method=method,
        ipc=ipc,
        concat_label=concat_label,
        seed=seed,
        architecture=architecture,
    )

    # Se cargan los datos reconstruidos para evaluacion de los modelos
    x_train_disti, y_train_disti = load_reconstructed_data(
        df_reconstruct,
        metadata_path,
        num_scaler=num_scaler,
        cat_encoder=cat_encoder,
        label_encoder=label_encoder,
    )

    # Se utilizan los datos en el espacio latente para la evaluación de los modelos
    flatten_train_z, flatten_test_z = flatten_vectors(distill_z), flatten_vectors(
        test_z
    )
    latent_df, latent_test_metrics, _ = evaluate_models(
        flatten_train_z,
        distill_y,
        flatten_test_z,
        test_y,
        ckpt_dir=checkpoint_path,
        method=method,
        random_state=seed,
        ipc=ipc,
        space="latent",
    )

    # Se agregan el resto de estadisticas a los resultados de la evaluación en latente
    latent_df = add_meta(
        latent_df,
        mle_space="latent",
        ipc=ipc,
        seed=seed,
        pretrain_time=pretrain_time,
        finetune_time=finetune_time,
        encoder_time=encoder_inference_time,
        decoder_time=decoder_inference_time,
        destillation_time=destillation_time,
    )

    # Evaluacion de los modelos con los datos reconstruidos
    original_df, original_test_metrics, _ = evaluate_models(
        x_train_disti,
        y_train_disti,
        X_test_pre,
        y_test_pre,
        ckpt_dir=checkpoint_path,
        method=method,
        random_state=seed,
        ipc=ipc,
    )

    # Se agregan el resto de estadisticas a los resultados de la evaluación en original
    original_df = add_meta(
        original_df,
        ipc=ipc,
        seed=seed,
        pretrain_time=pretrain_time,
        finetune_time=finetune_time,
        encoder_time=encoder_inference_time,
        decoder_time=decoder_inference_time,
        destillation_time=destillation_time,
    )

    return original_df, latent_df, latent_test_metrics, original_test_metrics


def main(args):
    print("Corriendo el pipeline de destilación con la siguiente configuración:")

    metadata_path = args.metadata_path
    distillation_space = args.distillation_space
    epochs_latent = args.epochs_latent
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    kmeans_type = args.kmeans_type
    epochs_fine_tuning_latent = args.epochs_fine_tuning_latent
    hyperparams_vae_path = args.hyperparams_vae
    concat_label = args.concat_label_vae
    architecture = args.architecture

    # Se crea la carpeta donde se guardarán los checkpoints
    os.makedirs(checkpoint_path, exist_ok=True)

    # Entrenar en GPU si está disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Leer el archivo de configuración JSON con los datos del dataset
    config = read_json_config(metadata_path)
    base_dir = os.path.dirname(metadata_path)

    # Se utiliza un test de 0.1 porque luego se aplicar validacion cruzada que ya incluye validacion
    X_train_init, y_train_init, X_test_init, y_test_init = split_train_test_custom(
        config, base_dir, test_size=0.1, random_state=STATIC_SEED
    )

    set_data = (X_train_init, y_train_init, X_test_init, y_test_init)

    # Preprocesamiento de los datos
    (
        X_train_pre,
        X_test_pre,
        y_train_pre,
        y_test_pre,
        num_scaler,
        cat_encoder,
        label_encoder,
    ) = preprocessing(
        config,
        X_train_init,
        y_train_init,
        X_test_init,
        y_test_init,
        encoding="one-hot",
        concat=True,
        random_state=SEED,
    )

    # Entrenamiento de los modelos de evaluación con todos los datos
    full_df, _, _ = evaluate_models(
         X_train_pre,
         y_train_pre,
         X_test_pre,
         y_test_pre,
         ckpt_dir=checkpoint_path,
         method="Full-data",
         random_state=SEED,
         n_trials=30,
     )

    # # Se agrega información para la generación de los resultados
    full_df = add_meta(full_df)
    #full_df = pd.DataFrame()
    # Se realiza entrenamiento del VAE sobre las diferentes semillas
    print("\n\n\n----------------FINALIZO-------------\n\n\n")
    # Medición de tiempos en el entrenamiento del VAE en las diferentes semillas
    times = {}

    # Inicialización de los tiempos
    pretrain_time, finetune_time, encoder_inference_time = 0, 0, 0
    if distillation_space == "latent":

        # Se carga el diccionario con los parametros del VAE
        with open(hyperparams_vae_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
            hyperparams_vae = data_dict["best_params"]

        # Se define el directorio donde se guardarán los checkpoints del VAE
        vae_dir = os.path.join(checkpoint_path, "vae")
        os.makedirs(vae_dir, exist_ok=True)

        for seed in RANDOM_SEED_EVALUATE:
            _, _, pretrain_time, finetune_time, encoder_inference_time = main_vae(
                config,
                base_dir,
                set_data,
                encoding="ordinal",
                random_state=seed,
                batch_size=batch_size,
                pretrain_epochs=epochs_latent,
                finetune_epochs=epochs_fine_tuning_latent,
                ckpt_dir=vae_dir,
                hyperparams=hyperparams_vae,
                concat_label=concat_label,
                architecture=architecture,
            )
            times[seed] = (pretrain_time, finetune_time, encoder_inference_time)

    # Proceso de destilación utilizando el espacio latente o el espacio original
    for ipc in IPC_LIST:

        # Destilación con sample random class
        for seed in RANDOM_SEED_EVALUATE:

            # Destilación utilizando muesttras aleatorias de cada clase
            X_random, y_random = distill_random(
                X_train_pre, y_train_pre, n_per_class=ipc, random_state=seed
            )

            random_df, _, _ = evaluate_models(
                X_random,
                y_random,
                X_test_pre,
                y_test_pre,
                ckpt_dir=checkpoint_path,
                method="random",
                random_state=seed,
                ipc=ipc,
            )
            random_df = add_meta(random_df, ipc=ipc, seed=seed)

            full_df = pd.concat([full_df, random_df], axis=0, ignore_index=True)

            # ------------------------------------------------------
            # destilación en el espacio latente
            # ------------------------------------------------------
            if distillation_space == "latent":
                pretrain_time, finetune_time, encoder_inference_time = times[seed]

                # Se cargan los datos de destilacion
                train_z_path, train_y_path = os.path.join(
                    vae_dir, f"train_z_seed_{seed}.npy"
                ), os.path.join(vae_dir, f"train_y_seed_{seed}.npy")
                test_z_path, test_y_path = os.path.join(
                    vae_dir, f"test_z_seed_{seed}.npy"
                ), os.path.join(vae_dir, f"test_y_seed_{seed}.npy")

                train_z, train_y = np.load(train_z_path), np.load(train_y_path)
                test_z, test_y = np.load(test_z_path), np.load(test_y_path)

                # ---------------------------------------------------------------
                # Destilación con K-means en el espacio latente y reconstruyendo
                # ---------------------------------------------------------------
                if kmeans_type == "centroid":
                    begin = time.time()
                    distill_z, distill_y = distill_with_kmeans(
                        train_z,
                        train_y,
                        num_centroids=ipc,
                        get_closest=False,
                        seed=seed,
                    )
                    end = time.time()
                    destillation_time = end - begin

                    # Se reconstruyen los datos que son destilados y se entregan los dataframes
                    # con los resultados de la evaluación en el espacio original y latente
                    k_means_df, k_means_latent_df, _, _ = recontructed_distillation(
                        distill_z,
                        distill_y,
                        test_z,
                        test_y,
                        X_test_pre,
                        y_test_pre,
                        "k-means",
                        metadata_path,
                        hyperparams_vae,
                        checkpoint_path,
                        vae_dir,
                        device,
                        ipc,
                        num_scaler,
                        cat_encoder,
                        label_encoder,
                        pretrain_time,
                        finetune_time,
                        encoder_inference_time,
                        concat_label,
                        seed,
                        destillation_time,
                        architecture=architecture,
                    )
                # ------------------------------------------
                # Destilación con K-center y recontruyendo
                # ------------------------------------------
                begin = time.time()
                distill_z, distill_y = distill_with_kcenters(
                    train_z, train_y, num_centroids=ipc, seed=seed
                )
                end = time.time()
                destillation_time = end - begin
                k_center_df, k_center_latent_df, _, _ = recontructed_distillation(
                    distill_z,
                    distill_y,
                    test_z,
                    test_y,
                    X_test_pre,
                    y_test_pre,
                    "k-center",
                    metadata_path,
                    hyperparams_vae,
                    checkpoint_path,
                    vae_dir,
                    device,
                    ipc,
                    num_scaler,
                    cat_encoder,
                    label_encoder,
                    pretrain_time,
                    finetune_time,
                    encoder_inference_time,
                    concat_label,
                    seed,
                    destillation_time,
                    architecture=architecture,
                )
                # ------------------------------------------
                # Destilacion con AG y reconstruyendo
                # ------------------------------------------
                begin = time.time()
                distill_z, distill_y = distill_with_agglomerative(
                    train_z, train_y, num_clusters=ipc, get_closest=False, seed=seed
                )
                end = time.time()
                destillation_time = end - begin
                ag_df, ag_latent_df, _, _ = recontructed_distillation(
                    distill_z,
                    distill_y,
                    test_z,
                    test_y,
                    X_test_pre,
                    y_test_pre,
                    "ag",
                    metadata_path,
                    hyperparams_vae,
                    checkpoint_path,
                    vae_dir,
                    device,
                    ipc,
                    num_scaler,
                    cat_encoder,
                    label_encoder,
                    pretrain_time,
                    finetune_time,
                    encoder_inference_time,
                    concat_label,
                    seed,
                    destillation_time,
                    architecture=architecture,
                )
                # --------------------------------------------------
                # Destilación con Least Confidence y recontruyendo
                # --------------------------------------------------
                begin = time.time()
                distill_z, distill_y = distill_least_confidence(
                    train_z, train_y, num_samples=ipc, seed=seed
                )
                end = time.time()
                destillation_time = end - begin
                lc_df, lc_latent_df, _, _ = recontructed_distillation(
                    distill_z,
                    distill_y,
                    test_z,
                    test_y,
                    X_test_pre,
                    y_test_pre,
                    "lc",
                    metadata_path,
                    hyperparams_vae,
                    checkpoint_path,
                    vae_dir,
                    device,
                    ipc,
                    num_scaler,
                    cat_encoder,
                    label_encoder,
                    pretrain_time,
                    finetune_time,
                    encoder_inference_time,
                    concat_label,
                    seed,
                    destillation_time,
                    architecture=architecture,
                )
                full_df = pd.concat(
                    [
                        full_df,
                        k_means_df,
                        k_means_latent_df,
                        k_center_df,
                        k_center_latent_df,
                        ag_df,
                        ag_latent_df,
                        lc_df,
                        lc_latent_df,
                    ],
                    axis=0,
                    ignore_index=True,
                )

            # ------------------------------------------------------
            # destilación en el espacio original
            # ------------------------------------------------------
            elif distillation_space == "original":
                # -------------------------------------------------
                # Destilación con K-means en el espacio original
                # -------------------------------------------------
                begin = time.time()
                distill_z, distill_y = distill_with_kmeans(
                    X_train_pre,
                    y_train_pre,
                    num_centroids=ipc,
                    get_closest=False,
                    seed=seed,
                )
                end = time.time()
                destillation_time = end - begin
                k_means_df, _, _ = evaluate_models(
                    distill_z,
                    distill_y,
                    X_test_pre,
                    y_test_pre,
                    ckpt_dir=checkpoint_path,
                    method="k-means",
                    random_state=seed,
                    ipc=ipc,
                )

                k_means_df = add_meta(
                    k_means_df, ipc=ipc, seed=seed, destillation_time=destillation_time
                )

                # -------------------------------------------------
                # Destilación con AG
                # -------------------------------------------------
                begin = time.time()
                distill_z, distill_y = distill_with_agglomerative(
                    X_train_pre,
                    y_train_pre,
                    num_clusters=ipc,
                    get_closest=False,
                    seed=seed,
                )
                end = time.time()
                destillation_time = end - begin
                ag_df, _, _ = evaluate_models(
                    distill_z,
                    distill_y,
                    X_test_pre,
                    y_test_pre,
                    ckpt_dir=checkpoint_path,
                    method="ag",
                    random_state=seed,
                    ipc=ipc,
                )

                ag_df = add_meta(
                    ag_df, ipc=ipc, seed=seed, destillation_time=destillation_time
                )

                # -------------------------------------------------
                # Destilación con k-centers
                # -------------------------------------------------
                begin = time.time()
                distill_z, distill_y = distill_with_kcenters(
                    X_train_pre, y_train_pre, num_centroids=ipc, seed=seed
                )
                end = time.time()
                destillation_time = end - begin
                k_center_df, _, _ = evaluate_models(
                    distill_z,
                    distill_y,
                    X_test_pre,
                    y_test_pre,
                    ckpt_dir=checkpoint_path,
                    method="k_center",
                    random_state=seed,
                    ipc=ipc,
                )

                k_center_df = add_meta(
                    k_center_df, ipc=ipc, seed=seed, destillation_time=destillation_time
                )

                # -------------------------------------------------
                # Destilación con LC
                # -------------------------------------------------
                begin = time.time()
                distill_z, distill_y = distill_least_confidence(
                    X_train_pre, y_train_pre, num_samples=ipc, seed=seed
                )
                end = time.time()
                destillation_time = end - begin
                lc_df, _, _ = evaluate_models(
                    distill_z,
                    distill_y,
                    X_test_pre,
                    y_test_pre,
                    ckpt_dir=checkpoint_path,
                    method="lc",
                    random_state=seed,
                    ipc=ipc,
                )

                lc_df = add_meta(
                    lc_df, ipc=ipc, seed=seed, destillation_time=destillation_time
                )

                full_df = pd.concat(
                    [full_df, k_means_df, ag_df, k_center_df, lc_df],
                    axis=0,
                    ignore_index=True,
                )

        # Se guarda
        full_df.to_csv(os.path.join(checkpoint_path, "metrics.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
