import argparse
from train_vae import main as main_vae
from distill import distill_with_kmeans, distill_random, distill_with_agglomerative, distill_with_kcenters, distill_least_confidence, distill_with_craig
from utils import preprocessing
import os
import numpy as np
from utils import evaluate_models, reconstruct_data, load_reconstructed_data, split_train_test_custom, compute_relative_regret
import torch
from utils import read_json_config
import pandas as pd
import json

IPC_LIST = list(range(10,110,10))

#IPC_LIST = list(range(10,30,10)) # Solo para depuración
RANDOM_SEED_EVALUATE = range(5,10)
#RANDOM_SEED_EVALUATE = range(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Distillation Configuration")

    parser.add_argument('--metadata_path', type=str, required=True, help='Path to the metadata file or dataset.')
    
    # parser.add_argument('--method_distillation', type=str, choices=['original', 'k-means', 'craig', 'k-center'], required=True,
    #                     help='Distillation method to use.')
    
    parser.add_argument('--distillation_space', type=str, choices=['latent', 'original'], required=True,
                        help='Space in which to perform distillation.')

    # parser.add_argument('--fine_tuning', type=str, choices=['true', 'false'], required=True,
    #                     help='Whether to apply fine-tuning after distillation.')

    parser.add_argument('--epochs_latent', type=int, required=True,
                        help='Number of epochs to train in the latent space.')

    parser.add_argument('--epochs_fine_tuning_latent', type=int, required=False,
                        help='Number of epochs to train fine tunning in the latent space.')

    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size for training.')

    # parser.add_argument('--token_dimension', type=int, required=True,
    #                     help='Dimension of token embeddings.')

    # parser.add_argument('--alpha', type=float, default=None,
    #                     help='Alpha value for fine-tuning (only used if fine_tuning is true).')

    parser.add_argument('--concat_label_vae', type=int, required=False, default=False,
                        help='Number of examples to use for distillation.')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='')
    
    parser.add_argument('--kmeans_type', type=str, required=True,
                        help='')

    parser.add_argument('--hyperparams_vae', type=str, required=False,
                        help='')

    args = parser.parse_args()

    return args

META_COLS = [
    "MLE_space"
    "IPC",
    "seed",
    "vae_pretrain_time",
    "vae_fine_tunning_time",
    "encoder_inference_time",
    "decoder_inference_time",
]

def add_meta(df, mle_space='original',ipc=None, seed=None,
             pretrain_time=None, finetune_time=None,
             encoder_time=None, decoder_time=None):
    """
    Añade de un tirón las columnas META_COLS al df, con los valores
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
    
    #x = x[:,1:,:]
    
    n, t, d = x.shape
    return x.reshape(n, t * d)

def recontructed_distillation(distill_z, distill_y, test_z, test_y, X_test_pre, y_test_pre,method, 
                              metadata_path, hyperparams_vae, checkpoint_path, 
                              vae_dir, device, ipc, num_scaler, cat_encoder, label_encoder, 
                              pretrain_time, finetune_time, encoder_inference_time, concat_label,
                              seed, ):
    
    df_reconstruct, decoder_inference_time = reconstruct_data(distill_z, distill_y, models_paths=vae_dir,device=device,
                                                                json_config_path=metadata_path, latent_space=True, 
                                                                hyperparams=hyperparams_vae, method=method, ipc=ipc, concat_label=concat_label, seed=seed)

    x_train_disti, y_train_disti = load_reconstructed_data(df_reconstruct, metadata_path, 
                                                                                        num_scaler=num_scaler,
                                                                                        cat_encoder=cat_encoder,label_encoder=label_encoder)
    
    # Evaluacion con espacio latente
    flatten_train_z, flatten_test_z = flatten_vectors(distill_z), flatten_vectors(test_z)
    latent_df, latent_test_metrics, _ = evaluate_models(flatten_train_z, distill_y, flatten_test_z, test_y, 
                                                            ckpt_dir=checkpoint_path, method=method, random_state=seed, 
                                                            ipc=ipc, space='latent')

    latent_df = add_meta(latent_df,
                        mle_space='latent',
                        ipc=ipc,
                        seed=seed,
                        pretrain_time=pretrain_time,
                        finetune_time=finetune_time,
                        encoder_time=encoder_inference_time,
                        decoder_time=decoder_inference_time)
    
    # # Evaluacion con reconstrucción
    original_df, original_test_metrics, _ = evaluate_models(x_train_disti, y_train_disti, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method=method, random_state=seed, ipc=ipc)

    original_df = add_meta(original_df,
                        ipc=ipc,
                        seed=seed,
                        pretrain_time=pretrain_time,
                        finetune_time=finetune_time,
                        encoder_time=encoder_inference_time,
                        decoder_time=decoder_inference_time)
    
    return original_df, latent_df, latent_test_metrics, original_test_metrics


def main(args):
    print("Running distillation pipeline with the following configuration:")

    metadata_path             = args.metadata_path
    distillation_space        = args.distillation_space
    epochs_latent             = args.epochs_latent
    batch_size                = args.batch_size
    checkpoint_path           = args.checkpoint_path
    kmeans_type              = args.kmeans_type
    epochs_fine_tuning_latent = args.epochs_fine_tuning_latent
    hyperparams_vae_path      = args.hyperparams_vae
    concat_label              = args.concat_label_vae   

    os.makedirs(checkpoint_path, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = read_json_config(metadata_path)
    base_dir = os.path.dirname(metadata_path)

    STATIC_SEED = 0
    SEED = 0

    # Se utiliza un test de 0.1 porque luego se aplicar validacion cruzada que ya incluye validacion
    X_train_init, y_train_init, X_test_init, y_test_init = split_train_test_custom(config, base_dir, test_size=0.1, random_state=STATIC_SEED)

    set_data = (X_train_init, y_train_init, X_test_init, y_test_init)

    (
        X_train_pre, X_test_pre,
        y_train_pre, y_test_pre,
        num_scaler,  cat_encoder,
        label_encoder
                        ) = preprocessing(config, X_train_init, y_train_init, X_test_init, y_test_init, encoding='one-hot', concat=True, random_state=SEED)

    # Entrenamiento con todos los datos
    #full_df, _, _ = evaluate_models(X_train_pre, y_train_pre, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='Full-data', random_state=SEED)
    #full_df = add_meta(full_df)
    full_df = pd.DataFrame()
    # Se realiza entrenamiento del VAE sobre las diferentes semillas
    if distillation_space == 'latent':

        # Se carga el diccionario con los parametros del VAE
        with open(hyperparams_vae_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            hyperparams_vae = data_dict['best_params']

        vae_dir = os.path.join(checkpoint_path, 'vae') 
        os.makedirs(vae_dir, exist_ok=True)

        times = {}
        pretrain_time, finetune_time, encoder_inference_time = 0,0,0
        for seed in RANDOM_SEED_EVALUATE:
            # _, _,  pretrain_time, finetune_time, encoder_inference_time = main_vae(config, base_dir, set_data, encoding='ordinal', random_state=seed, 
            #                             batch_size=batch_size, pretrain_epochs=epochs_latent, 
            #                             finetune_epochs=epochs_fine_tuning_latent, ckpt_dir=vae_dir, hyperparams=hyperparams_vae, concat_label=concat_label)
            times[seed] = (pretrain_time, finetune_time, encoder_inference_time)
    print(times)
    for ipc in IPC_LIST:

        # Destilación con sample random class
        for seed in RANDOM_SEED_EVALUATE:
            pretrain_time, finetune_time, encoder_inference_time = times[seed]

            X_random, y_random = distill_random(X_train_pre, y_train_pre, n_per_class=ipc, random_state=seed)

            random_df, _, _ = evaluate_models(X_random, y_random, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='random', random_state=seed, ipc=ipc)
            random_df = add_meta(random_df, ipc=ipc, seed=seed)

            full_df = pd.concat([full_df, random_df], axis=0, ignore_index=True)

            if distillation_space == 'latent':
                # Se cargan los datos de destilacion
                train_z_path, train_y_path= os.path.join(vae_dir,f'train_z_seed_{seed}.npy'), os.path.join(vae_dir,f'train_y_seed_{seed}.npy')
                test_z_path, test_y_path= os.path.join(vae_dir,f'test_z_seed_{seed}.npy'), os.path.join(vae_dir,f'test_y_seed_{seed}.npy')

                train_z, train_y = np.load(train_z_path), np.load(train_y_path)
                test_z, test_y = np.load(test_z_path), np.load(test_y_path)

                # Destilación con K-means en el espacio latente y reconstruyendo
                if kmeans_type == 'centroid':
                    distill_z, distill_y = distill_with_kmeans(train_z, train_y, num_centroids=ipc, get_closest=False, seed=seed)

                    k_means_df, k_means_latent_df, _, _ = recontructed_distillation(distill_z, distill_y, test_z, test_y, X_test_pre, y_test_pre, 'k-means', 
                                                                                                                    metadata_path, hyperparams_vae, checkpoint_path, 
                                                                                                                    vae_dir, device, ipc, num_scaler, cat_encoder, label_encoder, 
                                                                                                                    pretrain_time, finetune_time, encoder_inference_time,concat_label,
                                                                                                                    seed)

                # Destilación con K-centers y recontruyendo
                distill_z, distill_y = distill_with_kcenters(train_z, train_y, num_centroids=ipc, seed=seed)
                k_center_df, k_center_latent_df, _, _ = recontructed_distillation(distill_z, distill_y, test_z, test_y,X_test_pre, y_test_pre, 'k-center', 
                                                                                                                metadata_path, hyperparams_vae, checkpoint_path, 
                                                                                                                vae_dir, device, ipc, num_scaler, cat_encoder, label_encoder, 
                                                                                                                pretrain_time, finetune_time, encoder_inference_time,concat_label,
                                                                                                                seed)

                # Destilacion con AG y reconstruyendo

                distill_z, distill_y = distill_with_agglomerative(train_z, train_y, num_clusters=ipc, get_closest=False, seed=seed)
                ag_df, ag_latent_df, _, _ = recontructed_distillation(distill_z, distill_y, test_z, test_y, X_test_pre, y_test_pre,'ag', 
                                                                                                                metadata_path, hyperparams_vae, checkpoint_path, 
                                                                                                                vae_dir, device, ipc, num_scaler, cat_encoder, label_encoder, 
                                                                                                                pretrain_time, finetune_time, encoder_inference_time,concat_label,
                                                                                                                seed)
                
                # Destilación con Least Confidence y recontruyendo

                distill_z, distill_y = distill_least_confidence(train_z, train_y, num_samples=ipc, seed=seed)
                lc_df, lc_latent_df, _, _ = recontructed_distillation(distill_z, distill_y, test_z, test_y, X_test_pre, y_test_pre,'lc', 
                                                                                                                metadata_path, hyperparams_vae, checkpoint_path, 
                                                                                                                vae_dir, device, ipc, num_scaler, cat_encoder, label_encoder, 
                                                                                                                pretrain_time, finetune_time, encoder_inference_time,concat_label,
                                                                                                                seed)
                # # Destilación con craig y recontruyendo
                distill_z, distill_y = distill_with_craig(train_z, train_y, num_samples=ipc, seed=seed)
                craig_df, craig_latent_df, _, _ = recontructed_distillation(distill_z, distill_y, test_z, test_y,X_test_pre, y_test_pre, 'craig', 
                                                                                                                metadata_path, hyperparams_vae, checkpoint_path, 
                                                                                                                vae_dir, device, ipc, num_scaler, cat_encoder, label_encoder, 
                                                                                                                pretrain_time, finetune_time, encoder_inference_time,concat_label,
                                                                                                                seed)

                full_df = pd.concat([full_df, k_means_df, k_means_latent_df, 
                                     k_center_df, k_center_latent_df, 
                                     ag_df, ag_latent_df, 
                                     lc_df, lc_latent_df, 
                                     craig_df, craig_latent_df], axis=0, ignore_index=True)
                
            elif distillation_space == 'original':

                # Destilación con K-means en el espacio original

                distill_z, distill_y = distill_with_kmeans(X_train_pre, y_train_pre, num_centroids=ipc, get_closest=False, seed=seed)
                k_means_df, _, _ = evaluate_models(distill_z, distill_y, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='k-means', random_state=seed, ipc=ipc)

                k_means_df = add_meta(k_means_df,
                                    ipc=ipc,
                                    seed=seed)

                # Destilación con AG
                distill_z, distill_y = distill_with_agglomerative(X_train_pre, y_train_pre, num_clusters=ipc, get_closest=False, seed=seed)
                ag_df, _, _ = evaluate_models(distill_z, distill_y, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='ag', random_state=seed, ipc=ipc)
                
                ag_df = add_meta(ag_df,
                                    ipc=ipc,
                                    seed=seed)
                
                # Destilación con k-centers
                distill_z, distill_y = distill_with_kcenters(X_train_pre, y_train_pre, num_centroids=ipc, seed=seed)
                k_center_df, _, _ = evaluate_models(distill_z, distill_y, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='k_center', random_state=seed, ipc=ipc)
            
                k_center_df = add_meta(k_center_df,
                                    ipc=ipc,
                                    seed=seed)

                # Destilación con LC
                distill_z, distill_y = distill_least_confidence(X_train_pre, y_train_pre, num_samples=ipc, seed=seed)
                lc_df, _, _ = evaluate_models(distill_z, distill_y, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='lc', random_state=seed, ipc=ipc)
                            
                lc_df = add_meta(lc_df,
                                    ipc=ipc,
                                    seed=seed)

                # Destilación CRAIG
                distill_z, distill_y = distill_with_craig(X_train_pre, y_train_pre, num_samples=ipc, seed=seed)
                craig_df, _, _ = evaluate_models(distill_z, distill_y, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='craig', random_state=seed, ipc=ipc)
            
                craig_df = add_meta(craig_df,
                                    ipc=ipc,
                                    seed=seed)
                
                full_df = pd.concat([full_df, k_means_df, ag_df, k_center_df, craig_df, lc_df], axis=0, ignore_index=True)
            
            # Se guarda constantemente para un analisis previo de los resultados
            full_df.to_csv(os.path.join(checkpoint_path, "metrics.csv"), index=False)

    #full_df.to_csv(os.path.join(checkpoint_path, "metrics.csv"), index=False) # Descomentar y comentar la línea de arriba finalizada la depuracion

if __name__ == '__main__':
    args = parse_args()
    main(args)