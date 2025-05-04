import argparse
from train_vae import main as main_vae
from train_ft_vae import main as main_ft_vae
from distill import distill_with_kmeans, distill_random, distill_with_agglomerative, distill_with_kcenters, distill_least_confidence, distill_with_craig
from utils import preprocessing
import os
import numpy as np
from utils import evaluate_models, reconstruct_data, load_reconstructed_data, split_train_test_custom, compute_relative_regret
import torch
from utils import read_json_config
import pandas as pd
import json

IPC_LIST = list(range(10,100,10)) + [200,500,1000]
IPC_LIST = list(range(10,20,10))
RANDOM_SEED_EVALUATE = range(5)
RANDOM_SEED_EVALUATE = range(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Distillation Configuration")

    parser.add_argument('--metadata_path', type=str, required=True, help='Path to the metadata file or dataset.')
    
    parser.add_argument('--method_distillation', type=str, choices=['original', 'k-means', 'craig', 'k-center'], required=True,
                        help='Distillation method to use.')
    
    parser.add_argument('--distillation_space', type=str, choices=['latent', 'original'], required=True,
                        help='Space in which to perform distillation.')

    parser.add_argument('--fine_tuning', type=str, choices=['true', 'false'], required=True,
                        help='Whether to apply fine-tuning after distillation.')

    parser.add_argument('--epochs_latent', type=int, required=True,
                        help='Number of epochs to train in the latent space.')

    parser.add_argument('--epochs_fine_tuning_latent', type=int, required=False,
                        help='Number of epochs to train fine tunning in the latent space.')

    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size for training.')

    parser.add_argument('--token_dimension', type=int, required=True,
                        help='Dimension of token embeddings.')

    parser.add_argument('--alpha', type=float, default=None,
                        help='Alpha value for fine-tuning (only used if fine_tuning is true).')

    parser.add_argument('--examples_for_distillation', type=int, required=True,
                        help='Number of examples to use for distillation.')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='')
    
    parser.add_argument('--kmeans_type', type=str, required=True,
                        help='')

    parser.add_argument('--hyperparams_vae', type=str, required=False,
                        help='')

    args = parser.parse_args()

    # Convert fine_tuning string to boolean
    args.fine_tuning = args.fine_tuning.lower() == 'true'

    # If fine_tuning is true, alpha must be set
    if args.fine_tuning and args.alpha is None:
        parser.error("--alpha must be set when fine_tuning is true")

    return args

META_COLS = [
    "IPC",
    "seed",
    "vae_pretrain_time",
    "vae_fine_tunning_time",
    "encoder_inference_time",
    "decoder_inference_time",
]

def add_meta(df, ipc=None, seed=None,
             pretrain_time=None, finetune_time=None,
             encoder_time=None, decoder_time=None):
    """
    Añade de un tirón las columnas META_COLS al df, con los valores
    pasados o None por defecto.
    """
    return df.assign(
        IPC=ipc,
        seed=seed,
        vae_pretrain_time=pretrain_time,
        vae_fine_tunning_time=finetune_time,
        encoder_inference_time=encoder_time,
        decoder_inference_time=decoder_time,
    )

def main(args):
    print("Running distillation pipeline with the following configuration:")

    metadata_path             = args.metadata_path
    method_distillation       = args.method_distillation
    distillation_space        = args.distillation_space
    fine_tuning               = args.fine_tuning
    epochs_latent             = args.epochs_latent
    batch_size                = args.batch_size
    token_dimension           = args.token_dimension
    alpha                     = args.alpha
    examples_for_distillation = args.examples_for_distillation
    checkpoint_path           = args.checkpoint_path
    kmeans_type              = args.kmeans_type
    epochs_fine_tuning_latent = args.epochs_fine_tuning_latent
    hyperparams_vae_path      = args.hyperparams_vae

    os.makedirs(checkpoint_path, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = read_json_config(metadata_path)
    base_dir = os.path.dirname(metadata_path)

    STATIC_SEED = 0
    SEED =0

    # Se utiliza un test de 0.1 porque luego se aplicar validacion cruzada que ya incluye validacion
    X_train_init, y_train_init, X_test_init, y_test_init = split_train_test_custom(config, base_dir, test_size=0.1, random_state=STATIC_SEED)

    set_data = (X_train_init, y_train_init, X_test_init, y_test_init)

    (
        X_train_pre, X_test_pre,
        y_train_pre, y_test_pre,
        num_scaler,  cat_encoder,
        label_encoder
                        ) = preprocessing(config, X_train_init, y_train_init, X_test_init, y_test_init, encoding='one-hot', concat=True, random_state=SEED)

    # Entrenamiento con todos los datos para el relative regret
    
    full_df, test_metrics_all, _ = evaluate_models(X_train_pre, y_train_pre, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='Full-data', random_state=SEED)
    full_df = add_meta(full_df)
    
    all_dfs = pd.DataFrame()
    full_df = pd.DataFrame()

    for ipc in IPC_LIST:
        METRICS_KMEANS = []
        METRICS_RANDOM = []
        METRICS_AGGLOMERATIVE_CLUSTERING = []
        METRICS_K_CENTER = []
        METRICS_LEAST_CONFIDENCE = []
        METRICS_CRAIG = []

        # Destilación con sample random class
        for seed in RANDOM_SEED_EVALUATE:
            X_random, y_random = distill_random(X_train_pre, y_train_pre, n_per_class=ipc, random_state=seed)
            random_df, test_metrics_random, _ = evaluate_models(X_random, y_random, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='random', random_state=seed)
            random_df = add_meta(random_df, ipc=ipc, seed=seed)

            full_df = pd.concat([full_df, random_df], axis=0, ignore_index=True)

            if distillation_space == 'latent':

                # Se carga el diccionario con los parametros del VAE
                with open(hyperparams_vae_path, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                    hyperparams_vae = data_dict['best_params']

                # Call the distillation function
                train_z, train_y,  pretrain_time, finetune_time, encoder_inference_time = main_vae(config, base_dir, set_data, encoding='ordinal', random_state=seed, 
                                            batch_size=batch_size, pretrain_epochs=epochs_latent, 
                                            finetune_epochs=epochs_fine_tuning_latent, ckpt_dir=checkpoint_path, hyperparams=hyperparams_vae)

                # Destilación con Least Confidence y recontruyendo

                distill_z, distill_y = distill_least_confidence(train_z, train_y, num_samples=ipc, seed=seed)
                df_reconstruct, decoder_inference_time = reconstruct_data(distill_z, distill_y, models_paths=checkpoint_path,device=device,json_config_path=metadata_path, latent_space=True, hyperparams=hyperparams_vae, method='LC', seed=seed)
                
                x_train_disti, y_train_disti, x_test_disti, y_test_disti = load_reconstructed_data(df_reconstruct, metadata_path, checkpoint_path, random_state=seed)
                
                lc_df, test_metrics_lc, _ = evaluate_models(x_train_disti, y_train_disti, x_test_disti, y_test_disti, ckpt_dir=checkpoint_path, method='lc', random_state=seed)
            
                lc_df = add_meta(lc_df,
                                    ipc=ipc,
                                    seed=seed,
                                    pretrain_time=pretrain_time,
                                    finetune_time=finetune_time,
                                    encoder_time=encoder_inference_time,
                                    decoder_time=decoder_inference_time)
                
                # Destilación con craig y recontruyendo

                distill_z, distill_y = distill_with_craig(train_z, train_y, num_samples=ipc, seed=seed)
                df_reconstruct, decoder_inference_time = reconstruct_data(distill_z, distill_y, models_paths=checkpoint_path,device=device,json_config_path=metadata_path, latent_space=True, hyperparams=hyperparams_vae, method='LC', seed=seed)
                
                x_train_disti, y_train_disti, x_test_disti, y_test_disti = load_reconstructed_data(df_reconstruct, metadata_path, checkpoint_path, random_state=seed)
                
                craig_df, test_metrics_craig, _ = evaluate_models(x_train_disti, y_train_disti, x_test_disti, y_test_disti, ckpt_dir=checkpoint_path, method='craig', random_state=seed)
            
                craig_df = add_meta(craig_df,
                                    ipc=ipc,
                                    seed=seed,
                                    pretrain_time=pretrain_time,
                                    finetune_time=finetune_time,
                                    encoder_time=encoder_inference_time,
                                    decoder_time=decoder_inference_time)

                # Destilación con K-centers y recontruyendo

                distill_z, distill_y = distill_with_kcenters(train_z, train_y, num_centroids=ipc, seed=seed)
                df_reconstruct, decoder_inference_time = reconstruct_data(distill_z, distill_y, models_paths=checkpoint_path,device=device,json_config_path=metadata_path, latent_space=True, hyperparams=hyperparams_vae, method='k-centers', seed=seed)
                
                x_train_disti, y_train_disti, x_test_disti, y_test_disti = load_reconstructed_data(df_reconstruct, metadata_path, checkpoint_path, random_state=seed)
                
                k_center_df, test_metrics_k_center, _ = evaluate_models(x_train_disti, y_train_disti, x_test_disti, y_test_disti, ckpt_dir=checkpoint_path, method='k_center', random_state=seed)
            
                k_center_df = add_meta(k_center_df,
                                    ipc=ipc,
                                    seed=seed,
                                    pretrain_time=pretrain_time,
                                    finetune_time=finetune_time,
                                    encoder_time=encoder_inference_time,
                                    decoder_time=decoder_inference_time)

                # Destilación con K-means en el espacio latente y reconstruyendo
                if kmeans_type == 'centroid':
                    distill_z, distill_y = distill_with_kmeans(train_z, train_y, num_centroids=ipc, get_closest=False, seed=seed)

                    df_reconstruct, decoder_inference_time = reconstruct_data(distill_z, distill_y, models_paths=checkpoint_path,device=device,json_config_path=metadata_path, latent_space=True, hyperparams=hyperparams_vae, method='k-means', seed=seed)
                    
                    x_train_disti, y_train_disti, x_test_disti, y_test_disti = load_reconstructed_data(df_reconstruct, metadata_path, checkpoint_path, random_state=seed)
                    
                    kmeans_df, test_metrics_k_means, _ = evaluate_models(x_train_disti, y_train_disti, x_test_disti, y_test_disti, ckpt_dir=checkpoint_path, method='k-means', random_state=seed)
                
                    kmeans_df = add_meta(kmeans_df,
                                        ipc=ipc,
                                        seed=seed,
                                        pretrain_time=pretrain_time,
                                        finetune_time=finetune_time,
                                        encoder_time=encoder_inference_time,
                                        decoder_time=decoder_inference_time)

                # Destilacion con AG y reconstruyendo

                distill_z, distill_y = distill_with_agglomerative(train_z, train_y, num_clusters=ipc, get_closest=False, seed=seed)
                df_reconstruct, decoder_inference_time = reconstruct_data(distill_z, distill_y, models_paths=checkpoint_path,device=device,json_config_path=metadata_path, latent_space=True, hyperparams=hyperparams_vae, method='ag', seed=seed)
                
                x_train_disti, y_train_disti, x_test_disti, y_test_disti = load_reconstructed_data(df_reconstruct, metadata_path, checkpoint_path, random_state=seed)
                
                ag_df, test_metrics_ag, _ = evaluate_models(x_train_disti, y_train_disti, x_test_disti, y_test_disti, ckpt_dir=checkpoint_path, method='ag', random_state=seed)
            
                ag_df = add_meta(ag_df,
                                    ipc=ipc,
                                    seed=seed,
                                    pretrain_time=pretrain_time,
                                    finetune_time=finetune_time,
                                    encoder_time=encoder_inference_time,
                                    decoder_time=decoder_inference_time)
                
            elif distillation_space == 'original':

                # Destilación con LC
                distill_z, distill_y = distill_least_confidence(X_train_pre, y_train_pre, num_samples=ipc, seed=seed)
                lc_df, test_metrics_lc, _ = evaluate_models(distill_z, distill_y, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='lc', random_state=seed)
                            
                lc_df = add_meta(lc_df,
                                    ipc=ipc,
                                    seed=seed)

                # Destilación CRAIG
                distill_z, distill_y = distill_with_craig(X_train_pre, y_train_pre, num_samples=ipc, seed=seed)
                craig_df, test_metrics_craig, _ = evaluate_models(distill_z, distill_y, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='craig', random_state=seed)
            
                craig_df = add_meta(craig_df,
                                    ipc=ipc,
                                    seed=seed)

                # Destilación con K-means en el espacio original

                distill_z, distill_y = distill_with_kmeans(X_train_pre, y_train_pre, num_centroids=ipc, get_closest=False, seed=seed)
                kmeans_df, test_metrics_k_means, _ = evaluate_models(distill_z, distill_y, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='k-means', random_state=seed)

                kmeans_df = add_meta(kmeans_df,
                                    ipc=ipc,
                                    seed=seed)

                # Destilación con AG
                distill_z, distill_y = distill_with_agglomerative(X_train_pre, y_train_pre, num_clusters=ipc, get_closest=False, seed=seed)
                ag_df, test_metrics_ag, _ = evaluate_models(distill_z, distill_y, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='ag', random_state=seed)
                
                ag_df = add_meta(ag_df,
                                    ipc=ipc,
                                    seed=seed)
                
                # Destilación con k-centers
                distill_z, distill_y = distill_with_kcenters(X_train_pre, y_train_pre, num_centroids=ipc, seed=seed)
                k_center_df, test_metrics_k_center, _ = evaluate_models(distill_z, distill_y, X_test_pre, y_test_pre, ckpt_dir=checkpoint_path, method='k_center', random_state=seed)
            
                k_center_df = add_meta(k_center_df,
                                    ipc=ipc,
                                    seed=seed)
            full_df = pd.concat([full_df, kmeans_df, ag_df, k_center_df, craig_df, lc_df], axis=0, ignore_index=True)
            full_df.to_csv(os.path.join(checkpoint_path, "metrics.csv"), index=False)
            
            METRICS_RANDOM.append(test_metrics_random)
            METRICS_LEAST_CONFIDENCE.append(test_metrics_lc)
            METRICS_CRAIG.append(test_metrics_craig)
            METRICS_K_CENTER.append(test_metrics_k_center)
            METRICS_KMEANS.append(test_metrics_k_means)
            METRICS_AGGLOMERATIVE_CLUSTERING.append(test_metrics_ag)


        df1 = compute_relative_regret(test_metrics_all, METRICS_RANDOM, METRICS_KMEANS, 'k-means', ipc)
        df2 = compute_relative_regret(test_metrics_all, METRICS_RANDOM, METRICS_RANDOM, 'random', ipc)
        df3 = compute_relative_regret(test_metrics_all, METRICS_RANDOM, [test_metrics_all]*len(RANDOM_SEED_EVALUATE), 'original', ipc)

        df4 = compute_relative_regret(test_metrics_all, METRICS_RANDOM, METRICS_AGGLOMERATIVE_CLUSTERING, 'ag', ipc)
        df5 = compute_relative_regret(test_metrics_all, METRICS_RANDOM, METRICS_K_CENTER, 'k_center', ipc)
        df6 = compute_relative_regret(test_metrics_all, METRICS_RANDOM, METRICS_LEAST_CONFIDENCE, 'lc', ipc)
        df7 = compute_relative_regret(test_metrics_all, METRICS_RANDOM, METRICS_CRAIG, 'craig', ipc)
            
        all_dfs = pd.concat([all_dfs,df1,df2,df3,df4,df5,df6,df7], axis=0, ignore_index=True) 

        csv_path = os.path.join(checkpoint_path, "relative_regret.csv")
        all_dfs.to_csv(csv_path, index=False)
        print("Relative regret guardado en", csv_path)

    # guardas a CSV
    csv_path = os.path.join(checkpoint_path, "relative_regret.csv")
    all_dfs.to_csv(csv_path, index=False)
    print("Relative regret guardado en", csv_path)

    full_df.to_csv(os.path.join(checkpoint_path, "metrics.csv"), index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)

    # python main.py --metadata_path data/shoppers/metadata.json --method_distillation original --distillation_space original --fine_tuning false --epochs_latent 100 --batch_size 4069 --token_dimension 128 --alpha 0.5 --examples_for_distillation 10 --checkpoint_path checkpoint_prueba_A --kmeans_type centroid

    """
    python main.py \
        --metadata_path data/shoppers/metadata.json \
        --method_distillation k-means \
        --distillation_space latent \
        --fine_tuning false \
        --epochs_latent 5 \
        --epochs_fine_tuning_latent 0 \
        --batch_size 4069 \
        --token_dimension 4 --alpha 0.5 \
        --examples_for_distillation 10 \
        --checkpoint_path checkpoint_prueba_B \
        --kmeans_type centroid \
        --hyperparams_vae tune_vae/tune_vae_AAD/tune_adult/best_hyperparams.json
    """