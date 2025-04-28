import argparse
from train_vae import main as main_vae
from train_ft_vae import main as main_ft_vae
from distill import distill_with_kmeans
from utils import preprocessing
import os
import numpy as np
from utils import evaluate_models, reconstruct_data, load_reconstructed_data, split_train_test_custom
import torch
from utils import read_json_config

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

    args = parser.parse_args()

    # Convert fine_tuning string to boolean
    args.fine_tuning = args.fine_tuning.lower() == 'true'

    # If fine_tuning is true, alpha must be set
    if args.fine_tuning and args.alpha is None:
        parser.error("--alpha must be set when fine_tuning is true")

    return args



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

    os.makedirs(checkpoint_path, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = read_json_config(metadata_path)
    base_dir = os.path.dirname(metadata_path)


    # Se utiliza un test de 0.1 porque luego se aplicar validacion cruzada que ya incluye validacion
    X_train, y_train, X_test, y_test = split_train_test_custom(metadata_path, test_size=0.1, random_state=0)

    (
        X_train_enc, X_test_enc,
        y_train_enc, y_test_enc,
        num_scaler,  cat_encoder,
        label_encoder
                        ) = preprocessing(config, X_train, y_train, X_test, y_test, encoding='one-hot', concat=True, random_state=0)

    # Entrenamiento con todos los datos para el relative regret
    results_df, test_metrics_all, best_models = evaluate_models(X_train_enc, y_train_enc, X_test_enc, y_test_enc, ckpt_dir=checkpoint_path)

    # if distillation_space == 'latent': 
    #     if fine_tuning:
    #         # Call the fine-tuning function
    #         pass
    #     else:
    #         # Call the distillation function
    #         #train_z, train_y = main_vae(metadata_path, encoding='ordinal', random_state=0, batch_size=batch_size, ckpt_dir=checkpoint_path)
    #         train_z, train_y = main_vae(metadata_path, encoding='ordinal', random_state=0, batch_size=batch_size, pretrain_epochs=1000, finetune_epochs=200, ckpt_dir=checkpoint_path, fine_tune=True)
    #         #train_z = np.load(os.path.join(checkpoint_path, 'train_z.npy'))
    #         #train_y = np.load(os.path.join(checkpoint_path, 'train_y.npy'))
            
    #         #for i in range(10,1100,10):
    #         #print("Evaluacion with {} examples for distillation".format(i))
    #         if method_distillation == 'original':
    #             # Call the original distillation function
    #             pass
    #         elif method_distillation == 'k-means':
    #             if kmeans_type == 'centroid':
    #                 distill_z, distill_y = distill_with_kmeans(train_z, train_y, num_centroids=examples_for_distillation, return_indices=False)
    #                 #distill_z, distill_y = distill_with_kmeans(train_z, train_y, num_centroids=i, return_indices=False)

    #         reconstruct_data(distill_z, distill_y, models_paths=checkpoint_path,device=device,json_config_path=metadata_path, latent_space=True)

    #         x_train, y_train, x_test, y_test = load_reconstructed_data(metadata_path, checkpoint_path)
    #         evaluate_models(x_train, y_train, x_test, y_test)

    #     if args.method_distillation == 'original':
    #         # Call the original distillation function
    #         pass
    #     elif args.method_distillation == 'k-means':
    #         pass

    # elif distillation_space == 'original':
    #     (
    #     X_num_train, X_cat_train,
    #     X_num_test,  X_cat_test,
    #     y_train_enc, y_test_enc,
    #     num_scaler,  cat_encoder,
    #     label_encoder
    #                     )  = preprocessing(metadata_path, encoding='one-hot', random_state=0)
        
    #     x_train = np.concatenate((X_num_train, X_cat_train), axis=1)
    #     x_test = np.concatenate((X_num_test, X_cat_test), axis=1)
    
    #     x_train ,y_train = distill_with_kmeans(x_train,y_train_enc, num_centroids=examples_for_distillation, return_indices=False)
    #     print(x_train.shape)
    #     evaluate_models(x_train, y_train, x_test, y_test_enc)

if __name__ == '__main__':
    args = parse_args()
    main(args)

# python main.py --metadata_path data/shoppers/metadata.json --method_distillation original --distillation_space original --fine_tuning false --epochs_latent 100 --batch_size 4069 --token_dimension 128 --alpha 0.5 --examples_for_distillation 10 --checkpoint_path checkpoint_prueba_A --kmeans_type centroid

# python main.py --metadata_path data/shoppers/metadata.json --method_distillation k-means --distillation_space latent --fine_tuning false --epochs_latent 100 --batch_size 4069 --token_dimension 128 --alpha 0.5 --examples_for_distillation 10 --checkpoint_path checkpoint_prueba_A --kmeans_type centroid