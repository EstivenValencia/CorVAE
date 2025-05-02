import os
import argparse
import json
import numpy as np
import torch
import optuna
import sys

# Añadir carpeta padre al path para importar tu modelo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from train_vae import main as main_vae
from utils import read_json_config, split_train_test_custom, reconstruct_data, load_reconstructed_data, evaluate_models, evaluate_one_model
from distill import distill_with_kmeans

def objective(trial, args):
    # Construir hyperparams usando los límites definidos en args.search_space
    sp = args.search_space
    hyperparams = {
        'max_beta': trial.suggest_loguniform('max_beta', sp['max_beta'][0], sp['max_beta'][1]),
        'min_beta': trial.suggest_loguniform('min_beta', sp['min_beta'][0], sp['min_beta'][1]),
        'lambda_': trial.suggest_categorical('lambda_', sp['lambda_']),
        'lr_pretrain': trial.suggest_loguniform('lr_pretrain', sp['lr_pretrain'][0], sp['lr_pretrain'][1]),
        'wd_pretrain': trial.suggest_categorical('wd_pretrain', sp['wd_pretrain']),
        'd_token': trial.suggest_categorical('d_token', sp['d_token']),
        'token_bias': True,
        'n_head': trial.suggest_categorical('n_head', sp['n_head']),
        'factor': trial.suggest_categorical('factor', sp['factor']),
        'num_layers': trial.suggest_int('num_layers', sp['num_layers'][0], sp['num_layers'][1]),
        'dropout_ft': trial.suggest_categorical('dropout_ft', sp['dropout_ft']),
        'alpha_ft': trial.suggest_categorical('alpha_ft', sp['alpha_ft']),
        'lr_ft': trial.suggest_loguniform('lr_ft', sp['lr_ft'][0], sp['lr_ft'][1]),
        'wd_ft': trial.suggest_categorical('wd_ft', sp['wd_ft'])
    }
    hyperparams['early_stop_counter_ft'] = 15
    hyperparams['early_stop_counter_pretrain'] = 15

    # Ejecutar pipeline VAE con fine-tuning
    train_z, train_y = main_vae(
        config=args.config,
        base_dir=args.base_dir,
        set_data=args.set_data,
        encoding='ordinal',
        random_state=args.seed,
        batch_size=args.batch_size,
        pretrain_epochs=args.epochs_latent,
        finetune_epochs=args.epochs_fine_tuning_latent,
        ckpt_dir=args.checkpoint_path,
        hyperparams=hyperparams
    )
    distill_z, distill_y = distill_with_kmeans(train_z, train_y, num_centroids=args.IPC, get_closest=False, seed=args.seed)
    reconstruct_data(distill_z, distill_y, models_paths=args.checkpoint_path, device='cpu',json_config_path=metadata_path, latent_space=True, hyperparams=hyperparams)
                    
    x_train_disti, y_train_disti, x_test_disti, y_test_disti = load_reconstructed_data(metadata_path, args.checkpoint_path, random_state=args.seed)
                    
    _,_, test_metrics_k_means, _ = evaluate_one_model(x_train_disti, y_train_disti, x_test_disti, y_test_disti, random_state=args.seed)

    return test_metrics_k_means['roc_auc']

if __name__ == '__main__':

# python --metadata_path /mnt/d/home-2/Documentos/master/practica-deusto-tech/TFM/MyTabsyn/CorVAE/data/shoppers/metadata.json \
# --checkpoint_path tune_shoppers --epochs_latent 200 --epochs_fine_tuning_latent 0 --batch_size 4096 

    # Definición manual de todos los argumentos
    class Args: pass
    args = Args()

    dataset = 'default'
    metadata_path = '../../data/default/metadata.json'
    STATIC_SEED = 0

    # __file__ es la ruta (relativa o absoluta) del script actual
    script_path = os.path.abspath(__file__)              # ruta completa al archivo .py
    root_dir_tune = os.path.dirname(script_path)   
    

    config = read_json_config(metadata_path)
    base_dir = os.path.dirname(metadata_path)

    X_train_init, y_train_init, X_test_init, y_test_init = split_train_test_custom(config, base_dir, test_size=0.1, random_state=STATIC_SEED)
    set_data = (X_train_init, y_train_init, X_test_init, y_test_init)

    # Rutas y configuración manual
    args.config = config                      # Ruta al JSON de configuración
    args.base_dir = base_dir               # Directorio base inferido
    args.checkpoint_path = f'{root_dir_tune}/tune_{dataset}'                             # Carpeta de checkpoints

    # Parámetros de entrenamiento
    args.epochs_latent = 500                                    # Épocas de pre-entrenamiento VAE
    args.epochs_fine_tuning_latent = 200                        # Épocas de fine-tuning
    args.batch_size = 4096                                       # Tamaño de batch
    args.seed = 0                                              # Semilla global
    args.encoding = 'ordinal'                                  # Esquema de codificación
    args.set_data = set_data        # Tu tupla de datos (define o carga arriba)
    args.n_trials = 30
    args.IPC = 50

    # Definir límites y categorías de búsqueda para Optuna
    args.search_space = {
        'max_beta': [1e-3, 0.1],
        'min_beta': [1e-7, 1e-3],
        'lambda_': [0.5,0.7,0.9],
        'lr_pretrain': [1e-4, 1e-2],
        'wd_pretrain': [0,1e-4,1e-3],
        'd_token': [4, 8],
        'n_head': [1, 2, 4],
        'factor': [16, 32],
        'num_layers': [1, 2],
        'dropout_ft': [0.1, 0.2, 0.3],
        'alpha_ft': [0.3, 0.5, 0.7],
        'lr_ft': [1e-5, 1e-3],
        'wd_ft': [0,1e-4,1e-3],
    }

    # Asegurar carpeta
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Ejecutar Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    # Guardar resultados en JSON, incluyendo todos los argumentos definidos
    output = {
        'best_params': study.best_params,
        'search_space': args.search_space,

        # Todos los argumentos manuales:
        'checkpoint_path': args.checkpoint_path,
        'epochs_latent': args.epochs_latent,
        'epochs_fine_tuning_latent': args.epochs_fine_tuning_latent,
        'batch_size': args.batch_size,
        'seed': args.seed
    }

    params_file = os.path.join(args.checkpoint_path, 'best_hyperparams.json')
    with open(params_file, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Mejores hiperparámetros y parámetros de configuración guardados en {params_file}")
    print('Mejores hiperparámetros:', study.best_params)
    print('Mejor pérdida de validación:', study.best_value)


