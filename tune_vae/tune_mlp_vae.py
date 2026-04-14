"""
Script de Optuna para buscar los mejores hiperparámetros del MLP-VAE
(estudio de ablación: Transformer vs. MLP).

Uso:
    cd tune_vae
    python tune_mlp_vae.py --dataset adult --n_trials 30 --epochs 700

Guarda los mejores hiperparámetros en:
    tune_vae/tune_mlp_{dataset}/best_hyperparams_mlp.json
"""

import os
import sys
import json
import argparse
import optuna
import numpy as np

# Añadir carpeta padre al path para importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_vae import main as main_vae
from utils import (
    read_json_config,
    split_train_test_custom,
    reconstruct_data,
    load_reconstructed_data,
    evaluate_one_model,
    preprocessing,
)
from distill import distill_with_kmeans


# ─── Datasets disponibles y sus rutas relativas ───────────────────────────────
DATASETS = {
    "adult":            "../data/adult/metadata.json",
    "default":          "../data/default/metadata.json",
    "magic_telescope":  "../data/magic_telescope/metadata.json",
    "paysim":           "../data/paysim/metadata.json",
    "shoppers":         "../data/shoppers/metadata.json",
}


def objective(trial, args):
    """Función objetivo para Optuna: entrena MLP-VAE, destila, evalúa y retorna ROC-AUC."""

    hyperparams = {
        # Hiperparámetros específicos del MLP
        'n_hidden_layers': trial.suggest_int('n_hidden_layers', 1, 4),
        'hidden_dim':      trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
        'mlp_dropout':     trial.suggest_float('mlp_dropout', 0.1, 0.5),

        # Hiperparámetros del VAE (comunes)
        'max_beta':     trial.suggest_float('max_beta', 1e-3, 0.1, log=True),
        'min_beta':     trial.suggest_float('min_beta', 1e-7, 1e-3, log=True),
        'lr_pretrain':  trial.suggest_float('lr_pretrain', 1e-4, 1e-2, log=True),
        'wd_pretrain':  trial.suggest_categorical('wd_pretrain', [0, 1e-4, 1e-3]),
        'd_token':      trial.suggest_categorical('d_token', [4, 8]),
        'token_bias':   True,

        # Parámetros fijos (no usados por MLP pero requeridos por el pipeline)
        'n_head': 1,
        'factor': 32,
        'num_layers': 1,
        'early_stop_counter_pretrain': 15,
        'early_stop_counter_ft': 15,
    }

    # Entrenar MLP-VAE
    train_z, train_y, _, _, _ = main_vae(
        config=args.config,
        base_dir=args.base_dir,
        set_data=args.set_data,
        encoding='ordinal',
        random_state=args.seed,
        batch_size=args.batch_size,
        pretrain_epochs=args.epochs,
        finetune_epochs=0,
        ckpt_dir=args.checkpoint_path,
        hyperparams=hyperparams,
        architecture='mlp',
    )

    # Destilar con K-Means (IPC=50 como evaluación rápida)
    distill_z, distill_y = distill_with_kmeans(
        train_z, train_y, num_centroids=args.ipc, get_closest=False, seed=args.seed
    )

    # Reconstruir datos destilados al espacio original
    df_recon, _ = reconstruct_data(
        distill_z, distill_y,
        models_paths=args.checkpoint_path,
        device='cpu',
        json_config_path=args.metadata_path,
        latent_space=True,
        hyperparams=hyperparams,
        architecture='mlp',
    )

    # Cargar datos reconstruidos, usando los encoders one-hot pre-ajustados
    # (igual que hace main.py antes de llamar a evaluate_models)
    x_train_disti, y_train_disti = load_reconstructed_data(
        df_recon, args.metadata_path,
        num_scaler=args.num_scaler,
        cat_encoder=args.cat_encoder,
        label_encoder=args.label_encoder,
    )

    # Evaluar con XGBoost sobre el test set pre-procesado
    _, _, test_metrics, _ = evaluate_one_model(
        x_train_disti, y_train_disti,
        args.X_test_pre, args.y_test_pre,
        random_state=args.seed,
    )

    return test_metrics['roc_auc']


def main():
    parser = argparse.ArgumentParser(description='Optuna tuning para MLP-VAE (ablación)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASETS.keys()),
                        help='Nombre del dataset a usar.')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Número de trials de Optuna.')
    parser.add_argument('--epochs', type=int, default=700,
                        help='Épocas de pre-entrenamiento por trial.')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Tamaño de batch.')
    parser.add_argument('--ipc', type=int, default=50,
                        help='Instances Per Class para evaluación rápida.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Semilla global.')
    args_cli = parser.parse_args()

    # Rutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(script_dir, DATASETS[args_cli.dataset])
    metadata_path = os.path.abspath(metadata_path)

    # Cargar configuración y datos
    config = read_json_config(metadata_path)
    base_dir = os.path.dirname(metadata_path)

    X_train, y_train, X_test, y_test = split_train_test_custom(
        config, base_dir, test_size=0.1, random_state=0
    )
    set_data = (X_train, y_train, X_test, y_test)

    # Pre-procesar datos con one-hot encoding para evaluación downstream
    # (igual que hace main.py). Los encoders ajustados se pasan al objetivo.
    (X_train_pre, X_test_pre, y_train_pre, y_test_pre,
     num_scaler, cat_encoder, label_encoder) = preprocessing(
        config, X_train, y_train, X_test, y_test,
        encoding='one-hot', concat=True, random_state=0,
    )

    # Carpeta de checkpoints del tuning
    ckpt_dir = os.path.join(script_dir, f'tune_mlp_{args_cli.dataset}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Empaquetar argumentos
    class Args:
        pass
    args = Args()
    args.config = config
    args.base_dir = base_dir
    args.set_data = set_data
    args.metadata_path = metadata_path
    args.checkpoint_path = ckpt_dir
    args.epochs = args_cli.epochs
    args.batch_size = args_cli.batch_size
    args.ipc = args_cli.ipc
    args.seed = args_cli.seed
    # Encoders y datos de test pre-procesados (one-hot)
    args.num_scaler = num_scaler
    args.cat_encoder = cat_encoder
    args.label_encoder = label_encoder
    args.X_test_pre = X_test_pre
    args.y_test_pre = y_test_pre

    # Ejecutar Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args), n_trials=args_cli.n_trials)

    # Guardar resultados
    output = {
        'best_params': study.best_params,
        'best_roc_auc': study.best_value,
        'dataset': args_cli.dataset,
        'epochs': args_cli.epochs,
        'batch_size': args_cli.batch_size,
        'n_trials': args_cli.n_trials,
        'seed': args_cli.seed,
    }

    params_file = os.path.join(ckpt_dir, 'best_hyperparams_mlp.json')
    with open(params_file, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Dataset: {args_cli.dataset}")
    print(f"Mejores hiperparámetros MLP-VAE guardados en: {params_file}")
    print(f"Mejor ROC-AUC: {study.best_value:.4f}")
    print(f"Parámetros: {study.best_params}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
