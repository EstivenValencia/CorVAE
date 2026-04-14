"""
Estudio de Sensibilidad de Hiperparámetros del CorVAE (Transformer VAE).

Para cada hiperparámetro clave, se varía su valor manteniendo el resto en los
valores óptimos encontrados por Optuna. Se registra el rendimiento (ROC-AUC)
en el espacio latente y reconstruido para cada configuración.

Uso:
    python sensitivity_study.py --dataset shoppers --epochs 700

Salida:
    tune_vae/sensitivity_{dataset}/sensitivity_results.csv
    tune_vae/sensitivity_{dataset}/sensitivity_plots.png
"""

import os
import sys
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = {
    "adult":            "../data/adult/metadata.json",
    "default":          "../data/default/metadata.json",
    "magic_telescope":  "../data/magic_telescope/metadata.json",
    "paysim":           "../data/paysim/metadata.json",
    "shoppers":         "../data/shoppers/metadata.json",
}

# ── Valores óptimos de referencia (baseline) ──────────────────────────────────
# Estos son los mejores hiperparámetros encontrados por Optuna para shoppers.
# Se usan como punto de partida y se varía uno a la vez.
BASELINE_HYPERPARAMS = {
    'max_beta':      0.007715720500892705,
    'min_beta':      0.00014144050146933074,
    'lambda_':       0.9,
    'lr_pretrain':   0.005105439988316641,
    'wd_pretrain':   0,
    'd_token':       8,
    'token_bias':    True,
    'n_head':        2,
    'factor':        16,
    'num_layers':    1,
    'early_stop_counter_pretrain': 30,
    'early_stop_counter_ft': 30,
}

# ── Espacio de sensibilidad: hiperparámetro → valores a probar ────────────────
# Cada lista incluye el valor óptimo (marcado como baseline) y variaciones.
SENSITIVITY_SPACE = {
    'max_beta':    [1e-4, 1e-3, 5e-3, 0.0077, 0.01, 0.05, 0.1],
    'min_beta':    [1e-7, 1e-6, 1e-5, 0.00014, 1e-3],
    'num_layers':  [1, 2, 3, 4],
    'n_head':      [1, 2, 4, 8],
    'd_token':     [4, 8, 16, 32],
    'factor':      [4, 8, 16, 32, 64],
    'lr_pretrain': [1e-4, 5e-4, 1e-3, 0.005, 0.01],
    'lambda_':     [0.1, 0.3, 0.5, 0.7, 0.9],
}

# Nombres legibles para las gráficas
PARAM_LABELS = {
    'max_beta':    r'$\beta_{max}$',
    'min_beta':    r'$\beta_{min}$',
    'num_layers':  'Transformer Layers',
    'n_head':      'Attention Heads',
    'd_token':     'Token Dimension ($d$)',
    'factor':      'FFN Factor',
    'lr_pretrain': 'Learning Rate',
    'lambda_':     r'$\lambda$ (Annealing)',
}


def run_single_experiment(hyperparams, args):
    """Entrena CorVAE, destila con K-Means, reconstruye y evalúa. Retorna métricas."""
    try:
        # Entrenar VAE
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
            architecture='transformer',
        )

        # Destilar con K-Means
        distill_z, distill_y = distill_with_kmeans(
            train_z, train_y, num_centroids=args.ipc, get_closest=False, seed=args.seed
        )

        # ── Evaluación en espacio latente ──
        from utils import evaluate_one_model
        flat_train = train_z.reshape(train_z.shape[0], -1) if train_z.ndim == 3 else train_z
        flat_distill = distill_z.reshape(distill_z.shape[0], -1) if distill_z.ndim == 3 else distill_z

        # Preparar test latente
        import pickle
        pre_path = os.path.join(args.checkpoint_path, f'pre_encoders_seed_{args.seed}.pkl')
        with open(pre_path, 'rb') as f:
            encoders = pickle.load(f)

        # ── Evaluación en espacio reconstruido ──
        df_recon, _ = reconstruct_data(
            distill_z, distill_y,
            models_paths=args.checkpoint_path,
            device='cpu',
            json_config_path=args.metadata_path,
            latent_space=True,
            hyperparams=hyperparams,
            architecture='transformer',
        )

        x_train_disti, y_train_disti = load_reconstructed_data(
            df_recon, args.metadata_path,
            num_scaler=args.num_scaler,
            cat_encoder=args.cat_encoder,
            label_encoder=args.label_encoder,
        )

        _, _, test_metrics_recon, _ = evaluate_one_model(
            x_train_disti, y_train_disti,
            args.X_test_pre, args.y_test_pre,
            random_state=args.seed,
        )

        return {
            'roc_auc_recon': test_metrics_recon.get('roc_auc', np.nan),
            'balanced_acc_recon': test_metrics_recon.get('balanced', np.nan),
            'macro_f1_recon': test_metrics_recon.get('macro_f1', np.nan),
            'status': 'ok',
        }

    except Exception as e:
        print(f"  ⚠️ Experimento falló: {e}")
        return {
            'roc_auc_recon': np.nan,
            'balanced_acc_recon': np.nan,
            'macro_f1_recon': np.nan,
            'status': f'error: {str(e)[:100]}',
        }


def plot_sensitivity(results_df, output_dir):
    """Genera gráficas de sensibilidad para cada hiperparámetro."""
    params = results_df['param_name'].unique()
    n_params = len(params)
    fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(5 * ((n_params + 1) // 2), 10))
    axes = axes.flatten()

    for idx, param in enumerate(params):
        ax = axes[idx]
        subset = results_df[results_df['param_name'] == param].sort_values('param_value')

        # Identificar el valor baseline
        baseline_val = BASELINE_HYPERPARAMS.get(param)

        ax.plot(range(len(subset)), subset['roc_auc_recon'].values, 'o-', color='#2196F3',
                linewidth=2, markersize=8, label='ROC-AUC (Recon)')
        ax.plot(range(len(subset)), subset['balanced_acc_recon'].values, 's--', color='#FF9800',
                linewidth=2, markersize=7, label='Bal. Acc (Recon)')

        # Marcar baseline con línea vertical
        if baseline_val is not None:
            vals_list = subset['param_value'].tolist()
            # Buscar el valor más cercano al baseline
            closest_idx = min(range(len(vals_list)), key=lambda i: abs(float(vals_list[i]) - float(baseline_val)))
            ax.axvline(x=closest_idx, color='red', linestyle=':', alpha=0.7, label='Optimal')

        ax.set_xticks(range(len(subset)))
        x_labels = subset['param_value'].values
        # Formatear labels según tipo
        if param in ('max_beta', 'min_beta', 'lr_pretrain'):
            x_labels = [f'{float(v):.1e}' for v in x_labels]
        else:
            x_labels = [str(v) for v in x_labels]
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.set_title(PARAM_LABELS.get(param, param), fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(True, alpha=0.3)

    # Ocultar ejes sobrantes
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Hyperparameter Sensitivity Analysis — CorVAE (Shoppers Dataset)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'sensitivity_plots.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Gráficas guardadas en: {plot_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Estudio de sensibilidad de hiperparámetros del CorVAE')
    parser.add_argument('--dataset', type=str, default='shoppers',
                        choices=list(DATASETS.keys()))
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--ipc', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--baseline_json', type=str, default=None,
                        help='Ruta al JSON con hiperparámetros baseline. Si no se provee, usa los defaults internos.')
    args_cli = parser.parse_args()

    # ── Rutas ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.abspath(os.path.join(script_dir, DATASETS[args_cli.dataset]))
    config = read_json_config(metadata_path)
    base_dir = os.path.dirname(metadata_path)

    output_dir = os.path.join(script_dir, f'sensitivity_{args_cli.dataset}')
    ckpt_dir = os.path.join(output_dir, 'ckpt_sensitivity')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Cargar baseline desde JSON si se provee ──
    baseline = dict(BASELINE_HYPERPARAMS)
    if args_cli.baseline_json and os.path.isfile(args_cli.baseline_json):
        with open(args_cli.baseline_json, 'r') as f:
            loaded = json.load(f)
        if 'best_params' in loaded:
            baseline.update(loaded['best_params'])
        else:
            baseline.update(loaded)
        print(f"Baseline cargado desde: {args_cli.baseline_json}")

    # ── Cargar datos ──
    X_train, y_train, X_test, y_test = split_train_test_custom(
        config, base_dir, test_size=0.1, random_state=0
    )
    set_data = (X_train, y_train, X_test, y_test)

    # Preprocesar para evaluación downstream (one-hot)
    (X_train_pre, X_test_pre, y_train_pre, y_test_pre,
     num_scaler, cat_encoder, label_encoder) = preprocessing(
        config, X_train, y_train, X_test, y_test,
        encoding='one-hot', concat=True, random_state=0,
    )

    # Empaquetar argumentos
    class Args: pass
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
    args.num_scaler = num_scaler
    args.cat_encoder = cat_encoder
    args.label_encoder = label_encoder
    args.X_test_pre = X_test_pre
    args.y_test_pre = y_test_pre

    # ── Ejecutar estudio de sensibilidad ──
    print(f"\n{'='*70}")
    print(f" ESTUDIO DE SENSIBILIDAD — Dataset: {args_cli.dataset}")
    print(f" Baseline: {baseline}")
    print(f" Hiperparámetros a estudiar: {list(SENSITIVITY_SPACE.keys())}")
    print(f"{'='*70}\n")

    all_results = []
    total_experiments = sum(len(vals) for vals in SENSITIVITY_SPACE.values())
    exp_counter = 0

    for param_name, param_values in SENSITIVITY_SPACE.items():
        print(f"\n{'─'*50}")
        print(f"  Variando: {PARAM_LABELS.get(param_name, param_name)}")
        print(f"  Valores: {param_values}")
        print(f"{'─'*50}")

        for value in param_values:
            exp_counter += 1
            print(f"\n  [{exp_counter}/{total_experiments}] {param_name} = {value}")

            # Crear hiperparámetros: baseline + variación del parámetro actual
            hyperparams = dict(baseline)
            hyperparams[param_name] = value

            # Validación de compatibilidad: d_token debe ser divisible por n_head
            if hyperparams['d_token'] % hyperparams['n_head'] != 0:
                print(f"    ⏭️ Saltando: d_token={hyperparams['d_token']} no divisible por n_head={hyperparams['n_head']}")
                continue

            metrics = run_single_experiment(hyperparams, args)

            row = {
                'param_name': param_name,
                'param_value': value,
                'is_baseline': (value == baseline.get(param_name)),
                **metrics,
            }
            all_results.append(row)
            print(f"    ROC-AUC={metrics['roc_auc_recon']:.4f}  BalAcc={metrics['balanced_acc_recon']:.4f}")

    # ── Guardar resultados ──
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'sensitivity_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n📄 Resultados guardados en: {csv_path}")

    # ── Generar gráficas ──
    valid_results = results_df[results_df['status'] == 'ok']
    if len(valid_results) > 0:
        plot_sensitivity(valid_results, output_dir)
    else:
        print("⚠️ No hay resultados válidos para generar gráficas.")

    # ── Resumen ──
    print(f"\n{'='*70}")
    print(f" ESTUDIO COMPLETADO")
    print(f" Total experimentos: {len(results_df)}")
    print(f" Exitosos: {len(valid_results)}")
    print(f" CSV: {csv_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
