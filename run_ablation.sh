#!/bin/bash
# =============================================================================
# run_ablation.sh — Estudio de ablación: CorVAE (Transformer) vs. MLP-VAE
#
# Este script ejecuta de forma secuencial, para los 5 datasets, el pipeline
# completo de entrenamiento y destilación del VAE basado en MLP.
#
# Fases por dataset:
#   1. Búsqueda de hiperparámetros con Optuna (si no existen ya)
#   2. Entrenamiento + destilación con main.py --architecture mlp
#
# Uso:
#   chmod +x run_ablation.sh
#   nohup bash run_ablation.sh > ablation_full_log.txt 2>&1 &
# =============================================================================

set -e  # Detener ejecución si algún comando falla

# ── Configuración ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1

DATASETS=("adult" "default" "magic_telescope" "shoppers" "paysim")

# Parámetros de entrenamiento
EPOCHS_LATENT=3000
BATCH_SIZE=4096
KMEANS_TYPE="centroid"
EPOCHS_FT=0

# Parámetros de Optuna
OPTUNA_TRIALS=30
OPTUNA_EPOCHS=700

# Directorio raíz del proyecto (asume que el script está en la raíz del repo)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TUNE_DIR="${PROJECT_DIR}/tune_vae"
LOGS_DIR="${PROJECT_DIR}/jobs/ablation_logs"
mkdir -p "${LOGS_DIR}"

echo "============================================================"
echo " ESTUDIO DE ABLACIÓN: CorVAE (Transformer) vs. MLP-VAE"
echo " Fecha: $(date)"
echo " GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"
echo ""

# ── Bucle principal sobre datasets ───────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "************************************************************"
    echo " DATASET: ${DATASET}"
    echo "************************************************************"
    echo ""

    METADATA_PATH="${PROJECT_DIR}/data/${DATASET}/metadata.json"
    CHECKPOINT_PATH="${PROJECT_DIR}/checkpoint_${DATASET}_MLP_Dlatent"
    HYPERPARAMS_FILE="${TUNE_DIR}/tune_mlp_${DATASET}/best_hyperparams_mlp.json"

    # ── Fase 1: Búsqueda de hiperparámetros con Optuna ───────────────────
    if [ ! -f "${HYPERPARAMS_FILE}" ]; then
        echo "[$(date +%H:%M:%S)] Fase 1: Buscando hiperparámetros MLP para ${DATASET}..."
        echo "  Trials: ${OPTUNA_TRIALS}, Epochs/trial: ${OPTUNA_EPOCHS}"

        cd "${TUNE_DIR}"
        python tune_mlp_vae.py \
            --dataset "${DATASET}" \
            --n_trials ${OPTUNA_TRIALS} \
            --epochs ${OPTUNA_EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            > "${LOGS_DIR}/tune_mlp_${DATASET}.txt" 2>&1

        cd "${PROJECT_DIR}"

        echo "[$(date +%H:%M:%S)] Fase 1 completada para ${DATASET}."
        echo "  Hiperparámetros guardados en: ${HYPERPARAMS_FILE}"
    else
        echo "[$(date +%H:%M:%S)] Fase 1: Hiperparámetros MLP ya existen para ${DATASET}."
        echo "  Usando: ${HYPERPARAMS_FILE}"
    fi

    # ── Fase 2: Entrenamiento + destilación con MLP-VAE ──────────────────
    echo "[$(date +%H:%M:%S)] Fase 2: Entrenando MLP-VAE y destilando para ${DATASET}..."
    echo "  Epochs: ${EPOCHS_LATENT}, Batch size: ${BATCH_SIZE}"
    echo "  Checkpoint: ${CHECKPOINT_PATH}"

    python main.py \
        --metadata_path "${METADATA_PATH}" \
        --distillation_space latent \
        --epochs_latent ${EPOCHS_LATENT} \
        --epochs_fine_tuning_latent ${EPOCHS_FT} \
        --batch_size ${BATCH_SIZE} \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --kmeans_type ${KMEANS_TYPE} \
        --hyperparams_vae "${HYPERPARAMS_FILE}" \
        --architecture mlp \
        > "${LOGS_DIR}/train_mlp_${DATASET}.txt" 2>&1

    echo "[$(date +%H:%M:%S)] Fase 2 completada para ${DATASET}."
    echo "  Resultados en: ${CHECKPOINT_PATH}/metrics.csv"
    echo ""

done

echo ""
echo "============================================================"
echo " ABLACIÓN COMPLETA — $(date)"
echo " Resultados disponibles en los directorios checkpoint_*_MLP_Dlatent"
echo "============================================================"
