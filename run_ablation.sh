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
#   nohup bash run_ablation.sh > jobs/ablation_logs/ablation_full_log.txt 2>&1 &
# =============================================================================

echo "PID del proceso principal (Bash): $$"
echo "------------------------------------------------------------"

# ── Configuración ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=1

#DATASETS=("adult" "default" "magic_telescope" "shoppers" "paysim")
#DATASETS=("default" "magic_telescope" "shoppers" "paysim")
DATASETS=("paysim")

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
echo " PID: $$"
echo " GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

# # =============================================================================
# # SMOKE TEST: Validación rápida con 'adult' (2 epochs, 1 trial Optuna)
# # =============================================================================
# echo ""
# echo "============================================================"
# echo " SMOKE TEST: Validación rápida con dataset 'adult'"
# echo " (2 epochs de entrenamiento, 1 trial de Optuna)"
# echo "============================================================"
# echo ""

# SMOKE_DATASET="shoppers"
# SMOKE_METADATA="${PROJECT_DIR}/data/${SMOKE_DATASET}/metadata.json"
# SMOKE_CKPT="${PROJECT_DIR}/checkpoint_smoke_test_mlp"
# SMOKE_HYPER="${TUNE_DIR}/tune_mlp_smoke_test/best_hyperparams_mlp.json"

# # Fase 1 Smoke: Optuna con 1 trial y 2 epochs
# echo "[$(date +%H:%M:%S)] Smoke Test Fase 1: Optuna (1 trial, 2 epochs)..."
# cd "${TUNE_DIR}"
# python tune_mlp_vae.py \
#     --dataset "${SMOKE_DATASET}" \
#     --n_trials 1 \
#     --epochs 2 \
#     --batch_size 512 \
#     --ipc 10 \
#     2>&1 | tee "${LOGS_DIR}/smoke_test_tune.txt"

# SMOKE_TEST_TUNE_EXIT=$?
# cd "${PROJECT_DIR}"

# if [ ${SMOKE_TEST_TUNE_EXIT} -ne 0 ]; then
#     echo ""
#     echo "❌ SMOKE TEST FALLÓ en Fase 1 (Optuna). Revisa el log anterior."
#     echo "   Abortando ejecución."
#     exit 1
# fi
# echo "[$(date +%H:%M:%S)] ✅ Smoke Test Fase 1 completada."

# # Fase 2 Smoke: Entrenamiento + destilación con 2 epochs
# # Usar hiperparámetros del smoke test de Optuna
# SMOKE_HYPER_ACTUAL="${TUNE_DIR}/tune_mlp_${SMOKE_DATASET}/best_hyperparams_mlp.json"
# echo "[$(date +%H:%M:%S)] Smoke Test Fase 2: Entrenamiento MLP-VAE (2 epochs)..."
# python main.py \
#     --metadata_path "${SMOKE_METADATA}" \
#     --distillation_space latent \
#     --epochs_latent 2 \
#     --epochs_fine_tuning_latent 0 \
#     --batch_size 512 \
#     --checkpoint_path "${SMOKE_CKPT}" \
#     --kmeans_type ${KMEANS_TYPE} \
#     --hyperparams_vae "${SMOKE_HYPER_ACTUAL}" \
#     --architecture mlp \
#     2>&1 | tee "${LOGS_DIR}/smoke_test_train.txt"

# SMOKE_TEST_TRAIN_EXIT=$?

# if [ ${SMOKE_TEST_TRAIN_EXIT} -ne 0 ]; then
#     echo ""
#     echo "❌ SMOKE TEST FALLÓ en Fase 2 (Entrenamiento). Revisa el log anterior."
#     echo "   Abortando ejecución."
#     exit 1
# fi

# echo ""
# echo "============================================================"
# echo " ✅ SMOKE TEST COMPLETADO EXITOSAMENTE"
# echo "    El pipeline MLP-VAE funciona correctamente."
# echo "    Continuando con los datasets completos..."
# echo "============================================================"
# echo ""

# Limpiar checkpoint del smoke test
#rm -rf "${SMOKE_CKPT}"

# =============================================================================
# EJECUCIÓN PRINCIPAL: Todos los datasets con parámetros completos
# =============================================================================

# ── Bucle principal sobre datasets ───────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "************************************************************"
    echo " DATASET: ${DATASET}"
    echo "************************************************************"
    echo ""

    METADATA_PATH="${PROJECT_DIR}/data/${DATASET}/metadata.json"
    CHECKPOINT_PATH="${PROJECT_DIR}/results/checkpoint/checkpoint_${DATASET}_MLP_Dlatent"
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
    echo "  Resultados en: results/checkpoint/checkpoint_${DATASET}_MLP_Dlatent/metrics.csv"
    echo ""

done

echo ""
echo "============================================================"
echo " ABLACIÓN COMPLETA — $(date)"
echo " Resultados disponibles en results/checkpoint/checkpoint_*_MLP_Dlatent/"
echo "============================================================"
