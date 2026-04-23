#!/bin/bash
# =============================================================================
# run_sensitivity.sh — Estudio de sensibilidad de hiperparámetros del CorVAE
#
# Ejecuta sensitivity_study.py para el dataset de shoppers.
# Varía un hiperparámetro a la vez desde los valores óptimos.
#
# Uso:
#   chmod +x run_sensitivity.sh
#   nohup bash run_sensitivity.sh > sensitivity_log.txt 2>&1 &
# =============================================================================

echo "PID del proceso principal (Bash): $$"
echo "------------------------------------------------------------"

export CUDA_VISIBLE_DEVICES=1

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
TUNE_DIR="${PROJECT_DIR}/tune_vae"
LOGS_DIR="${PROJECT_DIR}/jobs/sensitivity_logs"
mkdir -p "${LOGS_DIR}"

echo "============================================================"
echo " ESTUDIO DE SENSIBILIDAD DE HIPERPARÁMETROS — CorVAE"
echo " Fecha: $(date)"
echo " PID: $$"
echo " GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"
echo ""

# ── Parámetros ───────────────────────────────────────────────────────────────
DATASET="shoppers"
EPOCHS=700
BATCH_SIZE=4096
IPC=50
SEED=0

# Ruta al JSON con hiperparámetros óptimos (baseline)
BASELINE_JSON="${TUNE_DIR}/tune_vae_AAC/tune_shoppers/best_hyperparams.json"

echo "Dataset: ${DATASET}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "IPC: ${IPC}"
echo "Baseline JSON: ${BASELINE_JSON}"
echo ""

# ── Ejecutar estudio ─────────────────────────────────────────────────────────
cd "${TUNE_DIR}"

python sensitivity_study.py \
    --dataset "${DATASET}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --ipc ${IPC} \
    --seed ${SEED} \
    --baseline_json "${BASELINE_JSON}" \
    2>&1 | tee "${LOGS_DIR}/sensitivity_${DATASET}.txt"

EXIT_CODE=$?
cd "${PROJECT_DIR}"

if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "❌ El estudio de sensibilidad falló. Revisa el log:"
    echo "   ${LOGS_DIR}/sensitivity_${DATASET}.txt"
    exit 1
fi

echo ""
echo "============================================================"
echo " ✅ ESTUDIO DE SENSIBILIDAD COMPLETADO — $(date)"
echo " Resultados: ${TUNE_DIR}/sensitivity_${DATASET}/"
echo "   - sensitivity_results.csv"
echo "   - sensitivity_plots.png"
echo "============================================================"
