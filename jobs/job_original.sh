#!/bin/bash
# Definir la GPU primero
export CUDA_VISIBLE_DEVICES=1

nohup python main.py \
    --metadata_path data/paysim/metadata.json \
    --distillation_space original \
    --epochs_latent 3000 \
    --batch_size 4069 \
    --checkpoint_path results/checkpoint/checkpoint_paysim_Doriginal_I \
    --kmeans_type centroid \
    --epochs_fine_tuning_latent 0 \
    --hyperparams_vae tune_vae/best_hyperparams.json > jobs/ablation_logs/checkpoint_paysim_Doriginal_I.txt 2>&1 &
    
# Capturar el PID inmediatamente
PID_PROCESO=$!

# Forzar la escritura al archivo
echo "PID del proceso: $PID_PROCESO" >> jobs/ablation_logs/checkpoint_paysim_Doriginal_I.txt

# Opcional: Imprimir en pantalla para confirmar
echo "Proceso lanzado con PID: $PID_PROCESO"
