#!/bin/bash
# Definir la GPU primero
export CUDA_VISIBLE_DEVICES=1

nohup python main.py \
    --metadata_path data/paysim/metadata.json \
    --distillation_space original \
    --epochs_latent 3000 \
    --batch_size 4069 \
    --checkpoint_path checkpoint_paysim_Doriginal_I_SVC \
    --kmeans_type centroid \
    --epochs_fine_tuning_latent 0 \
    --hyperparams_vae tune_vae/best_hyperparams.json > checkpoint_paysim_Doriginal_I_SVC.txt 2>&1 &
