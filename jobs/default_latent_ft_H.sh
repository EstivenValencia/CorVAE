export export CUDA_VISIBLE_DEVICES=5;cd ..;python3.12 main_percentages.py \
    --metadata_path data/default/metadata.json \
    --distillation_space latent \
    --epochs_latent 3000 \
    --epochs_fine_tuning_latent 1000 \
    --batch_size 4069 \
    --checkpoint_path checkpoint_default_Dlatent_ft_H \
    --kmeans_type centroid \
    --hyperparams_vae tune_vae/tune_vae_AAG/tune_default/best_hyperparams.json > jobs/out/Dlatent_default_ft_H.txt 2>&1 &

wait
