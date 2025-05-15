export export CUDA_VISIBLE_DEVICES=3;cd ..;python3.12 main_percentages.py \
    --metadata_path data/shoppers/metadata.json \
    --distillation_space latent \
    --epochs_latent 3000 \
    --epochs_fine_tuning_latent 1000 \
    --batch_size 4069 \
    --checkpoint_path checkpoint_shoppers_Dlatent_ft_F \
    --kmeans_type centroid \
    --hyperparams_vae tune_vae/tune_vae_AAC/tune_shoppers/best_hyperparams.json > jobs/out/Dlatent_shoppers_ft_F.txt 2>&1 &

wait
