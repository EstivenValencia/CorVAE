export export CUDA_VISIBLE_DEVICES=4;cd ..;python3.12 main.py \
    --metadata_path data/adult/metadata.json \
    --distillation_space latent \
    --epochs_latent 3000 \
    --epochs_fine_tuning_latent 0 \
    --batch_size 4069 \
    --checkpoint_path checkpoint_adult_Dlatent_G \
    --kmeans_type centroid \
    --hyperparams_vae tune_vae/tune_vae_AAD/tune_adult/best_hyperparams.json > jobs/out/Dlatent_adult_G.txt 2>&1 &
    
wait
