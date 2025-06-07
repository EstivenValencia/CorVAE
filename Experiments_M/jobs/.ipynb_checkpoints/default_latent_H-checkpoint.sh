export export CUDA_VISIBLE_DEVICES=5;cd ..;python3.12 main.py \
    --metadata_path data/default/metadata.json \
    --distillation_space latent \
    --epochs_latent 3000 \
    --epochs_fine_tuning_latent 0 \
    --batch_size 4069 \
    --checkpoint_path checkpoint_default_Dlatent_G \
    --kmeans_type centroid --concat_label_vae 0\
    --hyperparams_vae tune_vae/tune_vae_AAC/tune_shoppers/best_hyperparams.json > jobs/out/Dlatent_default_G.txt 2>&1 &
    
wait
