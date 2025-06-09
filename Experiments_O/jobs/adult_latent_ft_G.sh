<<<<<<< HEAD
export export CUDA_VISIBLE_DEVICES=4;cd ..;python3.12 main_percentages \
=======
export CUDA_VISIBLE_DEVICES=4;cd ..;python3.12 main.py \
>>>>>>> 500c31b3c96244a93d5af5329b5536191d23f1e5
    --metadata_path data/adult/metadata.json \
    --distillation_space latent \
    --epochs_latent 3000 \
    --epochs_fine_tuning_latent 1000 \
    --batch_size 4069 \
    --checkpoint_path checkpoint_adult_Dlatent_ft_G \
    --kmeans_type centroid \
    --concat_label_vae 0 \
    --hyperparams_vae tune_vae/tune_vae_AAC/tune_shoppers/best_hyperparams.json > jobs/out/Dlatent_adult_ft_G.txt 2>&1 &

wait
