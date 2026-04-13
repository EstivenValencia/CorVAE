# export CUDA_VISIBLE_DEVICES=3;python3.12 main.py \
#     --metadata_path data/shoppers/metadata.json \
#     --method_distillation k-means \
#     --distillation_space latent \
#     --fine_tuning false \
#     --epochs_latent 3000 \
#     --epochs_fine_tuning_latent 0 \
#     --batch_size 4069 \
#     --token_dimension 4 --alpha 0.5 \
#     --examples_for_distillation 10 \
#     --checkpoint_path checkpoint_shoppers_Dlatent_Roriginal_B \
#     --kmeans_type centroid \
#     --hyperparams_vae tune_vae/tune_vae_AAB/tune_shoppers/best_hyperparams.json > Dlatent_Roriginal_shoppers_B.txt 2>&1

nohup export CUDA_VISIBLE_DEVICES=3;python main.py \
    --metadata_path data/paysim/metadata.json \
    --distillation_space latent \
    --epochs_latent 3000 \
    --batch_size 4069 \
    --checkpoint_path checkpoint_paysim_Dlatent_I \
    --kmeans_type centroid
    --hyperparams_vae tune_vae/best_hyperparams.json > checkpoint_paysim_Dlatent_I.txt 2>&1 &

export export CUDA_VISIBLE_DEVICES=4;python3.12 main.py \
    --metadata_path data/shoppers/metadata.json \
    --method_distillation k-means \
    --distillation_space original \
    --fine_tuning false \
    --epochs_latent 3000 \
    --epochs_fine_tuning_latent 0 \
    --batch_size 4069 \
    --token_dimension 4 --alpha 0.5 \
    --examples_for_distillation 10 \
    --checkpoint_path checkpoint_shoppers_Doriginal_Roriginal_B \
    --kmeans_type centroid > Doriginal_Roriginal_shoppers_B.txt 2>&1 &

# ----------------------------------------

# export export CUDA_VISIBLE_DEVICES=4;python3.12 main_random.py \
#     --metadata_path data/shoppers/metadata.json \
#     --method_distillation k-means \
#     --distillation_space original \
#     --fine_tuning false \
#     --epochs_latent 3000 \
#     --epochs_fine_tuning_latent 0 \
#     --batch_size 4069 \
#     --token_dimension 4 --alpha 0.5 \
#     --examples_for_distillation 10 \
#     --checkpoint_path checkpoint_shoppers_Doriginal_Roriginal_C \
#     --kmeans_type centroid > Doriginal_Roriginal_shoppers_C.txt 2>&1 &

#-------------------------------------

# export export CUDA_VISIBLE_DEVICES=4;python3.12 main.py \
#     --metadata_path data/adult/metadata.json \
#     --method_distillation k-means \
#     --distillation_space latent \
#     --fine_tuning false \
#     --epochs_latent 3000 \
#     --epochs_fine_tuning_latent 0 \
#     --batch_size 4069 \
#     --token_dimension 4 --alpha 0.5 \
#     --examples_for_distillation 10 \
#     --checkpoint_path checkpoint_adult_Dlatent_Roriginal \
#     --kmeans_type centroid \
#     --hyperparams_vae tune_vae/tune_vae_AAD/tune_adult/best_hyperparams.json > Dlatent_Roriginal_adult.txt 2>&1 &

# export export CUDA_VISIBLE_DEVICES=5;python3.12 main.py \
#     --metadata_path data/adult/metadata.json \
#     --method_distillation k-means \
#     --distillation_space latent \
#     --fine_tuning false \
#     --epochs_latent 3000 \
#     --epochs_fine_tuning_latent 1000 \
#     --batch_size 4069 \
#     --token_dimension 4 --alpha 0.5 \
#     --examples_for_distillation 10 \
#     --checkpoint_path checkpoint_adult_Dlatent_Roriginal_ft \
#     --kmeans_type centroid \
#     --hyperparams_vae tune_vae/tune_vae_AAE/tune_adult/best_hyperparams.json > Dlatent_Roriginal_adult_ft.txt 2>&1 &

# export export CUDA_VISIBLE_DEVICES=5;python3.12 main.py \
#     --metadata_path data/adult/metadata.json \
#     --method_distillation k-means \
#     --distillation_space original \
#     --fine_tuning false \
#     --epochs_latent 3000 \
#     --epochs_fine_tuning_latent 0 \
#     --batch_size 4069 \
#     --token_dimension 4 --alpha 0.5 \
#     --examples_for_distillation 10 \
#     --checkpoint_path checkpoint_adult_Doriginal_Roriginal \
#     --kmeans_type centroid  > Doriginal_Roriginal_adult.txt 2>&1 &

wait
