export export CUDA_VISIBLE_DEVICES=4;cd ..;python3.12 main.py \
    --metadata_path data/adult/metadata.json \
    --distillation_space original \
    --epochs_latent 3000 \
    --epochs_fine_tuning_latent 0 \
    --batch_size 4069 \
    --checkpoint_path checkpoint_adult_Doriginal_G \
    --kmeans_type centroid > jobs/out/Doriginal_adult_G.txt 2>&1 &

wait
