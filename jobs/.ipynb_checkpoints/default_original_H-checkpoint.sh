export export CUDA_VISIBLE_DEVICES=5;cd ..;python3.12 main.py \
    --metadata_path data/default/metadata.json \
    --distillation_space original \
    --epochs_latent 3000 \
    --epochs_fine_tuning_latent 0 \
    --batch_size 4069 \
    --checkpoint_path checkpoint_default_Doriginal_H \
    --kmeans_type centroid > jobs/out/Doriginal_default_H.txt 2>&1 &

wait
