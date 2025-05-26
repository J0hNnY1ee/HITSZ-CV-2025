python main.py --data_root CamVid \
    --model_name deeplabv3plus \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.001 \
    --unet_base_c 64 \
    --unet_bilinear \
    --output_base_dir results/deeplabv3plus \
    --experiment_name deeplabv3plus_v1 \
    --evaluate_on_test

python main.py --data_root CamVid \
    --model_name simple \
    --epochs 1 \
    --batch_size 16 \
    --lr 0.001 \
    --unet_base_c 64 \
    --unet_bilinear \
    --output_base_dir results/Simple \
    --experiment_name Simple \
    --evaluate_on_test

python main.py --data_root CamVid \
    --model_name segformer_simple \
    --img_height 256 --img_width 256 \
    --epochs 150 \
    --batch_size 8 \
    --lr 6e-5 \
    --optimizer adamw \
    --segformer_embed_dims 32 64 160 256 \
    --segformer_num_heads 1 2 5 8 \
    --segformer_depths 2 2 2 2 \
    --segformer_patch_sizes 4 2 2 2 \
    --output_base_dir ./results/segformer \
    --experiment_name segformer_simple \
    --evaluate_on_test

python main.py --model_name UNet --camvid_root_dir CamVid \
    --use_augmentation --epochs 100 --batch_size 8 --lr 0.001 \
    --experiment_dir results/unet_with_aug_100 --unet_base_c 32



python main.py --data_root CamVid \
               --model_name segnet \
               --img_height 224 --img_width 224 \
               --epochs 200 \
               --batch_size 32 \
               --lr 0.001 \
               --optimizer adam \
               --scheduler_step_size 30 \
               --output_base_dir ./results/segnet \
               --experiment_name segnet_run200 \
               --evaluate_on_test