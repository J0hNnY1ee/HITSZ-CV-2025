python main.py --data_root CamVid \
               --model_name unet \
               --img_height 224 --img_width 224 \
               --epochs 50 \
               --batch_size 32 \
               --lr 0.001 \
               --optimizer adamw \
               --scheduler_step_size 30 \
               --output_base_dir ./results/UNet \
               --experiment_name UNet_50_1e3_adamw \
               --evaluate_on_test

