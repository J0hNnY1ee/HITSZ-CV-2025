cd /home/u220110828/HITSZ-CV-2025/CV_Project
# Base: lr=0.001
python main.py --data_root CamVid \
               --model_name deeplabv3plus \
               --img_height 224 --img_width 224 \
               --epochs 50 \
               --batch_size 32 \
               --lr 0.001 \
               --optimizer adam \
               --scheduler_step_size 30 \
               --output_base_dir ./results/deeplabv3plus \
               --experiment_name deeplabv3plus_50_1e3_adamw \
               --evaluate_on_test

# lr=0.0001
python main.py --data_root CamVid \
               --model_name deeplabv3plus \
               --img_height 224 --img_width 224 \
               --epochs 50 \
               --batch_size 32 \
               --lr 0.0001 \
               --optimizer adam \
               --scheduler_step_size 30 \
               --output_base_dir ./results/deeplabv3plus \
               --experiment_name deeplabv3plus_50_1e4_adamw \
               --evaluate_on_test

# lr=0.005
python main.py --data_root CamVid \
               --model_name deeplabv3plus \
               --img_height 224 --img_width 224 \
               --epochs 50 \
               --batch_size 32 \
               --lr 0.005 \
               --optimizer adam \
               --scheduler_step_size 30 \
               --output_base_dir ./results/deeplabv3plus \
               --experiment_name deeplabv3plus_50_5e3_adamw \
               --evaluate_on_test

# lr=0.01
python main.py --data_root CamVid \
               --model_name deeplabv3plus \
               --img_height 224 --img_width 224 \
               --epochs 50 \
               --batch_size 32 \
               --lr 0.01 \
               --optimizer adam \
               --scheduler_step_size 30 \
               --output_base_dir ./results/deeplabv3plus \
               --experiment_name deeplabv3plus_50_1e2_adamw \
               --evaluate_on_test