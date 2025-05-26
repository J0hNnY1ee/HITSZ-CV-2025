cd /home/u220110828/HITSZ-CV-2025/CV_Project
# Base: epochs=50
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

# epochs=10
python main.py --data_root CamVid \
               --model_name deeplabv3plus \
               --img_height 224 --img_width 224 \
               --epochs 10 \
               --batch_size 32 \
               --lr 0.001 \
               --optimizer adam \
               --scheduler_step_size 30 \
               --output_base_dir ./results/deeplabv3plus \
               --experiment_name deeplabv3plus_10_1e3_adamw \
               --evaluate_on_test

# epochs=20
python main.py --data_root CamVid \
               --model_name deeplabv3plus \
               --img_height 224 --img_width 224 \
               --epochs 20 \
               --batch_size 32 \
               --lr 0.001 \
               --optimizer adam \
               --scheduler_step_size 30 \
               --output_base_dir ./results/deeplabv3plus \
               --experiment_name deeplabv3plus_20_1e3_adamw \
               --evaluate_on_test

# epochs=100
python main.py --data_root CamVid \
               --model_name deeplabv3plus \
               --img_height 224 --img_width 224 \
               --epochs 100 \
               --batch_size 32 \
               --lr 0.001 \
               --optimizer adam \
               --scheduler_step_size 30 \
               --output_base_dir ./results/deeplabv3plus \
               --experiment_name deeplabv3plus_100_1e3_adamw \
               --evaluate_on_test