cd /home/u220110828/HITSZ-CV-2025/CV_Project
# Base: optimizer=adamw
python main.py --data_root CamVid \
               --model_name deeplabv3plus \
               --img_height 224 --img_width 224 \
               --epochs 50 \
               --batch_size 16 \
               --lr 0.001 \
               --optimizer adamw \
               --scheduler_step_size 30 \
               --output_base_dir ./results/deeplabv3plus \
               --experiment_name deeplabv3plus_50_1e3_adamw \
               --evaluate_on_test

# optimizer=adam
python main.py --data_root CamVid \
               --model_name deeplabv3plus \
               --img_height 224 --img_width 224 \
               --epochs 50 \
               --batch_size 16 \
               --lr 0.001 \
               --optimizer adam \
               --scheduler_step_size 30 \
               --output_base_dir ./results/deeplabv3plus \
               --experiment_name deeplabv3plus_50_1e3_adam \
               --evaluate_on_test

# optimizer=sgd
python main.py --data_root CamVid \
               --model_name deeplabv3plus \
               --img_height 224 --img_width 224 \
               --epochs 50 \
               --batch_size 16 \
               --lr 0.001 \
               --optimizer sgd \
               --scheduler_step_size 30 \
               --output_base_dir ./results/deeplabv3plus \
               --experiment_name deeplabv3plus_50_1e3_sgd \
               --evaluate_on_test