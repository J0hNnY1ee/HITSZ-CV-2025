python main.py --data_root CamVid \
               --epochs 1 \
               --batch_size 40 \
               --lr 0.01 \
               --optimizer adam \
               --output_base_dir ./results/Simple \
               --experiment_name simple_model_v1_lr001 \
               --use_tensorboard \
               --evaluate_on_test