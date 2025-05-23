# config.py

import torch

# 数据集路径
DATA_DIR = 'CamVid/' # 请确保这是CamVid数据集的根目录
CLASS_CSV_PATH = DATA_DIR + 'class_dict.csv'

# 图像参数
IMAGE_HEIGHT = 224 # 调整图像高度
IMAGE_WIDTH = 224  # 调整图像宽度

# 训练参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 10 # 简单演示，实际可能需要更多
LEARNING_RATE = 1e-3
NUM_WORKERS = 2 # Dataloader的工作进程数

# 模型保存路径 (可选)
MODEL_SAVE_PATH = './simple_segmentation_model.pth'