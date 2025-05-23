import kagglehub
import os

# 获取当前工作目录
current_dir = os.getcwd()

# 下载数据集到当前目录
path = kagglehub.dataset_download("carlolepelaars/camvid", download_dir=current_dir)

print("Path to dataset files:", path)
