# utils.py

import os
import random
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt # 新增导入

# ==============================================================================
# 随机种子设置
# ==============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子已设置为: {seed}")

# ==============================================================================
# 检查点管理 (与之前版本相同)
# ==============================================================================
def save_checkpoint(state: dict, is_best: bool, experiment_dir: str, filename_prefix: str = "ckpt"):
    if not os.path.exists(experiment_dir):
        print(f"警告: 实验目录 {experiment_dir} 不存在，正在创建...")
        os.makedirs(experiment_dir, exist_ok=True)
    filepath = os.path.join(experiment_dir, f"{filename_prefix}_latest.pth.tar")
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(experiment_dir, f"{filename_prefix}_best.pth.tar")
        torch.save(state, best_filepath)

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None,
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None, device: torch.device = None,
                    weights_only: bool = False):
    if not os.path.exists(checkpoint_path):
        print(f"警告: 检查点文件 {checkpoint_path} 未找到。将从头开始训练。")
        return 0, float('-inf')
    print(f"正在从 {checkpoint_path} 加载检查点 (weights_only={weights_only})...")
    if device is None:
        try: device = next(model.parameters()).device
        except StopIteration: device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
    try:
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint: model.load_state_dict(checkpoint['state_dict'])
        else: model.load_state_dict(checkpoint); print("警告: 检查点只含模型状态。"); return 0, float('-inf')
        print("模型状态加载成功。")
    except Exception as e:
        print(f"加载模型状态时出错: {e}。尝试部分加载。"); model_dict = model.state_dict()
        source_state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        if source_state_dict:
            pretrained_dict = {k: v for k, v in source_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict); model.load_state_dict(model_dict) # type: ignore
            print(f"已加载 {len(pretrained_dict)}/{len(source_state_dict)} 匹配层。")
        else: print("错误：检查点中无模型状态字典。")
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try: optimizer.load_state_dict(checkpoint['optimizer_state_dict']); print("优化器状态加载成功。")
        except Exception as e: print(f"加载优化器状态时出错: {e}。")
    elif optimizer: print("警告: 检查点中无优化器状态。")
    if scheduler and 'scheduler_state_dict' in checkpoint:
        try: scheduler.load_state_dict(checkpoint['scheduler_state_dict']); print("学习率调度器状态加载成功。")
        except Exception as e: print(f"加载学习率调度器状态时出错: {e}。")
    elif scheduler: print("警告: 检查点中无学习率调度器状态。")
    start_epoch = checkpoint.get('epoch', 0)
    best_metric_val = checkpoint.get('best_metric_val', float('-inf'))
    if isinstance(best_metric_val, (np.generic, np.ndarray)): best_metric_val = float(best_metric_val)
    metric_name = checkpoint.get('metric_name', 'unknown_metric')
    print(f"检查点加载完成。将从 Epoch {start_epoch} 继续，当前最佳 {metric_name}: {best_metric_val:.4f}")
    return start_epoch, best_metric_val

# ==============================================================================
# 掩码可视化 (与之前版本相同)
# ==============================================================================
def mask_to_rgb(mask_indices: np.ndarray, label_to_color: dict) -> np.ndarray:
    if mask_indices.ndim != 2: raise ValueError(f"mask_indices 必须是2D数组, 得到维度: {mask_indices.ndim}")
    height, width = mask_indices.shape; rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for label_idx, color_tuple in label_to_color.items():
        if not (isinstance(color_tuple, tuple) and len(color_tuple) == 3): continue
        rgb_mask[mask_indices == label_idx] = color_tuple
    return rgb_mask

def tensor_mask_to_pil_rgb(mask_tensor: torch.Tensor, label_to_color_map: dict) -> Image.Image:
    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1: mask_tensor = mask_tensor.squeeze(0)
    elif mask_tensor.ndim != 2: raise ValueError(f"mask_tensor 期望形状为 (H,W) 或 (1,H,W), 得到 {mask_tensor.shape}")
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8); rgb_array = mask_to_rgb(mask_np, label_to_color_map)
    return Image.fromarray(rgb_array)

# ==============================================================================
# 保存字典到JSON文件 (与之前版本相同)
# ==============================================================================
def save_dict_to_json(data_dict: dict, json_path: str):
    try:
        def convert_to_json_serializable(item):
            if isinstance(item, np.ndarray): return item.tolist()
            if isinstance(item, torch.Tensor): return item.cpu().tolist()
            if isinstance(item, (np.float32, np.float64)): return float(item)
            if isinstance(item, (np.int32, np.int64)): return int(item)
            return item
        serializable_dict = {k: convert_to_json_serializable(v) for k, v in data_dict.items()}
        with open(json_path, 'w') as f: json.dump(serializable_dict, f, indent=4)
        print(f"字典已保存至: {json_path}")
    except Exception as e: print(f"保存字典到 {json_path} 时出错: {e}")

# ==============================================================================
# 绘制和保存训练历史图表
# ==============================================================================
def plot_and_save_history(history: dict, save_path_prefix: str, primary_metric_name: str = "mIoU"):
    """
    绘制训练和验证的损失以及一个主要指标的曲线，并保存为图片。

    参数:
        history (dict): 包含列表的字典，例如:
                        {
                            'train_loss': [epoch1_loss, epoch2_loss, ...],
                            'val_loss': [epoch1_loss, epoch2_loss, ...],
                            'val_metric': [epoch1_metric, epoch2_metric, ...]
                        }
        save_path_prefix (str): 保存图表的文件路径前缀 (不含.png)。
                               例如 "experiment_dir/training_plots"
                               会生成 "experiment_dir/training_plots_loss.png" 和
                                       "experiment_dir/training_plots_metric.png"
        primary_metric_name (str): history中 'val_metric' 键对应的指标名称，用于图表标题。
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.style.use('seaborn-v0_8-whitegrid') # 使用一个好看的样式

    # --- 绘制损失曲线 ---
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        loss_fig_path = f"{save_path_prefix}_loss.png"
        plt.savefig(loss_fig_path)
        plt.close() # 关闭图形，防止在某些后端中累积
        print(f"训练/验证损失曲线图已保存至: {loss_fig_path}")
    except Exception as e:
        print(f"绘制损失曲线图时出错: {e}")

    # --- 绘制验证集主要指标曲线 ---
    if 'val_metric' in history and history['val_metric']:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['val_metric'], 'go-', label=f'Validation {primary_metric_name}')
            plt.title(f'Validation {primary_metric_name}')
            plt.xlabel('Epochs')
            plt.ylabel(primary_metric_name)
            plt.legend()
            plt.grid(True)
            metric_fig_path = f"{save_path_prefix}_{primary_metric_name.lower().replace(' ', '_')}.png"
            plt.savefig(metric_fig_path)
            plt.close()
            print(f"验证集{primary_metric_name}曲线图已保存至: {metric_fig_path}")
        except Exception as e:
            print(f"绘制{primary_metric_name}曲线图时出错: {e}")

