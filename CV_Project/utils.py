# utils.py

import os
import random
import json # 用于保存字典
import numpy as np
import torch
from PIL import Image

# ==============================================================================
# 随机种子设置
# ==============================================================================
def set_seed(seed: int = 42):
    """
    设置随机种子以确保实验的可复现性。

    参数:
        seed (int): 要设置的随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # 下面两行通常用于更严格的可复现性，但可能会降低性能
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    print(f"随机种子已设置为: {seed}")


# ==============================================================================
# 检查点管理
# ==============================================================================
def save_checkpoint(state: dict, is_best: bool, experiment_dir: str, filename_prefix: str = "ckpt"):
    """
    保存模型检查点到指定的实验目录。

    参数:
        state (dict): 包含模型状态和其他信息的字典 (例如 epoch, optimizer state, best_metric_val)。
        is_best (bool): 当前模型是否是验证集上表现最好的模型。
        experiment_dir (str): 保存检查点的实验根目录 (例如 "experiments_output/my_run_timestamp/").
        filename_prefix (str): 检查点文件名的前缀。
    """
    # 确保实验目录存在 (通常在main.py中已创建，这里作为双重检查)
    if not os.path.exists(experiment_dir):
        print(f"警告: 实验目录 {experiment_dir} 不存在，正在创建...")
        os.makedirs(experiment_dir, exist_ok=True)

    # 最新检查点路径
    filepath = os.path.join(experiment_dir, f"{filename_prefix}_latest.pth.tar")
    torch.save(state, filepath)
    # 详细的保存信息通常由调用者（如Trainer）记录到日志

    if is_best:
        best_filepath = os.path.join(experiment_dir, f"{filename_prefix}_best.pth.tar")
        torch.save(state, best_filepath)
        # 详细的保存信息通常由调用者（如Trainer）记录到日志

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None,
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None, device: torch.device = None,weights_only: bool = False):
    """
    加载模型检查点。

    参数:
        checkpoint_path (str): 检查点文件的路径。
        model (torch.nn.Module): 要加载状态的模型。
        optimizer (torch.optim.Optimizer, optional): 要加载状态的优化器。
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 要加载状态的学习率调度器。
        device (torch.device, optional): 指定加载到的设备，如果为None，则使用模型当前的设备。
        weights_only (bool): torch.load 的 weights_only 参数。默认为 False 以兼容旧检查点。
    返回:
        int: 检查点保存时的 epoch 数 (如果可用)。
        float: 检查点保存时的最佳验证指标 (如果可用)。
    """
    if not os.path.exists(checkpoint_path):
        print(f"警告: 检查点文件 {checkpoint_path} 未找到。将从头开始训练。")
        return 0, float('-inf') # 返回初始epoch和负无穷大的指标

    print(f"正在从 {checkpoint_path} 加载检查点...")
    if device is None:
        # 尝试从模型推断设备，或者默认为CPU
        try:
            device = next(model.parameters()).device
        except StopIteration: # 模型没有参数
            device = torch.device('cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
    # 加载模型状态
    try:
        # 兼容旧的保存方式（可能直接保存了model.state_dict()）
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint: # 另一个常见的键名
             model.load_state_dict(checkpoint['state_dict'])
        else: # 假设整个文件就是 state_dict
            model.load_state_dict(checkpoint)
            print("警告: 检查点似乎只包含模型状态字典，其他信息（如epoch, optimizer）将丢失。")
            return 0, float('-inf') # 无法恢复 epoch 和 metric
        print("模型状态加载成功。")
    except Exception as e:
        print(f"加载模型状态字典时出错: {e}。尝试仅加载匹配的层。")
        # 尝试加载部分匹配的层 (如果模型结构有变动)
        model_dict = model.state_dict()
        # 获取源状态字典，处理多种可能的键名
        source_state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        if source_state_dict:
            pretrained_dict = {k: v for k, v in source_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict) # type: ignore
            print(f"已加载 {len(pretrained_dict)}/{len(source_state_dict)} 个匹配的层。")
        else:
            print("错误：检查点中无法找到模型状态字典。")


    # 加载优化器状态
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("优化器状态加载成功。")
        except Exception as e:
            print(f"加载优化器状态时出错: {e}。优化器将使用初始状态。")
    elif optimizer:
        print("警告: 检查点中未找到优化器状态。优化器将使用初始状态。")

    # 加载学习率调度器状态
    if scheduler and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("学习率调度器状态加载成功。")
        except Exception as e:
            print(f"加载学习率调度器状态时出错: {e}。调度器将使用初始状态。")
    elif scheduler:
        print("警告: 检查点中未找到学习率调度器状态。")


    start_epoch = checkpoint.get('epoch', 0)
    best_metric_val = checkpoint.get('best_metric_val', float('-inf'))
    metric_name = checkpoint.get('metric_name', 'unknown_metric') # 尝试获取指标名称

    print(f"检查点加载完成。将从 Epoch {start_epoch} 继续，当前最佳 {metric_name}: {best_metric_val:.4f}")
    return start_epoch, best_metric_val


# ==============================================================================
# 掩码可视化 (与 dataset.py 中的类似，但作为通用工具)
# ==============================================================================
def mask_to_rgb(mask_indices: np.ndarray, label_to_color: dict) -> np.ndarray:
    """
    将类别索引掩码转换回RGB图像，用于可视化。

    参数:
        mask_indices (np.ndarray): 包含类别索引的2D NumPy数组 (H, W)。
        label_to_color (dict): 类别索引到(R,G,B)元组的映射。

    返回:
        np.ndarray: RGB格式的3D NumPy数组 (H, W, 3)。
    """
    if mask_indices.ndim != 2:
        raise ValueError(f"mask_indices 必须是2D数组, 得到维度: {mask_indices.ndim}")

    height, width = mask_indices.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for label_idx, color_tuple in label_to_color.items():
        if not (isinstance(color_tuple, tuple) and len(color_tuple) == 3):
            # print(f"警告: utils.mask_to_rgb - 索引 {label_idx} 的颜色值 '{color_tuple}' 不是有效的RGB元组。跳过此颜色。")
            continue # 保持静默或只在调试时打印
        rgb_mask[mask_indices == label_idx] = color_tuple
    return rgb_mask

def tensor_mask_to_pil_rgb(mask_tensor: torch.Tensor, label_to_color_map: dict) -> Image.Image:
    """
    将单通道的类别索引张量 (H, W) 或 (1, H, W) 转换为可显示的PIL RGB图像。

    参数:
        mask_tensor (torch.Tensor): 类别索引掩码张量 (H,W) 或 (1,H,W)，应为 Long 或 Byte 类型。
        label_to_color_map (dict): 类别索引到 (R,G,B) 颜色的映射。

    返回:
        PIL.Image.Image: RGB格式的PIL图像。
    """
    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
        mask_tensor = mask_tensor.squeeze(0)
    elif mask_tensor.ndim != 2:
        raise ValueError(f"mask_tensor 期望形状为 (H,W) 或 (1,H,W), 得到 {mask_tensor.shape}")

    mask_np = mask_tensor.cpu().numpy().astype(np.uint8) # 确保是整数类型
    rgb_array = mask_to_rgb(mask_np, label_to_color_map)
    return Image.fromarray(rgb_array)

# ==============================================================================
# 保存字典到JSON文件
# ==============================================================================
def save_dict_to_json(data_dict: dict, json_path: str):
    """
    将字典保存为JSON文件。

    参数:
        data_dict (dict): 要保存的字典。
        json_path (str): JSON文件的保存路径。
    """
    try:
        # 将 NumPy 数组和张量转换为列表，以便JSON序列化
        def convert_to_json_serializable(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            if isinstance(item, torch.Tensor):
                return item.cpu().tolist()
            if isinstance(item, (np.float32, np.float64)): # 处理numpy标量
                return float(item)
            if isinstance(item, (np.int32, np.int64)): # 处理numpy标量
                return int(item)
            return item

        serializable_dict = {k: convert_to_json_serializable(v) for k, v in data_dict.items()}

        with open(json_path, 'w') as f:
            json.dump(serializable_dict, f, indent=4)
        print(f"字典已保存至: {json_path}")
    except Exception as e:
        print(f"保存字典到 {json_path} 时出错: {e}")


# ==============================================================================
# 测试代码 (可选)
# ==============================================================================
if __name__ == '__main__':
    print("开始 utils.py 测试...")

    # --- 测试 set_seed ---
    print("\n测试 set_seed...")
    set_seed(123)
    val1_np = np.random.rand()
    val1_torch = torch.rand(1).item()
    set_seed(123)
    val2_np = np.random.rand()
    val2_torch = torch.rand(1).item()
    assert val1_np == val2_np, "NumPy 随机种子设置失败"
    assert val1_torch == val2_torch, "PyTorch 随机种子设置失败"
    print("set_seed 测试通过。")

    # --- 测试检查点函数 (模拟) ---
    print("\n测试检查点函数...")
    class DummyModel(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear = torch.nn.Linear(10, 2)
        def forward(self, x): return self.linear(x)

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # 模拟一个实验目录
    temp_experiment_dir = "temp_utils_experiment"
    os.makedirs(temp_experiment_dir, exist_ok=True)

    state_to_save = {
        'epoch': 5, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric_val': 0.75, 'metric_name': 'test_mIoU'
    }
    save_checkpoint(state_to_save, is_best=True, experiment_dir=temp_experiment_dir, filename_prefix="test_model")

    new_model = DummyModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.1)
    checkpoint_file = os.path.join(temp_experiment_dir, "test_model_best.pth.tar")
    if os.path.exists(checkpoint_file):
        loaded_epoch, loaded_metric = load_checkpoint(checkpoint_file, new_model, new_optimizer, new_scheduler)
        assert loaded_epoch == 5, f"加载的epoch不匹配: 期望5, 得到{loaded_epoch}"
        assert loaded_metric == 0.75, f"加载的指标不匹配: 期望0.75, 得到{loaded_metric}"
        print("检查点加载和参数恢复的基本测试通过。")
    else:
        print(f"错误: 未找到期望的检查点文件 {checkpoint_file}，跳过加载测试。")
    if os.path.exists(temp_experiment_dir):
        import shutil; shutil.rmtree(temp_experiment_dir)
        print(f"临时实验目录 {temp_experiment_dir} 已删除。")

    # --- 测试掩码可视化 ---
    # (与之前版本相同，可以保留)
    print("\n测试掩码可视化函数...")
    test_palette_inverse = {0: (128, 128, 128), 1: (128, 0, 0), 11: (0, 0, 0)}
    mask_data = torch.tensor([[0, 1, 0, 11], [1, 1, 0, 0], [11, 0, 1, 11]], dtype=torch.long)
    pil_rgb_image = tensor_mask_to_pil_rgb(mask_data, test_palette_inverse)
    assert pil_rgb_image.mode == "RGB", "输出图像模式应为RGB"
    print("掩码可视化函数测试通过。")

    # --- 测试 save_dict_to_json ---
    print("\n测试 save_dict_to_json...")
    test_dict_data = {
        "name": "test_experiment", "value": 123.45,
        "metrics": {"iou": np.array([0.5, 0.6]), "loss": torch.tensor(0.123)},
        "is_valid": True, "nested": {"a":1, "b": [1,2,3]}
    }
    json_test_path = "temp_test_dict.json"
    save_dict_to_json(test_dict_data, json_test_path)
    assert os.path.exists(json_test_path), f"{json_test_path} 未创建"
    with open(json_test_path, 'r') as f_read:
        loaded_json_data = json.load(f_read)
    assert loaded_json_data["name"] == "test_experiment"
    assert loaded_json_data["metrics"]["iou"] == [0.5, 0.6] # NumPy array converted to list
    print("save_dict_to_json 测试通过。")
    if os.path.exists(json_test_path):
        os.remove(json_test_path)


    print("\nutils.py 测试结束。")