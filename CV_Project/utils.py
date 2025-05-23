# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import config # 导入配置

def get_class_colormap():
    """从class_dict.csv读取类别名称和RGB值，返回颜色图和类别名"""
    import pandas as pd
    class_df = pd.read_csv(config.CLASS_CSV_PATH)
    class_df[['r', 'g', 'b']] = class_df[['r', 'g', 'b']].astype(int)
    colormap = class_df[['r', 'g', 'b']].values.tolist()
    class_names = class_df['name'].tolist()
    return colormap, class_names

# 获取颜色图和类别名
CLASS_COLORMAP, CLASS_NAMES = get_class_colormap()
NUM_CLASSES = len(CLASS_NAMES)


def tensor_to_pil(tensor_image):
    """将归一化的Tensor图像转为PIL Image (用于显示原始图像)"""
    # 反归一化
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    pil_image = T.ToPILImage()(inv_normalize(tensor_image.cpu()))
    return pil_image

def mask_to_rgb(mask_tensor, colormap):
    """将类别索引掩码 (H, W) Tensor 转换为RGB PIL Image"""
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    rgb_image_np = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(colormap):
        rgb_image_np[mask_np == class_idx] = color
    return Image.fromarray(rgb_image_np)


def plot_segmentation_results(original_image_tensor, true_mask_tensor, pred_mask_tensor, num_samples=3):
    """
    显示原始图像、真实掩码和预测掩码的对比图.
    original_image_tensor: (B, C, H, W) or (C,H,W)
    true_mask_tensor: (B, H, W) or (H,W)
    pred_mask_tensor: (B, H, W) or (H,W) - 已经是类别索引
    """
    if original_image_tensor.ndim == 3: # 单张图
        original_image_tensor = original_image_tensor.unsqueeze(0)
        true_mask_tensor = true_mask_tensor.unsqueeze(0)
        pred_mask_tensor = pred_mask_tensor.unsqueeze(0)

    batch_size = original_image_tensor.shape[0]
    num_to_show = min(num_samples, batch_size)

    fig, axes = plt.subplots(num_to_show, 3, figsize=(15, num_to_show * 5))
    if num_to_show == 1: # plt.subplots behaves differently for 1 row
        axes = [axes]

    for i in range(num_to_show):
        img_pil = tensor_to_pil(original_image_tensor[i])
        true_mask_rgb = mask_to_rgb(true_mask_tensor[i], CLASS_COLORMAP)
        pred_mask_rgb = mask_to_rgb(pred_mask_tensor[i], CLASS_COLORMAP)

        axes[i][0].imshow(img_pil)
        axes[i][0].set_title("original_image")
        axes[i][0].axis('off')

        axes[i][1].imshow(true_mask_rgb)
        axes[i][1].set_title("real_mask")
        axes[i][1].axis('off')

        axes[i][2].imshow(pred_mask_rgb)
        axes[i][2].set_title("pred_mask")
        axes[i][2].axis('off')

    plt.tight_layout()
    plt.show()


# 简单的像素准确率计算
def pixel_accuracy(pred_mask, true_mask):
    """
    计算像素准确率.
    pred_mask: (B, H, W) or (H,W) 预测的类别索引
    true_mask: (B, H, W) or (H,W) 真实的类别索引
    """
    correct = (pred_mask == true_mask).sum().item()
    total = true_mask.nelement() # 总像素数
    return correct / total if total > 0 else 0.0

# --- 新增函数开始 ---
def calculate_miou_and_iou_per_class(pred_masks_all, true_masks_all, num_classes):
    """
    计算平均交并比 (mIoU) 和每个类别的 IoU，跨整个数据集.
    pred_masks_all: (N_samples, H, W) 预测的类别索引 Tensor (CPU or CUDA).
    true_masks_all: (N_samples, H, W) 真实的类别索引 Tensor (CPU or CUDA, 必须与pred_masks_all在同一设备).
    num_classes: 类别总数.
    Returns:
        miou (float): 平均IoU.
        iou_per_class_agg (Tensor): 每个类别的聚合IoU，形状 (num_classes,).
    """
    # 确保数据类型正确且在同一设备
    pred_masks_all = pred_masks_all.long()
    true_masks_all = true_masks_all.long()
    
    if pred_masks_all.device != true_masks_all.device:
        # 为简单起见，我们假设它们应该在同一设备上，或者将一个移到另一个
        # 这里可以添加一个警告或错误，或者自动移动
        print("警告: 预测掩码和真实掩码不在同一设备上。可能会导致错误。")

    # 用于累积每个类别的总交集和总并集
    # 使用 float64 以获得更精确的累加
    total_intersection_per_class = torch.zeros(num_classes, dtype=torch.float64, device=pred_masks_all.device)
    total_union_per_class = torch.zeros(num_classes, dtype=torch.float64, device=pred_masks_all.device)
    
    for cls_idx in range(num_classes):
        pred_is_class = (pred_masks_all == cls_idx)
        true_is_class = (true_masks_all == cls_idx)

        intersection = torch.logical_and(pred_is_class, true_is_class).sum()
        union = torch.logical_or(pred_is_class, true_is_class).sum()
        
        total_intersection_per_class[cls_idx] += intersection.double() # 确保使用double进行累加
        total_union_per_class[cls_idx] += union.double()

    # 计算每个类别的聚合 IoU
    iou_per_class_agg = torch.zeros(num_classes, dtype=torch.float32, device=pred_masks_all.device)
    
    # 仅为那些在数据集中至少出现一次的类别（即union > 0）计算IoU
    valid_classes_mask = total_union_per_class > 0
    
    iou_per_class_agg[valid_classes_mask] = \
        (total_intersection_per_class[valid_classes_mask] / total_union_per_class[valid_classes_mask]).float()
    
    # 对于在整个数据集中都未出现的类别（union=0），其IoU为NaN，不计入mIoU
    # torch.nanmean 会自动处理 NaN 值
    iou_per_class_agg[~valid_classes_mask] = float('nan')
        
    # 计算 mIoU 时忽略 NaN 值
    miou = torch.nanmean(iou_per_class_agg).item()
    
    # 如果所有类的IoU都是NaN（例如，num_classes=0或数据全空），nanmean可能返回NaN
    if np.isnan(miou):
        miou = 0.0 # 在这种不太可能的情况下，将 mIoU 设为 0
        
    return miou, iou_per_class_agg
# --- 新增函数结束 ---

# 为了符合 torchvision.transforms 的使用，这里也导入它
from torchvision import transforms as T

if __name__ == '__main__':
    print("测试 utils...")
    print(f"类别数量: {NUM_CLASSES}")
    print(f"颜色图条目数量: {len(CLASS_COLORMAP)}")
    print(f"类别名称: {CLASS_NAMES}")

    # 创建一个假的图像和掩码来测试绘图
    dummy_img = torch.rand(3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    dummy_true_mask = torch.randint(0, NUM_CLASSES, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), dtype=torch.long)
    dummy_pred_mask = torch.randint(0, NUM_CLASSES, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), dtype=torch.long)

    print("测试绘图功能 (如果 matplotlib 正常工作)...")
    # plot_segmentation_results(dummy_img, dummy_true_mask, dummy_pred_mask, num_samples=1)
    # 取消注释上面一行以在本地测试绘图

    acc = pixel_accuracy(dummy_pred_mask, dummy_true_mask)
    print(f"虚拟数据的像素准确率: {acc:.4f}")

    # 测试 mIoU 函数
    if NUM_CLASSES > 0:
        print("\n测试mIoU计算...")
        # 假设批次大小为2
        dummy_pred_batch = torch.stack([dummy_pred_mask, torch.randint(0, NUM_CLASSES, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), dtype=torch.long)])
        dummy_true_batch = torch.stack([dummy_true_mask, torch.randint(0, NUM_CLASSES, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), dtype=torch.long)])
        
        miou_test, iou_per_class_test = calculate_miou_and_iou_per_class(dummy_pred_batch, dummy_true_batch, NUM_CLASSES)
        print(f"虚拟数据的mIoU: {miou_test:.4f}")
        # print(f"虚拟数据每类IoU: {iou_per_class_test}")
        for i, iou_val in enumerate(iou_per_class_test):
            class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
            print(f"  IoU for '{class_name}': {iou_val:.4f}")
    else:
        print("NUM_CLASSES is 0, skipping mIoU test.")