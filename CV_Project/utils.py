# ==============================================================================
# 模块导入 (IMPORTS)
# ==============================================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # 用于可视化中的图像转换
from torchvision import transforms # 用于可视化中的图像转换

# 从 dataset.py 导入颜色映射 (确保 dataset.py 在PYTHONPATH中或同级目录)
# 为了 utils.py 能独立运行测试，这里可以重新定义或有条件导入
try:
    from dataset import VOC_COLORMAP, VOC_CLASSES
except ImportError:
    print("警告: dataset.py 未找到或无法导入 VOC_COLORMAP/VOC_CLASSES。可视化将使用默认颜色。")
    VOC_COLORMAP = [[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128]] # 示例，不完整
    VOC_CLASSES = ['class0', 'class1', 'class2', 'class3', 'class4'] # 示例

# ==============================================================================
# 评估指标计算函数 (METRIC CALCULATION FUNCTIONS)
# ==============================================================================
def calculate_iou_batch(pred_masks, true_masks, num_classes, ignore_index=255):
    """
    计算一个批次中每个类别的 IoU 以及平均 IoU (mIoU)。
    假设 pred_masks 和 true_masks 已经是 (B, H, W) 或 (N, H, W) 形状的 LongTensor。
    """
    # 初始化每个类别的交集和并集计数器
    intersection_per_class = torch.zeros(num_classes, device=pred_masks.device, dtype=torch.float64)
    union_per_class = torch.zeros(num_classes, device=pred_masks.device, dtype=torch.float64)

    # 展平掩码以便于计算
    pred_flat = pred_masks.view(-1)
    true_flat = true_masks.view(-1)

    # 创建有效像素的掩码 (忽略 ignore_index)
    valid_mask = (true_flat != ignore_index)
    pred_flat_valid = pred_flat[valid_mask]
    true_flat_valid = true_flat[valid_mask]

    # 计算每个类别的交集和并集
    for cls_id in range(num_classes):
        pred_is_cls = (pred_flat_valid == cls_id)
        true_is_cls = (true_flat_valid == cls_id)
        
        intersection_per_class[cls_id] = (pred_is_cls & true_is_cls).sum()
        union_per_class[cls_id] = (pred_is_cls | true_is_cls).sum()

    # 计算每个类别的 IoU (防止除以零)
    # union_per_class 为0时，iou也应为0 (或nan，这里处理为0)
    iou_per_class = torch.where(union_per_class > 0, intersection_per_class / union_per_class, torch.tensor(0.0, device=pred_masks.device))
    
    # 计算 mIoU (通常只对在真值中出现过的类别进行平均)
    # 这里我们对所有IoU > 0 的类别进行平均，或者对所有类别IoU求平均（如果作业要求如此）
    # 作业要求是 "求出每一类的IOU取平均值" -> 直接对所有类别（即使union为0，iou也为0）求平均
    # 如果某些类别在数据集中从未出现，它们的IoU将是0，这会拉低mIoU，这是合理的。
    # 但更常见的做法是只对 ground truth 中存在的类计算 mIoU
    valid_classes_for_miou_mask = union_per_class > 0 # 只考虑在GT中至少出现一次的类别
    if valid_classes_for_miou_mask.any():
        miou = iou_per_class[valid_classes_for_miou_mask].mean()
    else: # 如果没有任何有效类别（例如，所有像素都被忽略）
        miou = torch.tensor(0.0, device=pred_masks.device)
        
    return iou_per_class.cpu().numpy(), miou.item()


def calculate_pixel_accuracy_batch(pred_masks, true_masks, ignore_index=255):
    """
    计算一个批次的像素准确率 (Pixel Accuracy)。
    假设 pred_masks 和 true_masks 已经是 (B, H, W) 或 (N, H, W) 形状的 LongTensor。
    """
    pred_flat = pred_masks.view(-1)
    true_flat = true_masks.view(-1)
    
    valid_mask = (true_flat != ignore_index)
    
    correct_pixels = (pred_flat[valid_mask] == true_flat[valid_mask]).sum().item()
    total_valid_pixels = valid_mask.sum().item()
    
    pixel_accuracy = 0.0
    if total_valid_pixels > 0:
        pixel_accuracy = correct_pixels / total_valid_pixels
        
    return pixel_accuracy


# ==============================================================================
# 可视化辅助函数 (VISUALIZATION HELPER FUNCTIONS)
# ==============================================================================
def tensor_to_pil_image(tensor_image: torch.Tensor, unnormalize=True):
    """
    将归一化后的 Tensor 图像 (C, H, W) 转换回 PIL Image 以便显示。
    """
    img_tensor_proc = tensor_image.cpu().clone()
    if unnormalize:
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        img_tensor_proc = inv_normalize(img_tensor_proc)
    pil_img = transforms.ToPILImage()(img_tensor_proc)
    return pil_img


def mask_to_colored_pil(mask_tensor: torch.Tensor, colormap, num_classes):
    """
    将单通道类别索引掩码转换为彩色 PIL 图像。
    mask_tensor: (H, W),值为类别索引。
    """
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    
    for cls_id in range(num_classes):
        # 确保 colormap 长度足够
        color = colormap[cls_id % len(colormap)] 
        colored_mask[mask_np == cls_id] = color
    
    # 处理可能的忽略标签 (如255)，通常将其设为黑色或特定颜色
    # (在此简化版本中，如果255不在0-num_classes-1范围内，它不会被着色)
    return Image.fromarray(colored_mask)


def visualize_segmentation(image_tensor: torch.Tensor, 
                           pred_mask_tensor: torch.Tensor, 
                           true_mask_tensor: torch.Tensor = None, 
                           num_classes=len(VOC_CLASSES), 
                           colormap=VOC_COLORMAP,
                           figsize=(15,5)):
    """
    可视化原始图像、预测掩码和真实掩码 (如果提供)。
    image_tensor: (C, H, W)
    pred_mask_tensor: (H, W), 类别索引
    true_mask_tensor: (H, W), 类别索引
    """
    original_image_pil = tensor_to_pil_image(image_tensor)
    pred_colored_mask_pil = mask_to_colored_pil(pred_mask_tensor, colormap, num_classes)

    num_plots = 2
    if true_mask_tensor is not None:
        num_plots = 3
        true_colored_mask_pil = mask_to_colored_pil(true_mask_tensor, colormap, num_classes)

    plt.figure(figsize=figsize)
    
    plt.subplot(1, num_plots, 1)
    plt.imshow(original_image_pil)
    plt.title("原始图像 (Original Image)")
    plt.axis('off')

    plt.subplot(1, num_plots, 2)
    plt.imshow(pred_colored_mask_pil)
    plt.title("预测掩码 (Predicted Mask)")
    plt.axis('off')

    if true_mask_tensor is not None:
        plt.subplot(1, num_plots, 3)
        plt.imshow(true_colored_mask_pil)
        plt.title("真实掩码 (Ground Truth)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 主执行块 (测试用) (MAIN EXECUTION BLOCK - FOR TESTING)
# ==============================================================================
if __name__ == '__main__':
    print("开始测试工具函数...")
    
    # --- 测试指标计算 ---
    num_test_classes_metric = 3
    # 假设 B=1, H=2, W=3
    pred_m = torch.tensor([[[0, 1, 2], [2, 0, 1]]], dtype=torch.long) # 预测 (1, 2, 3)
    true_m = torch.tensor([[[0, 1, 1], [2, 255, 0]]], dtype=torch.long) # 真实 (1, 2, 3), 255是忽略
    
    iou_pc, miou_val = calculate_iou_batch(pred_m, true_m, num_test_classes_metric, ignore_index=255)
    # Valid comparison:
    # pred_flat_valid: [0, 1, 2, 2, 0]
    # true_flat_valid: [0, 1, 1, 2, 0]
    # Class 0: P:[T,F,F,F,T], T:[T,F,F,F,T]. Inter=2, Union=2. IoU=1.0
    # Class 1: P:[F,T,F,F,F], T:[F,T,T,F,F]. Inter=1, Union=2. IoU=0.5
    # Class 2: P:[F,F,T,T,F], T:[F,F,F,T,F]. Inter=1, Union=2. IoU=0.5
    # Valid classes for mIoU are 0, 1, 2. mIoU = (1.0 + 0.5 + 0.5) / 3 = 0.666...
    print(f"  IoU per class: {iou_pc}") # 期望 [1.0, 0.5, 0.5]
    print(f"  mIoU: {miou_val:.4f}")    # 期望 0.6667

    pa_val = calculate_pixel_accuracy_batch(pred_m, true_m, num_test_classes_metric, ignore_index=255)
    # Correct: (0==0), (1==1), (2!=1), (2==2), (0==0). Correct=4. Total valid=5. PA = 4/5 = 0.8
    print(f"  Pixel Accuracy: {pa_val:.4f}") # 期望 0.8000
    
    # --- 测试可视化 (需要一个假的图像张量和掩码) ---
    print("\n  测试可视化 (将弹出一个窗口)...")
    dummy_img_tensor_vis = torch.rand(3, 64, 64) # C, H, W
    dummy_pred_mask_vis = torch.randint(0, num_test_classes_metric, (64, 64), dtype=torch.long) # H, W
    dummy_true_mask_vis = torch.randint(0, num_test_classes_metric, (64, 64), dtype=torch.long) # H, W
    # 确保 colormap 长度至少为 num_test_classes
    test_colormap_vis = [[0,0,0],[128,0,0],[0,128,0],[0,0,128]] 
    
    visualize_segmentation(
        dummy_img_tensor_vis, 
        dummy_pred_mask_vis, 
        dummy_true_mask_vis, 
        num_classes=num_test_classes_metric, 
        colormap=test_colormap_vis,
        figsize=(12,4)
    )
    print("工具函数测试完成。请检查弹出的可视化窗口。")