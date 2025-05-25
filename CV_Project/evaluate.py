# ==============================================================================
# 模块导入 (IMPORTS)
# ==============================================================================
import torch
from tqdm import tqdm
import numpy as np
from utils import calculate_iou_batch, calculate_pixel_accuracy_batch # 确保 utils.py 可导入

# ==============================================================================
# 模型评估函数 (MODEL EVALUATION FUNCTION)
# ==============================================================================
def evaluate_model(model: torch.nn.Module, 
                   dataloader: torch.utils.data.DataLoader, 
                   criterion: torch.nn.Module, 
                   device: torch.device, 
                   num_classes: int, 
                   ignore_index: int = 255):
    """
    在给定的数据集上评估语义分割模型。

    参数:
        model: 要评估的模型。
        dataloader: 用于评估的数据加载器 (通常是验证集或测试集)。
        criterion: 损失函数。
        device: 计算设备 ('cpu' 或 'cuda')。
        num_classes: 类别总数。
        ignore_index: 计算指标时忽略的类别索引。

    返回:
        avg_loss (float): 平均损失。
        avg_pixel_accuracy (float): 平均像素准确率。
        avg_miou (float): 平均 mIoU。
        iou_all_classes (np.ndarray): 所有评估样本累积计算的每个类别的IoU。
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    
    # 用于累积所有预测和目标的列表
    all_preds_list = []
    all_targets_list = []

    with torch.no_grad():  # 在评估阶段不计算梯度
        progress_bar = tqdm(dataloader, desc="评估中 (Evaluating)", leave=False)
        for images, masks in progress_bar:
            images = images.to(device)
            # masks 原始是 (B, 1, H, W), float32 (但值是整数类别)
            # 需要转换为 (B, H, W), long 类型用于损失计算和指标计算
            targets = masks.squeeze(1).long().to(device) # (B, H, W)

            # --- 前向传播 ---
            outputs = model(images)  # (B, num_classes, H, W)
            
            # --- 计算损失 ---
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0) # 乘以 batch_size 得到这个批次的总损失

            # --- 获取预测类别 ---
            # outputs 的形状是 (B, num_classes, H, W)
            # torch.max(outputs, 1) 返回 (values, indices)
            # 我们需要 indices，其形状是 (B, H, W)
            _, predicted_masks = torch.max(outputs, 1) 
            
            all_preds_list.append(predicted_masks.cpu())
            all_targets_list.append(targets.cpu())

    # --- 计算平均损失 ---
    avg_loss = total_loss / len(dataloader.dataset)
    
    # --- 拼接所有批次的预测和目标 ---
    # (num_total_samples, H, W)
    all_preds_tensor = torch.cat(all_preds_list, dim=0) 
    all_targets_tensor = torch.cat(all_targets_list, dim=0)

    # --- 计算评估指标 ---
    iou_per_class, miou = calculate_iou_batch(all_preds_tensor, all_targets_tensor, num_classes, ignore_index)
    pixel_acc = calculate_pixel_accuracy_batch(all_preds_tensor, all_targets_tensor, ignore_index)
        
    return avg_loss, pixel_acc, miou, iou_per_class

# ==============================================================================
# 主执行块 (测试用) (MAIN EXECUTION BLOCK - FOR TESTING)
# ==============================================================================
if __name__ == '__main__':
    print("evaluate.py 包含模型评估逻辑。")
    print("通常在 main.py 中调用 evaluate_model。")
    # 可以在此添加更独立的测试，例如：
    # 1. 创建一个虚拟模型 (可从 model.py 导入 SimpleSegmentationNet)
    # 2. 创建虚拟数据和 DataLoader (或从 dataset.py 导入 get_voc_dataloaders 并使用小数据集)
    # 3. 创建虚拟损失函数
    # 4. 调用 evaluate_model 并打印结果
    # 示例 (伪代码):
    # from model import SimpleSegmentationNet
    # from dataset import get_voc_dataloaders # (确保可以下载或有数据)
    #
    # DEVICE_TEST = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NUM_CLASSES_TEST_EVAL = 21 
    # IGNORE_INDEX_TEST_EVAL = 255
    #
    # # 创建一个简单的模型实例
    # dummy_model_eval = SimpleSegmentationNet(num_classes=NUM_CLASSES_TEST_EVAL).to(DEVICE_TEST)
    # # 假设模型参数已随机初始化或加载
    #
    # # 创建损失函数
    # dummy_criterion_eval = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_TEST_EVAL)
    #
    # # 获取数据加载器 (使用非常小的配置进行测试)
    # # _, val_loader_eval, _ = get_voc_dataloaders(
    # #     data_root='./data_voc_eval_test', batch_size=1, image_size=64, num_workers=0
    # # )
    # # if val_loader_eval and len(val_loader_eval) > 0:
    # #     print("\n开始模拟评估 (需要数据)...")
    # #     avg_loss, pa, miou, iou_cls = evaluate_model(
    # #         dummy_model_eval, val_loader_eval, dummy_criterion_eval, DEVICE_TEST, NUM_CLASSES_TEST_EVAL, IGNORE_INDEX_TEST_EVAL
    # #     )
    # #     print(f"模拟评估完成: Loss={avg_loss:.4f}, PA={pa:.4f}, mIoU={miou:.4f}")
    # #     print(f"各类别IoU: {iou_cls}")
    # # else:
    # #     print("无法加载数据进行模拟评估，跳过。")
    pass