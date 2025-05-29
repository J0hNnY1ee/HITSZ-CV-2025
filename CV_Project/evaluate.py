# evaluate.py

import torch
import numpy as np

# ==============================================================================
# 语义分割评估指标计算
# ==============================================================================

def _fast_hist(label_true: np.ndarray, label_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    快速计算混淆矩阵。

    参数:
        label_true (np.ndarray): 真实的标签掩码 (扁平化的一维数组)。
        label_pred (np.ndarray): 预测的标签掩码 (扁平化的一维数组)。
        num_classes (int): 类别总数。

    返回:
        np.ndarray: 混淆矩阵，形状为 (num_classes, num_classes)。
                    行代表真实类别，列代表预测类别。
    """
    mask = (label_true >= 0) & (label_true < num_classes) # 忽略超出范围的标签
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def calculate_metrics(hist: np.ndarray) -> dict:
    """
    根据混淆矩阵计算各项评估指标。

    参数:
        hist (np.ndarray): 混淆矩阵 (num_classes, num_classes)。

    返回:
        dict: 包含以下指标的字典:
            'pixel_accuracy' (float): 像素准确率 (PA)
            'mean_pixel_accuracy' (float): 平均像素准确率 (MPA)
            'mean_iou' (float): 平均交并比 (mIoU)
            'iou_per_class' (np.ndarray): 每个类别的IoU
            'accuracy_per_class' (np.ndarray): 每个类别的准确率
    """
    # 像素准确率 (Pixel Accuracy, PA)
    # 正确分类的像素数 / 总像素数
    pixel_accuracy = np.diag(hist).sum() / hist.sum() if hist.sum() > 0 else 0.0

    # 每个类别的准确率 (Accuracy per class)
    # 对于类别i: hist[i,i] / sum(hist[i,:]) (真实为i的像素中，被正确预测为i的比例)
    accuracy_per_class = np.diag(hist) / hist.sum(axis=1)
    accuracy_per_class[np.isnan(accuracy_per_class)] = 0 # 处理分母为0的情况 (某类别在真值中未出现)
    mean_pixel_accuracy = np.nanmean(accuracy_per_class) # MPA: 对每个类别的准确率取平均

    # 交并比 (Intersection over Union, IoU)
    # 对于类别i: hist[i,i] / (sum(hist[i,:]) + sum(hist[:,i]) - hist[i,i])
    # (真实为i且预测为i) / (真实为i 或 预测为i)
    iou_per_class = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    iou_per_class[np.isnan(iou_per_class)] = 0 # 处理分母为0的情况
    mean_iou = np.nanmean(iou_per_class) # mIoU: 对每个类别的IoU取平均 (通常只对存在的类别计算)

    # 如果只想对在真值中实际出现的类别计算mIoU和MPA (更公平的做法)
    # present_classes_mask = hist.sum(axis=1) > 0 # 哪些类别在真值中出现过
    # mean_pixel_accuracy_present = np.nanmean(accuracy_per_class[present_classes_mask]) if np.any(present_classes_mask) else 0.0
    # mean_iou_present = np.nanmean(iou_per_class[present_classes_mask]) if np.any(present_classes_mask) else 0.0

    return {
        'pixel_accuracy': pixel_accuracy,
        'mean_pixel_accuracy': mean_pixel_accuracy, # 或者 mean_pixel_accuracy_present
        'mean_iou': mean_iou, # 或者 mean_iou_present
        'iou_per_class': iou_per_class,
        'accuracy_per_class': accuracy_per_class,
    }


def evaluate_segmentation(model: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          criterion: torch.nn.Module, # 损失函数
                          device: torch.device,
                          num_classes: int,
                          ignore_index: int = -1): # 通常损失函数会处理，这里用于评估时可能需要排除
    """
    评估语义分割模型在一个数据集上的性能。

    参数:
        model (torch.nn.Module): 要评估的模型。
        dataloader (torch.utils.data.DataLoader): 数据加载器 (通常是验证集或测试集)。
        criterion (torch.nn.Module): 损失函数，用于计算评估损失。
        device (torch.device): 'cuda' 或 'cpu'。
        num_classes (int): 分割任务的类别总数。
                           注意：这个num_classes是用于混淆矩阵的维度，
                           应与模型输出的类别维度一致，或为数据集中定义的最大类别索引+1。
        ignore_index (int): 在计算指标时要忽略的类别索引 (例如背景或未标注)。
                            如果为-1或None，则不特定忽略（除了混淆矩阵本身不包含的）。
                            真实标签中等于 ignore_index 的像素将不参与混淆矩阵计算。

    返回:
        tuple: (avg_loss, metrics_dict)
            avg_loss (float): 数据集上的平均损失。
            metrics_dict (dict): 包含各项评估指标的字典 (来自 calculate_metrics)。
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    print(f"开始评估，共有 {len(dataloader)} 个批次...")
    processed_batches = 0

    with torch.no_grad(): # 在评估时不计算梯度
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device) # targets 形状 (B, H, W), 值为类别索引

            outputs = model(images) # outputs 形状 (B, num_classes, H, W)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0) # 乘以batch size，因为loss通常是batch平均

            # --- 计算混淆矩阵 ---
            # outputs: (B, C, H, W), targets: (B, H, W)
            # 获取预测类别: 对类别维度取argmax
            preds = torch.argmax(outputs, dim=1) # preds 形状 (B, H, W)

            # 将Tensor转为NumPy数组，并扁平化
            targets_np = targets.cpu().numpy().flatten()
            preds_np = preds.cpu().numpy().flatten()

            hist_batch = _fast_hist(targets_np, preds_np, num_classes)
            confusion_matrix += hist_batch

            if (batch_idx + 1) % (len(dataloader) // 10 + 1) == 0 : # 每10%左右打印一次
                 print(f"  已评估 {batch_idx + 1}/{len(dataloader)} 批次...")
            processed_batches +=1

    if processed_batches == 0: # 防止 dataloader 为空
        return 0.0, {}

    avg_loss = total_loss / len(dataloader.dataset) # 平均到每个样本
    metrics = calculate_metrics(confusion_matrix)

    # (可选) 如果 ignore_index 是一个在 [0, num_classes-1] 范围内的特定类别，
    # 而我们又不想它参与 mIoU 和 MPA 的计算，可以在这里调整：
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        print(f"  评估时将忽略类别索引 {ignore_index} 的 IoU 和 Accuracy 来计算均值。")
        valid_classes_mask = np.ones(num_classes, dtype=bool)
        valid_classes_mask[ignore_index] = False

        iou_per_class_valid = metrics['iou_per_class'][valid_classes_mask]
        accuracy_per_class_valid = metrics['accuracy_per_class'][valid_classes_mask]

        metrics['mean_iou_adjusted'] = np.nanmean(iou_per_class_valid) if iou_per_class_valid.size > 0 else 0.0
        metrics['mean_pixel_accuracy_adjusted'] = np.nanmean(accuracy_per_class_valid) if accuracy_per_class_valid.size > 0 else 0.0
        # 通常我们会报告调整后的 mIoU 和 MPA
        print(f"  调整后 mIoU (忽略类别 {ignore_index}): {metrics['mean_iou_adjusted']:.4f}")
        print(f"  调整后 MPA (忽略类别 {ignore_index}): {metrics['mean_pixel_accuracy_adjusted']:.4f}")


    print("评估完成。")
    return avg_loss, metrics

