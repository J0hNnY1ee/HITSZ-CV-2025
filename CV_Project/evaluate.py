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
    # 初始化混淆矩阵 (num_classes x num_classes)
    # 这里的 num_classes 应该是模型输出 logits 的类别维度，
    # 或者数据集中定义的最大有效类别索引 + 1
    # 例如，如果类别是0-10，num_classes=11。如果包含类别11(unlabelled)，则为12。
    # 我们假设 num_classes 对应于我们关心的、用于构建混淆矩阵的类别范围 [0, num_classes-1]
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

            # 过滤掉 ignore_index (如果指定了)
            # _fast_hist 内部的 mask 逻辑 (label_true < num_classes) 会处理这个问题
            # 如果 targets 中有 ignore_index，并且 ignore_index >= num_classes，则会被mask掉
            # 如果 ignore_index < num_classes, 它会被当做一个普通类别计入。
            # 所以，如果 ignore_index 是一个我们不想评估的特定类别（例如CamVid的11 'Unlabelled'），
            # 并且这个 ignore_index 在 [0, num_classes-1] 范围内，
            # 我们需要在调用 _fast_hist 之前，或者在 calculate_metrics 之后处理。

            # 一个常见的做法是，num_classes 对应于所有可能的标签值（包括要忽略的）。
            # 然后在 calculate_metrics 时，从 IoU 和 accuracy 的平均中排除 ignore_index 对应的行/列。
            # 或者，在 _fast_hist 之前，将 targets_np 和 preds_np 中 ignore_index 的值替换掉
            # （例如替换为一个不会影响其他类别的值，或者直接 mask 掉这些像素）。

            # 当前 _fast_hist 的实现：
            # mask = (label_true >= 0) & (label_true < num_classes)
            # 这意味着如果 targets_np 中的值 >= num_classes (例如 ignore_index = 11, num_classes = 11),
            # 这些值会被排除。这是我们期望的。
            # 如果 ignore_index = 0 且 num_classes = 11, 0 会被视为一个正常类别。

            # 我们假设 num_classes 是用于评估的类别数量 (例如 0 到 N-1)
            # 并且真实标签中任何等于 ignore_index 的值，如果 ignore_index 不在 [0, num_classes-1] 范围内，
            # 或者如果 ignore_index >= num_classes, 它们会在 _fast_hist 中被过滤掉。
            # 如果 ignore_index < num_classes，它会被当成一个普通类。
            # CamVidDataset 通常将 ignore_index 设为11 (Unlabelled)，num_train_classes 设为11。
            # 如果模型输出11个类别 (0-10)，则这里的 num_classes 应该是11。
            # 此时，真实标签中的11会被 _fast_hist 的 mask 掉，不参与计算。
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


# ==============================================================================
# 测试代码 (可选)
# ==============================================================================
if __name__ == '__main__':
    print("开始 evaluate.py 测试...")

    # --- 测试 _fast_hist 和 calculate_metrics ---
    num_test_classes = 3 # 假设3个类别 0, 1, 2
    # 真实标签 (扁平化)
    # 类别0: 3个, 类别1: 2个, 类别2: 1个
    true_labels = np.array([0, 0, 0, 1, 1, 2, 0, 1]) # 增加一些例子
    # 预测标签 (扁平化)
    pred_labels = np.array([0, 0, 1, 1, 2, 2, 0, 0]) # 对应上面的例子

    # 期望的混淆矩阵:
    #       Pred: 0  1  2
    # True 0:     2  1  0   (sum=3)  (GT: 0,0,0,0 -> Pred: 0,0,1,0)  Real: 0: idx 0,1,2,6 -> Preds: 0,0,1,0
    # True 1:     1  1  0   (sum=2)  (GT: 1,1,1 -> Pred: 1,2,0)    Real: 1: idx 3,4,7 -> Preds: 1,2,0
    # True 2:     0  0  1   (sum=1)  (GT: 2 -> Pred: 2)             Real: 2: idx 5 -> Pred: 2

    # 根据上面的例子:
    # T=0, P=0: (0,0), (1,0), (6,0) => 3
    # T=0, P=1: (2,1) => 1
    # T=0, P=2: 0
    # T=1, P=0: (7,0) => 1
    # T=1, P=1: (3,1) => 1
    # T=1, P=2: (4,2) => 1
    # T=2, P=0: 0
    # T=2, P=1: 0
    # T=2, P=2: (5,2) => 1
    # Expected hist:
    # [[3, 1, 0],
    #  [1, 1, 1],
    #  [0, 0, 1]]

    print("\n测试混淆矩阵和指标计算...")
    hist = _fast_hist(true_labels, pred_labels, num_test_classes)
    print("计算得到的混淆矩阵:\n", hist)

    expected_hist = np.array([[3, 1, 0], [1, 1, 1], [0, 0, 1]])
    assert np.array_equal(hist, expected_hist), f"混淆矩阵计算错误! 期望:\n{expected_hist}\n得到:\n{hist}"
    print("混淆矩阵计算正确。")

    metrics = calculate_metrics(hist)
    print("\n计算得到的指标:")
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {np.round(v, 4)}")
        else:
            print(f"  {k}: {v:.4f}")

    # 手动验证一些指标
    # PA = (3+1+1) / 8 = 5/8 = 0.625
    assert abs(metrics['pixel_accuracy'] - 0.625) < 1e-6, "PA 计算错误"
    # Acc per class:
    # C0: 3 / (3+1+0) = 3/4 = 0.75
    # C1: 1 / (1+1+1) = 1/3 = 0.3333
    # C2: 1 / (0+0+1) = 1/1 = 1.0
    expected_acc_per_class = np.array([0.75, 1/3, 1.0])
    assert np.allclose(metrics['accuracy_per_class'], expected_acc_per_class), "Accuracy per class 计算错误"
    # MPA = (0.75 + 0.3333 + 1.0) / 3 = 2.0833 / 3 = 0.6944
    assert abs(metrics['mean_pixel_accuracy'] - np.mean(expected_acc_per_class)) < 1e-6, "MPA 计算错误"
    # IoU per class:
    # C0: TP=3, FP=(1+0)=1, FN=(1+0)=1.  IoU0 = 3 / (3+1+1) = 3/5 = 0.6
    # C1: TP=1, FP=(1+0)=1, FN=(1+1)=2.  IoU1 = 1 / (1+1+2) = 1/4 = 0.25
    # C2: TP=1, FP=(0+1)=1, FN=(0+0)=0.  IoU2 = 1 / (1+1+0) = 1/2 = 0.5
    expected_iou_per_class = np.array([0.6, 0.25, 0.5])
    assert np.allclose(metrics['iou_per_class'], expected_iou_per_class), "IoU per class 计算错误"
    # mIoU = (0.6 + 0.25 + 0.5) / 3 = 1.35 / 3 = 0.45
    assert abs(metrics['mean_iou'] - np.mean(expected_iou_per_class)) < 1e-6, "mIoU 计算错误"
    print("指标计算验证通过。")

    # --- 测试 evaluate_segmentation (需要模拟模型和数据加载器) ---
    print("\n(模拟) 测试 evaluate_segmentation 函数...")
    from model import SimplePixelClassifier # 从 model.py 导入
    from torch.utils.data import TensorDataset, DataLoader

    # 模拟参数
    eval_batch_size = 2
    eval_img_channels = 1 # 简化，用单通道
    eval_img_h, eval_img_w = 4, 4
    eval_num_classes = num_test_classes # 3
    eval_device = torch.device("cpu")

    # 创建模拟模型
    eval_model = SimplePixelClassifier(in_channels=eval_img_channels, num_classes=eval_num_classes).to(eval_device)

    # 创建模拟数据 (一批)
    # 图像 (B, C, H, W)
    dummy_images_eval = torch.randn(eval_batch_size, eval_img_channels, eval_img_h, eval_img_w)
    # 真实标签 (B, H, W) - 使用上面测试过的 true_labels 的一部分
    # 我们需要 (B, H, W) 格式，所以reshape
    # true_labels: [0, 0, 0, 1, 1, 2, 0, 0] (len 8)
    # pred_labels: [0, 0, 1, 1, 2, 2, 0, 0]
    # 假设 HxW = 4, B=2.  Total pixels = 2*4 = 8.
    # 调整 H,W 以匹配
    eval_img_h, eval_img_w = 2, 2 # 2x2=4 pixels per image
    dummy_images_eval = torch.randn(eval_batch_size, eval_img_channels, eval_img_h, eval_img_w)

    # 真实标签 (B,H,W)
    # Sample 1: [[0,0],[0,1]] -> TPs for C0=3, TP for C1=1
    # Sample 2: [[1,2],[0,0]] -> TPs for C1=1, TP for C2=1, TPs for C0=2
    # Total true: C0: 5, C1: 2, C2: 1. Total pixels = 8.
    #
    # 我们用之前测试过的扁平化标签来构造
    # true_labels_flat = np.array([0,0,0,1,1,2,0,0]) # 8 pixels
    # pred_labels_flat (from model) -> we will use the same "prediction logic"
    # For simplicity, let's make the model always predict `(true_label + 0) % num_classes`
    # to have a predictable output for testing metrics.
    # This part is tricky to test perfectly without a real model pass.

    # 让我们使用一个已知的混淆矩阵来伪造模型输出
    # 我们将使模型输出与 pred_labels_flat 对应
    # (B, C, H, W)
    mock_outputs = torch.zeros(eval_batch_size, eval_num_classes, eval_img_h, eval_img_w)
    mock_targets = torch.zeros(eval_batch_size, eval_img_h, eval_img_w, dtype=torch.long)

    # true_labels = np.array([0, 0, 0, 1, 1, 2, 0, 0])
    # pred_labels = np.array([0, 0, 1, 1, 2, 2, 0, 0])
    # Reshape them into (B, H, W)
    true_labels_reshaped = torch.from_numpy(true_labels).long().reshape(eval_batch_size, eval_img_h, eval_img_w)
    pred_labels_reshaped = torch.from_numpy(pred_labels).long().reshape(eval_batch_size, eval_img_h, eval_img_w)

    mock_targets = true_labels_reshaped
    # one-hot encode pred_labels to create mock_outputs
    for b in range(eval_batch_size):
        for r in range(eval_img_h):
            for c in range(eval_img_w):
                pred_class = pred_labels_reshaped[b, r, c].item()
                mock_outputs[b, pred_class, r, c] = 1.0 # Assign high logit to predicted class

    # 模拟数据集和数据加载器
    mock_dataset = TensorDataset(dummy_images_eval, mock_targets) # images don't matter for this mock
    mock_dataloader = DataLoader(mock_dataset, batch_size=eval_batch_size)

    # 模拟损失函数
    mock_criterion = torch.nn.CrossEntropyLoss()

    # 覆盖模型的 forward 方法，使其返回我们伪造的输出
    def mock_forward(self, x): # x is dummy_images_eval
        # 在这个测试中，我们不关心输入x，直接返回预设的 mock_outputs
        # 找到与输入x对应的批次的 mock_outputs
        # 假设dataloader只产生一个批次，这个批次就是我们构造的
        return mock_outputs.to(x.device)

    original_forward = eval_model.forward
    eval_model.forward = mock_forward.__get__(eval_model, SimplePixelClassifier) # 绑定方法

    print("运行模拟评估...")
    avg_loss, eval_metrics = evaluate_segmentation(
        eval_model, mock_dataloader, mock_criterion, eval_device, eval_num_classes
    )
    eval_model.forward = original_forward # 恢复原始 forward 方法

    print(f"  模拟评估平均损失: {avg_loss:.4f}")
    print("  模拟评估指标:")
    for k, v in eval_metrics.items():
        if isinstance(v, np.ndarray):
            print(f"    {k}: {np.round(v, 4)}")
        else:
            print(f"    {k}: {v:.4f}")

    # 检查指标是否与之前手动计算的一致
    assert abs(eval_metrics['pixel_accuracy'] - 0.625) < 1e-6, "模拟评估 PA 不匹配"
    assert abs(eval_metrics['mean_iou'] - 0.45) < 1e-6, "模拟评估 mIoU 不匹配"
    print("evaluate_segmentation 函数的基本模拟测试通过。")

    # 测试 ignore_index 功能
    print("\n测试 evaluate_segmentation 带 ignore_index 功能...")
    # 假设类别2是忽略类别
    ignore_this_idx = 2
    # 此时混淆矩阵维度仍是 num_test_classes=3
    # 我们期望 'mean_iou_adjusted' 和 'mean_pixel_accuracy_adjusted'
    # 只基于类别 0 和 1 计算
    avg_loss_ign, eval_metrics_ign = evaluate_segmentation(
        eval_model, mock_dataloader, mock_criterion, eval_device, eval_num_classes,
        ignore_index=ignore_this_idx
    )
    eval_model.forward = original_forward # 恢复

    print(f"  模拟评估 (忽略索引 {ignore_this_idx}) 平均损失: {avg_loss_ign:.4f}")
    print("  模拟评估指标 (忽略索引):")
    for k, v in eval_metrics_ign.items():
        if isinstance(v, np.ndarray):
            print(f"    {k}: {np.round(v, 4)}")
        else:
            print(f"    {k}: {v:.4f}")

    # mIoU (adj) = (IoU_C0 + IoU_C1) / 2 = (0.6 + 0.25) / 2 = 0.85 / 2 = 0.425
    expected_miou_adj = (expected_iou_per_class[0] + expected_iou_per_class[1]) / 2
    assert abs(eval_metrics_ign['mean_iou_adjusted'] - expected_miou_adj) < 1e-6, \
        f"模拟评估 mIoU (adjusted) 不匹配. Exp: {expected_miou_adj}, Got: {eval_metrics_ign['mean_iou_adjusted']}"

    # MPA (adj) = (Acc_C0 + Acc_C1) / 2 = (0.75 + 0.3333) / 2 = 1.0833 / 2 = 0.54165
    expected_mpa_adj = (expected_acc_per_class[0] + expected_acc_per_class[1]) / 2
    assert abs(eval_metrics_ign['mean_pixel_accuracy_adjusted'] - expected_mpa_adj) < 1e-6, \
        f"模拟评估 MPA (adjusted) 不匹配. Exp: {expected_mpa_adj}, Got: {eval_metrics_ign['mean_pixel_accuracy_adjusted']}"
    print("evaluate_segmentation 带 ignore_index 功能测试通过。")


    print("\nevaluate.py 测试结束。")