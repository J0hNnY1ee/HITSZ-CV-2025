# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time # 用于计时

# 从本地模块导入 (假设 dataset.py 在同一目录或PYTHONPATH中)
try:
    from dataset import NUM_CLASSES, VOC_COLORMAP, tensor_to_pil, mask_to_pil_color
except ImportError:
    print("Warning: Could not import from dataset.py. Using placeholder values.")
    NUM_CLASSES = 21 # Placeholder
    VOC_COLORMAP = [[0,0,0]]*21 # Placeholder
    def tensor_to_pil(x, **kwargs): return x
    def mask_to_pil_color(x, **kwargs): return x


# --------------------------------------------------------------------------------
# 1. 评估指标计算 (Pixel Accuracy, Mean Accuracy, Mean IoU, Frequency Weighted IoU)
# --------------------------------------------------------------------------------

def _fast_hist(label_true, label_pred, n_class):
    """
    计算混淆矩阵。
    label_true: 真实标签, 形状为 [N] (N = H * W * B)
    label_pred: 预测标签, 形状为 [N]
    n_class: 类别数量
    """
    mask = (label_true >= 0) & (label_true < n_class) # 确保标签在有效范围内
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def calculate_metrics(hist):
    """
    根据混淆矩阵计算各种评估指标。
    hist: 混淆矩阵, 形状为 [n_class, n_class]
    """
    # Pixel Accuracy (PA)
    pa = np.diag(hist).sum() / hist.sum()
    
    # Mean Pixel Accuracy (MPA) / Class Accuracy
    # 计算每个类别的准确率，然后取平均
    class_accuracy = np.diag(hist) / (hist.sum(axis=1) + 1e-8) # 每行的和是该类别的真实像素数
    mpa = np.nanmean(class_accuracy) # nanmean 忽略 NaN 值 (当某个类别在真实标签中不存在时)

    # Intersection over Union (IoU) / Jaccard Index
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-8)
    # hist.sum(axis=1): 按行求和，真实为i的像素数
    # hist.sum(axis=0): 按列求和，预测为i的像素数
    # np.diag(hist): 对角线元素，真实为i且预测为i的像素数 (Intersection)
    # Union = True_i + Pred_i - Intersection_i

    # Mean IoU (MIoU)
    miou = np.nanmean(iou)
    
    # Frequency Weighted IoU (FWIoU)
    # freq = hist.sum(axis=1) / hist.sum() # 每个类别的频率
    # fwiou = (freq[freq > 0] * iou[freq > 0]).sum() # 只考虑真实标签中存在的类别

    return {
        "PixelAcc": pa,
        "MeanAcc": mpa,
        "MeanIoU": miou,
        "ClassIoU": iou # 返回每个类别的IoU，方便后续分析
    }


# --------------------------------------------------------------------------------
# 2. 训练器类
# --------------------------------------------------------------------------------

class SemanticSegmenterTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cpu', num_classes=NUM_CLASSES):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion # 损失函数
        self.optimizer = optimizer # 优化器
        self.scheduler = scheduler # 学习率调度器 (可选)
        self.device = device
        self.num_classes = num_classes

        self.best_val_miou = -1.0 # 用于保存最佳模型的指标
        self.best_model_state = None

        # 存储训练过程中的指标
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = [] # 存储每个epoch的验证集指标字典


    def train_epoch(self):
        self.model.train() # 设置模型为训练模式
        total_loss = 0.0
        
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device) # masks 应该是 [B, H, W] 的 LongTensor
            
            self.optimizer.zero_grad() # 清空梯度
            outputs = self.model(images) # 前向传播, outputs: [B, num_classes, H, W]
            
            # 确保 masks 的类型是 LongTensor，并且没有类别通道维度
            # criterion (e.g., CrossEntropyLoss) expects target of shape [B, H, W] with class indices
            loss = self.criterion(outputs, masks.long()) 
            
            loss.backward() # 反向传播
            self.optimizer.step() # 更新参数
            
            total_loss += loss.item() * images.size(0) # 乘以 batch_size
            progress_bar.set_postfix(loss=loss.item()) # 更新进度条上的损失显示

        epoch_loss = total_loss / len(self.train_loader.dataset)
        self.train_losses.append(epoch_loss)
        return epoch_loss

    def validate_epoch(self):
        self.model.eval() # 设置模型为评估模式
        total_loss = 0.0
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad(): # 在评估模式下不计算梯度
            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device) # [B, H, W]
                
                outputs = self.model(images) # [B, num_classes, H, W]
                loss = self.criterion(outputs, masks.long())
                total_loss += loss.item() * images.size(0)
                
                # 计算预测的类别
                # outputs 是 [B, C, H, W]，preds 是 [B, H, W]
                preds = torch.argmax(outputs, dim=1) # 在类别维度上取最大值的索引
                
                # 更新混淆矩阵
                # 将 masks 和 preds展平，然后计算
                masks_np = masks.cpu().numpy().flatten()
                preds_np = preds.cpu().numpy().flatten()
                
                # 确保标签值在 [0, num_classes-1] 范围内
                # VOCDataset 中已经将 255 映射为 0，所以这里应该是安全的
                # 但可以加一个检查或clip
                masks_np = np.clip(masks_np, 0, self.num_classes - 1)
                preds_np = np.clip(preds_np, 0, self.num_classes - 1)

                hist_batch = _fast_hist(masks_np, preds_np, self.num_classes)
                confusion_matrix += hist_batch
                
                progress_bar.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(self.val_loader.dataset)
        self.val_losses.append(epoch_loss)
        
        metrics = calculate_metrics(confusion_matrix)
        self.val_metrics_history.append(metrics)
        
        return epoch_loss, metrics, confusion_matrix

    def train(self, num_epochs, model_save_path_best='best_model.pth'):
        print(f"Starting training for {num_epochs} epochs on {self.device}...")
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss, val_metrics, _ = self.validate_epoch() # 忽略混淆矩阵，只取指标
            
            epoch_duration = time.time() - epoch_start_time

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MIoU: {val_metrics['MeanIoU']:.4f} | "
                  f"Val PixelAcc: {val_metrics['PixelAcc']:.4f} | "
                  f"Time: {epoch_duration:.2f}s")

            # 如果使用了学习率调度器
            if self.scheduler:
                # 有些scheduler是基于epoch的，有些是基于metric的
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['MeanIoU']) # 或者用 val_loss
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr:.6f}")

            # 保存最佳模型 (基于验证集MIoU)
            if val_metrics['MeanIoU'] > self.best_val_miou:
                self.best_val_miou = val_metrics['MeanIoU']
                self.best_model_state = self.model.state_dict() # 保存模型参数
                torch.save(self.best_model_state, model_save_path_best)
                print(f"Epoch {epoch+1}: New best model saved with MIoU: {self.best_val_miou:.4f} to {model_save_path_best}")

        total_training_time = time.time() - start_time
        print(f"Training finished. Total time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
        print(f"Best Validation MIoU: {self.best_val_miou:.4f}")
        
        # 可以在训练结束后加载最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model weights from {model_save_path_best} for final evaluation.")


    def evaluate(self, data_loader, checkpoint_path=None):
        """
        在给定的 data_loader (例如测试集) 上评估模型。
        如果提供了 checkpoint_path，则加载该检查点的权重。
        否则，使用当前模型的权重（通常是训练结束后的最佳模型）。
        """
        if checkpoint_path:
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded model weights from {checkpoint_path} for evaluation.")
            except FileNotFoundError:
                print(f"Error: Checkpoint file not found at {checkpoint_path}. Using current model weights.")
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint_path}: {e}. Using current model weights.")
        
        self.model.eval() # 设置为评估模式
        total_loss = 0.0 # 如果损失函数也需要的话
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        # 存储一些样本用于可视化 (可选)
        sample_images_for_vis = []
        sample_masks_for_vis = []
        sample_preds_for_vis = []
        num_samples_to_vis = 5 # 可视化多少个样本

        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for i, (images, masks) in enumerate(progress_bar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                # 如果需要计算损失 (例如，在测试集上报告最终损失)
                if self.criterion:
                    loss = self.criterion(outputs, masks.long())
                    total_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                
                masks_np = masks.cpu().numpy().flatten()
                preds_np = preds.cpu().numpy().flatten()
                masks_np = np.clip(masks_np, 0, self.num_classes - 1)
                preds_np = np.clip(preds_np, 0, self.num_classes - 1)
                
                hist_batch = _fast_hist(masks_np, preds_np, self.num_classes)
                confusion_matrix += hist_batch

                # 保存一些样本用于可视化
                if i < num_samples_to_vis and len(sample_images_for_vis) < num_samples_to_vis :
                    sample_images_for_vis.append(images[0].cpu()) # 取batch中的第一个
                    sample_masks_for_vis.append(masks[0].cpu())
                    sample_preds_for_vis.append(preds[0].cpu())
        
        eval_loss = total_loss / len(data_loader.dataset) if self.criterion and len(data_loader.dataset) > 0 else None
        metrics = calculate_metrics(confusion_matrix)
        
        print("\n--- Evaluation Results ---")
        if eval_loss is not None:
            print(f"Evaluation Loss: {eval_loss:.4f}")
        print(f"Pixel Accuracy (PA): {metrics['PixelAcc']:.4f}")
        print(f"Mean Pixel Accuracy (MPA): {metrics['MeanAcc']:.4f}")
        print(f"Mean Intersection over Union (MIoU): {metrics['MeanIoU']:.4f}")
        print("Class-wise IoU:")
        try:
            from dataset import VOC_CLASSES # 尝试导入类别名称
            for i, iou_val in enumerate(metrics['ClassIoU']):
                if i < len(VOC_CLASSES):
                    print(f"  {VOC_CLASSES[i]:<15}: {iou_val:.4f}")
                else:
                    print(f"  Class {i}: {iou_val:.4f}")
        except ImportError:
             for i, iou_val in enumerate(metrics['ClassIoU']):
                print(f"  Class {i}: {iou_val:.4f}")
        print("--------------------------")

        return metrics, confusion_matrix, (sample_images_for_vis, sample_masks_for_vis, sample_preds_for_vis)

# --------------------------------------------------------------------------------
# 3. (可选) 主程序块，用于测试 Trainer (需要配合模型和数据加载器)
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    print("Testing trainer.py (This is a basic test, full test requires model and data)")

    # 模拟一些数据
    class DummyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.fc = nn.Linear(10, num_classes * 16 * 16) # 模拟输出特征图
            self.num_classes = num_classes
        def forward(self, x): # x: [B, C, H, W]
            b, _, h, w = x.shape
            # 简化：假设输入是10维特征，输出是展平的分割图
            # 实际上模型输出是 [B, num_classes, H, W]
            # 为了测试，我们假设模型输出正确形状
            # return torch.randn(b, self.num_classes, h, w)
            # 更简单的模拟：
            return torch.rand(x.size(0), self.num_classes, x.size(2), x.size(3))


    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, num_classes, height, width):
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.height = height
            self.width = width
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            img = torch.randn(3, self.height, self.width)
            mask = torch.randint(0, self.num_classes, (self.height, self.width), dtype=torch.long)
            return img, mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES_TEST = 5
    IMG_H_TEST, IMG_W_TEST = 32, 32 # 小尺寸测试
    
    model = DummyModel(NUM_CLASSES_TEST).to(device)
    train_ds = DummyDataset(20, NUM_CLASSES_TEST, IMG_H_TEST, IMG_W_TEST)
    val_ds = DummyDataset(10, NUM_CLASSES_TEST, IMG_H_TEST, IMG_W_TEST)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    trainer = SemanticSegmenterTrainer(model, train_loader, val_loader, criterion, optimizer, device=device, num_classes=NUM_CLASSES_TEST)
    
    print("\n--- Starting Dummy Training Test ---")
    trainer.train(num_epochs=2, model_save_path_best='dummy_best_model.pth')
    print("\n--- Dummy Training Test Finished ---")

    print("\n--- Starting Dummy Evaluation Test ---")
    # 假设 'dummy_best_model.pth' 被保存了
    metrics, _, viz_samples = trainer.evaluate(val_loader, checkpoint_path='dummy_best_model.pth')
    print("Dummy Evaluation MIoU:", metrics.get("MeanIoU", "N/A"))
    print(f"Number of visualization samples: {len(viz_samples[0])}")
    print("--- Dummy Evaluation Test Finished ---")