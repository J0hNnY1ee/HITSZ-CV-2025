# ==============================================================================
# 模块导入 (IMPORTS)
# ==============================================================================
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from evaluate import evaluate_model # 确保 evaluate.py 可导入

# ==============================================================================
# 训练器类 (TRAINER CLASS)
# ==============================================================================
class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, 
                 criterion: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler, # 允许 None
                 device: torch.device, 
                 num_classes: int, 
                 num_epochs: int, 
                 save_dir: str = 'checkpoints', 
                 print_every: int = 20, 
                 ignore_index: int = 255,
                 patience_lr_scheduler: int = 5): # 用于 ReduceLROnPlateau 的耐心值
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.print_every = print_every
        self.ignore_index = ignore_index
        self.patience_lr_scheduler = patience_lr_scheduler # 主要用于 ReduceLROnPlateau

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"创建检查点目录: {self.save_dir}")
            
        self.best_val_metric = 0.0 # 通常是 mIoU，越高越好
        self.history = {'train_loss': [], 'val_loss': [], 'val_pa': [], 'val_miou': [], 'lr': []}


    # --- 训练一个 Epoch (Train One Epoch) ---
    def _train_epoch(self, epoch_num):
        self.model.train()  # 设置模型为训练模式
        running_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, 
                            desc=f"Epoch {epoch_num+1}/{self.num_epochs} [训练中]", 
                            unit="batch", 
                            leave=False)

        for i, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = masks.squeeze(1).long().to(self.device) # (B, H, W)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            if (i + 1) % self.print_every == 0 or (i + 1) == len(self.train_loader):
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.1e}")
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss


    # --- 验证一个 Epoch (Validate One Epoch) ---
    def _validate_epoch(self, epoch_num):
        # 使用 evaluate_model 函数进行验证
        val_loss, val_pa, val_miou, _ = evaluate_model(
            self.model, 
            self.val_loader, 
            self.criterion, 
            self.device, 
            self.num_classes,
            self.ignore_index
        )
        return val_loss, val_pa, val_miou


    # --- 执行完整训练流程 (Execute Full Training Loop) ---
    def train(self):
        print(f"\n{'='*20} 开始训练 {'='*20}")
        print(f"总轮数: {self.num_epochs}, 设备: {self.device}")
        print(f"优化器: {self.optimizer.__class__.__name__}")
        if self.scheduler:
            print(f"学习率调度器: {self.scheduler.__class__.__name__}")
        print(f"{'='*50}\n")

        for epoch in range(self.num_epochs):
            # --- 训练 ---
            train_loss = self._train_epoch(epoch)
            
            # --- 验证 ---
            val_loss, val_pa, val_miou = self._validate_epoch(epoch)

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.num_epochs} 完成 | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val PA: {val_pa:.4f} | "
                  f"Val mIoU: {val_miou:.4f} | "
                  f"LR: {current_lr:.1e}")

            # --- 更新历史记录 ---
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_pa'].append(val_pa)
            self.history['val_miou'].append(val_miou)
            self.history['lr'].append(current_lr)

            # --- 学习率调度器步骤 ---
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_miou) # 通常监控验证集上的指标
                else:
                    self.scheduler.step() # 对于 StepLR, CosineAnnealingLR 等

            # --- 保存模型检查点 ---
            # 保存当前轮次模型 (可选，如果需要从特定轮次恢复)
            # current_epoch_save_path = os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pth')
            # torch.save(self.model.state_dict(), current_epoch_save_path)

            # 保存最佳模型 (基于验证集 mIoU)
            if val_miou > self.best_val_metric:
                self.best_val_metric = val_miou
                best_model_save_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_metric': self.best_val_metric, # 保存的是 mIoU
                    'history': self.history # 保存训练历史记录
                }, best_model_save_path)
                print(f"  -> 新的最佳模型 (mIoU: {self.best_val_metric:.4f}) 已保存至: {best_model_save_path}")
            
            print("-" * 70)

        print(f"\n{'='*20} 训练完成 {'='*20}")
        print(f"最佳验证 mIoU: {self.best_val_metric:.4f}")
        print(f"最终模型检查点保存在: {self.save_dir}")
        
        return self.history

# ==============================================================================
# 主执行块 (测试用) (MAIN EXECUTION BLOCK - FOR TESTING)
# ==============================================================================
if __name__ == '__main__':
    print("trainer.py 包含模型训练逻辑。")
    print("通常在 main.py 中实例化和调用 Trainer。")
    # 可以在此添加更独立的测试，例如：
    # from model import SimpleSegmentationNet
    # from dataset import get_voc_dataloaders
    #
    # DEVICE_TRAINER_TEST = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NUM_CLASSES_TRAINER_TEST = 21 
    # IGNORE_INDEX_TRAINER_TEST = 255
    #
    # # 1. 数据 (使用非常小的配置)
    # train_loader_t, val_loader_t, _ = get_voc_dataloaders(
    #     data_root='./data_voc_trainer_test', batch_size=1, image_size=64, num_workers=0
    # )
    #
    # # 2. 模型
    # model_t = SimpleSegmentationNet(num_classes=NUM_CLASSES_TRAINER_TEST).to(DEVICE_TRAINER_TEST)
    #
    # # 3. 损失函数和优化器
    # criterion_t = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX_TRAINER_TEST)
    # optimizer_t = optim.Adam(model_t.parameters(), lr=1e-4)
    # # scheduler_t = optim.lr_scheduler.StepLR(optimizer_t, step_size=1, gamma=0.1) # 测试用简单调度器
    # scheduler_t = None
    #
    # # 4. 训练器
    # if train_loader_t and len(train_loader_t) > 0 and val_loader_t and len(val_loader_t) > 0:
    #     trainer_instance_test = Trainer(
    #         model=model_t, train_loader=train_loader_t, val_loader=val_loader_t,
    #         criterion=criterion_t, optimizer=optimizer_t, scheduler=scheduler_t,
    #         device=DEVICE_TRAINER_TEST, num_classes=NUM_CLASSES_TRAINER_TEST, num_epochs=1, # 仅1轮测试
    #         save_dir='./checkpoints_trainer_test', print_every=1, ignore_index=IGNORE_INDEX_TRAINER_TEST
    #     )
    #     print("\n开始模拟训练 (1 epoch)...")
    #     history_test_run = trainer_instance_test.train()
    #     print("模拟训练完成。历史记录:")
    #     for key, val in history_test_run.items():
    #         print(f"  {key}: {val}")
    # else:
    #     print("无法加载数据进行模拟训练，跳过。请确保 ./data_voc_trainer_test 目录下有数据或允许下载。")
    pass