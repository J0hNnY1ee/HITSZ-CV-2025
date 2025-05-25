# trainer.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np # 用于指标格式化

from utils import save_checkpoint # load_checkpoint 通常在 main.py 中使用
from evaluate import evaluate_segmentation

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler = None,
                 device: torch.device = None,
                 num_epochs: int = 100,
                 num_classes_eval: int = -1,
                 ignore_index_eval: int = -1,
                 experiment_dir: str = "experiment_outputs", # 实验输出根目录
                 checkpoint_prefix: str = "model_ckpt",
                 start_epoch: int = 0,
                 best_metric_val: float = float('-inf'),
                 metric_to_optimize: str = 'mean_iou',
                 print_freq: int = 10,
                 save_summary_writer = None
                 ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

        self.num_epochs = num_epochs
        self.num_classes_eval = num_classes_eval
        self.ignore_index_eval = ignore_index_eval
        self.experiment_dir = experiment_dir # 所有输出都到这里
        self.checkpoint_prefix = checkpoint_prefix
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch
        self.best_metric_val = best_metric_val
        self.metric_to_optimize = metric_to_optimize
        self.print_freq = print_freq
        self.summary_writer = save_summary_writer

        # 训练日志文件路径
        self.training_log_file = os.path.join(self.experiment_dir, "training_log.txt")
        self._log_message(f"Trainer 初始化完成:", console_too=False) # 避免重复打印
        self._log_message(f"  设备: {self.device}", console_too=False)
        self._log_message(f"  总训练轮数: {self.num_epochs}", console_too=False)
        self._log_message(f"  将从 Epoch {self.start_epoch} 开始训练", console_too=False)
        self._log_message(f"  当前最佳 {self.metric_to_optimize}: {self.best_metric_val:.4f}", console_too=False)
        self._log_message(f"  评估时类别数: {self.num_classes_eval}", console_too=False)
        self._log_message(f"  评估时忽略索引: {self.ignore_index_eval}", console_too=False)
        self._log_message(f"  实验输出目录: {self.experiment_dir}", console_too=False)


    def _log_message(self, message: str, console_too: bool = True):
        """记录消息到日志文件和控制台。"""
        if console_too:
            print(message)
        try:
            with open(self.training_log_file, 'a') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        except Exception as e:
            print(f"写入日志文件失败: {e}")


    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [训练中]", leave=False)

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            if (batch_idx + 1) % self.print_freq == 0 or (batch_idx + 1) == len(self.train_loader):
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'AvgLoss': f"{total_loss / num_samples:.4f}",
                    'LR': f"{current_lr:.1e}"
                })
                if self.summary_writer:
                    step = self.current_epoch * len(self.train_loader) + batch_idx
                    self.summary_writer.add_scalar('Train/Batch_Loss', loss.item(), step)
                    self.summary_writer.add_scalar('Train/Learning_Rate', current_lr, step)

        avg_train_loss = total_loss / num_samples
        progress_bar.close()
        log_msg = f"Epoch {self.current_epoch + 1}/{self.num_epochs} [训练结束] - 平均训练损失: {avg_train_loss:.4f}"
        self._log_message(log_msg)

        if self.summary_writer:
            self.summary_writer.add_scalar('Train/Epoch_Loss', avg_train_loss, self.current_epoch + 1)
        return avg_train_loss


    def validate(self):
        self._log_message(f"Epoch {self.current_epoch + 1}/{self.num_epochs} [验证中]...")
        avg_val_loss, metrics = evaluate_segmentation(
            self.model, self.val_loader, self.criterion, self.device,
            num_classes=self.num_classes_eval, ignore_index=self.ignore_index_eval
        )

        self._log_message(f"Epoch {self.current_epoch + 1}/{self.num_epochs} [验证结束] - 平均验证损失: {avg_val_loss:.4f}")
        self._log_message(f"  验证指标:")
        for k, v in metrics.items():
            if isinstance(v, (float, np.float32, np.float64)): # np.float64 is for numpy scalars
                 self._log_message(f"    {k}: {v:.4f}", console_too=False) # 避免重复打印到控制台
            elif isinstance(v, np.ndarray) and k.endswith('_per_class'):
                self._log_message(f"    {k}: {np.round(v, 4).tolist()}", console_too=False)


        if self.summary_writer:
            self.summary_writer.add_scalar('Val/Epoch_Loss', avg_val_loss, self.current_epoch + 1)
            for k, v in metrics.items():
                if isinstance(v, (float, np.float32, np.float64)):
                    self.summary_writer.add_scalar(f'Val/{k}', v, self.current_epoch + 1)
        return avg_val_loss, metrics


    def run_training(self):
        self._log_message(f"\n{'='*20} 开始训练 {'='*20}", console_too=False) # main中已打印
        self._log_message(f"模型将训练 {self.num_epochs - self.start_epoch} 个轮次 (从 epoch {self.start_epoch+1} 到 {self.num_epochs})", console_too=False)

        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            _ = self.train_one_epoch()
            avg_val_loss, val_metrics = self.validate()

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    metric_for_scheduler = avg_val_loss if self.metric_to_optimize.lower().endswith('loss') \
                                           else val_metrics.get(self.metric_to_optimize, float('-inf'))
                    self.scheduler.step(metric_for_scheduler)
                else:
                    self.scheduler.step()

            current_metric_val = val_metrics.get(self.metric_to_optimize)
            if current_metric_val is None:
                self._log_message(f"警告: 无法在验证指标中找到 '{self.metric_to_optimize}'。将使用验证损失的负值进行比较。")
                current_metric_val = -avg_val_loss
                is_best = current_metric_val > self.best_metric_val
            else:
                is_best = current_metric_val > self.best_metric_val

            if is_best:
                self.best_metric_val = current_metric_val
                self._log_message(f"  🎉 新的最佳模型! {self.metric_to_optimize}: {self.best_metric_val:.4f} at epoch {self.current_epoch + 1}")

            checkpoint_state = {
                'epoch': self.current_epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_metric_val': self.best_metric_val,
                'metric_name': self.metric_to_optimize
            }
            if self.scheduler:
                checkpoint_state['scheduler_state_dict'] = self.scheduler.state_dict()

            # 保存检查点到实验目录
            save_checkpoint(checkpoint_state, is_best, self.experiment_dir, self.checkpoint_prefix)
            self._log_message(f"检查点已保存 (Epoch {self.current_epoch + 1}, IsBest: {is_best})", console_too=False)


            epoch_duration = time.time() - epoch_start_time
            self._log_message(f"Epoch {self.current_epoch + 1} 完成，用时: {epoch_duration:.2f} 秒")
            self._log_message(f"  当前学习率: {self.optimizer.param_groups[0]['lr']:.1e}")
            self._log_message(f"  当前最佳 {self.metric_to_optimize}: {self.best_metric_val:.4f}")
            self._log_message("-" * 50, console_too=False)


        self._log_message(f"\n{'='*20} 训练完成 {'='*20}")
        self._log_message(f"最佳 {self.metric_to_optimize} 在验证集上达到: {self.best_metric_val:.4f}")
        if self.summary_writer:
            self.summary_writer.close()