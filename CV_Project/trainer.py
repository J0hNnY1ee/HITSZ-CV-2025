# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from utils import pixel_accuracy as pa_utils
from utils import calculate_miou_and_iou_per_class

from model import UNet



class Trainer:
    def __init__(self, model, train_loader, val_loader, num_classes, device):
        # model 参数现在会是 UNet 的一个实例，从 main.ipynb 传入
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self._pixel_accuracy_func = pa_utils

    # ... (train_epoch, validate_epoch, fit 方法保持不变) ...
    def train_epoch(self):
        self.model.train() 
        running_loss = 0.0
        total_train_samples = 0 
        progress_bar = tqdm(self.train_loader, desc="训练中", leave=False)

        for images, masks in progress_bar:
            if images is None or masks is None: 
                continue
            
            total_train_samples += images.size(0) 

            images = images.to(self.device) 
            masks = masks.to(self.device)   

            self.optimizer.zero_grad() 
            outputs = self.model(images) 
            loss = self.criterion(outputs, masks) 
            loss.backward() 
            self.optimizer.step() 

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total_train_samples if total_train_samples > 0 else 0.0
        self.train_losses.append(epoch_loss)
        return epoch_loss

    def validate_epoch(self):
        self.model.eval() 
        running_loss = 0.0
        total_pixel_acc_numerator = 0.0 
        total_mask_elements = 0         
        total_val_samples_for_loss = 0  

        progress_bar = tqdm(self.val_loader, desc="验证中", leave=False)

        with torch.no_grad(): 
            for images, masks in progress_bar:
                if images is None or masks is None: 
                    continue

                total_val_samples_for_loss += images.size(0) 

                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images) 
                loss = self.criterion(outputs, masks)
                running_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1) 
                acc = self._pixel_accuracy_func(preds, masks) 
                
                current_mask_elements = masks.nelement()
                total_pixel_acc_numerator += acc * current_mask_elements 
                total_mask_elements += current_mask_elements

                progress_bar.set_postfix(loss=loss.item(), acc=f"{acc:.4f}")
        
        epoch_loss = running_loss / total_val_samples_for_loss if total_val_samples_for_loss > 0 else 0.0
        epoch_acc = total_pixel_acc_numerator / total_mask_elements if total_mask_elements > 0 else 0.0
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        return epoch_loss, epoch_acc

    def fit(self, num_epochs):
        print(f"开始在 {self.device} 上训练 {num_epochs} 个周期...")
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            if self.val_loader and (len(self.val_loader.dataset) > 0 if hasattr(self.val_loader, 'dataset') else True) :
                val_loss, val_acc = self.validate_epoch()
                print(f"周期 [{epoch+1}/{num_epochs}] | "
                      f"训练损失: {train_loss:.4f} | "
                      f"验证损失: {val_loss:.4f} | "
                      f"验证像素准确率: {val_acc:.4f}")
            else: 
                self.val_losses.append(float('nan')) 
                self.val_accuracies.append(float('nan'))
                print(f"周期 [{epoch+1}/{num_epochs}] | "
                      f"训练损失: {train_loss:.4f} | "
                      f"无验证数据")

        print("训练完成.")
        if config.MODEL_SAVE_PATH:
            try:
                torch.save(self.model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"最终模型已保存到 {config.MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"保存模型失败: {e}")
        return self.train_losses, self.val_losses, self.val_accuracies

# --- evaluate_model 函数 (保持不变，但它使用的 model 参数现在是 UNet 的实例) ---
def evaluate_model(
    model,
    dataloader,
    criterion,
    num_classes,
    class_names, 
    device,
    plot_segmentation_results_func=None, 
    num_vis_samples=3 
):
    # ... (evaluate_model 函数代码保持不变) ...
    model.eval() 

    total_loss = 0.0
    num_processed_samples_for_loss = 0
    all_preds_list = [] 
    all_trues_list = [] 
    
    vis_batch_images, vis_batch_true_masks = None, None
    if plot_segmentation_results_func and dataloader: 
        try:
            vis_iter = iter(dataloader) 
            temp_images, temp_masks = next(vis_iter)
            tries = 0
            max_tries_vis = 5 
            while (temp_images is None or temp_masks is None) and tries < max_tries_vis:
                temp_images, temp_masks = next(vis_iter)
                tries += 1
            
            if temp_images is not None and temp_masks is not None:
                vis_batch_images = temp_images 
                vis_batch_true_masks = temp_masks 
            else:
                print("评估：无法为可视化获取有效批次。")
        except StopIteration:
            print("评估：数据加载器为空或已耗尽，无法获取可视化批次。")
        except Exception as e:
            print(f"评估：获取可视化批次时出错: {e}")

    print("在评估数据上进行推理...")
    with torch.no_grad():
        for images, true_masks in tqdm(dataloader, desc="评估中"):
            if images is None or true_masks is None:
                continue

            num_processed_samples_for_loss += images.size(0)
            images_dev = images.to(device)
            true_masks_dev = true_masks.to(device)

            outputs = model(images_dev)
            loss = criterion(outputs, true_masks_dev)
            total_loss += loss.item() * images.size(0)

            pred_masks = torch.argmax(outputs, dim=1)
            all_preds_list.append(pred_masks.cpu())
            all_trues_list.append(true_masks_dev.cpu())

    if not all_preds_list:
        print("评估过程中没有收集到任何预测结果，无法计算指标。")
        return

    all_preds_tensor = torch.cat(all_preds_list, dim=0)
    all_trues_tensor = torch.cat(all_trues_list, dim=0)

    eval_pixel_acc = pa_utils(all_preds_tensor, all_trues_tensor) 
    eval_miou_score, eval_iou_per_class = calculate_miou_and_iou_per_class( 
        all_preds_tensor, 
        all_trues_tensor, 
        num_classes=num_classes
    )
    
    avg_loss = total_loss / num_processed_samples_for_loss if num_processed_samples_for_loss > 0 else 0.0

    print(f"\n--- 评估结果 ---")
    if num_processed_samples_for_loss > 0:
         print(f"  处理样本数: {num_processed_samples_for_loss}")
         print(f"  平均损失: {avg_loss:.4f}")
    else:
        print("  未处理任何样本。")

    print(f"  像素准确率 (Pixel Accuracy): {eval_pixel_acc:.4f}")
    print(f"  平均交并比 (mIoU): {eval_miou_score:.4f}")
    
    print(f"\n  各类别 IoU:")
    if class_names and len(class_names) >= num_classes : 
        for i, iou_val in enumerate(eval_iou_per_class):
            name = class_names[i] if i < len(class_names) else f"类别 {i}"
            print(f"    - {name:<20}: {iou_val:.4f}")
    else: 
        for i, iou_val in enumerate(eval_iou_per_class):
            print(f"    - 类别 {i}: {iou_val:.4f}")

    if plot_segmentation_results_func and vis_batch_images is not None and vis_batch_true_masks is not None:
        print("\n可视化评估集上的一些样本 (使用已加载的评估模型)...")
        try:
            vis_images_device = vis_batch_images.to(device)
            with torch.no_grad():
                vis_outputs = model(vis_images_device)
            vis_pred_masks = torch.argmax(vis_outputs, dim=1).cpu()

            plot_segmentation_results_func(
                vis_batch_images,       
                vis_batch_true_masks,   
                vis_pred_masks,         
                num_samples=min(num_vis_samples, vis_batch_images.size(0))
            )
        except Exception as e_vis:
            print(f"评估结果可视化过程中发生错误: {e_vis}")
            import traceback
            traceback.print_exc()
    elif plot_segmentation_results_func:
        print("未能获取用于可视化的数据批次，跳过评估结果可视化。")