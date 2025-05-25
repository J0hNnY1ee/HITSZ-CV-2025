# ==============================================================================
# 模块导入 (IMPORTS)
# ==============================================================================
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import json # 用于保存参数和历史记录
import matplotlib.pyplot as plt

from dataset import get_voc_dataloaders, VOC_CLASSES, VOC_COLORMAP # 确保 dataset.py 可导入
from model import SimpleSegmentationNet # 确保 model.py 可导入
from trainer import Trainer # 确保 trainer.py 可导入
from evaluate import evaluate_model # 确保 evaluate.py 可导入
from utils import visualize_segmentation # 确保 utils.py 可导入

# ==============================================================================
# 参数解析函数 (ARGUMENT PARSING FUNCTION)
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PASCAL VOC 语义分割训练与评估脚本")

    # --- 核心参数 ---
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help="运行模式: 'train' (训练并评估) 或 'eval' (仅评估)。")
    parser.add_argument('--data_root', type=str, default='./data_voc',
                        help="PASCAL VOC 数据集存放的根目录。")
    parser.add_argument('--experiment_dir', type=str, default='./experiment_voc',
                        help="实验结果（检查点、日志、图表）保存目录。")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="用于评估或继续训练的模型检查点路径 (通常是 best_model.pth)。")
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help="如果可用，则使用 CUDA。默认为 True。") # store_true: 出现则为True
    parser.add_argument('--no_cuda', action='store_false', dest='use_cuda',
                        help="不使用 CUDA，即使可用。")

    # --- 数据参数 ---
    parser.add_argument('--image_size', type=int, default=256, 
                        help="输入图像的目标尺寸 (正方形 H=W)。")
    parser.add_argument('--num_workers', type=int, default=min(4, os.cpu_count() // 2 if os.cpu_count() else 1), 
                        help="数据加载的子进程数。")

    # --- 训练参数 ---
    parser.add_argument('--epochs', type=int, default=50, help="训练的总轮数。")
    parser.add_argument('--batch_size', type=int, default=16, help="训练和评估的批处理大小。")
    parser.add_argument('--lr', type=float, default=1e-3, help="初始学习率。")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help="优化器类型。")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="优化器的权重衰减。")
    parser.add_argument('--lr_scheduler', type=str, default='steplr', choices=['steplr', 'cosine', 'plateau', 'none'],
                        help="学习率调度器类型 ('none' 表示不使用)。")
    parser.add_argument('--lr_step_size', type=int, default=15, help="对于StepLR，学习率衰减的步长（epochs）。")
    parser.add_argument('--lr_gamma', type=float, default=0.1, help="对于StepLR，学习率衰减的乘法因子。")
    parser.add_argument('--plateau_patience', type=int, default=5, help="对于ReduceLROnPlateau的patience。")


    # --- 其他参数 ---
    parser.add_argument('--print_every_batch', type=int, default=20, 
                        help="训练时每隔多少个 batch 打印一次训练信息。")
    parser.add_argument('--num_visualize_eval', type=int, default=3, 
                        help="在评估模式结束时可视化多少个样本。")
    parser.add_argument('--seed', type=int, default=42, help="随机种子，用于可复现性。")
    
    args = parser.parse_args()
    return args

# ==============================================================================
# 辅助函数：设置随机种子 (HELPER: SET RANDOM SEED)
# ==============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # 可复现性优先
    np.random.seed(seed)
    random.seed(seed)

# ==============================================================================
# 辅助函数：绘制训练历史曲线 (HELPER: PLOT TRAINING HISTORY)
# ==============================================================================
def plot_training_history(history, save_path):
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='训练损失 (Train Loss)')
    plt.plot(epochs_range, history['val_loss'], label='验证损失 (Validation Loss)')
    plt.xlabel('轮数 (Epochs)')
    plt.ylabel('损失 (Loss)')
    plt.legend()
    plt.title('损失曲线 (Loss Curves)')

    # mIoU 曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['val_miou'], label='验证 mIoU (Validation mIoU)')
    plt.xlabel('轮数 (Epochs)')
    plt.ylabel('mIoU')
    plt.legend()
    plt.title('验证 mIoU (Validation mIoU)')
    
    # 学习率曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['lr'], label='学习率 (Learning Rate)')
    plt.xlabel('轮数 (Epochs)')
    plt.ylabel('学习率 (Learning Rate)')
    plt.legend()
    plt.title('学习率变化 (Learning Rate Schedule)')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练历史曲线图已保存至: {save_path}")
    # plt.show() # 如果希望在脚本结束时显示

# ==============================================================================
# 主函数 (MAIN FUNCTION)
# ==============================================================================
def main():
    args = parse_args()

    # --- 0. 初始化和设置 ---
    set_seed(args.seed)
    
    # 实验目录
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
        print(f"创建实验目录: {args.experiment_dir}")
    
    # 保存参数配置
    args_path = os.path.join(args.experiment_dir, 'args_config.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"参数配置已保存至: {args_path}")

    # 设备配置
    use_cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if args.use_cuda and use_cuda_available else "cpu")
    print(f"使用设备: {device}")
    if args.use_cuda and not use_cuda_available:
        print("警告: 请求使用 CUDA 但 CUDA 不可用，将使用 CPU。")

    # --- 1. 数据准备 ---
    num_classes = len(VOC_CLASSES)
    ignore_index_dataset = 255 # PASCAL VOC 的忽略标签
    image_size_tuple = (args.image_size, args.image_size)

    print("\n[1/5] 正在加载数据集...")
    train_loader, val_loader, _ = get_voc_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size_tuple,
        num_workers=args.num_workers
    )
    print("数据集加载完成。")

    # --- 2. 模型构建 ---
    print("\n[2/5] 正在构建模型...")
    model = SimpleSegmentationNet(num_classes=num_classes).to(device)
    print(f"模型 '{model.__class__.__name__}' 构建完成。")
    
    # --- 3. 损失函数和优化器 ---
    print("\n[3/5] 正在配置损失函数和优化器...")
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index_dataset)
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else: # 理论上 argparse choices 会阻止这个
        raise ValueError(f"不支持的优化器: {args.optimizer}")
    print(f"  优化器: {args.optimizer.upper()}, 学习率: {args.lr}, 权重衰减: {args.weight_decay}")

    scheduler = None
    if args.lr_scheduler == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        print(f"  学习率调度器: StepLR (step_size={args.lr_step_size}, gamma={args.lr_gamma})")
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(1e-7, args.lr * 0.01))
        print("  学习率调度器: CosineAnnealingLR")
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_gamma, patience=args.plateau_patience, verbose=True)
        print(f"  学习率调度器: ReduceLROnPlateau (factor={args.lr_gamma}, patience={args.plateau_patience})")
    elif args.lr_scheduler == 'none':
        print("  不使用学习率调度器。")


    # --- 4. 训练或评估 ---
    if args.mode == 'train':
        print("\n[4/5] 开始训练模式...")
        trainer_instance = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            device=device, num_classes=num_classes, num_epochs=args.epochs,
            save_dir=args.experiment_dir, # 保存到实验目录
            print_every=args.print_every_batch,
            ignore_index=ignore_index_dataset,
            patience_lr_scheduler=args.plateau_patience 
        )
        training_history = trainer_instance.train()
        
        # 保存训练历史
        history_path = os.path.join(args.experiment_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        print(f"训练历史记录已保存至: {history_path}")

        # 绘制并保存训练曲线图
        plot_path = os.path.join(args.experiment_dir, 'training_curves.png')
        plot_training_history(training_history, plot_path)

        # 训练完成后，自动加载最佳模型进行一次最终评估 (可选，但推荐)
        print("\n[5/5] 加载最佳模型在验证集上进行最终评估...")
        best_model_path = os.path.join(args.experiment_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已从 '{best_model_path}' 加载最佳模型 "
                  f"(Epoch {checkpoint.get('epoch', 'N/A')}, mIoU {checkpoint.get('best_val_metric', 'N/A'):.4f})。")
            
            final_val_loss, final_val_pa, final_val_miou, final_iou_per_class = evaluate_model(
                model, val_loader, criterion, device, num_classes, ignore_index_dataset
            )
            print("\n最终验证集评估结果:")
            print(f"  平均损失: {final_val_loss:.4f}")
            print(f"  像素准确率 (PA): {final_val_pa:.4f}")
            print(f"  平均交并比 (mIoU): {final_val_miou:.4f}")
            print("  各类别 IoU:")
            for i, iou_val in enumerate(final_iou_per_class):
                print(f"    {VOC_CLASSES[i % len(VOC_CLASSES)]:<15}: {iou_val:.4f}")
        else:
            print(f"警告: 未找到最佳模型 '{best_model_path}'。")

    elif args.mode == 'eval':
        print("\n[4/5] 开始评估模式...")
        if not args.resume_checkpoint:
            print("错误: 评估模式需要通过 --resume_checkpoint 指定模型路径。")
            return
        if not os.path.exists(args.resume_checkpoint):
            print(f"错误: 找不到指定的模型文件 '{args.resume_checkpoint}'。")
            return
            
        print(f"正在从 '{args.resume_checkpoint}' 加载模型...")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        # 检查 checkpoint 是否包含 model_state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型加载成功 (来自Epoch {checkpoint.get('epoch', 'N/A')}, mIoU {checkpoint.get('best_val_metric', 'N/A'):.4f})。")
        else: # 假设直接保存的是 state_dict
            model.load_state_dict(checkpoint)
            print("模型加载成功 (假设直接加载 state_dict)。")
        
        eval_loss, eval_pa, eval_miou, eval_iou_per_class = evaluate_model(
            model, val_loader, criterion, device, num_classes, ignore_index_dataset
        )
        print("\n[5/5] 评估完成:")
        print(f"  平均损失: {eval_loss:.4f}")
        print(f"  像素准确率 (PA): {eval_pa:.4f}")
        print(f"  平均交并比 (mIoU): {eval_miou:.4f}")
        print("  各类别 IoU:")
        for i, iou_val in enumerate(eval_iou_per_class):
            print(f"    {VOC_CLASSES[i % len(VOC_CLASSES)]:<15}: {iou_val:.4f}")
            
        # --- 可视化部分验证集样本 ---
        if args.num_visualize_eval > 0:
            print(f"\n正在可视化 {args.num_visualize_eval} 个验证集样本...")
            model.eval()
            with torch.no_grad():
                count_visualized = 0
                for images_vis, masks_vis in val_loader: # 使用验证集进行可视化
                    if count_visualized >= args.num_visualize_eval:
                        break
                    
                    image_single_pil = images_vis[0].to(device) # 取批次中的第一张图 (C,H,W)
                    mask_single_true_pil = masks_vis[0].squeeze(0).long() # (H,W)
                    
                    output_single = model(image_single_pil.unsqueeze(0)) # (1, num_classes, H, W)
                    _, pred_single_mask = torch.max(output_single.squeeze(0), 0) # (H,W)
                    
                    print(f"  可视化样本 {count_visualized + 1}:")
                    visualize_segmentation(
                        image_single_pil.cpu(), 
                        pred_single_mask.cpu(), 
                        mask_single_true_pil.cpu(),
                        num_classes=num_classes,
                        colormap=VOC_COLORMAP # 使用 dataset 中定义的 colormap
                    )
                    count_visualized += 1
    else:
        print(f"错误: 不支持的模式 '{args.mode}'。请选择 'train' 或 'eval'。")

    print("\n脚本执行完毕。")

# ==============================================================================
# 主执行块 (MAIN EXECUTION BLOCK)
# ==============================================================================
if __name__ == '__main__':
    main()