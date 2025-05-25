# main.py (修正后的相关部分)

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import datetime
import json
import numpy as np

from dataset import CamVidDataset # 确保这个导入是正确的
from model import SimplePixelClassifier
from utils import set_seed, load_checkpoint, tensor_mask_to_pil_rgb, save_dict_to_json
from trainer import Trainer
from evaluate import evaluate_segmentation

def parse_args():
    parser = argparse.ArgumentParser(description="轻量级图像语义分割训练脚本")
    # ... (所有参数定义保持不变，这里省略以保持简洁) ...
    parser.add_argument('--data_root', type=str, default="path/to/your/CamVid", 
                        help="CamVid数据集的根目录")
    parser.add_argument('--output_base_dir', type=str, default="experiments_output",
                        help="所有实验输出的根目录")
    parser.add_argument('--experiment_name', type=str, default=None,
                        help="当前实验的名称 (可选, 用于构成实验子目录名)")
    parser.add_argument('--img_height', type=int, default=256, help="输入图像的高度")
    parser.add_argument('--img_width', type=int, default=384, help="输入图像的宽度")
    parser.add_argument('--batch_size', type=int, default=4, help="训练和验证的批次大小")
    parser.add_argument('--num_workers', type=int, default=2, help="数据加载器使用的工作线程数")
    parser.add_argument('--in_channels', type=int, default=3, help="模型输入通道数")
    parser.add_argument('--epochs', type=int, default=50, help="总训练轮数")
    parser.add_argument('--lr', type=float, default=1e-3, help="初始学习率")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help="选择优化器")
    parser.add_argument('--scheduler_step_size', type=int, default=20, help="学习率调度器的步长")
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help="学习率调度器的gamma值")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="从指定的检查点文件恢复训练")
    parser.add_argument('--metric_to_optimize', type=str, default='mean_iou_adjusted',
                        help="用于选择最佳模型的验证指标")
    parser.add_argument('--print_freq', type=int, default=20, help="训练时打印批次信息的频率")
    parser.add_argument('--no_cuda', action='store_true', default=False, help="禁用CUDA训练")
    parser.add_argument('--use_tensorboard', action='store_true', default=False, help="使用TensorBoard")
    parser.add_argument('--evaluate_on_test', action='store_true', default=False, help="训练结束后在测试集上评估最佳模型")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # --- 0. 创建实验目录和基本设置 ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.experiment_name:
        experiment_folder_name = f"{args.experiment_name}_{timestamp}"
    else:
        experiment_folder_name = timestamp
    
    experiment_dir = os.path.join(args.output_base_dir, experiment_folder_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"所有输出将保存到: {experiment_dir}")

    config_save_path = os.path.join(experiment_dir, "config.json")
    save_dict_to_json(vars(args), config_save_path)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"当前使用的设备: {device}")

    summary_writer = None
    if args.use_tensorboard:
        tensorboard_dir = os.path.join(experiment_dir, "tensorboard_logs")
        os.makedirs(tensorboard_dir, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard 日志将保存在: {tensorboard_dir}")

    # --- 1. 数据准备 ---
    print("\n--- 正在准备数据集 ---")
    if args.data_root == "path/to/your/CamVid" or not os.path.exists(args.data_root):
        print(f"错误: 请通过 --data_root 参数指定正确的CamVid数据集根目录 ('{args.data_root}' 无效)。")
        if summary_writer: summary_writer.close(); return
    # ... (transforms定义与之前一致) ...
    image_transform = T.Compose([
        T.Resize((args.img_height, args.img_width), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = T.Resize((args.img_height, args.img_width), interpolation=T.InterpolationMode.NEAREST)

    try:
        # 实例化数据集时，传入 void_class_names=['Void'] (或其他你的void类名称)
        train_dataset = CamVidDataset(
            root_dir=args.data_root, image_set='train', transform=image_transform,
            target_transform=target_transform, void_class_names=['Void']
        )
        val_dataset = CamVidDataset(
            root_dir=args.data_root, image_set='val', transform=image_transform,
            target_transform=target_transform, void_class_names=['Void']
        )
        if args.evaluate_on_test:
            test_dataset = CamVidDataset(
                root_dir=args.data_root, image_set='test', transform=image_transform,
                target_transform=target_transform, void_class_names=['Void']
            )
    except Exception as e:
        print(f"加载数据集时出错: {e}");
        if summary_writer: summary_writer.close(); return

    # **核心修正点**
    # 模型输出的类别数应该是数据集中定义的总类别数
    num_output_classes = train_dataset.num_total_classes
    # 用于评估的类别数（混淆矩阵维度）也应该是总类别数
    num_classes_for_eval = train_dataset.num_total_classes
    # 实际的忽略索引，从数据集中获取 (例如 "Void" 类的索引)
    ignore_index_actual = train_dataset.ignore_index

    if num_output_classes <= 0:
        print("错误: 从数据集中获取的总类别数无效。")
        if summary_writer: summary_writer.close(); return
    if ignore_index_actual == -1 : # 如果dataset说没有特定忽略类
        print("警告: 数据集未指定明确的 ignore_index (-1)。这意味着所有类别都可能被训练。")
        print("       如果希望忽略某个类别（如'Void'），请确保void_class_names参数正确设置，")
        print("       并且该类别存在于class_dict.csv中。")


    print(f"  训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}" + (f", 测试集: {len(test_dataset)}" if args.evaluate_on_test and 'test_dataset' in locals() else ""))
    print(f"  模型输出类别数 (num_output_classes): {num_output_classes}")
    print(f"  评估用类别数 (num_classes_for_eval): {num_classes_for_eval}")
    print(f"  实际忽略索引 (ignore_index_actual): {ignore_index_actual}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    if args.evaluate_on_test and 'test_dataset' in locals():
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)

    # --- 2. 模型、损失函数、优化器、调度器 ---
    print("\n--- 正在准备模型和优化器 ---")
    model = SimplePixelClassifier(in_channels=args.in_channels, num_classes=num_output_classes) # 使用总类别数
    model.to(device)

    # 损失函数: ignore_index 设置为数据集中识别出的忽略索引
    criterion_ignore_idx = ignore_index_actual
    if criterion_ignore_idx == -1: # 如果数据集中没有识别出特定的忽略索引
        # 这种情况下，CrossEntropyLoss不会忽略任何特定类别，除非标签值本身是-100
        # 通常我们期望有一个明确的 ignore_index (如 Void 类)
        print("警告: 损失函数没有设置特定的 ignore_index (因为数据集中未识别到)。")
        # 可以选择设为-100，或者如果确定没有需要忽略的，则不设置（但通常语义分割会有）
        criterion_ignore_idx = -100 # PyTorch CrossEntropyLoss 默认的 ignore_index
    criterion = nn.CrossEntropyLoss(ignore_index=criterion_ignore_idx)
    print(f"  损失函数: CrossEntropyLoss (ignore_index={criterion_ignore_idx})")

    # ... (优化器和调度器部分与之前一致) ...
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"不支持的优化器: {args.optimizer}")
    print(f"  优化器: {args.optimizer.upper()} (初始学习率: {args.lr})")
    scheduler = None
    if args.scheduler_step_size > 0 :
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        print(f"  学习率调度器: StepLR (step_size={args.scheduler_step_size}, gamma={args.scheduler_gamma})")


    # --- 3. 加载检查点 ---
    start_epoch = 0
    best_metric_val = float('-inf')
    if args.resume_checkpoint:
        # ... (与之前一致) ...
        if os.path.isfile(args.resume_checkpoint):
            print(f"  正在从检查点恢复: {args.resume_checkpoint}")
            start_epoch, best_metric_val = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler, device)
        else:
            print(f"  警告: 找不到指定的检查点文件 {args.resume_checkpoint}。将从头开始训练。")


    # --- 4. 实例化 Trainer 并开始训练 ---
    print("\n--- 准备训练器 ---")
    checkpoint_prefix_for_trainer = args.experiment_name if args.experiment_name else "camvid_run"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        num_classes_eval=num_classes_for_eval,    # **使用总类别数**
        ignore_index_eval=ignore_index_actual,   # **使用数据集中识别的忽略索引**
        experiment_dir=experiment_dir,
        checkpoint_prefix=checkpoint_prefix_for_trainer,
        start_epoch=start_epoch,
        best_metric_val=best_metric_val,
        metric_to_optimize=args.metric_to_optimize,
        print_freq=args.print_freq,
        save_summary_writer=summary_writer
    )
    trainer.run_training()

    # --- 5. 在测试集上评估最佳模型 ---
    if args.evaluate_on_test and 'test_loader' in locals() and len(test_loader) > 0:
        print("\n--- 正在测试集上评估最佳模型 ---")
        best_checkpoint_path = os.path.join(experiment_dir, f"{checkpoint_prefix_for_trainer}_best.pth.tar")
        if os.path.exists(best_checkpoint_path):
            print(f"  加载最佳模型检查点: {best_checkpoint_path}")
            test_model = SimplePixelClassifier(in_channels=args.in_channels, num_classes=num_output_classes) # **使用总类别数**
            # ... (模型加载逻辑与之前一致) ...
            checkpoint = torch.load(best_checkpoint_path, map_location=device,weights_only=False)
            if 'model_state_dict' in checkpoint: test_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint: test_model.load_state_dict(checkpoint['state_dict'])
            else: test_model.load_state_dict(checkpoint)
            test_model.to(device)
            test_model.eval()

            test_loss, test_metrics = evaluate_segmentation(
                model=test_model, dataloader=test_loader, criterion=criterion, device=device,
                num_classes=num_classes_for_eval, # **使用总类别数**
                ignore_index=ignore_index_actual  # **使用数据集中识别的忽略索引**
            )
            # ... (结果保存和可视化逻辑与之前一致) ...
            print(f"\n测试集评估结果 (保存到 {experiment_dir}/test_metrics.json):")
            print(f"  测试损失: {test_loss:.4f}")
            serializable_metrics = {}
            for k, v in test_metrics.items():
                if isinstance(v, np.ndarray): serializable_metrics[k] = v.tolist(); print(f"  {k}: {np.round(v, 4).tolist()}")
                elif isinstance(v, (float, np.float32, np.float64)): serializable_metrics[k] = float(v); print(f"  {k}: {v:.4f}")
                else: serializable_metrics[k] = str(v)
            serializable_metrics['test_loss'] = test_loss
            save_dict_to_json(serializable_metrics, os.path.join(experiment_dir, "test_metrics.json"))

            try:
                from matplotlib import pyplot as plt # Moved import here
                num_viz_samples = min(args.batch_size, 4); test_iter = iter(test_loader)
                if len(test_loader) == 0: print("测试集为空，跳过可视化。")
                else:
                    viz_images, viz_targets = next(test_iter)
                    viz_images = viz_images[:num_viz_samples].to(device)
                    with torch.no_grad(): viz_outputs = test_model(viz_images); viz_preds = torch.argmax(viz_outputs, dim=1)
                    mean_viz = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std_viz = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    viz_images_denorm = viz_images * std_viz + mean_viz; viz_images_denorm = viz_images_denorm.clamp(0, 1)
                    fig, axes = plt.subplots(num_viz_samples, 3, figsize=(15, 5 * num_viz_samples))
                    if num_viz_samples == 1: axes = np.array([axes]) # type: ignore
                    label_map_viz = train_dataset.label_to_color if hasattr(train_dataset, 'label_to_color') else {}
                    for i in range(num_viz_samples):
                        axes[i, 0].imshow(T.ToPILImage()(viz_images_denorm[i].cpu())); axes[i, 0].set_title("Input"); axes[i, 0].axis('off')
                        axes[i, 1].imshow(tensor_mask_to_pil_rgb(viz_targets[i].cpu(), label_map_viz)); axes[i, 1].set_title("Ground Truth"); axes[i, 1].axis('off')
                        axes[i, 2].imshow(tensor_mask_to_pil_rgb(viz_preds[i].cpu(), label_map_viz)); axes[i, 2].set_title("Prediction"); axes[i, 2].axis('off')
                    plt.tight_layout()
                    viz_save_path = os.path.join(experiment_dir, "test_visualization.png")
                    plt.savefig(viz_save_path); print(f"测试集可视化结果已保存至: {viz_save_path}")
            except Exception as e_viz: print(f"生成测试集可视化时出错: {e_viz}")
        else:
            print(f"  错误: 找不到最佳模型检查点，无法在测试集上评估。")

    if summary_writer:
        summary_writer.close()
    print(f"\n--- main.py 执行完毕. 结果保存在: {experiment_dir} ---")

if __name__ == "__main__":
    main()