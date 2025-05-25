# main.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import datetime
import json
import numpy as np
import matplotlib # 设置matplotlib后端以避免GUI问题
matplotlib.use('Agg') # 使用Agg后端，不显示图形窗口，只保存到文件
import matplotlib.pyplot as plt


from dataset import CamVidDataset
from model import SimplePixelClassifier, UNet, DeepLabV3Plus, SimplifiedSegFormer # 导入 SimplifiedSegFormer
from utils import set_seed, load_checkpoint, tensor_mask_to_pil_rgb, save_dict_to_json
from trainer import Trainer
from evaluate import evaluate_segmentation

def parse_args():
    parser = argparse.ArgumentParser(description="轻量级图像语义分割训练脚本")
    
    # --- 实验输出 ---
    parser.add_argument('--output_base_dir', type=str, default="experiments_output",
                        help="所有实验输出的根目录")
    parser.add_argument('--experiment_name', type=str, default=None,
                        help="当前实验的名称 (可选, 用于构成实验子目录名)")
    
    # --- 数据集参数 ---
    parser.add_argument('--data_root', type=str, default="path/to/your/CamVid", # !!! 需要修改为你的路径 !!!
                        help="CamVid数据集的根目录 (包含train/, val/, test/等文件夹)")
    parser.add_argument('--img_height', type=int, default=256, help="输入图像的高度")
    parser.add_argument('--img_width', type=int, default=256, help="输入图像的宽度 (Transformer通常喜欢正方形)")
    parser.add_argument('--batch_size', type=int, default=2, help="训练和验证的批次大小 (Transformer模型可能需要更小)")
    parser.add_argument('--num_workers', type=int, default=2, help="数据加载器使用的工作线程数")

    # --- 模型参数 ---
    parser.add_argument('--model_name', type=str, default='simple', 
                        choices=['simple', 'unet', 'deeplabv3plus', 'segformer_simple'],
                        help="选择模型架构: 'simple', 'unet', 'deeplabv3plus', or 'segformer_simple'")
    parser.add_argument('--in_channels', type=int, default=3, help="模型输入通道数 (RGB为3)")
    # UNet 特有参数
    parser.add_argument('--unet_bilinear', action='store_true', default=False, 
                        help="UNet是否使用双线性插值进行上采样 (否则使用转置卷积)")
    parser.add_argument('--unet_base_c', type=int, default=64,
                        help="UNet第一层卷积的输出通道数 (例如 32, 64)")
    # DeepLabV3+ 特有参数
    parser.add_argument('--deeplab_output_stride', type=int, default=16, choices=[8, 16],
                        help="DeepLabV3+ 编码器的输出步幅 (通常为8或16)")
    parser.add_argument('--deeplab_encoder_layers', type=int, nargs='+', default=[2,2,2,2], # 接收一个列表
                        help="DeepLabV3+ CustomResNetEncoder中每个stage的Bottleneck块数量 (例如: 2 2 2 2)")
    parser.add_argument('--deeplab_aspp_out_channels', type=int, default=256, help="ASPP模块的输出通道数")
    parser.add_argument('--deeplab_decoder_channels', type=int, default=256, help="DeepLabV3+解码器中间通道数")
    # SimplifiedSegFormer 特有参数
    parser.add_argument('--segformer_embed_dims', type=int, nargs='+', default=[32, 64, 160, 256], help="SegFormer各阶段embedding维度 (e.g., 32 64 160 256 for B0-like)")
    parser.add_argument('--segformer_num_heads', type=int, nargs='+', default=[1, 2, 5, 8], help="SegFormer各阶段注意力头数 (e.g., 1 2 5 8 for B0-like)")
    parser.add_argument('--segformer_mlp_ratios', type=int, nargs='+', default=[4, 4, 4, 4], help="SegFormer各阶段MLP扩展比率")
    parser.add_argument('--segformer_depths', type=int, nargs='+', default=[2, 2, 2, 2], help="SegFormer各阶段Transformer层数 (e.g., 2 2 2 2 for B0-like)")
    parser.add_argument('--segformer_patch_sizes', type=int, nargs='+', default=[4, 2, 2, 2], help="SegFormer各阶段patch_embed的步幅/尺寸 (e.g., 4 2 2 2)")
    parser.add_argument('--segformer_decoder_hidden_dim', type=int, default=256, help="SegFormer解码器MLP隐藏层维度")
    parser.add_argument('--segformer_drop_rate', type=float, default=0.1, help="SegFormer中的dropout率")


    # --- 训练参数 ---
    parser.add_argument('--epochs', type=int, default=100, help="总训练轮数 (Transformer可能需要更多)")
    parser.add_argument('--lr', type=float, default=6e-5, help="初始学习率 (Transformer通常需要较小学习率, e.g., 6e-5 for SegFormer)")
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help="选择优化器 (AdamW常用于Transformer)")
    parser.add_argument('--scheduler_step_size', type=int, default=0, help="学习率调度器的步长 (0表示不使用StepLR, Transformer常用其他策略)")
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help="学习率调度器的gamma值 (对于StepLR)")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="从指定的检查点文件恢复训练")
    parser.add_argument('--metric_to_optimize', type=str, default='mean_iou_adjusted',
                        help="用于选择最佳模型的验证指标 (例如 'mean_iou', 'pixel_accuracy', 'mean_iou_adjusted')")
    parser.add_argument('--print_freq', type=int, default=20, help="训练时打印批次信息的频率")
    parser.add_argument('--no_cuda', action='store_true', default=False, help="禁用CUDA训练 (即使可用)")
    parser.add_argument('--evaluate_on_test', action='store_true', default=False, help="训练结束后在测试集上评估最佳模型")

    args = parser.parse_args()

    # 校验 SegFormer 参数列表长度是否一致 (仅当选择segformer_simple时)
    if args.model_name.lower() == 'segformer_simple':
        segformer_list_param_names = ['segformer_embed_dims', 'segformer_num_heads', 
                                      'segformer_mlp_ratios', 'segformer_depths', 
                                      'segformer_patch_sizes']
        segformer_list_args_values = [getattr(args, name) for name in segformer_list_param_names]
        
        if not segformer_list_args_values: # Should not happen if defaults are set
            pass 
        else:
            it = iter(segformer_list_args_values)
            try:
                the_len = len(next(it))
                if not all(len(l) == the_len for l in it):
                    param_details = "\n".join([f"  {name}: length {len(getattr(args, name))}" for name in segformer_list_param_names])
                    raise ValueError(f'所有 segformer_* 列表参数的长度必须一致。当前长度:\n{param_details}')
            except StopIteration: # Only one list or empty list, should be fine or caught by len check
                pass
            if len(args.segformer_embed_dims) == 0 : # Ensure lists are not empty
                 raise ValueError('SegFormer参数列表 (如 segformer_embed_dims) 不能为空。')


    return args

def main():
    args = parse_args()

    # --- 0. 创建实验目录和基本设置 ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name_prefix = args.model_name 
    if args.experiment_name:
        experiment_folder_name = f"{exp_name_prefix}_{args.experiment_name}_{timestamp}"
    else:
        experiment_folder_name = f"{exp_name_prefix}_{timestamp}"
    
    experiment_dir = os.path.join(args.output_base_dir, experiment_folder_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"所有输出将保存到: {experiment_dir}")

    config_save_path = os.path.join(experiment_dir, "config.json")
    save_dict_to_json(vars(args), config_save_path)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"当前使用的设备: {device}")

    # --- 1. 数据准备 ---
    print("\n--- 正在准备数据集 ---")
    if args.data_root == "path/to/your/CamVid" or not os.path.exists(args.data_root):
        print(f"错误: 请通过 --data_root 参数或直接修改代码中的默认值，")
        print(f"      指定正确的CamVid数据集根目录 ('{args.data_root}' 无效)。")
        return

    image_transform = T.Compose([
        T.Resize((args.img_height, args.img_width), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = T.Resize((args.img_height, args.img_width), interpolation=T.InterpolationMode.NEAREST)

    try:
        train_dataset = CamVidDataset(
            root_dir=args.data_root, image_set='train', transform=image_transform,
            target_transform=target_transform, void_class_names=['Void'] # 确保与你的CSV匹配
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
    except FileNotFoundError as e:
        print(f"加载数据集时出错: {e}")
        print("请确保数据集路径正确，且包含 train/, train_labels/, val/, val_labels/, class_dict.csv 等文件夹。")
        return
    except Exception as e: 
        print(f"加载数据集时发生运行时错误: {e}")
        return


    num_output_classes = train_dataset.num_total_classes
    num_classes_for_eval = train_dataset.num_total_classes
    ignore_index_actual = train_dataset.ignore_index

    if num_output_classes <= 0:
        print("错误: 从数据集中获取的总类别数无效。")
        return
    if ignore_index_actual == -1 :
        print("警告: 数据集未指定明确的 ignore_index (-1)。这意味着所有类别都可能被训练，或依赖损失函数的默认行为。")


    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    if args.evaluate_on_test and 'test_dataset' in locals():
        print(f"  测试集样本数: {len(test_dataset)}")
    print(f"  模型输出类别数 (num_output_classes): {num_output_classes}")
    print(f"  评估用类别数 (num_classes_for_eval): {num_classes_for_eval}")
    print(f"  实际忽略索引 (ignore_index_actual): {ignore_index_actual}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False,
                              drop_last=True) # drop_last=True 对训练更稳定，尤其小batch或BN
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
    if args.evaluate_on_test and 'test_dataset' in locals():
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)


    # --- 2. 模型、损失函数、优化器、调度器 ---
    print("\n--- 正在准备模型和优化器 ---")
    
    if args.model_name.lower() == 'simple':
        model = SimplePixelClassifier(in_channels=args.in_channels, num_classes=num_output_classes)
    elif args.model_name.lower() == 'unet':
        model = UNet(in_channels=args.in_channels, num_classes=num_output_classes,
                     bilinear=args.unet_bilinear, base_c=args.unet_base_c)
    elif args.model_name.lower() == 'deeplabv3plus':
        model = DeepLabV3Plus(in_channels=args.in_channels, num_classes=num_output_classes,
                              output_stride=args.deeplab_output_stride,
                              encoder_layers=args.deeplab_encoder_layers,
                              aspp_out_channels=args.deeplab_aspp_out_channels,
                              decoder_channels=args.deeplab_decoder_channels)
    elif args.model_name.lower() == 'segformer_simple':
        model = SimplifiedSegFormer(in_chans=args.in_channels, num_classes=num_output_classes,
                                    img_size=(args.img_height, args.img_width),
                                    embed_dims=args.segformer_embed_dims,
                                    num_heads=args.segformer_num_heads,
                                    mlp_ratios=args.segformer_mlp_ratios,
                                    depths=args.segformer_depths,
                                    patch_sizes=args.segformer_patch_sizes,
                                    decoder_hidden_dim=args.segformer_decoder_hidden_dim,
                                    drop_rate=args.segformer_drop_rate)
    else:
        raise ValueError(f"不支持的模型名称: {args.model_name}.")
    model.to(device)

    criterion_ignore_idx = ignore_index_actual if ignore_index_actual != -1 else -100
    criterion = nn.CrossEntropyLoss(ignore_index=criterion_ignore_idx)
    print(f"  损失函数: CrossEntropyLoss (ignore_index={criterion_ignore_idx})")

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01) # AdamW 推荐配合 weight_decay
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"不支持的优化器: {args.optimizer}")
    print(f"  优化器: {args.optimizer.upper()} (初始学习率: {args.lr})")

    scheduler = None
    if args.scheduler_step_size > 0 : # StepLR
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        print(f"  学习率调度器: StepLR (step_size={args.scheduler_step_size}, gamma={args.scheduler_gamma})")
    # else: # 可以为Transformer模型考虑其他类型的调度器，例如PolynomialLR或CosineAnnealingLR
    #     print("  未使用固定的学习率调度器 (StepLR)。")


    # --- 3. 加载检查点 ---
    start_epoch = 0
    best_metric_val = float('-inf') 

    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            print(f"  正在从检查点恢复: {args.resume_checkpoint}")
            start_epoch, best_metric_val = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler, device, weights_only=False)
        else:
            print(f"  警告: 找不到指定的检查点文件 {args.resume_checkpoint}。将从头开始训练。")


    # --- 4. 实例化 Trainer 并开始训练 ---
    print("\n--- 准备训练器 ---")
    checkpoint_prefix_for_trainer = f"{args.model_name}_" + (args.experiment_name if args.experiment_name else "run")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        num_classes_eval=num_classes_for_eval,
        ignore_index_eval=ignore_index_actual,
        experiment_dir=experiment_dir,
        checkpoint_prefix=checkpoint_prefix_for_trainer,
        start_epoch=start_epoch,
        best_metric_val=best_metric_val,
        metric_to_optimize=args.metric_to_optimize,
        print_freq=args.print_freq
    )
    trainer.run_training()


    # --- 5. 在测试集上评估最佳模型 ---
    if args.evaluate_on_test and 'test_loader' in locals() and len(test_loader) > 0:
        print("\n--- 正在测试集上评估最佳模型 ---")
        best_checkpoint_path = os.path.join(experiment_dir, f"{checkpoint_prefix_for_trainer}_best.pth.tar")
        if os.path.exists(best_checkpoint_path):
            print(f"  加载最佳模型检查点: {best_checkpoint_path}")
            
            if args.model_name.lower() == 'simple':
                test_model = SimplePixelClassifier(in_channels=args.in_channels, num_classes=num_output_classes)
            elif args.model_name.lower() == 'unet':
                test_model = UNet(in_channels=args.in_channels, num_classes=num_output_classes,
                                  bilinear=args.unet_bilinear, base_c=args.unet_base_c)
            elif args.model_name.lower() == 'deeplabv3plus':
                test_model = DeepLabV3Plus(in_channels=args.in_channels, num_classes=num_output_classes,
                                           output_stride=args.deeplab_output_stride,
                                           encoder_layers=args.deeplab_encoder_layers,
                                           aspp_out_channels=args.deeplab_aspp_out_channels,
                                           decoder_channels=args.deeplab_decoder_channels)
            elif args.model_name.lower() == 'segformer_simple':
                test_model = SimplifiedSegFormer(in_chans=args.in_channels, num_classes=num_output_classes,
                                                img_size=(args.img_height, args.img_width),
                                                embed_dims=args.segformer_embed_dims,
                                                num_heads=args.segformer_num_heads,
                                                mlp_ratios=args.segformer_mlp_ratios,
                                                depths=args.segformer_depths,
                                                patch_sizes=args.segformer_patch_sizes,
                                                decoder_hidden_dim=args.segformer_decoder_hidden_dim,
                                                drop_rate=args.segformer_drop_rate) # 通常评估时drop_rate设为0，但这里为了与训练时一致
            else: 
                print(f"错误: 测试时遇到不支持的模型名称: {args.model_name}")
                return

            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                 test_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint: 
                 test_model.load_state_dict(checkpoint['state_dict'])
            else: 
                 test_model.load_state_dict(checkpoint)
            test_model.to(device)
            test_model.eval()

            test_loss, test_metrics = evaluate_segmentation(
                model=test_model,
                dataloader=test_loader,
                criterion=criterion, 
                device=device,
                num_classes=num_classes_for_eval,
                ignore_index=ignore_index_actual
            )
            
            print(f"\n测试集评估结果 (保存到 {experiment_dir}/test_metrics.json):")
            print(f"  测试损失: {test_loss:.4f}")
            
            serializable_metrics = {}
            for k, v in test_metrics.items():
                if isinstance(v, np.ndarray):
                    serializable_metrics[k] = v.tolist()
                    print(f"  {k}: {np.round(v, 4).tolist()}")
                elif isinstance(v, (float, np.float32, np.float64)):
                    serializable_metrics[k] = float(v)
                    print(f"  {k}: {v:.4f}")
                else:
                    serializable_metrics[k] = str(v)

            serializable_metrics['test_loss'] = test_loss
            save_dict_to_json(serializable_metrics, os.path.join(experiment_dir, "test_metrics.json"))

            # 可视化测试集样本
            try:
                num_viz_samples = min(args.batch_size if args.batch_size > 0 else 1, 4)
                test_iter = iter(test_loader)
                if len(test_loader) == 0:
                    print("测试集为空，跳过可视化。")
                else:
                    viz_images, viz_targets = next(test_iter)
                    viz_images = viz_images[:num_viz_samples].to(device)
                    with torch.no_grad():
                        viz_outputs = test_model(viz_images)
                        viz_preds = torch.argmax(viz_outputs, dim=1)

                    mean_viz = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std_viz = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    viz_images_denorm = viz_images * std_viz + mean_viz
                    viz_images_denorm = viz_images_denorm.clamp(0, 1)

                    fig, axes = plt.subplots(num_viz_samples, 3, figsize=(15, 5 * num_viz_samples), squeeze=False) # squeeze=False确保axes总是2D
                    
                    label_map_viz = train_dataset.label_to_color if hasattr(train_dataset, 'label_to_color') else {}

                    for i in range(num_viz_samples):
                        axes[i, 0].imshow(T.ToPILImage()(viz_images_denorm[i].cpu()))
                        axes[i, 0].set_title("Input Image")
                        axes[i, 0].axis('off')

                        axes[i, 1].imshow(tensor_mask_to_pil_rgb(viz_targets[i].cpu(), label_map_viz))
                        axes[i, 1].set_title("Ground Truth Mask")
                        axes[i, 1].axis('off')

                        axes[i, 2].imshow(tensor_mask_to_pil_rgb(viz_preds[i].cpu(), label_map_viz))
                        axes[i, 2].set_title("Predicted Mask")
                        axes[i, 2].axis('off')
                    plt.tight_layout()
                    viz_save_path = os.path.join(experiment_dir, "test_visualization.png")
                    plt.savefig(viz_save_path)
                    print(f"测试集可视化结果已保存至: {viz_save_path}")
                    plt.close(fig) 
            except Exception as e_viz:
                print(f"生成测试集可视化时出错: {e_viz}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  错误: 找不到最佳模型检查点 {best_checkpoint_path}，无法在测试集上评估。")

    print(f"\n--- main.py 执行完毕. 结果保存在: {experiment_dir} ---")

if __name__ == "__main__":
    main()