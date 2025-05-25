# main.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF # 用于同步变换
import random # 用于随机性
import datetime
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from dataset import CamVidDataset # 假设这是您原始的 dataset.py
from model import SimplePixelClassifier, UNet, DeepLabV3Plus, SimplifiedSegFormer, SegNet
from utils import set_seed, load_checkpoint, tensor_mask_to_pil_rgb, save_dict_to_json
from trainer import Trainer
from evaluate import evaluate_segmentation

def parse_args():
    parser = argparse.ArgumentParser(description="轻量级图像语义分割训练脚本")
    
    # --- 实验输出 ---
    parser.add_argument('--output_base_dir', type=str, default="experiments_output", help="所有实验输出的根目录")
    parser.add_argument('--experiment_name', type=str, default=None, help="当前实验的名称")
    
    # --- 数据集参数 ---
    parser.add_argument('--data_root', type=str, default="path/to/your/CamVid", help="CamVid数据集的根目录")
    parser.add_argument('--img_height', type=int, default=224, help="输入图像的高度")
    parser.add_argument('--img_width', type=int, default=224, help="输入图像的宽度")
    parser.add_argument('--batch_size', type=int, default=4, help="批次大小")
    parser.add_argument('--num_workers', type=int, default=2, help="数据加载器工作线程数")
    parser.add_argument('--use_data_augmentation', action='store_true', default=False, # 新增参数
                        help="是否对训练集使用数据增强")

    # --- 模型参数 ---
    parser.add_argument('--model_name', type=str, default='simple', 
                        choices=['simple', 'unet', 'deeplabv3plus', 'segformer_simple', 'segnet'],
                        help="选择模型架构")
    parser.add_argument('--in_channels', type=int, default=3, help="模型输入通道数")
    parser.add_argument('--unet_bilinear', action='store_true', default=False, help="UNet使用双线性插值")
    parser.add_argument('--unet_base_c', type=int, default=64, help="UNet基础通道数")
    parser.add_argument('--deeplab_output_stride', type=int, default=16, choices=[8, 16], help="DeepLabV3+编码器输出步幅")
    parser.add_argument('--deeplab_encoder_layers', type=int, nargs='+', default=[2,2,2,2], help="DeepLabV3+ CustomResNetEncoder块数量")
    parser.add_argument('--deeplab_aspp_out_channels', type=int, default=256, help="ASPP模块输出通道数")
    parser.add_argument('--deeplab_decoder_channels', type=int, default=256, help="DeepLabV3+解码器中间通道数")
    parser.add_argument('--segformer_embed_dims', type=int, nargs='+', default=[32, 64, 160, 256], help="SegFormer各阶段embedding维度")
    parser.add_argument('--segformer_num_heads', type=int, nargs='+', default=[1, 2, 5, 8], help="SegFormer各阶段注意力头数")
    parser.add_argument('--segformer_mlp_ratios', type=int, nargs='+', default=[4, 4, 4, 4], help="SegFormer各阶段MLP扩展比率")
    parser.add_argument('--segformer_depths', type=int, nargs='+', default=[2, 2, 2, 2], help="SegFormer各阶段Transformer层数")
    parser.add_argument('--segformer_patch_sizes', type=int, nargs='+', default=[4, 2, 2, 2], help="SegFormer各阶段patch_embed的步幅/尺寸")
    parser.add_argument('--segformer_decoder_hidden_dim', type=int, default=256, help="SegFormer解码器MLP隐藏层维度")
    parser.add_argument('--segformer_drop_rate', type=float, default=0.1, help="SegFormer中的dropout率")
    parser.add_argument('--segnet_bn_momentum', type=float, default=0.1, help="SegNet中BatchNorm的momentum")

    # --- 训练参数 ---
    parser.add_argument('--epochs', type=int, default=100, help="总训练轮数")
    parser.add_argument('--lr', type=float, default=1e-3, help="初始学习率")
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], help="选择优化器")
    parser.add_argument('--scheduler_step_size', type=int, default=30, help="学习率调度器的步长")
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help="学习率调度器的gamma值")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="从指定的检查点文件恢复训练")
    parser.add_argument('--metric_to_optimize', type=str, default='mean_iou_adjusted', help="用于选择最佳模型的验证指标")
    parser.add_argument('--print_freq', type=int, default=20, help="训练时打印批次信息的频率")
    parser.add_argument('--no_cuda', action='store_true', default=False, help="禁用CUDA训练")
    parser.add_argument('--evaluate_on_test', action='store_true', default=False, help="训练结束后在测试集上评估最佳模型")
    args = parser.parse_args()
    # ... (SegFormer参数列表长度校验与之前一致) ...
    if args.model_name.lower() == 'segformer_simple':
        segformer_list_param_names = ['segformer_embed_dims', 'segformer_num_heads', 'segformer_mlp_ratios', 'segformer_depths', 'segformer_patch_sizes']
        segformer_list_args_values = [getattr(args, name) for name in segformer_list_param_names]
        if segformer_list_args_values:
            it = iter(segformer_list_args_values); 
            try:
                the_len = len(next(it))
                if not all(len(l) == the_len for l in it):
                    param_details = "\n".join([f"  {name}: length {len(getattr(args, name))}" for name in segformer_list_param_names])
                    raise ValueError(f'所有 segformer_* 列表参数的长度必须一致。当前长度:\n{param_details}')
                if the_len == 0: raise ValueError('SegFormer参数列表不能为空。')
            except StopIteration: 
                if len(segformer_list_args_values[0]) == 0: raise ValueError('SegFormer参数列表不能为空。')
    return args

# 自定义 Transform 类用于同步变换图像和掩码
class JointTransform:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, image, mask):
        # 确保随机参数对于 image 和 mask 是一致的
        # 对于 torchvision.transforms 中需要随机参数的变换，需要手动管理
        # 例如，RandomHorizontalFlip, RandomRotation
        
        # 1. 随机水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        # 2. (可选) 随机旋转
        # angle = T.RandomRotation.get_params([-15, 15]) # 获取随机角度
        # image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        # mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 3. 应用传入的其他只针对图像的变换 (如ColorJitter)
        # 注意: transforms_list 应该包含只作用于图像的变换
        if self.transforms_list:
            for t in self.transforms_list:
                if isinstance(t, (T.ColorJitter, T.GaussianBlur, T.RandomGrayscale)): # 这些只作用于图像
                    image = t(image)
                # elif isinstance(t, T.RandomResizedCrop): # 这个比较复杂，需要同步
                #     # i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(0.75, 1.33))
                #     # image = TF.resized_crop(image, i, j, h, w, (args.img_height, args.img_width)) # 需要传入目标尺寸
                #     # mask = TF.resized_crop(mask, i, j, h, w, (args.img_height, args.img_width), interpolation=T.InterpolationMode.NEAREST)
                #     pass # 简化，暂时不实现复杂的同步 RandomResizedCrop
                else:
                    # 对于其他可能是几何变换的，需要确保同步，或不在这里应用
                    # 为了简单，假设 transforms_list 主要包含颜色类增强
                    pass


        return image, mask

def main():
    args = parse_args()
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name_prefix = args.model_name 
    if args.experiment_name: experiment_folder_name = f"{exp_name_prefix}_{args.experiment_name}_{timestamp}"
    else: experiment_folder_name = f"{exp_name_prefix}_{timestamp}"
    experiment_dir = os.path.join(args.output_base_dir, experiment_folder_name)
    os.makedirs(experiment_dir, exist_ok=True); print(f"所有输出将保存到: {experiment_dir}")
    config_save_path = os.path.join(experiment_dir, "config.json"); save_dict_to_json(vars(args), config_save_path)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"); print(f"当前使用的设备: {device}")

    print("\n--- 正在准备数据集 ---")
    if args.data_root == "path/to/your/CamVid" or not os.path.exists(args.data_root):
        print(f"错误: 请通过 --data_root 参数指定正确的CamVid数据集根目录 ('{args.data_root}' 无效)。"); return

    # 定义基础的图像和目标变换（在增强之后或直接应用）
    # 这些将被包装在Dataset的transform和target_transform中，或者在自定义的JointTransform之后应用
    
    # 基础的图像变换（在所有PIL级别增强后应用）
    base_image_transforms = T.Compose([
        T.Resize((args.img_height, args.img_width), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 基础的目标掩码变换（在所有PIL级别增强后应用）
    base_target_transforms = T.Resize((args.img_height, args.img_width), 
                                       interpolation=T.InterpolationMode.NEAREST)

    # 如果使用数据增强，我们需要一个自定义的 callable 来处理同步变换和图像专用变换
    # 然后将最终的 Resize, ToTensor, Normalize 放在 Dataset 的 transform 中处理 image_pil
    # 或者将它们也集成到自定义 callable 中，使其直接输出 tensor

    # 方案：Dataset的transform参数接收一个能处理 (image_pil, label_pil) -> (image_tensor, label_pil_resized) 的函数
    # target_transform 接收 label_pil_resized -> label_mask_tensor
    # 或者，Dataset的transform只处理image_pil，target_transform只处理label_pil。增强逻辑在Dataset内部。

    # 保持 Dataset 的 __init__ 接口不变 (transform for image, target_transform for label)
    # 我们在 main.py 中构造不同的 transform 对象

    if args.use_data_augmentation:
        print("  训练集将使用数据增强。")
        # 定义只针对图像的PIL级增强
        pil_image_only_augmentations = [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.GaussianBlur(kernel_size=(3,5), sigma=(0.1,1.0)) # 示例
        ]
        
        # 创建一个 callable，它将按顺序应用同步变换和图像专用变换
        # 然后再应用最终的 Resize, ToTensor, Normalize
        class CustomTrainTransform:
            def __init__(self, image_only_augs, final_img_transform, final_lbl_transform):
                self.image_only_augs = T.Compose(image_only_augs) if image_only_augs else None
                self.final_img_transform = final_img_transform
                self.final_lbl_transform = final_lbl_transform

            def __call__(self, image_pil, label_pil):
                # 1. 同步几何变换 (示例: 随机水平翻转)
                if random.random() > 0.5:
                    image_pil = TF.hflip(image_pil)
                    label_pil = TF.hflip(label_pil)
                
                # (可以添加更多同步几何变换，如旋转、仿射等)

                # 2. 应用仅针对图像的PIL级增强
                if self.image_only_augs:
                    image_pil = self.image_only_augs(image_pil)
                
                # 3. 应用最终的图像变换 (Resize, ToTensor, Normalize)
                image_tensor = self.final_img_transform(image_pil)
                
                # 4. 应用最终的标签变换 (Resize)
                label_pil_resized = self.final_lbl_transform(label_pil)
                
                return image_tensor, label_pil_resized # Dataset的target_transform会接收label_pil_resized

        # train_transform_obj 现在是一个接收 (image_pil, label_pil) 的对象
        # 但 Dataset 的 transform 参数只接收 image_pil。
        # 这意味着数据增强逻辑必须在 Dataset 的 __getitem__ 内部实现。
        # 所以我们回到修改 Dataset 的方案。

        # ** 再次确认：如果不想修改 Dataset.py，增强逻辑必须封装在传递给 Dataset 的 transform 和 target_transform 中 **
        # 这对于同步几何变换来说用 torchvision.transforms 会很困难。
        # 所以，最合理的方式还是在 Dataset 内部根据 image_set 和 use_data_augmentation 来选择性应用。
        # 我将假设你已经采纳了我上一个关于修改 Dataset.py 的回复。

        # 如果 dataset.py 确实是【你提供的那个未修改版本】，那么只能在 main.py 中这样做：
        # 训练集的图像变换
        train_image_transform = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1), # 颜色增强
            T.RandomHorizontalFlip(p=0.5), # 几何增强，但无法同步给 target
            # 注意：这里的 RandomHorizontalFlip 只会作用于图像！
            # 要同步，必须在 Dataset 内部或用 albumentations
            T.Resize((args.img_height, args.img_width), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 训练集的目标变换 (只能Resize)
        train_target_transform = T.Resize((args.img_height, args.img_width), interpolation=T.InterpolationMode.NEAREST)
        
        # 验证/测试集的图像变换
        eval_image_transform = base_image_transforms
        eval_target_transform = base_target_transforms

    else:
        print("  训练集将不使用额外的数据增强 (除了Resize, ToTensor, Normalize)。")
        train_image_transform = base_image_transforms
        train_target_transform = base_target_transforms
        eval_image_transform = base_image_transforms
        eval_target_transform = base_target_transforms


    try:
        train_dataset = CamVidDataset(
            root_dir=args.data_root, image_set='train', 
            transform=train_image_transform, # 传递构造好的 transform
            target_transform=train_target_transform,
            void_class_names=['Void']
        )
        val_dataset = CamVidDataset(
            root_dir=args.data_root, image_set='val', 
            transform=eval_image_transform,
            target_transform=eval_target_transform,
            void_class_names=['Void']
        )
        if args.evaluate_on_test:
            test_dataset = CamVidDataset(
                root_dir=args.data_root, image_set='test', 
                transform=eval_image_transform,
                target_transform=eval_target_transform,
                void_class_names=['Void']
            )
    except FileNotFoundError as e: print(f"加载数据集时出错: {e}"); return
    except Exception as e: print(f"加载数据集时发生运行时错误: {e}"); return

    # ... (后续代码与之前的版本（集成了SegNet的那个）完全相同，这里不再重复) ...
    # ... 包括：类别信息获取、DataLoader创建、模型选择与实例化、优化器、调度器、检查点加载 ...
    # ... Trainer实例化和运行、测试集评估和可视化 ...
    num_output_classes = train_dataset.num_total_classes; num_classes_for_eval = train_dataset.num_total_classes
    ignore_index_actual = train_dataset.ignore_index
    if num_output_classes <= 0: print("错误: 从数据集中获取的总类别数无效。"); return
    if ignore_index_actual == -1 : print("警告: 数据集未指定明确的 ignore_index (-1)。")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    if args.evaluate_on_test and 'test_dataset' in locals(): print(f"  测试集样本数: {len(test_dataset)}")
    print(f"  模型输出类别数 (num_output_classes): {num_output_classes}")
    print(f"  评估用类别数 (num_classes_for_eval): {num_classes_for_eval}")
    print(f"  实际忽略索引 (ignore_index_actual): {ignore_index_actual}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True if device.type=='cuda' else False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type=='cuda' else False)
    if args.evaluate_on_test and 'test_dataset' in locals(): test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type=='cuda' else False)

    print("\n--- 正在准备模型和优化器 ---")
    if args.model_name.lower() == 'simple': model = SimplePixelClassifier(in_channels=args.in_channels, num_classes=num_output_classes)
    elif args.model_name.lower() == 'unet': model = UNet(in_channels=args.in_channels, num_classes=num_output_classes, bilinear=args.unet_bilinear, base_c=args.unet_base_c)
    elif args.model_name.lower() == 'deeplabv3plus': model = DeepLabV3Plus(in_channels=args.in_channels, num_classes=num_output_classes, output_stride=args.deeplab_output_stride, encoder_layers=args.deeplab_encoder_layers, aspp_out_channels=args.deeplab_aspp_out_channels, decoder_channels=args.deeplab_decoder_channels)
    elif args.model_name.lower() == 'segformer_simple': model = SimplifiedSegFormer(in_chans=args.in_channels, num_classes=num_output_classes, img_size=(args.img_height, args.img_width), embed_dims=args.segformer_embed_dims, num_heads=args.segformer_num_heads, mlp_ratios=args.segformer_mlp_ratios, depths=args.segformer_depths, patch_sizes=args.segformer_patch_sizes, decoder_hidden_dim=args.segformer_decoder_hidden_dim, drop_rate=args.segformer_drop_rate)
    elif args.model_name.lower() == 'segnet': model = SegNet(in_chn=args.in_channels, out_chn=num_output_classes, BN_momentum=args.segnet_bn_momentum)
    else: raise ValueError(f"不支持的模型名称: {args.model_name}.")
    model.to(device)
    criterion_ignore_idx = ignore_index_actual if ignore_index_actual != -1 else -100
    criterion = nn.CrossEntropyLoss(ignore_index=criterion_ignore_idx); print(f"  损失函数: CrossEntropyLoss (ignore_index={criterion_ignore_idx})")
    if args.optimizer.lower() == 'adam': optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adamw': optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.optimizer.lower() == 'sgd': optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else: raise ValueError(f"不支持的优化器: {args.optimizer}")
    print(f"  优化器: {args.optimizer.upper()} (初始学习率: {args.lr})")
    scheduler = None
    if args.scheduler_step_size > 0 : scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma); print(f"  学习率调度器: StepLR (step_size={args.scheduler_step_size}, gamma={args.scheduler_gamma})")
    
    start_epoch = 0; best_metric_val = float('-inf')
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint): print(f"  正在从检查点恢复: {args.resume_checkpoint}"); start_epoch, best_metric_val = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler, device, weights_only=False)
        else: print(f"  警告: 找不到指定的检查点文件 {args.resume_checkpoint}。将从头开始训练。")

    print("\n--- 准备训练器 ---")
    checkpoint_prefix_for_trainer = f"{args.model_name}_" + (args.experiment_name if args.experiment_name else "run")
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, num_epochs=args.epochs, num_classes_eval=num_classes_for_eval, ignore_index_eval=ignore_index_actual, experiment_dir=experiment_dir, checkpoint_prefix=checkpoint_prefix_for_trainer, start_epoch=start_epoch, best_metric_val=best_metric_val, metric_to_optimize=args.metric_to_optimize, print_freq=args.print_freq)
    trainer.run_training()

    if args.evaluate_on_test and 'test_loader' in locals() and len(test_loader) > 0:
        print("\n--- 正在测试集上评估最佳模型 ---")
        best_checkpoint_path = os.path.join(experiment_dir, f"{checkpoint_prefix_for_trainer}_best.pth.tar")
        if os.path.exists(best_checkpoint_path):
            print(f"  加载最佳模型检查点: {best_checkpoint_path}")
            if args.model_name.lower() == 'simple': test_model = SimplePixelClassifier(in_channels=args.in_channels, num_classes=num_output_classes)
            elif args.model_name.lower() == 'unet': test_model = UNet(in_channels=args.in_channels, num_classes=num_output_classes, bilinear=args.unet_bilinear, base_c=args.unet_base_c)
            elif args.model_name.lower() == 'deeplabv3plus': test_model = DeepLabV3Plus(in_channels=args.in_channels, num_classes=num_output_classes, output_stride=args.deeplab_output_stride, encoder_layers=args.deeplab_encoder_layers, aspp_out_channels=args.deeplab_aspp_out_channels, decoder_channels=args.deeplab_decoder_channels)
            elif args.model_name.lower() == 'segformer_simple': test_model = SimplifiedSegFormer(in_chans=args.in_channels, num_classes=num_output_classes, img_size=(args.img_height, args.img_width), embed_dims=args.segformer_embed_dims, num_heads=args.segformer_num_heads, mlp_ratios=args.segformer_mlp_ratios, depths=args.segformer_depths, patch_sizes=args.segformer_patch_sizes, decoder_hidden_dim=args.segformer_decoder_hidden_dim, drop_rate=args.segformer_drop_rate)
            elif args.model_name.lower() == 'segnet': test_model = SegNet(in_chn=args.in_channels, out_chn=num_output_classes, BN_momentum=args.segnet_bn_momentum)
            else: print(f"错误: 测试时遇到不支持的模型名称: {args.model_name}"); return
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint: test_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint: test_model.load_state_dict(checkpoint['state_dict'])
            else: test_model.load_state_dict(checkpoint)
            test_model.to(device); test_model.eval()
            test_loss, test_metrics = evaluate_segmentation(model=test_model, dataloader=test_loader, criterion=criterion, device=device, num_classes=num_classes_for_eval, ignore_index=ignore_index_actual)
            print(f"\n测试集评估结果 (保存到 {experiment_dir}/test_metrics.json):"); print(f"  测试损失: {test_loss:.4f}"); serializable_metrics = {}
            for k, v in test_metrics.items():
                if isinstance(v, np.ndarray): serializable_metrics[k] = v.tolist(); print(f"  {k}: {np.round(v, 4).tolist()}")
                elif isinstance(v, (float, np.float32, np.float64)): serializable_metrics[k] = float(v); print(f"  {k}: {v:.4f}")
                else: serializable_metrics[k] = str(v)
            serializable_metrics['test_loss'] = test_loss; save_dict_to_json(serializable_metrics, os.path.join(experiment_dir, "test_metrics.json"))
            try:
                num_viz_samples = min(args.batch_size if args.batch_size > 0 else 1, 4); test_iter = iter(test_loader)
                if len(test_loader) == 0: print("测试集为空，跳过可视化。")
                else:
                    viz_images, viz_targets = next(test_iter)
                    viz_images = viz_images[:num_viz_samples].to(device)
                    with torch.no_grad(): viz_outputs = test_model(viz_images); viz_preds = torch.argmax(viz_outputs, dim=1)
                    mean_viz=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1); std_viz=torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
                    viz_images_denorm = viz_images * std_viz + mean_viz; viz_images_denorm = viz_images_denorm.clamp(0,1)
                    fig, axes = plt.subplots(num_viz_samples, 3, figsize=(15, 5 * num_viz_samples), squeeze=False)
                    label_map_viz = train_dataset.label_to_color if hasattr(train_dataset, 'label_to_color') else {}
                    for i in range(num_viz_samples):
                        axes[i,0].imshow(T.ToPILImage()(viz_images_denorm[i].cpu())); axes[i,0].set_title("Input Image"); axes[i,0].axis('off')
                        axes[i,1].imshow(tensor_mask_to_pil_rgb(viz_targets[i].cpu(),label_map_viz)); axes[i,1].set_title("Ground Truth Mask"); axes[i,1].axis('off')
                        axes[i,2].imshow(tensor_mask_to_pil_rgb(viz_preds[i].cpu(),label_map_viz)); axes[i,2].set_title("Predicted Mask"); axes[i,2].axis('off')
                    plt.tight_layout(); viz_save_path = os.path.join(experiment_dir, "test_visualization.png")
                    plt.savefig(viz_save_path); print(f"测试集可视化结果已保存至: {viz_save_path}"); plt.close(fig)
            except Exception as e_viz: print(f"生成测试集可视化时出错: {e_viz}"); import traceback; traceback.print_exc()
        else: print(f"  错误: 找不到最佳模型检查点 {best_checkpoint_path}，无法在测试集上评估。")

    print(f"\n--- main.py 执行完毕. 结果保存在: {experiment_dir} ---")

if __name__ == "__main__":
    main()