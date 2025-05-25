# ==============================================================================
# 模块导入 (IMPORTS)
# ==============================================================================
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF # 用于同步变换
import os
import random
from PIL import Image # 用于类型提示和检查
import matplotlib.pyplot as plt

# ==============================================================================
# 全局常量 (GLOBAL CONSTANTS)
# ==============================================================================
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]

# ==============================================================================
# 同步变换类 (SYNCHRONIZED TRANSFORMS CLASS)
# ==============================================================================
class SynchronizedTransform:
    """
    对图像和掩码应用同步的几何变换，以及各自的非几何变换。
    设计用于 torchvision.datasets.VOCSegmentation 的 'transforms' (复数) 参数。
    """
    def __init__(self, image_size, is_train=True):
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.is_train = is_train

        # 图像的非几何变换
        self.image_normalization = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 掩码的非几何变换 (P模式PIL转Tensor)
        self.mask_to_tensor = transforms.ToTensor()

    def __call__(self, image: Image.Image, mask: Image.Image):
        # --- 1. 同步几何变换 (应用于 PIL Image) ---
        # 统一缩放 (对于验证/测试集，或训练集增强前)
        if not self.is_train: # 验证或测试
            image = TF.resize(image, self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.image_size, interpolation=transforms.InterpolationMode.NEAREST)
        else: # 训练时的数据增强
            # 随机缩放裁剪
            # get_params 方法用于生成随机参数，然后分别应用于图像和掩码
            scale_range = (0.5, 2.0) # 缩放范围
            ratio_range = (3. / 4., 4. / 3.) # 高宽比范围
            i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale_range, ratio=ratio_range)
            image = TF.resized_crop(image, i, j, h, w, self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, self.image_size, interpolation=transforms.InterpolationMode.NEAREST)

            # 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        
        # --- 2. 图像特有的非几何变换 (应用于 PIL Image) ---
        if self.is_train:
            # 随机颜色抖动 (仅对图像)
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
            image = color_jitter(image)

        # --- 3. 转换为 Tensor 并进行归一化 ---
        image_tensor = self.image_normalization(image)
        
        # 对于P模式的PIL掩码，ToTensor会将其转换为 (1, H, W) 的 LongTensor，值保持原始类别索引
        # 如果掩码是L模式，ToTensor会缩放到[0,1]，这不是我们想要的。VOC的掩码是P模式。
        mask_tensor = self.mask_to_tensor(mask) # 输出 (1, H, W), dtype=torch.float32, 但值是整数索引

        return image_tensor, mask_tensor


# ==============================================================================
# 数据加载器函数 (DATALOADER FUNCTION)
# ==============================================================================
def get_voc_dataloaders(data_root, batch_size, image_size=(256, 256), num_workers=4):
    """
    获取 PASCAL VOC 2012 数据集的 DataLoader，包含训练时的数据增强。

    参数:
        data_root (str): 数据集存放的根目录。
        batch_size (int): 批处理大小。
        image_size (tuple or int): 图像和掩码统一调整到的大小 (height, width)。
        num_workers (int): 数据加载的子进程数量。

    返回:
        train_loader (DataLoader): 训练集 DataLoader。
        val_loader (DataLoader): 验证集 DataLoader。
        num_classes (int): 类别数量 (包括背景)。
    """
    # --- 为训练集和验证集定义同步变换 ---
    train_transforms = SynchronizedTransform(image_size, is_train=True)
    val_transforms = SynchronizedTransform(image_size, is_train=False)

    # --- 创建训练集 ---
    # download=True 如果数据集不存在，会自动下载
    # 'transforms' (复数) 参数接收一个可调用对象，该对象处理 (image, target) 对
    train_dataset = datasets.VOCSegmentation(
        root=data_root,
        year='2012',
        image_set='train',
        download=True,
        transforms=train_transforms # 使用同步变换
    )

    # --- 创建验证集 ---
    val_dataset = datasets.VOCSegmentation(
        root=data_root,
        year='2012',
        image_set='val',
        download=True,
        transforms=val_transforms # 使用同步变换
    )

    # --- 创建 DataLoader ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # 如果GPU可用，这可以加速数据传输
        drop_last=True   # 训练时可以丢弃最后一个不完整的批次
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, # 验证时batch_size可以设大一些如果显存允许
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(VOC_CLASSES) # 21 类

    print(f"数据集信息:")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    print(f"  类别数量: {num_classes}")

    return train_loader, val_loader, num_classes


# ==============================================================================
# 主执行块 (测试用) (MAIN EXECUTION BLOCK - FOR TESTING)
# ==============================================================================
if __name__ == '__main__':
    DATA_ROOT = './data_voc' # 测试用的数据路径
    BATCH_SIZE = 2
    IMAGE_SIZE = (256, 256)

    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    print("开始测试数据集加载...")
    train_loader_test, val_loader_test, num_classes_test = get_voc_dataloaders(
        DATA_ROOT, BATCH_SIZE, IMAGE_SIZE, num_workers=0 # 测试时num_workers设为0方便调试
    )

    print("\n测试训练数据加载器:")
    images_train, masks_train = next(iter(train_loader_test))
    print(f"  图像批次形状: {images_train.shape}") # 期望 (B, 3, H, W)
    print(f"  掩码批次形状: {masks_train.shape}")   # 期望 (B, 1, H, W) from ToTensor on P-mode PIL
    print(f"  图像数据类型: {images_train.dtype}")
    print(f"  掩码数据类型: {masks_train.dtype}")   # 应该是 torch.float32，但值是整数
    
    # 掩码处理： squeeze(1) 移除通道维度，.long() 转换为长整型用于损失计算
    masks_train_for_loss = masks_train.squeeze(1).long()
    print(f"  用于损失计算的训练掩码形状: {masks_train_for_loss.shape}") # 期望 (B, H, W)
    print(f"  用于损失计算的训练掩码数据类型: {masks_train_for_loss.dtype}") # 期望 torch.int64
    unique_train_mask_values = torch.unique(masks_train_for_loss)
    print(f"  训练掩码中的唯一值: {unique_train_mask_values}")
    if 255 in unique_train_mask_values:
        print("  训练掩码中包含255 (忽略标签)。")


    print("\n测试验证数据加载器:")
    images_val, masks_val = next(iter(val_loader_test))
    print(f"  图像批次形状: {images_val.shape}")
    print(f"  掩码批次形状: {masks_val.shape}")
    masks_val_for_loss = masks_val.squeeze(1).long()
    unique_val_mask_values = torch.unique(masks_val_for_loss)
    print(f"  验证掩码中的唯一值: {unique_val_mask_values}")


    print("\n数据集加载测试完成。")

    sample_img_tensor = images_train[0]
    sample_mask_tensor = masks_train_for_loss[0]
    inv_normalize = transforms.Normalize(
       mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
       std=[1/0.229, 1/0.224, 1/0.225]
    )
    sample_img_pil = transforms.ToPILImage()(inv_normalize(sample_img_tensor.cpu()))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_img_pil)
    plt.title("增强后的图像样本")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(sample_mask_tensor.cpu().numpy(), cmap='tab20', vmin=0, vmax=20) # tab20 有20种颜色
    plt.title("增强后的掩码样本")
    plt.axis('off')
    plt.show()