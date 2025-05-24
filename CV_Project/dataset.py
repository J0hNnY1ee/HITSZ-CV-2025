# dataset.py

import torch
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os

# --------------------------------------------------------------------------------
# 0. 定义一些常量和辅助数据
# --------------------------------------------------------------------------------

NUM_CLASSES = 21  # 20 物体类别 + 1 背景
IMG_WIDTH = 256
IMG_HEIGHT = 256

# (R, G, B) for PASCAL VOC
VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
    [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
    [0, 192, 0], [128, 192, 0], [0, 64, 128]
]
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "potted plant", "sheep", "sofa", "train",
    "tv/monitor"
]

# ImageNet 均值和标准差
normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]

# --------------------------------------------------------------------------------
# 1. 定义数据转换
# --------------------------------------------------------------------------------

def get_transform_image(img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    return T.Compose([
        T.Resize((img_height, img_width)),
        T.ToTensor(),
        T.Normalize(mean=normalize_mean, std=normalize_std)
    ])

def get_transform_mask(img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    return T.Compose([
        T.Resize((img_height, img_width), interpolation=T.InterpolationMode.NEAREST),
        T.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
    ])

# --------------------------------------------------------------------------------
# 2. 自定义 Dataset 类
# --------------------------------------------------------------------------------

class PascalVOCDataset(Dataset):
    def __init__(self, root, image_set='train', download=False, 
                 img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                 transform_img=None, transform_msk=None):
        self.voc_dataset = VOCSegmentation(root=root, year='2012', image_set=image_set, download=download)
        
        self.transform_img = transform_img if transform_img else get_transform_image(img_height, img_width)
        self.transform_msk = transform_msk if transform_msk else get_transform_mask(img_height, img_width)
        
        # 进一步处理掩码中的边界值 (255)
        # PASCAL VOC 掩码中的边界像素值为255。
        # 我们通常将其视为背景(0)或在损失函数中通过 ignore_index 忽略。
        # 这里我们选择将其映射到0 (背景类别)，这样我们的类别索引就是 0 到 NUM_CLASSES-1。
        # 如果损失函数设置了 ignore_index=255，则无需此步骤。
        # 但为了统一，这里处理掉。
        self.map_255_to_0 = True 
        # 或者，如果你想在损失函数中忽略255 (例如，如果某些模型或损失函数默认处理255)
        # self.ignore_index = 255 
        # self.map_255_to_0 = False

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        img, mask_pil = self.voc_dataset[idx] # img 和 mask_pil 都是 PIL Image

        img_tensor = self.transform_img(img)
        mask_tensor = self.transform_msk(mask_pil)
            
        if self.map_255_to_0:
            mask_tensor[mask_tensor == 255] = 0 # 将边界 (255) 映射到背景 (0)
            
        return img_tensor, mask_tensor

# --------------------------------------------------------------------------------
# 3. 获取 DataLoaders 的辅助函数
# --------------------------------------------------------------------------------

def get_dataloaders(data_root='./data', batch_size=8, num_workers=2, 
                    img_height=IMG_HEIGHT, img_width=IMG_WIDTH, download_data=True):
    """
    准备 PASCAL VOC 2012 的训练和验证 DataLoader。
    """
    os.makedirs(data_root, exist_ok=True)

    try:
        train_dataset = PascalVOCDataset(root=data_root, image_set='train', download=download_data,
                                         img_height=img_height, img_width=img_width)
        val_dataset = PascalVOCDataset(root=data_root, image_set='val', download=download_data,
                                       img_height=img_height, img_width=img_width)
        
        print(f"Successfully loaded datasets.")
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
        
        print(f"DataLoaders created with batch size: {batch_size}")
        return train_loader, val_loader

    except RuntimeError as e:
        print(f"Error loading dataset or creating DataLoaders: {e}")
        print("Please ensure you have a stable internet connection or the dataset is already downloaded and correctly placed.")
        print("If you are on a restricted network, you might need to download the dataset manually from:")
        print("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")
        print("And place it in the 'data' directory (or the specified data_root), then extract it.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

# --------------------------------------------------------------------------------
# 4. 可视化辅助函数 (可以保留在这里，方便在其他地方调用)
# --------------------------------------------------------------------------------
def denormalize(tensor, mean=normalize_mean, std=normalize_std):
    # 确保 mean 和 std 是 tensor 并且形状正确
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std)
    
    # 调整形状以进行广播
    if tensor.ndim == 4: # Batch of images [B, C, H, W]
        mean = mean.view(1, -1, 1, 1).to(tensor.device)
        std = std.view(1, -1, 1, 1).to(tensor.device)
    elif tensor.ndim == 3: # Single image [C, H, W]
        mean = mean.view(-1, 1, 1).to(tensor.device)
        std = std.view(-1, 1, 1).to(tensor.device)
    else:
        raise ValueError("Input tensor must have 3 or 4 dimensions")
        
    return tensor * std + mean

def tensor_to_pil(tensor_image, mean=normalize_mean, std=normalize_std):
    """将归一化后的图像Tensor转换为PIL Image (用于显示)"""
    if tensor_image.ndim == 4: # 如果是 [B, C, H, W]，取第一张
        tensor_image = tensor_image[0]
    
    denorm_image = denormalize(tensor_image.cpu(), mean, std)
    pil_image = T.ToPILImage()(denorm_image.clamp(0, 1)) # clamp确保值在[0,1]
    return pil_image

def mask_to_pil_color(mask_tensor, colormap=VOC_COLORMAP):
    """将类别索引的mask Tensor转换为彩色的PIL Image"""
    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1: # [1, H, W] -> [H, W]
        mask_tensor = mask_tensor.squeeze(0)
    
    if mask_tensor.ndim != 2:
        raise ValueError(f"Mask tensor must be 2D (H, W) or 3D (1, H, W), got {mask_tensor.shape}")

    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    
    for r in range(mask_np.shape[0]):
        for c in range(mask_np.shape[1]):
            class_idx = mask_np[r, c]
            if class_idx < len(colormap):
                colored_mask[r, c, :] = colormap[class_idx]
            # elif class_idx == 255 and 0 < len(colormap): # 如果255被映射到了背景
            #     colored_mask[r, c, :] = colormap[0] 
    
    return Image.fromarray(colored_mask)


# --------------------------------------------------------------------------------
# 5. (可选) 主程序块，用于测试 dataset.py 是否能独立运行并加载数据
# --------------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("Testing dataset.py...")
    
    # 测试获取 dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=4, download_data=True)

    if train_loader and val_loader:
        print("\nVisualizing a sample from training data using matplotlib...")
        try:
            sample_images, sample_masks = next(iter(train_loader))
            
            idx_to_show = 0
            img_to_show_pil = tensor_to_pil(sample_images[idx_to_show])
            mask_to_show_pil = mask_to_pil_color(sample_masks[idx_to_show])

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_to_show_pil)
            axes[0].set_title(f"Sample Image (Index {idx_to_show})")
            axes[0].axis('off')

            axes[1].imshow(mask_to_show_pil)
            axes[1].set_title(f"Sample Mask (Index {idx_to_show})")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()

            print(f"Image tensor shape: {sample_images[idx_to_show].shape}")
            print(f"Mask tensor shape: {sample_masks[idx_to_show].shape}")
            print(f"Mask tensor dtype: {sample_masks[idx_to_show].dtype}")
            print(f"Unique values in displayed mask: {torch.unique(sample_masks[idx_to_show])}")

        except Exception as e:
            print(f"Error during visualization test: {e}")
    else:
        print("Could not test visualization as DataLoaders were not created.")