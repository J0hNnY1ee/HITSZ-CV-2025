# dataset.py

import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import config # 导入配置

def get_class_rgb_values():
    """
    从class_dict.csv读取类别名称和RGB值.
    Returns:
        tuple: (rgb_values, class_names)
               rgb_values (list): 包含每个类别RGB值 [r,g,b] 的列表.
               class_names (list): 包含每个类别名称的列表.
    """
    try:
        class_df = pd.read_csv(config.CLASS_CSV_PATH)
        # 确保r, g, b列是整数类型
        class_df[['r', 'g', 'b']] = class_df[['r', 'g', 'b']].astype(int)
        rgb_values = class_df[['r', 'g', 'b']].values.tolist()
        class_names = class_df['name'].tolist()
        return rgb_values, class_names
    except FileNotFoundError:
        print(f"错误: 类别定义文件 {config.CLASS_CSV_PATH} 未找到!")
        return [], [] # 返回空列表以避免后续错误，但应在主脚本中处理
    except Exception as e:
        print(f"读取类别定义文件时发生错误: {e}")
        return [], []


class CamVidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_rgb_values=None, transform=None, mask_transform=None):
        """
        Args:
            image_dir (string): 图像文件目录路径.
            mask_dir (string): 标签掩码文件目录路径.
            class_rgb_values (list): 包含每个类别RGB值的列表. 如果为None, 会尝试从文件加载.
            transform (callable, optional): 应用于样本图像的可选转换.
            mask_transform (callable, optional): 应用于样本掩码的RGB PIL图像的可选尺寸调整转换.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform # 用于调整RGB掩码的PIL Image

        if class_rgb_values is None:
            print("警告: CamVidDataset 未提供 class_rgb_values, 尝试从文件加载...")
            loaded_rgb_values, _ = get_class_rgb_values()
            if not loaded_rgb_values:
                raise ValueError("无法加载类别RGB值，且未在初始化时提供。")
            self.class_rgb_values = loaded_rgb_values
        else:
            self.class_rgb_values = class_rgb_values

        self.images = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
        # 假设掩码文件名与图像文件名相同 (或可以通过某种方式匹配)
        # 为简单起见，我们这里直接使用与图像列表相同的排序，但实际应用中需要确保它们一一对应
        self.masks_filenames = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])

        if len(self.images) != len(self.masks_filenames):
            print(f"警告: 图像数量 ({len(self.images)}) 与掩码数量 ({len(self.masks_filenames)}) 不匹配。")
            # 你可能需要更复杂的逻辑来匹配图像和掩码
            # 例如，如果文件名相似但有不同后缀

        # 创建从RGB到类别索引的映射
        self.rgb_to_idx = {tuple(rgb): idx for idx, rgb in enumerate(self.class_rgb_values)}

    def __len__(self):
        return len(self.images) # 以图像数量为准

    def _rgb_to_idx_numpy(self, rgb_mask_pil):
        """将RGB PIL Image掩码转换为类别索引的numpy数组"""
        rgb_mask_np = np.array(rgb_mask_pil) # (H, W, C)
        idx_mask_np = np.zeros((rgb_mask_np.shape[0], rgb_mask_np.shape[1]), dtype=np.uint8) # uint8因为类别数 < 256

        for rgb_val, class_idx in self.rgb_to_idx.items():
            matches = np.all(rgb_mask_np == np.array(rgb_val), axis=-1)
            idx_mask_np[matches] = class_idx
        return idx_mask_np

    def __getitem__(self, idx):
        if idx >= len(self.images) or idx >= len(self.masks_filenames):
             print(f"索引 {idx} 超出范围。图像数: {len(self.images)}, 掩码文件名数: {len(self.masks_filenames)}")
             return None, None

        img_name = os.path.join(self.image_dir, self.images[idx])
        # 假设掩码文件名与图像文件名相同，或者可以通过images[idx]推断出来
        # 例如，如果掩码文件名是 image_name_mask.png，而图像是 image_name.png
        # 这里简单假设它们按顺序一一对应，且文件名相同（常见于CamVid）
        mask_filename = self.images[idx] # 直接使用图像文件名，因为CamVid通常是这样
        mask_path = os.path.join(self.mask_dir, mask_filename[:-4] + '_L.png') 

        try:
            image = Image.open(img_name).convert("RGB")
            mask_pil_rgb = Image.open(mask_path).convert("RGB")
        except FileNotFoundError:
            print(f"错误: 文件未找到。图像: {img_name} 或 掩码: {mask_path}")
            return None, None # Dataloader collate_fn 需要处理这个

        # 1. 对原始RGB掩码PIL图像应用尺寸变换 (如果定义了)
        if self.mask_transform:
            mask_pil_rgb_resized = self.mask_transform(mask_pil_rgb)
        else:
            mask_pil_rgb_resized = mask_pil_rgb # 如果没有变换，直接使用

        # 2. 将调整尺寸后的RGB PIL掩码转换为 NumPy 索引掩码
        idx_mask_numpy = self._rgb_to_idx_numpy(mask_pil_rgb_resized) # (H_resized, W_resized) dtype=uint8

        # 3. 将NumPy索引掩码转换为LongTensor (torch.int64)
        idx_mask_tensor = torch.from_numpy(idx_mask_numpy).long()

        # 4. 对原始图像应用变换
        if self.transform:
            image_tensor = self.transform(image)
        else: # 如果没有图像变换，至少需要转为Tensor
            image_tensor = T.ToTensor()(image)


        return image_tensor, idx_mask_tensor


# 定义数据转换
image_transforms = T.Compose([
    T.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
    T.ToTensor(), # 将PIL Image [0,255] 转为 FloatTensor [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 均值和标准差
])

# 对于掩码：只需要调整大小。Resize的插值方法对掩码很重要，通常用NEAREST
# 这个变换作用于原始的RGB掩码PIL图像，返回调整大小后的RGB掩码PIL图像
mask_transforms_for_resize = T.Compose([
    T.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH), interpolation=T.InterpolationMode.NEAREST),
])


if __name__ == '__main__':
    print("测试 CamVidDataset 和 get_class_rgb_values...")
    
    # 测试 get_class_rgb_values
    rgb_values_test, class_names_test = get_class_rgb_values()
    if not rgb_values_test:
        print("无法加载类别信息，测试中止。")
    else:
        print(f"共 {len(class_names_test)} 个类别: {class_names_test[:5]}...") # 打印前5个
        print(f"RGB值示例 (第一个类别): {rgb_values_test[0] if rgb_values_test else 'N/A'}")

        # 测试 CamVidDataset
        print("\n初始化 CamVidDataset...")
        # 确保 DATA_DIR/train 和 DATA_DIR/train_labels 存在且有内容
        # 为了测试，可以创建一个小的子集
        train_image_dir = os.path.join(config.DATA_DIR, 'train')
        train_mask_dir = os.path.join(config.DATA_DIR, 'train_labels')

        if not os.path.exists(train_image_dir) or not os.path.exists(train_mask_dir):
            print(f"错误: 训练数据目录 {train_image_dir} 或 {train_mask_dir} 未找到。请检查 config.DATA_DIR。")
        else:
            train_dataset = CamVidDataset(
                image_dir=train_image_dir,
                mask_dir=train_mask_dir,
                class_rgb_values=rgb_values_test, # 使用上面加载的
                transform=image_transforms,
                mask_transform=mask_transforms_for_resize
            )

            if len(train_dataset) > 0:
                print(f"数据集大小: {len(train_dataset)}")
                # 尝试获取第一个有效样本
                sample_idx = 0
                item = None
                # 循环查找有效样本，因为某些文件可能丢失
                while sample_idx < len(train_dataset):
                    item = train_dataset[sample_idx]
                    if item is not None and item[0] is not None:
                        break
                    sample_idx += 1
                
                if item is not None and item[0] is not None:
                    img, mask = item
                    print(f"成功获取样本 {sample_idx}:")
                    print(f"  图像尺寸: {img.shape}, 类型: {img.dtype}")
                    print(f"  掩码尺寸: {mask.shape}, 类型: {mask.dtype}")
                    print(f"  掩码唯一值: {torch.unique(mask)}")
                else:
                    print("无法从数据集中获取有效样本进行测试。检查文件是否存在或是否所有文件都已正确处理。")
            else:
                print("训练数据集为空或无法加载。")