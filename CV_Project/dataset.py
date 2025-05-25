# dataset.py

import os
import csv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# ==============================================================================
# 全局常量定义 (这些将作为无法从 class_dict.csv 加载时的后备)
# 这些默认值现在的重要性降低了，因为我们会优先使用 class_dict.csv
# ==============================================================================
DEFAULT_CAMVID_CLASS_NAMES = [
    'Sky', 'Building', 'Pole', 'Road', 'Pavement',
    'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist', 'Void' # 更改为Void以匹配常见情况
]
DEFAULT_CAMVID_PALETTE = {
    (128, 128, 128): 0, (128, 0, 0): 1, (192, 192, 128): 2, (128, 64, 128): 3,
    (60, 40, 222): 4, (128, 128, 0): 5, (192, 128, 128): 6, (64, 64, 128): 7,
    (64, 0, 128): 8, (64, 64, 0): 9, (0, 128, 192): 10, (0, 0, 0): 11
}
DEFAULT_CAMVID_LABEL_TO_COLOR = {v: k for k, v in DEFAULT_CAMVID_PALETTE.items()}
DEFAULT_NUM_TRAIN_CLASSES = 11 # 默认可训练类别数 (Sky to Bicyclist)
DEFAULT_IGNORE_INDEX = 11    # 默认忽略索引 (Void)

# ==============================================================================
# CamVidDataset 类定义
# ==============================================================================
class CamVidDataset(Dataset):
    """
    CamVid 语义分割数据集类。
    能够从 class_dict.csv 文件动态加载类别定义。

    假设数据集结构如下:
    root_dir/
    ├── train/                  # 训练集原始图像
    │   ├── 0001TP_006690.png
    │   └── ...
    ├── train_labels/           # 训练集标签图像 (RGB格式)
    │   ├── 0001TP_006690_L.png
    │   └── ...
    ├── val/
    ├── val_labels/
    ├── test/
    ├── test_labels/
    └── class_dict.csv          # 类别定义文件 (name,r,g,b)
    """

    def __init__(self,
                 root_dir: str,
                 image_set: str = 'train',
                 transform=None,
                 target_transform=None, # 主要用于尺寸调整PIL Label
                 img_suffix: str = ".png",
                 label_suffix_from_img: str = "_L.png", # 标签文件名相对于图像文件名的后缀
                 class_dict_filename: str = "class_dict.csv", # CSV文件名
                 void_class_names: list = None): # 用于指定哪些类名算作void/ignore
        """
        初始化 CamVidDataset。

        参数:
            root_dir (str): 数据集的根目录 (例如 "CamVid/")。
            image_set (str): 'train', 'val', 或 'test'，指定加载哪个数据集分割。
            transform (callable, optional): 应用于输入图像的转换 (例如 ToTensor, Normalize)。
            target_transform (callable, optional): 应用于PIL标签图像的转换 (通常是Resize)。
                                                   注意：最终目标掩码应为 LongTensor 类型。
            img_suffix (str): 原始图像文件的后缀 (例如 ".png")。
            label_suffix_from_img (str): 标签文件名是在图像文件名(去除img_suffix后)基础上添加的后缀。
            class_dict_filename (str): 类别定义CSV文件的名称 (应位于`root_dir`下)。
            void_class_names (list, optional): 一个包含应被视为"忽略"或"Void"的类别名称的列表 (不区分大小写)。
                                             例如: ['Void', 'Unlabelled', 'Background']。
                                             如果为None，则默认为 ['void', 'unlabelled', 'ignore', 'background']。
        """
        if image_set not in ['train', 'val', 'test']:
            raise ValueError(f"参数 'image_set' 必须是 'train', 'val', 或 'test'之一, 但得到了 '{image_set}'")

        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        # --- 确定图像和标签文件夹路径 ---
        self.img_folder = os.path.join(root_dir, image_set)
        self.label_folder = os.path.join(root_dir, f"{image_set}_labels")

        self.img_suffix = img_suffix
        self.label_suffix_from_img = label_suffix_from_img

        if void_class_names is None:
            self.void_class_names = ['void', 'unlabelled', 'ignore', 'background']
        else:
            # 确保 void_class_names 中的名称是小写的，以便进行不区分大小写的比较
            self.void_class_names = [name.lower() for name in void_class_names]


        if not os.path.isdir(self.img_folder):
            raise FileNotFoundError(f"图像文件夹未找到: {self.img_folder}")
        if not os.path.isdir(self.label_folder):
            raise FileNotFoundError(f"标签文件夹未找到: {self.label_folder}")

        # --- 加载类别定义 ---
        # self.num_total_classes: CSV中定义的类别总数
        # self.num_train_classes: 实际用于训练的类别数 (通常是 total - 1 if ignore_index is found)
        # self.ignore_index: 被识别为忽略类的索引，如果未找到则为 -1
        self.class_names, self.palette, self.label_to_color, \
        self.num_total_classes, self.num_train_classes, self.ignore_index = \
            self._load_class_definitions(root_dir, class_dict_filename)

        # --- 加载图像文件名 ---
        # 获取图像文件夹下所有以img_suffix结尾的文件
        self.image_filenames = sorted([
            f for f in os.listdir(self.img_folder) if f.endswith(self.img_suffix)
        ])

        if not self.image_filenames:
            raise RuntimeError(f"在 {self.img_folder} 中没有找到任何图像文件。")

        print(f"成功初始化 CamVidDataset ({image_set}集):")
        print(f"  图像文件夹: {self.img_folder}")
        print(f"  标签文件夹: {self.label_folder}")
        print(f"  图像数量: {len(self.image_filenames)}")
        print(f"  总类别数 (从CSV): {self.num_total_classes}")
        print(f"  可训练类别数: {self.num_train_classes}")
        ignore_class_name = "N/A"
        if self.ignore_index != -1 and self.ignore_index < len(self.class_names):
            ignore_class_name = f"'{self.class_names[self.ignore_index]}'"
        print(f"  忽略索引: {self.ignore_index} (对应类别: {ignore_class_name})")
        # print(f"  所有类别名称: {self.class_names}") # 可选打印


    def _load_class_definitions(self, root_dir, class_dict_filename):
        """ 从 class_dict.csv 加载类别名称、调色板等信息，如果失败则使用默认值。 """
        csv_path = os.path.join(root_dir, class_dict_filename)
        
        loaded_class_names = []
        loaded_palette = {}
        determined_ignore_idx = -1 # 默认为没有特定忽略索引
        num_total_cls_from_csv = 0
        
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f: # utf-8-sig to handle BOM
                    reader = csv.DictReader(f)
                    
                    # 动态获取列名，并转换为小写进行不区分大小写的匹配
                    fieldnames_lower = [field.lower() for field in reader.fieldnames if field]
                    if not all(col in fieldnames_lower for col in ['name', 'r', 'g', 'b']):
                         print(f"警告: {csv_path} 文件缺少 'name', 'r', 'g', 'b' 中的某些列 (不区分大小写)。")
                         print(f"       找到的列: {reader.fieldnames}。将尝试使用默认类别定义。")
                         raise ValueError("CSV文件列不匹配")

                    # 找到实际的列名 (保持原始大小写以便访问row)
                    name_col = reader.fieldnames[fieldnames_lower.index('name')]
                    r_col = reader.fieldnames[fieldnames_lower.index('r')]
                    g_col = reader.fieldnames[fieldnames_lower.index('g')]
                    b_col = reader.fieldnames[fieldnames_lower.index('b')]

                    for idx, row in enumerate(reader):
                        name_val = row[name_col].strip()
                        try:
                            r_val, g_val, b_val = int(row[r_col]), int(row[g_col]), int(row[b_col])
                        except ValueError:
                            print(f"警告: CSV文件 {csv_path} 行 {idx+2} 颜色值无法转换为整数: {row}. 跳过此行.")
                            continue
                        except TypeError: # 如果颜色值是None或空字符串
                            print(f"警告: CSV文件 {csv_path} 行 {idx+2} 颜色值为空: {row}. 跳过此行.")
                            continue


                        loaded_class_names.append(name_val)
                        loaded_palette[(r_val, g_val, b_val)] = idx # 类别索引基于CSV中的行顺序
                        num_total_cls_from_csv += 1

                        # 检查是否为void/ignore class
                        if name_val.lower() in self.void_class_names:
                            if determined_ignore_idx != -1:
                                print(f"警告: 在 {csv_path} 中找到多个可能的忽略类别 ('{loaded_class_names[determined_ignore_idx]}' 和 '{name_val}') 基于 void_class_names。将使用第一个找到的。")
                            else:
                                determined_ignore_idx = idx
                                # print(f"  找到忽略类别: '{name_val}' (索引: {idx})")


                if not loaded_class_names: # 如果CSV为空或所有行都有问题
                    raise ValueError("CSV文件中没有成功加载任何类别定义。")

                loaded_label_to_color = {v: k for k, v in loaded_palette.items()}
                
                if determined_ignore_idx != -1:
                    num_train_cls_final = num_total_cls_from_csv - 1
                else:
                    print(f"警告: 在 {csv_path} 中未根据提供的 void_class_names ('{self.void_class_names}') 找到明确的忽略类别。")
                    print(f"       将假设所有 {num_total_cls_from_csv} 个类别都可训练，并将 ignore_index 设置为 -1。")
                    num_train_cls_final = num_total_cls_from_csv
                    # determined_ignore_idx 保持为 -1

                print(f"成功从 {csv_path} 加载 {num_total_cls_from_csv} 个类别定义。")
                return (loaded_class_names, loaded_palette, loaded_label_to_color, 
                        num_total_cls_from_csv, num_train_cls_final, determined_ignore_idx)

            except Exception as e:
                print(f"从 {csv_path} 加载类别定义失败: {e}。将尝试使用内置的默认值。")
        
        # 如果加载失败或文件不存在，使用默认值
        print("警告: 未找到或无法解析 class_dict.csv，将使用内置的默认CamVid类别和调色板。")
        print("       这可能与您的数据集不完全匹配，请检查 class_dict.csv 文件。")
        return (DEFAULT_CAMVID_CLASS_NAMES, DEFAULT_CAMVID_PALETTE, DEFAULT_CAMVID_LABEL_TO_COLOR,
                len(DEFAULT_CAMVID_CLASS_NAMES), DEFAULT_NUM_TRAIN_CLASSES, DEFAULT_IGNORE_INDEX)

    def __len__(self) -> int:
        """返回数据集中样本的总数。"""
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据集中的一个样本。
        """
        if idx < 0 or idx >= len(self.image_filenames):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.image_filenames)-1}]")

        # --- 构造文件路径 ---
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.img_folder, img_filename)

        base_name = img_filename[:-len(self.img_suffix)] 
        label_filename = base_name + self.label_suffix_from_img
        label_path = os.path.join(self.label_folder, label_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件未找到: {img_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件未找到: {label_path} (期望基于图像名 {img_filename})")

        # --- 加载图像和标签 ---
        try:
            image_pil = Image.open(img_path).convert('RGB')
            label_pil = Image.open(label_path).convert('RGB') 
        except Exception as e:
            print(f"加载图像/标签时出错 (索引 {idx}, 文件名: {img_filename}): {e}")
            raise e

        # --- 应用针对PIL标签图像的转换 (例如Resize) ---
        if self.target_transform:
            label_pil = self.target_transform(label_pil) 

        # --- 将RGB标签转换为类别索引掩码 ---
        # 如果 self.ignore_index 是 -1 (表示没有特定忽略类或所有类都训练)，
        # 那么未匹配的像素会被标记为一个超出 num_total_classes 范围的值，
        # 这样它们通常会被损失函数或评估逻辑自然忽略。
        fill_value_for_unmatched_pixels = self.ignore_index if self.ignore_index != -1 else self.num_total_classes
        
        target_mask_np = self._rgb_to_mask(label_pil, self.palette, fill_value_for_unmatched_pixels)
        target_mask_tensor = torch.from_numpy(target_mask_np).long()

        # --- 应用针对图像的转换 (例如Resize, ToTensor, Normalize) ---
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            # 如果没有提供transform，至少需要转换为Tensor
            image_tensor = transforms.ToTensor()(image_pil)

        return image_tensor, target_mask_tensor

    def _rgb_to_mask(self, rgb_image: Image.Image, palette: dict, fill_value_for_unmatched: int) -> np.ndarray:
        """
        将RGB格式的PIL Image标签掩码转换为类别索引的NumPy数组。
        未匹配palette中任何颜色的像素将被赋予 fill_value_for_unmatched。
        """
        rgb_array = np.array(rgb_image, dtype=np.uint8)
        if rgb_array.ndim == 2: # 有些掩码可能是灰度图，直接用作索引
            # print("警告: 标签图像是灰度图，将直接使用其值作为类别索引。请确保这是期望的行为。")
            return rgb_array.astype(np.int64)
        if rgb_array.shape[2] != 3:
            raise ValueError(f"RGB图像应有3个通道，但得到 {rgb_array.shape[2]} 个通道。")

        height, width, _ = rgb_array.shape
        # 用指定的填充值初始化掩码
        mask = np.full((height, width), fill_value_for_unmatched, dtype=np.int64) 

        for rgb_tuple, class_index in palette.items():
            # 找到所有颜色匹配的像素
            matches = np.all(rgb_array == np.array(rgb_tuple, dtype=np.uint8), axis=-1)
            mask[matches] = class_index
        return mask

    def _mask_to_rgb(self, mask_indices: np.ndarray) -> np.ndarray:
        """
        (辅助函数) 将类别索引掩码转换回RGB图像，主要用于调试或可视化。
        使用 self.label_to_color 进行映射。
        """
        height, width = mask_indices.shape
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for label_idx, color_tuple in self.label_to_color.items():
            # 只为在CSV中定义的有效类别索引（0 到 num_total_classes-1）上色
            if 0 <= label_idx < self.num_total_classes : 
                 rgb_mask[mask_indices == label_idx] = color_tuple
        return rgb_mask

# ==============================================================================
# 测试代码 (可选，用于验证Dataset类是否正常工作)
# ==============================================================================
if __name__ == '__main__':
    print("开始 CamVidDataset (根据CSV动态加载类别) 测试...")

    # !!! 重要: 请将 CAMVID_ROOT_DIR 修改为你本地CamVid数据集的实际路径 !!!
    CAMVID_ROOT_DIR = "path/to/your/CamVid" # 例如: "D:/datasets/CamVid"
    if CAMVID_ROOT_DIR == "path/to/your/CamVid" or not os.path.exists(CAMVID_ROOT_DIR):
        print(f"警告: 请修改 CAMVID_ROOT_DIR ('{CAMVID_ROOT_DIR}') 为你的 CamVid 数据集根目录路径，然后重新运行测试。")
        print("该目录下应包含 train/, train_labels/, val/, val_labels/, class_dict.csv 等。")
        exit()

    # --- 定义图像和标签预处理 ---
    IMG_HEIGHT = 256
    IMG_WIDTH = 384 

    image_transforms = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_pil_transforms = transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.NEAREST)

    print(f"\n正在创建 'train' 数据集实例 (目标尺寸: {IMG_HEIGHT}x{IMG_WIDTH})...")
    try:
        # 指定 void_class_names 来匹配你的CSV中的 "Void" 类或其他你认为是忽略的类
        train_dataset = CamVidDataset(
            root_dir=CAMVID_ROOT_DIR,
            image_set='train',
            transform=image_transforms,
            target_transform=target_pil_transforms,
            void_class_names=['Void'] # 明确告诉它 "Void" 是忽略类
        )

        print(f"\n成功创建 'train' 数据集。")
        print(f"  总类别数 (num_total_classes): {train_dataset.num_total_classes}")
        print(f"  可训练类别数 (num_train_classes): {train_dataset.num_train_classes}")
        print(f"  忽略索引 (ignore_index): {train_dataset.ignore_index}")
        if train_dataset.ignore_index != -1 and train_dataset.ignore_index < len(train_dataset.class_names):
            print(f"  忽略的类别名称: '{train_dataset.class_names[train_dataset.ignore_index]}'")
        else:
            print(f"  没有特定的忽略类别被指定为训练时的ignore_index (当前为 {train_dataset.ignore_index})")


        if len(train_dataset) > 0:
            sample_idx = 0 #np.random.randint(0, len(train_dataset))
            print(f"\n获取第 {sample_idx} 个样本...")
            image, target_mask = train_dataset[sample_idx]

            print(f"  图像张量 (Image Tensor): 类型: {type(image)}, 形状: {image.shape}, 数据类型: {image.dtype}")
            print(f"  目标掩码张量 (Target Mask Tensor): 类型: {type(target_mask)}, 形状: {target_mask.shape}, 数据类型: {target_mask.dtype}")
            unique_labels = torch.unique(target_mask)
            print(f"  目标掩码唯一值: {unique_labels.tolist()}") # 打印所有唯一值

            assert target_mask.ndim == 2, "目标掩码应为2D (H, W)"
            assert target_mask.dtype == torch.long, "目标掩码数据类型应为 torch.long"
            assert image.shape[1:] == target_mask.shape, \
                   f"图像和掩码的H, W尺寸应一致。图像: {image.shape[1:]}, 掩码: {target_mask.shape}"

            # 检查标签值是否在 [0, num_total_classes - 1] 范围内，或者等于ignore_index (如果ignore_index是有效索引)
            # 或者等于 fill_value_for_unmatched (如果ignore_index是-1)
            max_val_in_mask = torch.max(unique_labels)
            min_val_in_mask = torch.min(unique_labels)

            if train_dataset.ignore_index != -1: # 有明确的忽略索引
                # 所有值要么小于 num_total_classes, 要么等于 ignore_index (它本身也 < num_total_classes)
                assert max_val_in_mask < train_dataset.num_total_classes, \
                   f"掩码中的最大类别索引 ({max_val_in_mask}) 应小于总类别数 {train_dataset.num_total_classes}。"
            else: # ignore_index is -1, fill_value_for_unmatched 可能是 num_total_classes
                 # 此时，有效类别是 [0, num_total_classes-1], 未匹配的是 num_total_classes
                assert max_val_in_mask <= train_dataset.num_total_classes, \
                    f"掩码中的最大类别索引 ({max_val_in_mask}) 超出预期范围 [0, {train_dataset.num_total_classes}]。"

            assert min_val_in_mask >= 0, f"掩码中的最小类别索引 ({min_val_in_mask}) 小于0"
            print("\n样本检查通过！")

            # (可选) 可视化一个样本
            try:
                import matplotlib.pyplot as plt
                print("\n尝试可视化样本...")
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_display = image.clone() * std + mean
                img_display = img_display.clamp(0, 1)
                img_display_pil = transforms.ToPILImage()(img_display)

                target_mask_np = target_mask.cpu().numpy()
                rgb_mask_display = train_dataset._mask_to_rgb(target_mask_np)
                rgb_mask_display_pil = Image.fromarray(rgb_mask_display)

                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(img_display_pil)
                ax[0].set_title(f"原始图像 (样本 {sample_idx}: {train_dataset.image_filenames[sample_idx]})")
                ax[0].axis('off')
                ax[1].imshow(rgb_mask_display_pil)
                ax[1].set_title(f"目标掩码 (样本 {sample_idx})")
                ax[1].axis('off')
                plt.suptitle(f"CamVid 数据集样本可视化 ({train_dataset.image_set} 集)")
                plt.tight_layout()
                plt.show()
                print("可视化完成。请检查弹出的图像窗口。")
            except ImportError:
                print("\nMatplotlib 未安装，跳过可视化。可使用 'pip install matplotlib' 安装。")
            except Exception as e_vis:
                print(f"\n可视化过程中发生错误: {e_vis}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保 CAMVID_ROOT_DIR 设置正确，并且数据集结构符合预期。")
    except Exception as e:
        print(f"创建或测试CamVidDataset时发生意外错误: {e}")
        import traceback
        traceback.print_exc()
    print("\nCamVidDataset 测试结束。")