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
                 transform=None, # 应用于输入图像的转换 (在内部调整尺寸后，如果提供了output_size)
                                 # 应包括 ToTensor(), Normalize() 等。
                 target_transform=None, # 应用于PIL标签图像的转换 (在内部调整尺寸后，如果提供了output_size)
                                        # 例如，用于PIL级别的标签增强。
                 output_size: tuple[int, int] = None, # 可选：(height, width) 用于内部图像和标签的初始调整尺寸。
                 img_suffix: str = ".png",
                 label_suffix_from_img: str = "_L.png", # 标签文件名相对于图像文件名的后缀
                 class_dict_filename: str = "class_dict.csv", # CSV文件名
                 void_class_names: list = None): # 用于指定哪些类名算作void/ignore
        """
        初始化 CamVidDataset。

        参数:
            root_dir (str): 数据集的根目录 (例如 "CamVid/")。
            image_set (str): 'train', 'val', 或 'test'，指定加载哪个数据集分割。
            transform (callable, optional): 应用于输入图像的转换。
                                           如果提供了 `output_size`，此转换将应用于已调整大小的图像。
                                           通常应包含 ToTensor() 和 Normalize()。
            target_transform (callable, optional): 应用于PIL标签图像的转换。
                                                   如果提供了 `output_size`，此转换将应用于已调整大小的PIL标签。
                                                   注意：最终目标掩码应为 LongTensor 类型。
            output_size (tuple[int, int], optional): 一个元组 (height, width) 指定图像和标签的输出尺寸。
                                                     如果提供，图像将使用 BILINEAR 插值调整大小，
                                                     标签将使用 NEAREST 插值调整大小。此调整在 `transform` 和
                                                     `target_transform` 之前应用。
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
        self.output_size = output_size # (height, width)

        # --- 确定图像和标签文件夹路径 ---
        self.img_folder = os.path.join(root_dir, image_set)
        self.label_folder = os.path.join(root_dir, f"{image_set}_labels")

        self.img_suffix = img_suffix
        self.label_suffix_from_img = label_suffix_from_img

        if void_class_names is None:
            self.void_class_names = ['void', 'unlabelled', 'ignore', 'background']
        else:
            self.void_class_names = [name.lower() for name in void_class_names]


        if not os.path.isdir(self.img_folder):
            raise FileNotFoundError(f"图像文件夹未找到: {self.img_folder}")
        if not os.path.isdir(self.label_folder):
            raise FileNotFoundError(f"标签文件夹未找到: {self.label_folder}")

        self.class_names, self.palette, self.label_to_color, \
        self.num_total_classes, self.num_train_classes, self.ignore_index = \
            self._load_class_definitions(root_dir, class_dict_filename)

        self.image_filenames = sorted([
            f for f in os.listdir(self.img_folder) if f.endswith(self.img_suffix)
        ])

        if not self.image_filenames:
            raise RuntimeError(f"在 {self.img_folder} 中没有找到任何图像文件。")

        print(f"成功初始化 CamVidDataset ({image_set}集):")
        print(f"  图像文件夹: {self.img_folder}")
        print(f"  标签文件夹: {self.label_folder}")
        print(f"  图像数量: {len(self.image_filenames)}")
        if self.output_size:
            print(f"  内部调整尺寸至: {self.output_size} (H x W)")
        else:
            print(f"  未指定内部调整尺寸 (output_size=None)。尺寸将由外部 transform 控制。")
        print(f"  总类别数 (从CSV): {self.num_total_classes}")
        print(f"  可训练类别数: {self.num_train_classes}")
        ignore_class_name = "N/A"
        if self.ignore_index != -1 and self.ignore_index < len(self.class_names):
            ignore_class_name = f"'{self.class_names[self.ignore_index]}'"
        print(f"  忽略索引: {self.ignore_index} (对应类别: {ignore_class_name})")

    def _load_class_definitions(self, root_dir, class_dict_filename):
        """ 从 class_dict.csv 加载类别名称、调色板等信息，如果失败则使用默认值。 """
        csv_path = os.path.join(root_dir, class_dict_filename)

        loaded_class_names = []
        loaded_palette = {}
        determined_ignore_idx = -1
        num_total_cls_from_csv = 0

        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    fieldnames_lower = [field.lower() for field in reader.fieldnames if field]
                    if not all(col in fieldnames_lower for col in ['name', 'r', 'g', 'b']):
                         print(f"警告: {csv_path} 文件缺少 'name', 'r', 'g', 'b' 中的某些列 (不区分大小写)。")
                         print(f"       找到的列: {reader.fieldnames}。将尝试使用默认类别定义。")
                         raise ValueError("CSV文件列不匹配")

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
                        except TypeError:
                            print(f"警告: CSV文件 {csv_path} 行 {idx+2} 颜色值为空: {row}. 跳过此行.")
                            continue

                        loaded_class_names.append(name_val)
                        loaded_palette[(r_val, g_val, b_val)] = idx
                        num_total_cls_from_csv += 1

                        if name_val.lower() in self.void_class_names:
                            if determined_ignore_idx != -1:
                                print(f"警告: 在 {csv_path} 中找到多个可能的忽略类别 ('{loaded_class_names[determined_ignore_idx]}' 和 '{name_val}') 基于 void_class_names。将使用第一个找到的。")
                            else:
                                determined_ignore_idx = idx
                if not loaded_class_names:
                    raise ValueError("CSV文件中没有成功加载任何类别定义。")

                loaded_label_to_color = {v: k for k, v in loaded_palette.items()}

                if determined_ignore_idx != -1:
                    num_train_cls_final = num_total_cls_from_csv - 1 # 如果有一个忽略类
                else:
                    print(f"警告: 在 {csv_path} 中未根据提供的 void_class_names ('{self.void_class_names}') 找到明确的忽略类别。")
                    print(f"       将假设所有 {num_total_cls_from_csv} 个类别都可训练，并将 ignore_index 设置为 -1。")
                    num_train_cls_final = num_total_cls_from_csv

                print(f"成功从 {csv_path} 加载 {num_total_cls_from_csv} 个类别定义。")
                return (loaded_class_names, loaded_palette, loaded_label_to_color,
                        num_total_cls_from_csv, num_train_cls_final, determined_ignore_idx)
            except Exception as e:
                print(f"从 {csv_path} 加载类别定义失败: {e}。将尝试使用内置的默认值。")

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

        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.img_folder, img_filename)
        base_name = img_filename[:-len(self.img_suffix)]
        label_filename = base_name + self.label_suffix_from_img
        label_path = os.path.join(self.label_folder, label_filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件未找到: {img_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件未找到: {label_path} (期望基于图像名 {img_filename})")

        try:
            image_pil = Image.open(img_path).convert('RGB')
            label_pil = Image.open(label_path).convert('RGB') # 假设标签是RGB格式
        except Exception as e:
            print(f"加载图像/标签时出错 (索引 {idx}, 文件名: {img_filename}): {e}")
            # 可以选择返回一个特殊的错误样本或者重新抛出异常
            raise e # 重新抛出，让DataLoader处理 (e.g. skip)

        # 1. 如果提供了 output_size，首先进行内部调整尺寸
        if self.output_size:
            # self.output_size is (height, width)
            # transforms.functional.resize expects size as (h, w)
            image_pil = transforms.functional.resize(image_pil, self.output_size,
                                                     interpolation=transforms.InterpolationMode.BILINEAR)
            label_pil = transforms.functional.resize(label_pil, self.output_size,
                                                     interpolation=transforms.InterpolationMode.NEAREST)

        # 2. 应用用户提供的 target_transform (针对 PIL 标签图像)
        if self.target_transform:
            label_pil = self.target_transform(label_pil) # label_pil 此时可能是原始尺寸或已由 output_size 调整过

        # 3. 将RGB标签转换为类别索引掩码
        # fill_value: 如果RGB值在调色板中找不到，则用此值填充。
        # 通常是 ignore_index (如果有效)，或者一个不会引起歧义的值 (如 num_total_classes，如果ignore_index是-1)
        fill_value_for_unmatched_pixels = self.ignore_index if self.ignore_index != -1 else self.num_total_classes
        target_mask_np = self._rgb_to_mask(label_pil, self.palette, fill_value_for_unmatched_pixels)
        target_mask_tensor = torch.from_numpy(target_mask_np).long() # 转换为 LongTensor

        # 4. 应用用户提供的 transform (针对图像)
        if self.transform:
            image_tensor = self.transform(image_pil) # image_pil 可能是原始尺寸或已由 output_size 调整过
        else:
            # 如果没有提供transform，至少需要转换为Tensor
            image_tensor = transforms.ToTensor()(image_pil)


        return image_tensor, target_mask_tensor

    def _rgb_to_mask(self, rgb_image: Image.Image, palette: dict, fill_value_for_unmatched: int) -> np.ndarray:
        """
        将 PIL RGB 图像根据调色板转换为单通道的类别索引掩码 (NumPy数组)。
        """
        rgb_array = np.array(rgb_image, dtype=np.uint8)
        if rgb_array.ndim == 2: # 如果已经是单通道灰度图 (可能已经是索引了)
            return rgb_array.astype(np.int64) # 直接返回，假设它是类别索引
        if rgb_array.shape[2] != 3:
            raise ValueError(f"RGB图像应有3个通道，但得到 {rgb_array.shape[2]} 个通道。")

        height, width, _ = rgb_array.shape
        # 初始化掩码，使用 fill_value (通常是 ignore_index 或 num_total_classes)
        mask = np.full((height, width), fill_value_for_unmatched, dtype=np.int64)

        for rgb_tuple, class_index in palette.items():
            # rgb_tuple is (R, G, B)
            # class_index is the integer label
            matches = np.all(rgb_array == np.array(rgb_tuple, dtype=np.uint8), axis=-1)
            mask[matches] = class_index
        return mask

    def _mask_to_rgb(self, mask_indices: np.ndarray) -> np.ndarray:
        """
        将单通道的类别索引掩码 (NumPy数组) 转换为 PIL RGB 图像。
        主要用于可视化。
        """
        height, width = mask_indices.shape
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for label_idx, color_tuple in self.label_to_color.items():
            # 确保 label_idx 是有效的，并且在 num_total_classes 范围内 (如果需要严格检查)
            if 0 <= label_idx < self.num_total_classes : # 只映射已知的类别索引
                 rgb_mask[mask_indices == label_idx] = color_tuple
        return rgb_mask

# ==============================================================================
# 测试代码
# ==============================================================================
if __name__ == '__main__':
    print("开始 CamVidDataset (根据CSV动态加载类别) 测试...")

    CAMVID_ROOT_DIR = "path/to/your/CamVid" # 例如: "D:/datasets/CamVid"
    if CAMVID_ROOT_DIR == "path/to/your/CamVid" or not os.path.exists(CAMVID_ROOT_DIR):
        print(f"警告: 请修改 CAMVID_ROOT_DIR ('{CAMVID_ROOT_DIR}') 为你的 CamVid 数据集根目录路径，然后重新运行测试。")
        print("该目录下应包含 train/, train_labels/, val/, val_labels/, class_dict.csv 等。")
        exit()

    # --- 定义图像和标签预处理 ---
    TARGET_HEIGHT = 256
    TARGET_WIDTH = 384

    # 演示1: 使用 output_size 参数进行内部调整尺寸
    print(f"\n--- 演示1: 使用 output_size=({TARGET_HEIGHT}, {TARGET_WIDTH}) ---")
    image_transforms_for_output_size = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_pil_transforms_for_output_size = None # 例如，没有额外的PIL级标签变换

    try:
        train_dataset_demo1 = CamVidDataset(
            root_dir=CAMVID_ROOT_DIR,
            image_set='train',
            output_size=(TARGET_HEIGHT, TARGET_WIDTH), # <--- 使用 output_size
            transform=image_transforms_for_output_size,
            target_transform=target_pil_transforms_for_output_size,
            void_class_names=['Void'] # 假设 'Void' 是 class_dict.csv 中要忽略的类
        )
        print(f"\n成功创建 'train' 数据集 (使用 output_size)。")
        if len(train_dataset_demo1) > 0:
            img_demo1, mask_demo1 = train_dataset_demo1[0]
            print(f"  样本图像张量形状: {img_demo1.shape}") # 应为 (C, TARGET_HEIGHT, TARGET_WIDTH)
            print(f"  样本目标掩码形状: {mask_demo1.shape}") # 应为 (TARGET_HEIGHT, TARGET_WIDTH)
            assert img_demo1.shape[1:] == mask_demo1.shape, "图像和掩码尺寸不匹配 (Demo 1)"
            assert img_demo1.shape[1] == TARGET_HEIGHT and img_demo1.shape[2] == TARGET_WIDTH, "图像尺寸不等于output_size (Demo 1)"

    except Exception as e:
        print(f"演示1时发生错误: {e}")
        import traceback
        traceback.print_exc()

    # 演示2: 不使用 output_size (旧行为), Resize 在外部 transform 中定义
    print(f"\n--- 演示2: output_size=None (Resize 在外部 transform 中) ---")
    image_transforms_external_resize = transforms.Compose([
        transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_pil_transforms_external_resize = transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH),
                                                              interpolation=transforms.InterpolationMode.NEAREST)
    try:
        train_dataset_demo2 = CamVidDataset(
            root_dir=CAMVID_ROOT_DIR,
            image_set='train',
            output_size=None, # <--- 不使用 output_size
            transform=image_transforms_external_resize,
            target_transform=target_pil_transforms_external_resize,
            void_class_names=['Void']
        )
        print(f"\n成功创建 'train' 数据集 (output_size=None)。")
        if len(train_dataset_demo2) > 0:
            img_demo2, mask_demo2 = train_dataset_demo2[0]
            print(f"  样本图像张量形状: {img_demo2.shape}")
            print(f"  样本目标掩码形状: {mask_demo2.shape}")
            assert img_demo2.shape[1:] == mask_demo2.shape, "图像和掩码尺寸不匹配 (Demo 2)"
            assert img_demo2.shape[1] == TARGET_HEIGHT and img_demo2.shape[2] == TARGET_WIDTH, "图像尺寸不等于目标尺寸 (Demo 2)"
    except Exception as e:
        print(f"演示2时发生错误: {e}")
        import traceback
        traceback.print_exc()


    # --- 以下为详细的单个数据集测试和可视化 (使用演示1的配置) ---
    print(f"\n正在使用 output_size=({TARGET_HEIGHT}, {TARGET_WIDTH}) 配置进行详细测试...")
    try:
        train_dataset = CamVidDataset(
            root_dir=CAMVID_ROOT_DIR,
            image_set='train',
            output_size=(TARGET_HEIGHT, TARGET_WIDTH),
            transform=image_transforms_for_output_size,
            target_transform=target_pil_transforms_for_output_size,
            void_class_names=['Void']
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
            sample_idx = 0 # 测试第一个样本
            print(f"\n获取第 {sample_idx} 个样本...")
            image, target_mask = train_dataset[sample_idx]

            print(f"  图像张量 (Image Tensor): 类型: {type(image)}, 形状: {image.shape}, 数据类型: {image.dtype}")
            print(f"  目标掩码张量 (Target Mask Tensor): 类型: {type(target_mask)}, 形状: {target_mask.shape}, 数据类型: {target_mask.dtype}")
            unique_labels = torch.unique(target_mask)
            print(f"  目标掩码唯一值: {unique_labels.tolist()}")

            assert target_mask.ndim == 2, "目标掩码应为2D (H, W)"
            assert target_mask.dtype == torch.long, "目标掩码数据类型应为 torch.long"
            assert image.shape[1:] == target_mask.shape, \
                   f"图像和掩码的H, W尺寸应一致。图像: {image.shape[1:]}, 掩码: {target_mask.shape}"
            assert image.shape[1] == TARGET_HEIGHT and image.shape[2] == TARGET_WIDTH, \
                   f"图像尺寸 ({image.shape[1]}x{image.shape[2]}) 与期望尺寸 ({TARGET_HEIGHT}x{TARGET_WIDTH}) 不符"


            max_val_in_mask = torch.max(unique_labels)
            min_val_in_mask = torch.min(unique_labels)

            # 如果 ignore_index 是一个有效类别（例如 CamVid 的 Void 类，索引为11），
            # 并且它包含在 num_total_classes 中，那么它可能出现在 target_mask 中。
            # CrossEntropyLoss 会处理 ignore_index。
            # 这里的检查是确保掩码中的值在预期范围内 [0, num_total_classes-1] 或者等于 ignore_index。
            if train_dataset.ignore_index != -1:
                 # 允许值达到 num_total_classes-1，或者等于ignore_index
                assert torch.all((unique_labels >= 0) & \
                                 ((unique_labels < train_dataset.num_total_classes) | \
                                  (unique_labels == train_dataset.ignore_index)) ), \
                    f"掩码中的值超出范围 [0, {train_dataset.num_total_classes-1}] 或不等于 ignore_index ({train_dataset.ignore_index}). 得到: {unique_labels.tolist()}"
            else: # 如果没有 ignore_index，则所有值应在 [0, num_total_classes-1]
                assert max_val_in_mask < train_dataset.num_total_classes, \
                    f"掩码中的最大类别索引 ({max_val_in_mask}) 应小于总类别数 {train_dataset.num_total_classes} (当无ignore_index时)。"

            assert min_val_in_mask >= 0, f"掩码中的最小类别索引 ({min_val_in_mask}) 小于0"
            print("\n样本检查通过！")

            try:
                import matplotlib.pyplot as plt
                print("\n尝试可视化样本...")
                # 反归一化图像以供显示
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_display = image.clone() * std + mean # clone以避免修改原始张量
                img_display = img_display.clamp(0, 1) # 将值限制在[0,1]范围
                img_display_pil = transforms.ToPILImage()(img_display)

                # 将目标掩码转换为RGB图像以供显示
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
                plt.suptitle(f"CamVid 数据集样本可视化 ({train_dataset.image_set} 集, 尺寸: {TARGET_HEIGHT}x{TARGET_WIDTH})")
                plt.tight_layout()
                plt.show() # 在某些环境下可能需要 plt.savefig() 代替
                print("可视化完成。请检查弹出的图像窗口 (如果环境支持)。")
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