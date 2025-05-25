# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 非常简单的单层卷积模型 (占位模型)
# ==============================================================================

class SimplePixelClassifier(nn.Module):
    """
    一个极其简单的像素级分类器，使用单个1x1卷积层。
    主要用于快速搭建训练流程，后续可以替换为更复杂的分割模型。
    """
    def __init__(self, in_channels: int, num_classes: int):
        """
        初始化模型。

        参数:
            in_channels (int): 输入图像的通道数 (例如，RGB图像为3)。
            num_classes (int): 分割任务的总类别数。
                               注意：这个 num_classes 应该是包括了所有要预测的类别，
                               例如 CamVid 的11个可训练类别，如果模型输出覆盖所有类别，
                               则这里应该是11。如果模型输出包括了背景/忽略类，则需要对应调整。
                               通常，损失函数会处理 ignore_index。
        """
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes 必须是正整数")
        if in_channels <= 0:
            raise ValueError("in_channels 必须是正整数")

        # 单个1x1卷积层，直接将输入通道映射到类别得分通道
        # 输出的空间维度 (H, W) 与输入相同
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)

        print(f"SimplePixelClassifier 初始化完成:")
        print(f"  输入通道数 (in_channels): {in_channels}")
        print(f"  输出类别数 (num_classes): {num_classes}")
        print(f"  模型结构: nn.Conv2d({in_channels}, {num_classes}, kernel_size=1)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (torch.Tensor): 输入图像张量，形状为 (B, C_in, H, W)。

        返回:
            torch.Tensor: 输出的类别得分张量，形状为 (B, num_classes, H, W)。
        """
        return self.classifier(x)

# ==============================================================================
# 测试代码 (可选)
# ==============================================================================
if __name__ == '__main__':
    print("开始 model.py 测试...")

    # --- 定义参数 ---
    batch_size = 2
    img_channels = 3  # RGB图像
    img_height = 256
    img_width = 256
    num_output_classes = 11 # 假设CamVid有11个可训练类别

    # --- 创建模型实例 ---
    print(f"\n创建 SimplePixelClassifier 实例 (num_classes={num_output_classes})...")
    model = SimplePixelClassifier(in_channels=img_channels, num_classes=num_output_classes)
    print(model) # 打印模型结构

    # --- 创建一个虚拟输入 ---
    print("\n创建虚拟输入张量...")
    dummy_input = torch.randn(batch_size, img_channels, img_height, img_width)
    print(f"  虚拟输入形状: {dummy_input.shape}")

    # --- 进行一次前向传播 ---
    print("\n执行前向传播...")
    try:
        with torch.no_grad(): # 在测试/推理时不计算梯度
            output = model(dummy_input)
        print(f"  输出张量形状: {output.shape}")

        # --- 检查输出形状是否符合预期 ---
        expected_output_shape = (batch_size, num_output_classes, img_height, img_width)
        assert output.shape == expected_output_shape, \
            f"输出形状不匹配! 期望: {expected_output_shape}, 得到: {output.shape}"
        print("  输出形状符合预期。")

    except Exception as e:
        print(f"模型前向传播时发生错误: {e}")
        import traceback
        traceback.print_exc()

    print("\nmodel.py 测试结束。")