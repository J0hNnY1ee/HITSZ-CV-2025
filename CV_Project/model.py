# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFCN(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(SimpleFCN, self).__init__()
        
        # 编码器 (下采样路径)
        # 我们将使用预训练的 ResNet 的一部分或一个简单的自定义CNN作为主干
        # 为了“轻量级”，我们这里构建一个非常简单的自定义编码器
        
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64xH/2xW/2

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 128xH/4xW/4

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # 额外一层
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 256xH/8xW/8
        
        # Block 4 (Bottleneck / 替换全连接层)
        # 在FCN中，这里通常是继续卷积，而不是全连接层
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 512xH/16xW/16
        # 为了简单，我们可以在这里就输出类别分数，然后上采样
        # 或者再加一个池化层，然后用1x1卷积调整通道数
        
        # 分类层 (1x1 卷积)
        # 将特征图的通道数调整为 num_classes
        self.score_conv = nn.Conv2d(512, num_classes, kernel_size=1) # num_classes x H/8 x W/8
        
        # 解码器 (上采样路径)
        # 我们需要将 H/8 x W/8 的特征图上采样回 H x W
        # 简单的实现是使用一个转置卷积，stride=8
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, 
                                           kernel_size=16, stride=8, 
                                           padding=4, output_padding=0) 
        # kernel_size, stride, padding 的选择是为了使输出尺寸恢复
        # output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
        # H_out = (H_in - 1)*8 - 2*4 + 16 = 8*H_in - 8 - 8 + 16 = 8*H_in
        # W_out = (W_in - 1)*8 - 2*4 + 16 = 8*W_in - 8 - 8 + 16 = 8*W_in
        # 这样可以将 H/8 x W/8 上采样到 H x W

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 编码器
        h = self.relu1_1(self.conv1_1(x))
        h = self.relu1_2(self.conv1_2(h))
        # s1 = h # 可以保存用于skip connection，但简单FCN可以不用
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        # s2 = h
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        # s3 = h
        h = self.pool3(h) # 此时尺寸为 H/8, W/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h)) # 512 x H/8 x W/8
        
        # 分类与上采样
        h = self.score_conv(h)  # num_classes x H/8 x W/8
        
        # 使用转置卷积进行上采样
        out = self.upsample(h) # num_classes x H x W
        
        # 也可以用双线性插值上采样，通常效果也不错，而且参数更少
        # out = F.interpolate(h, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                # 通常转置卷积的初始化也需要小心
                # 一种常见的做法是双线性核，但这里我们用简单的初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # 如果使用BN层
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# (可选) 测试模型结构
if __name__ == '__main__':
    # 从 dataset.py 导入 NUM_CLASSES 和图像尺寸
    # 假设 dataset.py 和 model.py 在同一目录下
    try:
        from dataset import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH
    except ImportError:
        print("Warning: dataset.py not found or NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH not defined.")
        print("Using default values for testing model.")
        NUM_CLASSES = 21
        IMG_HEIGHT = 256
        IMG_WIDTH = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化模型
    model = SimpleFCN(num_classes=NUM_CLASSES).to(device)
    print(model)

    # 创建一个虚拟输入
    # batch_size=2, channels=3, height=IMG_HEIGHT, width=IMG_WIDTH
    dummy_input = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
    
    # 前向传播测试
    try:
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}") # 应该是 [batch_size, num_classes, IMG_HEIGHT, IMG_WIDTH]
        
        # 检查输出尺寸是否与输入图像尺寸匹配 (通道数是类别数)
        assert output.shape[0] == dummy_input.shape[0]
        assert output.shape[1] == NUM_CLASSES
        assert output.shape[2] == dummy_input.shape[2]
        assert output.shape[3] == dummy_input.shape[3]
        print("Model forward pass test successful and output shape is correct.")

    except Exception as e:
        print(f"Error during model forward pass test: {e}")

    # 打印模型参数数量 (一个简单的轻量级衡量指标)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")