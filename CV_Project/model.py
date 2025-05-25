# ==============================================================================
# 模块导入 (IMPORTS)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 轻量级语义分割模型 (SimpleSegmentationNet)
# ==============================================================================
class SimpleSegmentationNet(nn.Module):
    """
    一个简单的基于U-Net结构的轻量级语义分割模型。
    包含编码器（下采样）、瓶颈层和解码器（上采样与跳跃连接）。
    """
    def __init__(self, num_classes):
        super(SimpleSegmentationNet, self).__init__()
        
        # --- 编码器 (Encoder / Down-sampling Path) ---
        # 输入: (B, 3, H, W)
        self.enc_block1 = self._make_enc_block(3, 64)      # 输出 H/2, W/2
        self.enc_block2 = self._make_enc_block(64, 128)    # 输出 H/4, W/4
        self.enc_block3 = self._make_enc_block(128, 256)   # 输出 H/8, W/8
        
        # --- 瓶颈层 (Bottleneck) ---
        self.bottleneck = self._make_conv_block(256, 512) # 输出 H/8, W/8

        # --- 解码器 (Decoder / Up-sampling Path) ---
        self.dec_block1 = self._make_dec_block(512, 256, 256) # 输入来自 bottleneck + enc_block3, 输出 H/4, W/4
        self.dec_block2 = self._make_dec_block(256, 128, 128) # 输入来自 dec_block1 + enc_block2, 输出 H/2, W/2
        self.dec_block3 = self._make_dec_block(128, 64, 64)   # 输入来自 dec_block2 + enc_block1, 输出 H, W
        
        # --- 输出层 (Output Layer) ---
        # 最终输出每个像素的类别得分
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1) # (B, num_classes, H, W)

    # --- 辅助函数：构建卷积块 ---
    def _make_conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False), # bias=False 因为用了BN
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # --- 辅助函数：构建编码器块 (卷积 + 池化) ---
    def _make_enc_block(self, in_channels, out_channels):
        return nn.Sequential(
            self._make_conv_block(in_channels, out_channels),
            self._make_conv_block(out_channels, out_channels), # 增加一层卷积
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    # --- 辅助函数：构建解码器块 (上采样 + 跳跃连接 + 卷积) ---
    def _make_dec_block(self, in_channels_up, in_channels_skip, out_channels):
        # in_channels_up: 来自上一个解码器层或瓶颈层的通道数
        # in_channels_skip: 来自对应编码器层的跳跃连接的通道数
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels_up, out_channels, kernel_size=2, stride=2), # 上采样
            self._make_conv_block(out_channels + in_channels_skip, out_channels), # 拼接后卷积
            self._make_conv_block(out_channels, out_channels)  # 再一层卷积
        )

    # --- 前向传播 (Forward Pass) ---
    def forward(self, x):
        # --- 编码器 ---
        e1_out = self.enc_block1[0:2](x)  # 获取第一个编码器块卷积部分的输出用于跳跃连接
        e1_pool = self.enc_block1[2](e1_out)
        
        e2_out = self.enc_block2[0:2](e1_pool)
        e2_pool = self.enc_block2[2](e2_out)
        
        e3_out = self.enc_block3[0:2](e2_pool)
        e3_pool = self.enc_block3[2](e3_out)

        # --- 瓶颈层 ---
        bottleneck_out = self.bottleneck(e3_pool)
        
        # --- 解码器 ---
        # 上采样 + 拼接 + 卷积
        d1_up = self.dec_block1[0](bottleneck_out)
        d1_cat = torch.cat([d1_up, e3_out], dim=1) # 跳跃连接
        d1_out = self.dec_block1[1:](d1_cat)      # 解码器卷积部分
        
        d2_up = self.dec_block2[0](d1_out)
        d2_cat = torch.cat([d2_up, e2_out], dim=1)
        d2_out = self.dec_block2[1:](d2_cat)
        
        d3_up = self.dec_block3[0](d2_out)
        d3_cat = torch.cat([d3_up, e1_out], dim=1)
        d3_out = self.dec_block3[1:](d3_cat)

        # --- 输出层 ---
        out = self.output_conv(d3_out)
        return out

# ==============================================================================
# 主执行块 (测试用) (MAIN EXECUTION BLOCK - FOR TESTING)
# ==============================================================================
if __name__ == '__main__':
    num_classes_test = 21
    dummy_input = torch.randn(2, 3, 256, 256) # (B, C, H, W)
    
    print("开始测试模型构建及前向传播...")
    model = SimpleSegmentationNet(num_classes=num_classes_test)
    # print(model) # 取消注释以查看模型结构
    
    # 测试模型参数量 (可选)
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"模型参数量: {num_params / 1e6:.2f} M")

    output = model(dummy_input)
    
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}") # 期望: (B, num_classes, H, W)
    assert output.shape == (dummy_input.shape[0], num_classes_test, dummy_input.shape[2], dummy_input.shape[3]), \
        "模型输出形状不正确！"
    print("模型测试通过！")