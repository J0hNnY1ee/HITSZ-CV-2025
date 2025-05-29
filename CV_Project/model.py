# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math # 如果其他模型需要

# ==============================================================================
# 基础模块 (所有模型共享)
# ==============================================================================
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# ==============================================================================
# ResNet 风格的 Bottleneck 块 (用于自定义编码器)
# ==============================================================================
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # ConvBNReLU uses bias=False by default in its Conv2d
        self.conv1 = ConvBNReLU(inplanes, planes, kernel_size=1, padding=0) # 1x1
        self.conv2 = ConvBNReLU(planes, planes, kernel_size=3, stride=stride,
                                padding=dilation, dilation=dilation) # 3x3
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False) # 1x1, no ReLU yet
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # self.stride = stride # Not used explicitly in forward of this version
        # self.dilation = dilation # Not used explicitly in forward of this version

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ==============================================================================
# 自定义类 ResNet 编码器
# ==============================================================================
class CustomResNetEncoder(nn.Module):
    def __init__(self, block, layers, in_channels=3, output_stride=16):
        super(CustomResNetEncoder, self).__init__()
        self.inplanes = 64

        if output_stride == 16:
            strides = [1, 2, 2, 1]; dilations = [1, 1, 1, 2]; aspp_dilations = [6, 12, 18]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]; dilations = [1, 1, 2, 4]; aspp_dilations = [12, 24, 36]
        else:
            raise ValueError("Output stride must be 8 or 16 for this ResNet encoder.")

        self.conv1 = ConvBNReLU(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3) # H/2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # H/4 (low_level_features origin)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.low_level_channels = 64 * block.expansion

        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])

        self.final_encoder_channels = 512 * block.expansion
        self.aspp_dilations = aspp_dilations

        print(f"CustomResNetEncoder initialized: output_stride={output_stride}")
        print(f"  Layer strides: {strides}, Layer dilations: {dilations}")
        print(f"  Low-level feature channels (from layer1): {self.low_level_channels}")
        print(f"  Final encoder output channels (to ASPP): {self.final_encoder_channels}")

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        low_level_feat = self.layer1(x) # This is the H/4 feature map
        # x_layer1_out = self.layer1(x) # Variable name changed for clarity
        x_layer2_out = self.layer2(low_level_feat) # layer2 takes output of layer1
        x_layer3_out = self.layer3(x_layer2_out)
        x_layer4_out = self.layer4(x_layer3_out)
        return x_layer4_out, low_level_feat # Return final encoder output and low_level_feat

# ==============================================================================
# ASPP 模块
# ==============================================================================
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        size = x.shape[-2:]; x_pooled = super(ASPPPooling, self).forward(x)
        return F.interpolate(x_pooled, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5))
    def forward(self, x):
        return self.project(torch.cat([conv(x) for conv in self.convs], dim=1))

# ==============================================================================
# DeepLabV3+ 解码器
# ==============================================================================
class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_level_channels, aspp_channels, num_classes, decoder_channels=256):
        super(DeepLabV3PlusDecoder, self).__init__()
        self.conv_low_level = ConvBNReLU(low_level_channels, 48, kernel_size=1, padding=0)
        self.conv_concat = nn.Sequential(
            ConvBNReLU(aspp_channels + 48, decoder_channels, kernel_size=3, padding=1),
            nn.Dropout(0.5),
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.Dropout(0.1)
        )
        self.final_conv = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
    def forward(self, x_aspp, x_low_level):
        x_low_level_proc = self.conv_low_level(x_low_level)
        x_aspp_upsampled = F.interpolate(x_aspp, size=x_low_level_proc.shape[-2:], mode='bilinear', align_corners=False)
        x_concat = torch.cat((x_aspp_upsampled, x_low_level_proc), dim=1)
        x_fused = self.conv_concat(x_concat)
        return self.final_conv(x_fused)

# ==============================================================================
# DeepLabV3+ 模型
# ==============================================================================
class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, output_stride: int = 16,
                 encoder_layers: list = None,
                 aspp_out_channels: int = 256, decoder_channels: int = 256):
        super(DeepLabV3Plus, self).__init__()
        if encoder_layers is None: encoder_layers = [2, 2, 2, 2] # Default shallow encoder

        self.encoder = CustomResNetEncoder(Bottleneck, encoder_layers, in_channels, output_stride)
        self.aspp = ASPP(self.encoder.final_encoder_channels, aspp_out_channels, self.encoder.aspp_dilations)
        self.decoder = DeepLabV3PlusDecoder(self.encoder.low_level_channels, aspp_out_channels, num_classes, decoder_channels)

        print(f"DeepLabV3+ (CustomResNetEncoder) 初始化完成:")
        print(f"  输入通道数: {in_channels}, 输出类别数: {num_classes}, OS: {output_stride}")
        print(f"  编码器层配置: {encoder_layers}, ASPP输出/膨胀率: {aspp_out_channels}/{self.encoder.aspp_dilations}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        encoder_output, low_level_features = self.encoder(x)
        x_aspp = self.aspp(encoder_output)
        logits_at_decoder_scale = self.decoder(x_aspp, low_level_features)
        logits = F.interpolate(logits_at_decoder_scale, size=input_size, mode='bilinear', align_corners=False)
        return logits

# ==============================================================================
# UNet 基础卷积块, 下采样, 上采样, 输出层
# ==============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2) # Mid channels reduced
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels) # Full channels for conv after cat

    def forward(self, x1, x2): # x1 from upsample, x2 from skip connection
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

# ==============================================================================
# UNet 模型定义
# ==============================================================================
class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, bilinear: bool = True, base_c: int = 64):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor) # Bottleneck
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.outc = OutConv(base_c, num_classes)
        print(f"UNet 初始化完成: IN={in_channels}, OUT={num_classes}, Bilinear={bilinear}, BaseC={base_c}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# ==============================================================================
# SimplePixelClassifier 模型
# ==============================================================================
class SimplePixelClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        if num_classes <= 0: raise ValueError("num_classes 必须是正整数")
        if in_channels <= 0: raise ValueError("in_channels 必须是正整数")
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        print(f"SimplePixelClassifier 初始化: IN={in_channels}, OUT={num_classes}")
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.classifier(x)
