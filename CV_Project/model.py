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
# SegNet 模型定义
# ==============================================================================
class SegNet(nn.Module):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.1):
        super(SegNet, self).__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn

        # ENCODING
        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)

        # DECODING
        self.MaxDe = nn.MaxUnpool2d(2, stride=2)

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        # 最后一层通常不加 BatchNorm 和 ReLU

        print(f"SegNet 初始化完成: in_chn={self.in_chn}, out_chn={self.out_chn}, BN_momentum={BN_momentum}")

    def forward(self, x):
        # ENCODE
        x = F.relu(self.BNEn11(self.ConvEn11(x)))
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x); size1 = x.size()

        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        x, ind2 = self.MaxEn(x); size2 = x.size()

        x = F.relu(self.BNEn31(self.ConvEn31(x)))
        x = F.relu(self.BNEn32(self.ConvEn32(x)))
        x = F.relu(self.BNEn33(self.ConvEn33(x)))
        x, ind3 = self.MaxEn(x); size3 = x.size()

        x = F.relu(self.BNEn41(self.ConvEn41(x)))
        x = F.relu(self.BNEn42(self.ConvEn42(x)))
        x = F.relu(self.BNEn43(self.ConvEn43(x)))
        x, ind4 = self.MaxEn(x); size4 = x.size()

        x = F.relu(self.BNEn51(self.ConvEn51(x)))
        x = F.relu(self.BNEn52(self.ConvEn52(x)))
        x = F.relu(self.BNEn53(self.ConvEn53(x)))
        x, ind5 = self.MaxEn(x) # size5 not strictly needed if last MaxUnpool doesn't take output_size

        # DECODE
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))

        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        x = self.MaxDe(x, ind1) # MaxUnpool to original size (or size before first MaxPool)
                                # If input to network had padding, output_size might be needed
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x) # Final logits, no ReLU/Softmax
        return x

# ==============================================================================
# Transformer Encoder Layer (标准实现)
# ==============================================================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# ==============================================================================
# 简化的 SegFormer 风格编码器中的一个 Stage
# ==============================================================================
class SegFormerEncoderStage(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.img_size_h, self.img_size_w = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.feature_map_h = self.img_size_h // self.patch_size
        self.feature_map_w = self.img_size_w // self.patch_size
        # self.num_patches = self.feature_map_h * self.feature_map_w # Not strictly needed if pos_embed is adaptive or not used

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim)) # Optional
        # nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                    dim_feedforward=int(embed_dim * mlp_ratio), dropout=drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # print(f"  EncoderStage: input_size=({self.img_size_h},{self.img_size_w}), patch_size={self.patch_size}, embed_dim={embed_dim}")

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        H_new, W_new = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        # if hasattr(self, 'pos_embed'): x = x + self.pos_embed
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_new, W_new)
        return x

# ==============================================================================
# 简化的 SegFormer 风格模型
# ==============================================================================
class SimplifiedSegFormer(nn.Module):
    def __init__(self, in_chans=3, num_classes=11,
                 img_size=(256, 256),
                 embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
                 mlp_ratios=[4, 4, 4, 4], depths=[2, 2, 2, 2],
                 patch_sizes=[4, 2, 2, 2],
                 decoder_hidden_dim=256, drop_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.img_size_h, self.img_size_w = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        current_in_chans = in_chans
        current_h, current_w = self.img_size_h, self.img_size_w
        self.encoder_stages = nn.ModuleList()

        for i in range(len(depths)):
            stage = SegFormerEncoderStage(
                img_size=(current_h, current_w), patch_size=patch_sizes[i],
                in_chans=current_in_chans, embed_dim=embed_dims[i],
                depth=depths[i], num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i], drop_rate=drop_rate
            )
            self.encoder_stages.append(stage)
            current_in_chans = embed_dims[i]
            current_h //= patch_sizes[i]
            current_w //= patch_sizes[i]

        self.decoder_head = nn.ModuleList()
        for i in range(len(embed_dims)):
            self.decoder_head.append(
                nn.Conv2d(embed_dims[i], decoder_hidden_dim, kernel_size=1)
            )

        self.fusion_conv = ConvBNReLU(decoder_hidden_dim * len(embed_dims), decoder_hidden_dim, kernel_size=3, padding=1)
        self.predict_conv = nn.Conv2d(decoder_hidden_dim, num_classes, kernel_size=1)

        print(f"SimplifiedSegFormer 初始化完成:")
        print(f"  输入通道: {in_chans}, 输出类别: {num_classes}, 图像尺寸: {img_size}")
        print(f"  编码器层深度: {depths}, Embedding维度: {embed_dims}, Patch尺寸(步幅): {patch_sizes}")
        print(f"  解码器隐藏层维度: {decoder_hidden_dim}")

    def forward(self, x):
        B, _, H_in, W_in = x.shape
        features_from_encoder = []
        current_x = x
        for stage in self.encoder_stages:
            current_x = stage(current_x)
            features_from_encoder.append(current_x)

        target_size = features_from_encoder[0].shape[-2:] # Upsample to size of first stage output
        decoded_features = []
        for i in range(len(features_from_encoder)):
            feat = self.decoder_head[i](features_from_encoder[i])
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            decoded_features.append(feat)

        fused_features = torch.cat(decoded_features, dim=1)
        fused_features = self.fusion_conv(fused_features)
        logits = self.predict_conv(fused_features)
        logits = F.interpolate(logits, size=(H_in, W_in), mode='bilinear', align_corners=False)
        return logits

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

# ==============================================================================
# 测试代码
# ==============================================================================
if __name__ == '__main__':
    print("开始 model.py 测试...")
    B, C_IN, H, W, N_CLASSES = 2, 3, 256, 256, 11
    dummy_input = torch.randn(B, C_IN, H, W)

    models_to_test = {
        "SimplePixelClassifier": SimplePixelClassifier(C_IN, N_CLASSES),
        "UNet (bilinear)": UNet(C_IN, N_CLASSES, bilinear=True, base_c=64),
        "UNet (convtranspose)": UNet(C_IN, N_CLASSES, bilinear=False, base_c=32), # Smaller base_c for convtranspose
        "DeepLabV3+ (OS16, [2,2,2,2])": DeepLabV3Plus(C_IN, N_CLASSES, output_stride=16, encoder_layers=[2,2,2,2]),
        "DeepLabV3+ (OS8, [1,1,1,1])": DeepLabV3Plus(C_IN, N_CLASSES, output_stride=8, encoder_layers=[1,1,1,1]),
        "SegNet": SegNet(in_chn=C_IN, out_chn=N_CLASSES, BN_momentum=0.1),
        "SimplifiedSegFormer": SimplifiedSegFormer(
            in_chans=C_IN, num_classes=N_CLASSES, img_size=(H,W),
            embed_dims=[32, 64, 128, 160], num_heads=[1, 2, 4, 5], # Example params
            mlp_ratios=[2, 2, 2, 2], depths=[1, 1, 1, 1], patch_sizes=[4,2,2,2]
        )
    }

    for name, model_instance in models_to_test.items():
        print(f"\n--- Testing {name} ---")
        try:
            # Some models might print during init, this is fine
            output = model_instance(dummy_input)
            print(f"  Output shape: {output.shape}")
            assert output.shape == (B, N_CLASSES, H, W), f"{name} output shape mismatch!"
            print(f"  {name} test PASSED.")
        except Exception as e:
            print(f"  Error testing {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nmodel.py 测试结束。")