# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# 基础模块 (复用之前的，或者根据需要调整)
# ==============================================================================
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

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

        self.activation = F.relu if activation == "relu" else F.gelu # GELU is common in Transformers

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: (Batch, SequenceLength, EmbeddingDim)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# ==============================================================================
# 简化的 SegFormer 风格编码器中的一个 Stage
# (Patch Embedding + Transformer Encoder Layers)
# ==============================================================================
class SegFormerEncoderStage(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        # img_size is expected to be a tuple (current_H, current_W)
        # patch_size is an int, representing the stride and kernel size of the patch embedding conv
        self.img_size_h, self.img_size_w = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 计算下采样后的特征图尺寸
        self.feature_map_h = self.img_size_h // self.patch_size
        self.feature_map_w = self.img_size_w // self.patch_size
        self.num_patches = self.feature_map_h * self.feature_map_w # 正确计算 num_patches

        # Patch Embedding: 使用卷积实现
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # (可选) 位置编码，如果需要的话
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim)) 
        # nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                    dim_feedforward=int(embed_dim * mlp_ratio), dropout=drop_rate)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # print(f"  EncoderStage: input_size=({self.img_size_h},{self.img_size_w}), patch_size={self.patch_size}, num_patches={self.num_patches}, embed_dim={embed_dim}")


    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x) # (B, embed_dim, feature_map_h, feature_map_w)
        H_new, W_new = x.shape[-2:] # 获取卷积后的实际 H, W
        
        # 确保计算的 num_patches 与展平后的一致 (用于位置编码等)
        # 如果 patch_embed 可能有 padding 导致 H_new, W_new 与预期不同，则需要基于 H_new, W_new 更新
        # 但对于 stride=kernel_size 的卷积，并且输入尺寸能被kernel_size整除，H_new, W_new 应该是准确的
        
        x = x.flatten(2).transpose(1, 2)  # (B, H_new*W_new, embed_dim)

        # if hasattr(self, 'pos_embed'):
        #     if x.shape[1] != self.pos_embed.shape[1]: # 动态调整位置编码长度 (不推荐，最好保证一致)
        #         print(f"Warning: Positional embedding size mismatch. Expected {self.pos_embed.shape[1]} patches, got {x.shape[1]}. This might happen if input image size to the stage is not perfectly divisible by patch_size or if patch_embed has padding effects not accounted for.")
        #         # A more robust pos_embed would handle dynamic sizes, or ensure input sizes are fixed/padded.
        #     x = x + self.pos_embed[:, :x.shape[1]] # 截断或填充pos_embed (简单处理)


        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.norm(x)
        # Reshape back: (B, embed_dim, H_new, W_new)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_new, W_new) 
        return x

# ==============================================================================
# 简化的 SegFormer 风格模型
# ==============================================================================
class SimplifiedSegFormer(nn.Module):
    def __init__(self, in_chans=3, num_classes=11,
                 img_size=(256, 256), # 输入图像尺寸
                 # 为每个stage定义参数: embed_dims, num_heads, mlp_ratios, depths
                 # patch_sizes 定义了每个stage的下采样因子 (相对于上一阶段或原始输入)
                 # SegFormer B0-like (非常简化)
                 embed_dims=[32, 64, 160, 256], # SegFormer B0 uses [32, 64, 160, 256]
                 num_heads=[1, 2, 5, 8],        # SegFormer B0 uses [1, 2, 5, 8]
                 mlp_ratios=[4, 4, 4, 4],       # Standard MLP ratio
                 depths=[2, 2, 2, 2],           # SegFormer B0 uses [2, 2, 2, 2] or [3,3,6,3] for B1
                 patch_sizes=[4, 2, 2, 2],      # Strides for patch embedding at each stage
                                                # Stage 1: 256/4 = 64x64
                                                # Stage 2: 64/2 = 32x32
                                                # Stage 3: 32/2 = 16x16
                                                # Stage 4: 16/2 = 8x8
                 decoder_hidden_dim=256,        # MLP Decoder的隐藏层维度
                 drop_rate=0.1):                # Dropout rate
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.img_size_h, self.img_size_w = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        # --- Encoder Stages ---
        current_in_chans = in_chans
        current_h, current_w = self.img_size_h, self.img_size_w
        self.encoder_stages = nn.ModuleList()

        for i in range(len(depths)):
            stage = SegFormerEncoderStage(
                img_size=(current_h, current_w), # Pass tuple for img_size
                patch_size=patch_sizes[i],
                in_chans=current_in_chans,
                embed_dim=embed_dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                drop_rate=drop_rate
            )
            self.encoder_stages.append(stage)
            current_in_chans = embed_dims[i]
            current_h //= patch_sizes[i]
            current_w //= patch_sizes[i]

        # --- MLP Decoder Head (SegFormer style) ---
        self.decoder_head = nn.ModuleList()
        for i in range(len(embed_dims)): # For each encoder stage output
            self.decoder_head.append(
                nn.Conv2d(embed_dims[i], decoder_hidden_dim, kernel_size=1)
            )
        
        # Final fusion and prediction layer
        self.fusion_conv = ConvBNReLU(decoder_hidden_dim * len(embed_dims), decoder_hidden_dim, kernel_size=3, padding=1)
        self.predict_conv = nn.Conv2d(decoder_hidden_dim, num_classes, kernel_size=1)

        print(f"SimplifiedSegFormer 初始化完成:")
        print(f"  输入通道: {in_chans}, 输出类别: {num_classes}")
        print(f"  图像尺寸: {img_size}")
        print(f"  编码器层深度: {depths}, Embedding维度: {embed_dims}, Patch尺寸(步幅): {patch_sizes}")
        print(f"  解码器隐藏层维度: {decoder_hidden_dim}")


    def forward(self, x):
        B, _, H_in, W_in = x.shape
        features_from_encoder = []

        # Encoder
        current_x = x
        for stage in self.encoder_stages:
            current_x = stage(current_x)
            features_from_encoder.append(current_x)
            # print(f"Encoder stage output shape: {current_x.shape}")

        # Decoder
        decoded_features = []
        # Target size for upsampling (largest feature map from encoder, usually the first one after patch embed)
        # Or simply upsample all to 1/4 of original input size, like SegFormer does for its MLP head.
        # For example, if input is 256, first patch_size is 4, so H_target = 256/4 = 64
        # H_target = H_in // self.encoder_stages[0].patch_size 
        # W_target = W_in // self.encoder_stages[0].patch_size
        # For SegFormer, all features are upsampled to size of feature map from stage 1 (1/4 of input)
        target_size = features_from_encoder[0].shape[-2:]


        for i in range(len(features_from_encoder)):
            feat = self.decoder_head[i](features_from_encoder[i])
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            decoded_features.append(feat)
            # print(f"Decoded feature {i} shape: {feat.shape}")

        # Concatenate and fuse
        fused_features = torch.cat(decoded_features, dim=1)
        # print(f"Fused features shape: {fused_features.shape}")
        fused_features = self.fusion_conv(fused_features)
        # print(f"After fusion_conv shape: {fused_features.shape}")

        # Prediction and final upsampling
        logits = self.predict_conv(fused_features)
        # print(f"Logits before final upsample shape: {logits.shape}")
        logits = F.interpolate(logits, size=(H_in, W_in), mode='bilinear', align_corners=False)
        # print(f"Final logits shape: {logits.shape}")
        
        return logits
    
# ==============================================================================
# 基础卷积块 (之前为UNet定义，DeepLabV3+的组件也会用到)
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
    expansion = 4 # 输出通道相对于输入中间通道的倍数

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBNReLU(inplanes, planes, kernel_size=1, padding=0) # 1x1 conv
        self.conv2 = ConvBNReLU(planes, planes, kernel_size=3, stride=stride, 
                                padding=dilation, dilation=dilation) # 3x3 conv
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False) # 1x1 conv
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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
        self.inplanes = 64 # 初始通道数

        if output_stride == 16:
            strides = [1, 2, 2, 1] # 第一个2对应第一个stage的pool, 第二个2对应第二个stage的第一个block的stride
            dilations = [1, 1, 1, 2] # 最后一个stage使用空洞卷积
            aspp_dilations = [6, 12, 18] # for output_stride 16
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
            aspp_dilations = [12, 24, 36] # for output_stride 8
        else:
            raise ValueError("Output stride must be 8 or 16 for this ResNet encoder.")

        self.conv1 = ConvBNReLU(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3) # H/2, W/2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # H/4, W/4 (これがlow_level_featuresの元)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.low_level_channels = 64 * block.expansion # layer1の出力チャンネル数

        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3])
        
        self.final_encoder_channels = 512 * block.expansion
        self.aspp_dilations = aspp_dilations # Store for ASPP configuration

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
            layers.append(block(self.inplanes, planes, dilation=dilation)) # Subsequent blocks in layer have stride 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)       # Output: H/2, W/2
        x = self.maxpool(x)     # Output: H/4, W/4
        
        low_level_feat = self.layer1(x) # Output: H/4, W/4 (stride=1 in layer1)
        
        x = self.layer2(low_level_feat if self.layer1[0].stride == 1 else self.layer2(x)) # if layer1 had stride > 1, x is already processed
                                                                  # Corrected: layer2 always takes output of layer1
        x = self.layer2(low_level_feat) # layer2 takes output of layer1. stride in layer2 is 2. Output: H/8, W/8
        x = self.layer3(x)      # stride in layer3 is 2 (for OS16) or 1 (for OS8). Output: H/16 or H/8
        x = self.layer4(x)      # stride in layer4 is 1 (for OS16) or 1 (for OS8, but with more dilation). Output: H/16 or H/8
        
        return x, low_level_feat

# ==============================================================================
# ASPP 模块 (与之前版本类似，但确保atrous_rates可以从encoder获取)
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
        size = x.shape[-2:]; x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]): # rates will be passed from encoder
        super(ASPP, self).__init__()
        modules = [nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))]
        for rate in atrous_rates: modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.5))
    def forward(self, x):
        return self.project(torch.cat([conv(x) for conv in self.convs], dim=1))

# ==============================================================================
# DeepLabV3+ 解码器 (与之前版本类似)
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
        x_low_level = self.conv_low_level(x_low_level)
        x_aspp_upsampled = F.interpolate(x_aspp, size=x_low_level.shape[-2:], mode='bilinear', align_corners=False)
        x_concat = torch.cat((x_aspp_upsampled, x_low_level), dim=1)
        x_fused = self.conv_concat(x_concat)
        return self.final_conv(x_fused)

# ==============================================================================
# "标准" DeepLabV3+ 模型 (使用自定义 ResNet 风格编码器)
# ==============================================================================
class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, output_stride: int = 16,
                 encoder_layers: list = None, # 例如 [3, 4, 6, 3] for ResNet50-like
                 aspp_out_channels: int = 256, decoder_channels: int = 256):
        super(DeepLabV3Plus, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.output_stride = output_stride

        if encoder_layers is None:
            encoder_layers = [2, 2, 2, 2] # A shallower ResNet-like encoder (like ResNet-18 if using BasicBlock)
                                         # For Bottleneck, [3,4,6,3] is ResNet50. Let's use a smaller one.
                                         # E.g., a "ResNet-26" like structure for Bottleneck: [2,2,2,2]
                                         # (1+2*3) + (1+2*3) + (1+2*3) + (1+2*3) = 7*4 = 28 convs approx
                                         # Let's use something smaller for "lightweight" requirement.
                                         # How about layers=[2,2,2,2] for Bottleneck? It's custom.
            print("Using default encoder_layers=[2,2,2,2] for CustomResNetEncoder with Bottleneck blocks.")


        # 1. Encoder (Backbone)
        self.encoder = CustomResNetEncoder(Bottleneck, encoder_layers, in_channels, output_stride)
        
        low_level_channels = self.encoder.low_level_channels
        encoder_output_channels = self.encoder.final_encoder_channels
        aspp_actual_dilations = self.encoder.aspp_dilations # Get dilations based on output_stride

        # 2. ASPP
        self.aspp = ASPP(encoder_output_channels, out_channels=aspp_out_channels, atrous_rates=aspp_actual_dilations)

        # 3. Decoder
        self.decoder = DeepLabV3PlusDecoder(low_level_channels, aspp_out_channels, num_classes, decoder_channels)
        
        print(f"DeepLabV3+ (CustomResNetEncoder) 初始化完成:")
        print(f"  输入通道数: {in_channels}, 输出类别数: {num_classes}")
        print(f"  编码器输出步幅: {output_stride}, 编码器层配置: {encoder_layers}")
        print(f"  ASPP 输出通道: {aspp_out_channels}, ASPP膨胀率: {aspp_actual_dilations}")
        print(f"  解码器中间通道: {decoder_channels}, 低级特征通道: {low_level_channels}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        encoder_output, low_level_features = self.encoder(x)
        x_aspp = self.aspp(encoder_output)
        logits_at_decoder_scale = self.decoder(x_aspp, low_level_features)
        logits = F.interpolate(logits_at_decoder_scale, size=input_size, mode='bilinear', align_corners=False)
        return logits

# ==============================================================================
# 基础卷积块 (用于UNet)
# ==============================================================================
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# ==============================================================================
# UNet 下采样块 (编码器部分)
# ==============================================================================
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# ==============================================================================
# UNet 上采样块 (解码器部分)
# ==============================================================================
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels) # in_channels here is after cat with skip connection

    def forward(self, x1, x2):
        # x1 is from previous upsampling layer, x2 is from skip connection (encoder)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match x2's dimensions
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ==============================================================================
# UNet 输出层
# ==============================================================================
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ==============================================================================
# UNet 模型定义
# ==============================================================================
class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, bilinear: bool = True, base_c: int = 64):
        """
        UNet模型。

        参数:
            in_channels (int): 输入图像的通道数 (例如RGB为3)。
            num_classes (int): 输出的类别数量。
            bilinear (bool): 是否使用双线性插值进行上采样。如果为False，则使用转置卷积。
            base_c (int): UNet第一层卷积的输出通道数，后续层通道数会基于此加倍。
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1 # factor for mid_channels in Up if bilinear
        self.down4 = Down(base_c * 8, base_c * 16 // factor) #瓶颈层

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.outc = OutConv(base_c, num_classes)

        print(f"UNet 初始化完成:")
        print(f"  输入通道数 (in_channels): {self.in_channels}")
        print(f"  输出类别数 (num_classes): {self.num_classes}")
        print(f"  上采样方式 (bilinear): {self.bilinear}")
        print(f"  基础通道数 (base_c): {base_c}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)    # (B, base_c, H, W)
        x2 = self.down1(x1) # (B, base_c*2, H/2, W/2)
        x3 = self.down2(x2) # (B, base_c*4, H/4, W/4)
        x4 = self.down3(x3) # (B, base_c*8, H/8, W/8)
        x5 = self.down4(x4) # (B, base_c*16 // factor, H/16, W/16) - 瓶颈

        x = self.up1(x5, x4) # 输入 x5 (上采样后) 和 x4 (跳跃连接)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x) # (B, num_classes, H, W)
        return logits

# ==============================================================================
# 非常简单的单层卷积模型 (占位模型，保持不变)
# ==============================================================================
class SimplePixelClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        if num_classes <= 0: raise ValueError("num_classes 必须是正整数")
        if in_channels <= 0: raise ValueError("in_channels 必须是正整数")
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0)
        print(f"SimplePixelClassifier 初始化完成:")
        print(f"  输入通道数 (in_channels): {in_channels}")
        print(f"  输出类别数 (num_classes): {num_classes}")
        print(f"  模型结构: nn.Conv2d({in_channels}, {num_classes}, kernel_size=1)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ==============================================================================
# 测试代码
# ==============================================================================
if __name__ == '__main__':
    print("开始 model.py 测试...")
    B, C_IN, H, W, N_CLASSES = 2, 3, 256, 256, 11
    dummy_input = torch.randn(B, C_IN, H, W)

    print("\n--- Testing SimplePixelClassifier ---")
    simple_model = SimplePixelClassifier(C_IN, N_CLASSES)
    try: output = simple_model(dummy_input); print(f"Output shape: {output.shape}"); assert output.shape == (B, N_CLASSES, H, W)
    except Exception as e: print(f"Error: {e}")

    print("\n--- Testing UNet (bilinear=True) ---")
    unet_model_b = UNet(C_IN, N_CLASSES, bilinear=True, base_c=64)
    try: output = unet_model_b(dummy_input); print(f"Output shape: {output.shape}"); assert output.shape == (B, N_CLASSES, H, W)
    except Exception as e: print(f"Error: {e}")
    
    print("\n--- Testing DeepLabV3+ (CustomResNetEncoder, OS=16, layers=[2,2,2,2]) ---")
    # 使用较浅的 encoder_layers=[2,2,2,2] 进行测试
    deeplab_custom_os16 = DeepLabV3Plus(C_IN, N_CLASSES, output_stride=16, encoder_layers=[2,2,2,2])
    try: output = deeplab_custom_os16(dummy_input); print(f"Output shape: {output.shape}"); assert output.shape == (B, N_CLASSES, H, W)
    except Exception as e: print(f"Error: {e}")

    print("\n--- Testing DeepLabV3+ (CustomResNetEncoder, OS=8, layers=[1,1,1,1]) ---")
    # 使用更浅的 encoder_layers=[1,1,1,1] 进行测试
    deeplab_custom_os8 = DeepLabV3Plus(C_IN, N_CLASSES, output_stride=8, encoder_layers=[1,1,1,1])
    try: output = deeplab_custom_os8(dummy_input); print(f"Output shape: {output.shape}"); assert output.shape == (B, N_CLASSES, H, W)
    except Exception as e: print(f"Error: {e}")

    print("\nmodel.py 测试结束。")