from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Handle pad if dimensions are slightly off
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, in_channels: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        # Load Pretrained ResNet34 from torchvision
        encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first layer to accept 4 channels (3 for EO, 1 for SAR)
        original_conv1 = encoder.conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_conv1.weight
            if in_channels > 3:
                # Initialize the extra channels with the mean of the RGB weights
                # This gives the network a head start instead of random noise
                self.conv1.weight[:, 3:] = original_conv1.weight.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)

        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool

        self.layer1 = encoder.layer1 # output: 64 channels
        self.layer2 = encoder.layer2 # output: 128 channels
        self.layer3 = encoder.layer3 # output: 256 channels
        self.layer4 = encoder.layer4 # output: 512 channels

        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256, dropout)
        self.dec3 = DecoderBlock(256, 128, 128, dropout)
        self.dec2 = DecoderBlock(128, 64, 64, dropout)
        self.dec1 = DecoderBlock(64, 64, 64, dropout=0.0)

        # Final upsampling block to match original image size
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)
        e1 = self.layer1(x1)  # 64
        e2 = self.layer2(e1)  # 128
        e3 = self.layer3(e2)  # 256
        e4 = self.layer4(e3)  # 512

        # Decoder
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)
        out = self.head(out)

        # Ensure output size exactly matches input size
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


class LateFusionUNet(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        # EO Stream (ResNet34) - Robust feature extractor for 3-channel optical
        self.eo_encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # SAR Stream (ResNet18) - Lighter feature extractor for 1-channel radar
        self.sar_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Modify SAR conv1 to accept 1 channel instead of 3
        old_conv1 = self.sar_encoder.conv1
        self.sar_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.sar_encoder.conv1.weight[:] = old_conv1.weight.sum(dim=1, keepdim=True)

        # Decoder blocks (in_channels and skip_channels are doubled because of concatenation)
        # e4_eo(512) + e4_sar(512) = 1024. e3_eo(256) + e3_sar(256) = 512.
        self.dec4 = DecoderBlock(1024, 512, 256, dropout)
        self.dec3 = DecoderBlock(256, 256, 128, dropout)
        self.dec2 = DecoderBlock(128, 128, 64, dropout)
        self.dec1 = DecoderBlock(64, 128, 64, dropout=0.0)

        # Final upsampling
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split inputs
        eo = x[:, :3, :, :]
        sar = x[:, 3:, :, :]

        # EO Forward
        x0_eo = self.eo_encoder.relu(self.eo_encoder.bn1(self.eo_encoder.conv1(eo)))
        x1_eo = self.eo_encoder.maxpool(x0_eo)
        e1_eo = self.eo_encoder.layer1(x1_eo)
        e2_eo = self.eo_encoder.layer2(e1_eo)
        e3_eo = self.eo_encoder.layer3(e2_eo)
        e4_eo = self.eo_encoder.layer4(e3_eo)

        # SAR Forward Ensure SAR stream doesn't crash on spatial size
        x0_sar = self.sar_encoder.relu(self.sar_encoder.bn1(self.sar_encoder.conv1(sar)))
        x1_sar = self.sar_encoder.maxpool(x0_sar)
        e1_sar = self.sar_encoder.layer1(x1_sar)
        e2_sar = self.sar_encoder.layer2(e1_sar)
        e3_sar = self.sar_encoder.layer3(e2_sar)
        e4_sar = self.sar_encoder.layer4(e3_sar)

        # Late Fusion (Concatenation at every level)
        e4 = torch.cat([e4_eo, e4_sar], dim=1)
        e3 = torch.cat([e3_eo, e3_sar], dim=1)
        e2 = torch.cat([e2_eo, e2_sar], dim=1)
        e1 = torch.cat([e1_eo, e1_sar], dim=1)
        x0 = torch.cat([x0_eo, x0_sar], dim=1)

        # Decoding
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)
        out = self.head(out)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


class DifferenceFusionUNet(nn.Module):
    """
    EO/SAR dual-encoder U-Net with explicit absolute feature-difference fusion.
    This gives the decoder direct change cues instead of relying on concatenation alone.
    """
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.eo_encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.sar_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv1 = self.sar_encoder.conv1
        self.sar_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.sar_encoder.conv1.weight[:] = old_conv1.weight.sum(dim=1, keepdim=True)

        # Fused channels are [EO, SAR, abs(EO-SAR)] at each level.
        self.dec4 = DecoderBlock(1536, 768, 384, dropout)
        self.dec3 = DecoderBlock(384, 384, 192, dropout)
        self.dec2 = DecoderBlock(192, 192, 96, dropout)
        self.dec1 = DecoderBlock(96, 192, 64, dropout=0.0)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _fuse(eo: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        return torch.cat([eo, sar, torch.abs(eo - sar)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eo = x[:, :3, :, :]
        sar = x[:, 3:, :, :]

        x0_eo = self.eo_encoder.relu(self.eo_encoder.bn1(self.eo_encoder.conv1(eo)))
        x1_eo = self.eo_encoder.maxpool(x0_eo)
        e1_eo = self.eo_encoder.layer1(x1_eo)
        e2_eo = self.eo_encoder.layer2(e1_eo)
        e3_eo = self.eo_encoder.layer3(e2_eo)
        e4_eo = self.eo_encoder.layer4(e3_eo)

        x0_sar = self.sar_encoder.relu(self.sar_encoder.bn1(self.sar_encoder.conv1(sar)))
        x1_sar = self.sar_encoder.maxpool(x0_sar)
        e1_sar = self.sar_encoder.layer1(x1_sar)
        e2_sar = self.sar_encoder.layer2(e1_sar)
        e3_sar = self.sar_encoder.layer3(e2_sar)
        e4_sar = self.sar_encoder.layer4(e3_sar)

        e4 = self._fuse(e4_eo, e4_sar)
        e3 = self._fuse(e3_eo, e3_sar)
        e2 = self._fuse(e2_eo, e2_sar)
        e1 = self._fuse(e1_eo, e1_sar)
        x0 = self._fuse(x0_eo, x0_sar)

        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)
        out = self.head(out)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


class TransformerBottleneck(nn.Module):
    def __init__(self, channels: int, num_layers: int = 2, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        # PyTorch built-in Transformer layer for our bottleneck
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Convert spatial feature map [B, C, H, W] into a sequence of tokens [B, H*W, C]
        tokens = x.flatten(2).transpose(1, 2)
        # Pass tokens through the Self-Attention layers for global context
        out_tokens = self.transformer(tokens)
        # Reconstruct the spatial feature map [B, C, H, W]
        return out_tokens.transpose(1, 2).view(b, c, h, w)


class TransLateFusionUNet(nn.Module):
    """
    State-of-the-Art Architecture: CNN Two-Stream Encoder + Transformer Bottleneck + CNN Decoder
    Captures both fine-grained local textures (via ResNet) and high-level global context (via Transformer).
    """
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        # EO Stream (ResNet34) - Robust feature extractor for 3-channel optical
        self.eo_encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # SAR Stream (ResNet18) - Lighter feature extractor for 1-channel radar
        self.sar_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Modify SAR conv1 to accept 1 channel instead of 3
        old_conv1 = self.sar_encoder.conv1
        self.sar_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.sar_encoder.conv1.weight[:] = old_conv1.weight.sum(dim=1, keepdim=True)

        # TRANSFORMER BOTTLENECK
        # e4_eo(512) + e4_sar(512) = 1024 channels.
        self.bottleneck_proj = nn.Conv2d(1024, 512, kernel_size=1) # Project to 512 to save memory in attention
        self.transformer = TransformerBottleneck(channels=512, num_layers=2, nhead=8, dropout=dropout)

        # Decoder blocks
        # We start decoding from the 512-dim transformer output, concatenating with e3 (256+256)
        self.dec4 = DecoderBlock(512, 512, 256, dropout)
        self.dec3 = DecoderBlock(256, 256, 128, dropout)
        self.dec2 = DecoderBlock(128, 128, 64, dropout)
        self.dec1 = DecoderBlock(64, 128, 64, dropout=0.0)

        # Final upsampling
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split inputs
        eo = x[:, :3, :, :]
        sar = x[:, 3:, :, :]

        # EO Forward
        x0_eo = self.eo_encoder.relu(self.eo_encoder.bn1(self.eo_encoder.conv1(eo)))
        x1_eo = self.eo_encoder.maxpool(x0_eo)
        e1_eo = self.eo_encoder.layer1(x1_eo)
        e2_eo = self.eo_encoder.layer2(e1_eo)
        e3_eo = self.eo_encoder.layer3(e2_eo)
        e4_eo = self.eo_encoder.layer4(e3_eo)

        # SAR Forward Ensure SAR stream doesn't crash on spatial size
        x0_sar = self.sar_encoder.relu(self.sar_encoder.bn1(self.sar_encoder.conv1(sar)))
        x1_sar = self.sar_encoder.maxpool(x0_sar)
        e1_sar = self.sar_encoder.layer1(x1_sar)
        e2_sar = self.sar_encoder.layer2(e1_sar)
        e3_sar = self.sar_encoder.layer3(e2_sar)
        e4_sar = self.sar_encoder.layer4(e3_sar)

        # Late Fusion at deep features
        e4 = torch.cat([e4_eo, e4_sar], dim=1)

        # Apply Vision Transformer Bottleneck over deep joined features
        e4 = self.bottleneck_proj(e4)
        e4_transformed = self.transformer(e4)

        # Skip connections
        e3 = torch.cat([e3_eo, e3_sar], dim=1)
        e2 = torch.cat([e2_eo, e2_sar], dim=1)
        e1 = torch.cat([e1_eo, e1_sar], dim=1)
        x0 = torch.cat([x0_eo, x0_sar], dim=1)

        # Decoding with Transformer Features mapped downwards
        # dec4 takes (x, skip) computes up(x) + skip
        d4 = self.dec4(e4_transformed, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)
        out = self.head(out)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    name = model_cfg.get("name", "unet").lower()
    if name not in ["unet", "resnet_unet", "late_fusion_unet", "difference_fusion_unet", "trans_late_fusion_unet"]:
        raise ValueError(f"Unsupported model: {name}")

    in_channels = int(data_cfg.get("input_channels", 4))
    dropout = float(model_cfg.get("dropout", 0.1))

    if name == "trans_late_fusion_unet":
        return TransLateFusionUNet(dropout=dropout)
    if name == "difference_fusion_unet":
        return DifferenceFusionUNet(dropout=dropout)
    if name == "late_fusion_unet":
        return LateFusionUNet(dropout=dropout)
    return ResNetUNet(in_channels=in_channels, dropout=dropout)
