import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import matplotlib.pyplot as plt
import os
import numpy as np

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden_channels = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.fc(self.avg_pool(x))
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.conv(x_cat)

class FrequencyAttention(nn.Module):
    def __init__(self, in_channels, mode, vis_prefix=None):
        super().__init__()
        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.vis_prefix = vis_prefix
        self.h = 0
        self.l = 0

    def forward(self, x):
        freq = torch.fft.fft2(x, norm='ortho')
        amp = torch.abs(freq)
        B, C, H, W = amp.shape
        mask = torch.zeros_like(amp, device=amp.device)
        h_center, w_center = H // 2, W // 2
        h_half, w_half = int(H * 0.25 // 2), int(W * 0.25 // 2)
        if self.mode == 'low':
            mask[:, :, h_center - h_half:h_center + h_half, w_center - w_half:w_center + w_half] = 1
        elif self.mode == 'high':
            mask[:, :, :] = 1
            mask[:, :, h_center - h_half:h_center + h_half, w_center - w_half:w_center + w_half] = 0

        amp_masked = amp * mask

        attn = self.conv(amp_masked)
        return x * attn

class DualBranchFrequencyBlock(nn.Module):
    def __init__(self, in_channels, vis_prefix=None):
        super().__init__()
        self.high_attn = FrequencyAttention(in_channels, mode='high', vis_prefix=vis_prefix)
        self.low_attn = FrequencyAttention(in_channels, mode='low', vis_prefix=vis_prefix)
        self.fuse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_high = self.high_attn(x)
        x_low = self.low_attn(x)
        x_cat = torch.cat([x_high, x_low], dim=1)
        gate = self.fuse(x_cat)
        return x_high * gate + x_low * (1 - gate)

class MambaBlock(nn.Module):
    def __init__(self, dim, vis_prefix=None):
        """
        vis_prefix: 可视化文件名前缀（如 '256_enc2'）
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(dim)
        self.freq_attn = DualBranchFrequencyBlock(dim, vis_prefix=vis_prefix)
        self.vis_prefix = vis_prefix
        self.i = 0
    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x_2d = x.transpose(1, 2).view(B, C, H, W)
        x_2d = self.freq_attn(x_2d)
        x = x_2d.flatten(2).transpose(1, 2)
        return self.mamba(self.norm(x))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, vis_prefix=None):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.block = MambaBlock(out_ch, vis_prefix=vis_prefix)

    def forward(self, x):
        x = self.pool(self.proj(x))
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.block(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, block_dim, vis_prefix=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.block = MambaBlock(block_dim, vis_prefix=vis_prefix)
        self.out_conv = nn.Conv2d(block_dim, out_ch, kernel_size=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.block(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.out_conv(x)
        return x

class MambaUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_dim=32, prefix="256"):
        super().__init__()
        self.prefix = prefix  # 用于可视化标识

        self.inc = nn.Conv2d(in_ch, base_dim, 3, padding=1)
        self.down1 = Down(base_dim, base_dim * 2, vis_prefix=f"{prefix}_enc1")
        self.down2 = Down(base_dim * 2, base_dim * 4, vis_prefix=f"{prefix}_enc2")
        self.down3 = Down(base_dim * 4, base_dim * 8, vis_prefix=f"{prefix}_enc3")
        self.down4 = Down(base_dim * 8, base_dim * 8, vis_prefix=f"{prefix}_enc4")

        self.up1 = Up(base_dim * 8, base_dim * 4, base_dim * 8 + base_dim * 4, vis_prefix=f"{prefix}_dec4")
        self.up2 = Up(base_dim * 4, base_dim * 2, base_dim * 4 + base_dim * 2, vis_prefix=f"{prefix}_dec3")
        self.up3 = Up(base_dim * 2, base_dim, base_dim * 2 + base_dim, vis_prefix=f"{prefix}_dec2")
        self.up4 = Up(base_dim, base_dim, base_dim + base_dim, vis_prefix=f"{prefix}_dec1")
        self.outc = nn.Conv2d(base_dim, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x), [x, x1, x2, x3]

class MultiScaleMambaFusion(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.attn_pan_c = ChannelAttention(in_ch)
        self.attn_pan_s = SpatialAttention()
        self.attn_ms_c = ChannelAttention(in_ch)
        self.attn_ms_s = SpatialAttention()

        self.unet_256 = MambaUNet(in_ch * 2, out_ch, prefix="256")
        self.unet_128 = MambaUNet(in_ch * 2, out_ch, prefix="128")
        self.unet_64 = MambaUNet(in_ch * 2, out_ch, prefix="64")

        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch * 3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, pan, ms):
        pan = self.attn_pan_s(self.attn_pan_c(pan))
        ms = self.attn_ms_s(self.attn_ms_c(ms))

        ms_256 = F.interpolate(ms, size=(256, 256), mode='bilinear', align_corners=True)
        x1 = torch.cat([pan, ms_256], dim=1)
        out_256, feat_256 = self.unet_256(x1)

        pan_128 = F.interpolate(pan, size=(128, 128), mode='bilinear', align_corners=True)
        ms_128 = F.interpolate(ms, size=(128, 128), mode='bilinear', align_corners=True)
        x2 = torch.cat([pan_128, ms_128], dim=1)
        out_128, feat_128 = self.unet_128(x2)
        out_128_up = F.interpolate(out_128, size=(256, 256), mode='bilinear', align_corners=True)

        pan_64 = F.interpolate(pan, size=(64, 64), mode='bilinear', align_corners=True)
        x3 = torch.cat([pan_64, ms], dim=1)
        out_64, feat_64 = self.unet_64(x3)
        out_64_up = F.interpolate(out_64, size=(256, 256), mode='bilinear', align_corners=True)

        fused = torch.cat([out_256, out_128_up, out_64_up], dim=1)
        out = self.fusion(fused)

        return out, {"out_256": out_256, "out_128": out_128, "out_64": out_64,
                     "feat_256": feat_256, "feat_128": feat_128, "feat_64": feat_64}


