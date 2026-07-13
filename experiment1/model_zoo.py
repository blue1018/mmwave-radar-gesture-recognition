from __future__ import annotations

import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torchvision.models.video import swin3d_t

from experiment_config import EXTERNAL_MOBILEVIT_MODEL_ID, model_batch_size
from training_tools import cleanup_accelerator

# -----------------------------
# Gesture model implementations
# -----------------------------

def group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    # GroupNorm is more stable than BatchNorm for full models with tiny MPS batches.
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class AttentionMeanMaxPooling(nn.Module):
    # Sequence pooling that preserves salient gesture frames better than plain mean pooling.
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = dim * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        weights = torch.softmax(self.score(normed).squeeze(-1), dim=1).unsqueeze(-1)
        attention = (x * weights).sum(dim=1)
        mean = x.mean(dim=1)
        max_value = x.max(dim=1).values
        return self.dropout(torch.cat([attention, mean, max_value], dim=-1))


def motion_difference_view(x: torch.Tensor) -> torch.Tensor:
    # Frame-to-frame absolute difference used as an explicit motion channel.
    diff = torch.zeros_like(x)
    diff[:, 1:] = x[:, 1:] - x[:, :-1]
    return diff.abs()


MODEL_MOTION_INPUT_ENABLED = True


def set_model_motion_input_enabled(enabled: bool) -> None:
    # Set the default motion-channel behaviour for configured model construction.
    global MODEL_MOTION_INPUT_ENABLED
    MODEL_MOTION_INPUT_ENABLED = bool(enabled)


def motion_enhanced_view(x: torch.Tensor, enabled: bool | None = None) -> torch.Tensor:
    # Return raw + temporal-difference channels for Transformer-family models.
    enabled = MODEL_MOTION_INPUT_ENABLED if enabled is None else enabled
    if not enabled:
        return x
    return torch.cat([x, motion_difference_view(x)], dim=2)


def transformer_input_channels(enabled: bool | None = None) -> int:
    enabled = MODEL_MOTION_INPUT_ENABLED if enabled is None else enabled
    return 2 if enabled else 1


def mps_safe_time_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    # Shift frames without torch.roll, which is unsupported on this MPS setup.
    if shift == 0:
        return x
    shifted = torch.zeros_like(x)
    if shift > 0:
        shifted[:, shift:] = x[:, :-shift]
    else:
        shifted[:, :shift] = x[:, -shift:]
    return shifted


def augment_transformer_batch(
    x: torch.Tensor,
    model_name: str | None,
    transformer_model_names: set[str],
    enabled: bool = True,
) -> torch.Tensor:
    # Apply DRAI-safe augmentation only to Transformer-family training batches.
    if not enabled or model_name not in transformer_model_names:
        return x
    if random.random() < 0.50:
        shift = random.randint(-2, 2)
        if shift:
            x = mps_safe_time_shift(x, shift)
    if random.random() < 0.45:
        x = x + 0.012 * torch.randn_like(x)
    if random.random() < 0.35:
        keep = (torch.rand(x.size(0), x.size(1), 1, 1, 1, device=x.device) > 0.08).to(x.dtype)
        x = x * keep
    if random.random() < 0.35:
        height, width = x.shape[-2:]
        erase_h = random.randint(4, 8)
        erase_w = random.randint(4, 8)
        top = random.randint(0, max(0, height - erase_h))
        left = random.randint(0, max(0, width - erase_w))
        x = x.clone()
        x[..., top : top + erase_h, left : left + erase_w] = 0
    return x



class Conv2dFrameEncoder(nn.Module):
    # Shared frame encoder: converts each 32x32 DRAI frame to one feature vector.
    def __init__(self, embed_dim: int = 128, drop_p: float = 0.3, in_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64 * 3 * 3, embed_dim)
        self.drop_p = drop_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x.shape
        x = x.reshape(batch_size * time_steps, channels, height, width)
        x = self.conv(x).reshape(batch_size * time_steps, -1)
        x = F.dropout(self.fc(x), p=self.drop_p, training=self.training)
        return x.reshape(batch_size, time_steps, -1)


class CRNNBaseline(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden_size: int = 192, num_classes: int = 7):
        super().__init__()
        self.encoder = Conv2dFrameEncoder(embed_dim=embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        sequence_output, _ = self.rnn(features)
        return self.classifier(sequence_output[:, -1, :])


class CNNTransformer(nn.Module):
    def __init__(self, seq_len: int = 32, embed_dim: int = 128, num_classes: int = 7):
        super().__init__()
        self.encoder = Conv2dFrameEncoder(embed_dim=embed_dim, in_channels=transformer_input_channels())
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(motion_enhanced_view(x))
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.position[:, : x.size(1)]
        x = self.transformer(x)
        return self.classifier(self.norm(x[:, 0]))


class CNN3DClassifier(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert [B, T, C, H, W] to [B, C, T, H, W] for Conv3d.
        x = x.permute(0, 2, 1, 3, 4)
        return self.classifier(self.features(x))


class MLP(nn.Module):
    # Transformer feed-forward block shared by several models.
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransCNN1D(nn.Module):
    # TRANS-CNN proxy: temporal self-attention followed by local 1D convolutions.
    def __init__(self, seq_len: int = 32, embed_dim: int = 128, num_classes: int = 7):
        super().__init__()
        self.encoder = Conv2dFrameEncoder(embed_dim=embed_dim, drop_p=0.2, in_channels=transformer_input_channels())
        self.position = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, 4, dropout=0.1, batch_first=True)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            group_norm(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            group_norm(embed_dim),
            nn.GELU(),
        )
        self.post_conv_norm = nn.LayerNorm(embed_dim)
        self.post_conv_mlp = MLP(embed_dim, dropout=0.1)
        self.head_norm = nn.LayerNorm(embed_dim)
        self.temporal_pool = AttentionMeanMaxPooling(embed_dim, dropout=0.15)
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(self.temporal_pool.out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(motion_enhanced_view(x)) + self.position[:, : x.size(1)]
        normed = self.norm(x)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attended
        x = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.post_conv_mlp(self.post_conv_norm(x))
        return self.classifier(self.temporal_pool(self.head_norm(x)))


class LocalPyramidBlock(nn.Module):
    # Local perception block: global token attention plus local depthwise convolution.
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.norm_local = nn.LayerNorm(dim)
        self.local_ffn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm_attn(tokens)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + self.attn_drop(attended)

        # The local FFN returns image-shaped features, then we fold them back to tokens.
        local = self.norm_local(tokens).transpose(1, 2).reshape(batch_size, channels, height, width)
        local = self.local_ffn(local).flatten(2).transpose(1, 2)
        tokens = tokens + local
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


class LPVTLite(nn.Module):
    # DRAI-compatible Local Pyramid Vision Transformer lite model.
    def __init__(self, seq_len: int = 32, num_classes: int = 7):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(transformer_input_channels(), 32, kernel_size=4, stride=4), nn.BatchNorm2d(32), nn.GELU())
        self.block1 = LocalPyramidBlock(dim=32, num_heads=2)
        self.stage2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU())
        self.block2 = LocalPyramidBlock(dim=64, num_heads=4)
        self.stage3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU())
        self.block3 = LocalPyramidBlock(dim=128, num_heads=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.position = nn.Parameter(torch.zeros(1, seq_len, 128))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=384,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.norm = nn.LayerNorm(128)
        self.temporal_pool = AttentionMeanMaxPooling(128, dropout=0.15)
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(self.temporal_pool.out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = motion_enhanced_view(x)
        batch_size, time_steps, _, height, width = x.shape
        x = x.reshape(batch_size * time_steps, x.size(2), height, width)
        x = self.block1(self.stage1(x))
        x = self.block2(self.stage2(x))
        x = self.block3(self.stage3(x))
        x = self.pool(x).flatten(1).reshape(batch_size, time_steps, -1)
        x = self.temporal(x + self.position[:, :time_steps])
        return self.classifier(self.temporal_pool(self.norm(x)))


class LPVTFull(nn.Module):
    # Higher-capacity LPVT-inspired model used in the full experiment.
    def __init__(self, seq_len: int = 32, num_classes: int = 7):
        super().__init__()
        dims = [48, 96, 192, 256]
        self.stage1 = nn.Sequential(nn.Conv2d(transformer_input_channels(), dims[0], kernel_size=4, stride=4), group_norm(dims[0]), nn.GELU())
        self.blocks1 = nn.ModuleList([LocalPyramidBlock(dim=dims[0], num_heads=3) for _ in range(2)])
        self.stage2 = nn.Sequential(nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1), group_norm(dims[1]), nn.GELU())
        self.blocks2 = nn.ModuleList([LocalPyramidBlock(dim=dims[1], num_heads=4) for _ in range(2)])
        self.stage3 = nn.Sequential(nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1), group_norm(dims[2]), nn.GELU())
        self.blocks3 = nn.ModuleList([LocalPyramidBlock(dim=dims[2], num_heads=6) for _ in range(4)])
        self.stage4 = nn.Sequential(nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=1), group_norm(dims[3]), nn.GELU())
        self.blocks4 = nn.ModuleList([LocalPyramidBlock(dim=dims[3], num_heads=8) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.position = nn.Parameter(torch.zeros(1, seq_len, dims[-1]))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dims[-1],
            nhead=8,
            dim_feedforward=dims[-1] * 4,
            dropout=0.15,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=3)
        self.norm = nn.LayerNorm(dims[-1])
        self.temporal_pool = AttentionMeanMaxPooling(dims[-1], dropout=0.2)
        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(self.temporal_pool.out_dim, num_classes))

    def _run_stage(self, x: torch.Tensor, stage: nn.Module, blocks: nn.ModuleList) -> torch.Tensor:
        x = stage(x)
        for block in blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = motion_enhanced_view(x)
        batch_size, time_steps, _, height, width = x.shape
        x = x.reshape(batch_size * time_steps, x.size(2), height, width)
        x = self._run_stage(x, self.stage1, self.blocks1)
        x = self._run_stage(x, self.stage2, self.blocks2)
        x = self._run_stage(x, self.stage3, self.blocks3)
        x = self._run_stage(x, self.stage4, self.blocks4)
        x = self.pool(x).flatten(1).reshape(batch_size, time_steps, -1)
        x = self.temporal(x + self.position[:, :time_steps])
        return self.classifier(self.temporal_pool(self.norm(x)))


class MobileViTBlock(nn.Module):
    # Small MobileViT-style block with local convolution and global attention.
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, use_group_norm: bool = False):
        super().__init__()
        norm = group_norm if use_group_norm else nn.BatchNorm2d
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            norm(dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1),
            norm(dim),
            nn.SiLU(inplace=True),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)
        self.fusion = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=1), norm(dim), nn.SiLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        local = self.local(x)
        batch_size, channels, height, width = local.shape
        tokens = local.flatten(2).transpose(1, 2)
        normed = self.norm1(tokens)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        global_map = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        return residual + self.fusion(torch.cat([local, global_map], dim=1))


class MobileViTLite(nn.Module):
    # Lightweight CNN-Transformer model for real-time DRAI sequence experiments.
    def __init__(self, seq_len: int = 32, num_classes: int = 7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(transformer_input_channels(), 24, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(24), nn.SiLU(inplace=True))
        self.down1 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(48), nn.SiLU(inplace=True))
        self.block1 = MobileViTBlock(dim=48, num_heads=4)
        self.down2 = nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(96), nn.SiLU(inplace=True))
        self.block2 = MobileViTBlock(dim=96, num_heads=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.position = nn.Parameter(torch.zeros(1, seq_len, 96))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=96,
            nhead=4,
            dim_feedforward=288,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.norm = nn.LayerNorm(96)
        self.temporal_pool = AttentionMeanMaxPooling(96, dropout=0.15)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.temporal_pool.out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = motion_enhanced_view(x)
        batch_size, time_steps, _, height, width = x.shape
        x = x.reshape(batch_size * time_steps, x.size(2), height, width)
        x = self.stem(x)
        x = self.block1(self.down1(x))
        x = self.block2(self.down2(x))
        x = self.pool(x).flatten(1).reshape(batch_size, time_steps, -1)
        x = self.temporal(x + self.position[:, :time_steps])
        return self.classifier(self.temporal_pool(self.norm(x)))


class MobileViTFull(nn.Module):
    # Higher-capacity MobileViT-inspired CNN/Transformer sequence model.
    def __init__(self, seq_len: int = 32, num_classes: int = 7):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(transformer_input_channels(), 32, kernel_size=3, stride=2, padding=1), group_norm(32), nn.SiLU(inplace=True))
        self.down1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), group_norm(64), nn.SiLU(inplace=True))
        self.blocks1 = nn.ModuleList([MobileViTBlock(dim=64, num_heads=4, use_group_norm=True) for _ in range(2)])
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), group_norm(128), nn.SiLU(inplace=True))
        self.blocks2 = nn.ModuleList([MobileViTBlock(dim=128, num_heads=4, use_group_norm=True) for _ in range(3)])
        self.down3 = nn.Sequential(nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1), group_norm(192), nn.SiLU(inplace=True))
        self.blocks3 = nn.ModuleList([MobileViTBlock(dim=192, num_heads=6, use_group_norm=True) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.position = nn.Parameter(torch.zeros(1, seq_len, 192))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=6,
            dim_feedforward=768,
            dropout=0.15,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=3)
        self.norm = nn.LayerNorm(192)
        self.temporal_pool = AttentionMeanMaxPooling(192, dropout=0.2)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.temporal_pool.out_dim, num_classes))

    def _run_blocks(self, x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = motion_enhanced_view(x)
        batch_size, time_steps, _, height, width = x.shape
        x = x.reshape(batch_size * time_steps, x.size(2), height, width)
        x = self.stem(x)
        x = self._run_blocks(self.down1(x), self.blocks1)
        x = self._run_blocks(self.down2(x), self.blocks2)
        x = self._run_blocks(self.down3(x), self.blocks3)
        x = self.pool(x).flatten(1).reshape(batch_size, time_steps, -1)
        x = self.temporal(x + self.position[:, :time_steps])
        return self.classifier(self.temporal_pool(self.norm(x)))


class PaperMV2Block(nn.Module):
    # Inverted residual block used by the paper-adapted MobileViT models.
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 2):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            group_norm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, groups=hidden_channels, bias=False),
            group_norm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            group_norm(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_residual else out


class PaperMobileViTBranch(nn.Module):
    # One branch of the paper-style RTM/DTM design, adapted here to DRAI/motion-DRAI.
    def __init__(self, stem_dim: int, branch_dim: int, repeat_blocks: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            group_norm(stem_dim),
            nn.SiLU(inplace=True),
        )
        blocks = [PaperMV2Block(stem_dim, branch_dim, stride=2, expansion=2)]
        blocks.extend(PaperMV2Block(branch_dim, branch_dim, stride=1, expansion=2) for _ in range(repeat_blocks))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class PaperMobileViTAdapted(nn.Module):
    # Paper-inspired teacher/student: dual DRAI branches, MV2 compression, MobileViT fusion, sequence pooling.
    def __init__(
        self,
        seq_len: int = 32,
        num_classes: int = 7,
        stem_dim: int = 16,
        branch_dim: int = 32,
        fusion_dim: int = 64,
        repeat_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.raw_branch = PaperMobileViTBranch(stem_dim=stem_dim, branch_dim=branch_dim, repeat_blocks=repeat_blocks)
        self.motion_branch = PaperMobileViTBranch(stem_dim=stem_dim, branch_dim=branch_dim, repeat_blocks=repeat_blocks)
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_dim * 2, fusion_dim, kernel_size=1, bias=False),
            group_norm(fusion_dim),
            nn.SiLU(inplace=True),
        )
        self.mobilevit = MobileViTBlock(dim=fusion_dim, num_heads=num_heads, dropout=0.1, use_group_norm=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.position = nn.Parameter(torch.zeros(1, seq_len, fusion_dim))
        self.temporal_norm = nn.LayerNorm(fusion_dim)
        self.temporal_pool = AttentionMeanMaxPooling(fusion_dim, dropout=dropout * 0.5)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.temporal_pool.out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        motion = motion_difference_view(x)
        batch_size, time_steps, _, height, width = x.shape
        raw = x.reshape(batch_size * time_steps, 1, height, width)
        motion = motion.reshape(batch_size * time_steps, 1, height, width)
        raw_features = self.raw_branch(raw)
        motion_features = self.motion_branch(motion)
        fused = self.fusion(torch.cat([raw_features, motion_features], dim=1))
        fused = self.mobilevit(fused)
        sequence = self.pool(fused).flatten(1).reshape(batch_size, time_steps, -1)
        sequence = self.temporal_norm(sequence + self.position[:, :time_steps])
        return self.classifier(self.temporal_pool(sequence))


class PaperMobileViTTeacher(PaperMobileViTAdapted):
    def __init__(self, seq_len: int = 32, num_classes: int = 7):
        super().__init__(
            seq_len=seq_len,
            num_classes=num_classes,
            stem_dim=16,
            branch_dim=32,
            fusion_dim=64,
            repeat_blocks=2,
            num_heads=4,
            dropout=0.25,
        )


class PaperMobileViTStudent(PaperMobileViTAdapted):
    def __init__(self, seq_len: int = 32, num_classes: int = 7):
        super().__init__(
            seq_len=seq_len,
            num_classes=num_classes,
            stem_dim=8,
            branch_dim=16,
            fusion_dim=32,
            repeat_blocks=1,
            num_heads=2,
            dropout=0.2,
        )


def drai_frames_as_imagenet_batch(x: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    # Convert DRAI frames to normalized 3-channel images for ImageNet-pretrained backbones.
    batch_size, time_steps, _, height, width = x.shape
    frames = x.reshape(batch_size * time_steps, 1, height, width).float()
    frame_min = frames.amin(dim=(-1, -2, -3), keepdim=True)
    frame_max = frames.amax(dim=(-1, -2, -3), keepdim=True)
    frames = (frames - frame_min) / (frame_max - frame_min + 1e-6)
    frames = frames.repeat(1, 3, 1, 1)
    frames = F.interpolate(frames, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device, dtype=frames.dtype).view(1, 3, 1, 1)
    return (frames - mean) / std


class ExternalMobileViTFrozenSequence(nn.Module):
    # External ImageNet-pretrained MobileViT baseline: frozen frame features plus temporal pooling.
    def __init__(
        self,
        seq_len: int = 32,
        num_classes: int = 7,
        embed_dim: int = 256,
        freeze_backbone: bool = True,
        model_id: str = EXTERNAL_MOBILEVIT_MODEL_ID,
    ):
        super().__init__()
        try:
            from transformers import MobileViTForImageClassification
        except ImportError as exc:
            raise ValueError(
                "MobileViT (External Frozen) requires the `transformers` package. "
                "Use the pytorch-real kernel/environment or install transformers in the active environment."
            ) from exc

        try:
            pretrained = MobileViTForImageClassification.from_pretrained(model_id, local_files_only=True)
        except OSError:
            try:
                pretrained = MobileViTForImageClassification.from_pretrained(model_id)
            except OSError as exc:
                raise ValueError(
                    f"Could not load Hugging Face pretrained weights `{model_id}`. "
                    "Make sure the checkpoint is cached or that the notebook can access Hugging Face for the first download."
                ) from exc

        self.model_id = model_id
        self.frame_backbone = pretrained.mobilevit
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            self.frame_backbone.eval()
            for parameter in self.frame_backbone.parameters():
                parameter.requires_grad_(False)
        self.image_size = int(getattr(pretrained.config, "image_size", 256) or 256)
        feature_dim = int(getattr(pretrained.config, "neck_hidden_sizes", [640])[-1])
        self.proj = nn.Sequential(nn.Linear(feature_dim, embed_dim), nn.GELU(), nn.Dropout(0.2))
        self.position = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.temporal_pool = AttentionMeanMaxPooling(embed_dim, dropout=0.2)
        self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(self.temporal_pool.out_dim, num_classes))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.frame_backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps = x.shape[:2]
        frames = drai_frames_as_imagenet_batch(x, image_size=self.image_size)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.frame_backbone(pixel_values=frames, return_dict=True).pooler_output
        else:
            features = self.frame_backbone(pixel_values=frames, return_dict=True).pooler_output
        sequence = self.proj(features).reshape(batch_size, time_steps, -1)
        sequence = self.norm(sequence + self.position[:, :time_steps])
        return self.classifier(self.temporal_pool(sequence))


def temporal_difference_view(x: torch.Tensor) -> torch.Tensor:
    # Motion stream: absolute frame-to-frame difference.
    diff = torch.zeros_like(x)
    diff[:, 1:] = x[:, 1:] - x[:, :-1]
    return diff.abs()


def adaptive_topk_view(x: torch.Tensor, keep_ratio: float = 0.15) -> torch.Tensor:
    # Keep the highest-energy DRAI cells as a simple adaptive top-k proxy.
    batch_size, time_steps, channels, height, width = x.shape
    flat = x.abs().reshape(batch_size, time_steps, -1)
    keep_count = max(1, int(flat.size(-1) * keep_ratio))
    indices = flat.topk(keep_count, dim=-1).indices
    mask = torch.zeros_like(flat).scatter_(-1, indices, 1.0)
    return x * mask.reshape(batch_size, time_steps, channels, height, width)


class Small3DStreamBranch(nn.Module):
    def __init__(self, out_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.permute(0, 2, 1, 3, 4))


class AdaptiveTopKMultiStreamCNN(nn.Module):
    # CNN proxy baseline using DRAI, temporal-difference, and top-k streams.
    def __init__(self, num_classes: int = 7):
        super().__init__()
        branch_dim = 48
        self.raw_branch = Small3DStreamBranch(out_dim=branch_dim)
        self.diff_branch = Small3DStreamBranch(out_dim=branch_dim)
        self.topk_branch = Small3DStreamBranch(out_dim=branch_dim)
        self.classifier = nn.Sequential(
            nn.Linear(branch_dim * 3, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_branch(x)
        diff = self.diff_branch(temporal_difference_view(x))
        topk = self.topk_branch(adaptive_topk_view(x))
        return self.classifier(torch.cat([raw, diff, topk], dim=1))

class FactorizedTimeSformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_spatial = nn.LayerNorm(embed_dim)
        self.norm_temporal = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = MLP(embed_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, patch_count, embed_dim = x.shape

        spatial = self.norm_spatial(x).reshape(batch_size * time_steps, patch_count, embed_dim)
        spatial_out, _ = self.spatial_attn(spatial, spatial, spatial, need_weights=False)
        x = x + spatial_out.reshape(batch_size, time_steps, patch_count, embed_dim)

        temporal = self.norm_temporal(x).permute(0, 2, 1, 3).reshape(batch_size * patch_count, time_steps, embed_dim)
        temporal_out, _ = self.temporal_attn(temporal, temporal, temporal, need_weights=False)
        temporal_out = temporal_out.reshape(batch_size, patch_count, time_steps, embed_dim).permute(0, 2, 1, 3)
        x = x + temporal_out

        return x + self.mlp(self.norm_mlp(x))


class TimeSformerLite(nn.Module):
    def __init__(self, seq_len: int = 32, image_size: int = 32, patch_size: int = 8, embed_dim: int = 128, num_classes: int = 7):
        super().__init__()
        self.patch_embed = nn.Conv2d(transformer_input_channels(), embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_count = (image_size // patch_size) ** 2
        self.temporal_pos = nn.Parameter(torch.zeros(1, seq_len, 1, embed_dim))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.patch_count, embed_dim))
        self.blocks = nn.ModuleList([FactorizedTimeSformerBlock(embed_dim, num_heads=4) for _ in range(2)])
        self.norm = nn.LayerNorm(embed_dim)
        self.token_pool = AttentionMeanMaxPooling(embed_dim, dropout=0.15)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.token_pool.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = motion_enhanced_view(x)
        batch_size, time_steps, _, height, width = x.shape
        x = x.reshape(batch_size * time_steps, x.size(2), height, width)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x.reshape(batch_size, time_steps, self.patch_count, -1)
        x = self.dropout(x + self.temporal_pos[:, :time_steps] + self.spatial_pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).reshape(batch_size, time_steps * self.patch_count, -1)
        return self.classifier(self.token_pool(x))


class TimeSformerFull(nn.Module):
    # Higher-capacity divided space-time transformer for DRAI sequences.
    def __init__(
        self,
        seq_len: int = 32,
        image_size: int = 32,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        num_classes: int = 7,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(transformer_input_channels(), embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_count = (image_size // patch_size) ** 2
        self.temporal_pos = nn.Parameter(torch.zeros(1, seq_len, 1, embed_dim))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.patch_count, embed_dim))
        self.blocks = nn.ModuleList(
            [FactorizedTimeSformerBlock(embed_dim, num_heads=num_heads, dropout=0.15) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.token_pool = AttentionMeanMaxPooling(embed_dim, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.token_pool.out_dim, embed_dim), nn.GELU(), nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = motion_enhanced_view(x)
        batch_size, time_steps, _, height, width = x.shape
        x = x.reshape(batch_size * time_steps, x.size(2), height, width)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x.reshape(batch_size, time_steps, self.patch_count, -1)
        x = self.dropout(x + self.temporal_pos[:, :time_steps] + self.spatial_pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).reshape(batch_size, time_steps * self.patch_count, -1)
        return self.classifier(self.token_pool(x))


class VideoSwinClassifier(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.model = swin3d_t(weights=None, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Torchvision video models expect [B, C, T, H, W] and 3 input channels.
        x = x.permute(0, 2, 1, 3, 4).repeat(1, 3, 1, 1, 1)
        return self.model(x)


def frequency_magnitude_view(x: torch.Tensor) -> torch.Tensor:
    # Frequency stream for the MFF proxy branch.
    magnitude = torch.fft.fft2(x.squeeze(2), norm="ortho").abs().unsqueeze(2)
    magnitude = magnitude / (magnitude.amax(dim=(-1, -2, -3, -4), keepdim=True) + 1e-6)
    return magnitude.float()


class MobileNetFrameBranch(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(576, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _, height, width = x.shape
        x = x.reshape(batch_size * time_steps, 1, height, width).repeat(1, 3, 1, 1)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x).reshape(batch_size, time_steps, -1)


class InterferenceAwareTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.interference_gate = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.mlp = MLP(embed_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        gated = normed * self.interference_gate(normed)
        attn_out, _ = self.attn(gated, gated, gated, need_weights=False)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class MFFCNNIAT(nn.Module):
    # Single-stream DRAI proxy for the multi-feature fusion + IAT idea.
    def __init__(self, seq_len: int = 32, num_classes: int = 7, branch_dim: int = 96, embed_dim: int = 192):
        super().__init__()
        self.raw_branch = MobileNetFrameBranch(out_dim=branch_dim)
        self.diff_branch = MobileNetFrameBranch(out_dim=branch_dim)
        self.freq_branch = MobileNetFrameBranch(out_dim=branch_dim)
        self.fuse = nn.Linear(branch_dim * 3, embed_dim)
        self.position = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.blocks = nn.ModuleList([InterferenceAwareTransformerBlock(embed_dim=embed_dim, num_heads=4) for _ in range(2)])
        self.norm = nn.LayerNorm(embed_dim)
        self.temporal_pool = AttentionMeanMaxPooling(embed_dim, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.temporal_pool.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_branch(x)
        diff = self.diff_branch(temporal_difference_view(x))
        freq = self.freq_branch(frequency_magnitude_view(x))
        x = self.fuse(torch.cat([raw, diff, freq], dim=-1)) + self.position[:, : x.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.dropout(self.temporal_pool(self.norm(x))))

def build_gesture_model(
    model_name: str,
    seq_len: int = 32,
    image_size: int = 32,
    num_classes: int = 7,
    motion_input_enabled: bool = True,
) -> nn.Module:
    # Build one gesture model using the active experiment settings.
    set_model_motion_input_enabled(motion_input_enabled)
    if model_name == "crnn":
        return CRNNBaseline(num_classes=num_classes)
    if model_name == "cnn_transformer":
        return CNNTransformer(seq_len=seq_len, num_classes=num_classes)
    if model_name == "cnn3d":
        return CNN3DClassifier(num_classes=num_classes)
    if model_name == "trans_cnn_1d":
        return TransCNN1D(seq_len=seq_len, num_classes=num_classes)
    if model_name == "lpvt_lite":
        return LPVTLite(seq_len=seq_len, num_classes=num_classes)
    if model_name == "lpvt_full":
        return LPVTFull(seq_len=seq_len, num_classes=num_classes)
    if model_name == "mobilevit_lite":
        return MobileViTLite(seq_len=seq_len, num_classes=num_classes)
    if model_name == "mobilevit_full":
        return MobileViTFull(seq_len=seq_len, num_classes=num_classes)
    if model_name == "paper_mobilevit_teacher":
        return PaperMobileViTTeacher(seq_len=seq_len, num_classes=num_classes)
    if model_name == "paper_mobilevit_student":
        return PaperMobileViTStudent(seq_len=seq_len, num_classes=num_classes)
    if model_name == "external_mobilevit_frozen":
        return ExternalMobileViTFrozenSequence(seq_len=seq_len, num_classes=num_classes)
    if model_name == "adaptive_topk_multistream_cnn":
        return AdaptiveTopKMultiStreamCNN(num_classes=num_classes)
    if model_name == "timesformer_lite":
        return TimeSformerLite(seq_len=seq_len, image_size=image_size, num_classes=num_classes)
    if model_name == "timesformer_full":
        return TimeSformerFull(seq_len=seq_len, image_size=image_size, num_classes=num_classes)
    if model_name == "mff_cnn_iat":
        return MFFCNNIAT(seq_len=seq_len, num_classes=num_classes)
    if model_name == "video_swin":
        return VideoSwinClassifier(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model: nn.Module) -> float:
    # Return model size in millions of parameters.
    return sum(param.numel() for param in model.parameters()) / 1e6


def model_registry_frame(
    model_names,
    display_names: dict[str, str],
    build_model_fn,
    batch_size_fn=model_batch_size,
) -> pd.DataFrame:
    # Build a compact parameter and batch-size table for selected models.
    # Import pandas locally to keep the registry function self-contained.
    import pandas as pd

    rows = []
    for model_name in model_names:
        model = None
        try:
            model = build_model_fn(model_name)
            params_m = round(count_parameters(model), 3)
            available = True
            registry_error = ""
        except (RuntimeError, NotImplementedError, ValueError) as exc:
            params_m = float("nan")
            available = False
            registry_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "model": model_name,
                "display_name": display_names[model_name],
                "params_m": params_m,
                "batch_size": batch_size_fn(model_name),
                "available": available,
                "registry_error": registry_error,
            }
        )
        cleanup_accelerator(model)
    return pd.DataFrame(rows)
