import torch
import torch.nn as nn
import torch.nn.functional as F

# Swish Activation Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze-and-Excitation Module
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

# Inverted Residual Block
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, kernel_size, se_ratio=0.25, drop_rate=0.0):
        super(InvertedResidualBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_rate = drop_rate

        hidden_channels = in_channels * expansion
        self.expand_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.swish = Swish()

        self.depthwise_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.se = SqueezeExcitation(hidden_channels, reduction=int(1 / se_ratio)) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.swish(self.bn1(self.expand_conv(x)))
        x = self.swish(self.bn2(self.depthwise_conv(x)))
        x = self.se(x)
        x = self.bn3(self.project_conv(x))

        if self.use_residual:
            if self.drop_rate > 0.0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x += residual
        return x

# EfficientNet Model
class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):
        super(EfficientNet, self).__init__()

        def round_filters(filters):
            return int(filters * width_coefficient)

        def round_repeats(repeats):
            return int(repeats * depth_coefficient)

        self.swish = Swish()

        # Stem
        out_channels = round_filters(32)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.swish
        )

        # Configuration for blocks
        block_configs = [
            # (in_channels, out_channels, expansion, stride, kernel_size, repeats)
            (32, 16, 1, 1, 3, 1),
            (16, 24, 6, 2, 3, 2),
            (24, 40, 6, 2, 5, 2),
            (40, 80, 6, 2, 3, 3),
            (80, 112, 6, 1, 5, 3),
            (112, 192, 6, 2, 5, 4),
            (192, 320, 6, 1, 3, 1),
        ]

        # Build blocks
        blocks = []
        for in_channels, out_channels, expansion, stride, kernel_size, repeats in block_configs:
            blocks.append(InvertedResidualBlock(round_filters(in_channels), round_filters(out_channels), expansion, stride, kernel_size))
            for _ in range(round_repeats(repeats) - 1):
                blocks.append(InvertedResidualBlock(round_filters(out_channels), round_filters(out_channels), expansion, 1, kernel_size))
        self.blocks = nn.Sequential(*blocks)

        # Head
        in_channels = round_filters(320)
        out_channels = round_filters(1280)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            self.swish
        )

        # Final Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
