import torch
import torch.nn as nn

from blocks.bam_blocks import BAMBlock
from config import REDUCTION_RATIO, DILATION


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class BAMStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

        self.bam = BAMBlock(
            in_channels=out_channels,
            reduction_ratio=REDUCTION_RATIO,
            dilation=DILATION
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bam(x)      
        x = self.pool(x)
        return x


class BAMModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage1 = BAMStage(in_channels, 64)
        self.stage2 = BAMStage(64, 128)
        self.stage3 = BAMStage(128, 256)
        self.stage4 = BAMStage(256, 512)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
