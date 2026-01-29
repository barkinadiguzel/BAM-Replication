import torch
import torch.nn as nn

from blocks.bam_blocks import BAMBlock
from config import REDUCTION_RATIO, DILATION, IN_CHANNELS, NUM_BAM_BLOCKS


class BAMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.bam_blocks = nn.ModuleList([
            BAMBlock(
                in_channels=IN_CHANNELS,
                reduction_ratio=REDUCTION_RATIO,
                dilation=DILATION
            )
            for _ in range(NUM_BAM_BLOCKS)
        ])

    def forward(self, x):
        for bam in self.bam_blocks:
            x = bam(x)
        return x
