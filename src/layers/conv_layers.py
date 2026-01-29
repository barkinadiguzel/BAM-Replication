import torch
import torch.nn as nn


def conv1x1(in_channels, out_channels, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=bias
    )


def conv3x3_dilated(in_channels, out_channels, dilation=1, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=dilation,
        dilation=dilation,
        bias=bias
    )


def bn(channels):
    return nn.BatchNorm2d(channels)


def relu():
    return nn.ReLU(inplace=True)
