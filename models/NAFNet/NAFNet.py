import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


# Define a Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# Define a Residual Block with Channel Attention
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        out += residual
        return out


# Define a Channel Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, padding=0)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = x * self.sigmoid(avg_out)
        return out


# Define the NafNet Model
class NafNet(nn.Module):
    def __init__(self, depth=17, in_channels=64, image_channels=3, kernel_size=3):
        super(NafNet, self).__init__()
        self.initial_conv = ConvBlock(image_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.res_blocks = nn.Sequential(*[ResidualBlock(in_channels) for _ in range(depth-2)])
        self.final_conv = nn.Conv2d(in_channels, image_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        out = self.initial_conv(x)
        out = self.res_blocks(out)
        out = self.final_conv(out)
        return self.tanh(out)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def create_NAFNet(depth=17, in_channels=64, image_channels=3, kernel_size=3):
    return NafNet(depth=depth, in_channels=in_channels, image_channels=image_channels, kernel_size=kernel_size)
