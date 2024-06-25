import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_rate, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out1 = self.lrelu(self.conv1(x))
        out2 = self.lrelu(self.conv2(torch.cat((x, out1), 1)))
        out3 = self.lrelu(self.conv3(torch.cat((x, out1, out2), 1)))
        out4 = self.lrelu(self.conv4(torch.cat((x, out1, out2, out3), 1)))
        out5 = self.conv5(torch.cat((x, out1, out2, out3, out4), 1))
        return out5 * 0.2 + x

# Residual Dense Network (RDN)
class RDN(nn.Module):
    def __init__(self, in_channels, num_blocks=16, growth_rate=32):
        super(RDN, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, growth_rate, 3, 1, 1)
        self.rdbs = nn.ModuleList([ResidualDenseBlock(growth_rate) for _ in range(num_blocks)])
        self.conv_after_rdbs = nn.Conv2d(num_blocks * growth_rate, growth_rate, 1, 1, 0)
        self.final_conv = nn.Conv2d(growth_rate, in_channels, 3, 1, 1)

    def forward(self, x):
        out = self.initial_conv(x)
        concat_rdbs = torch.cat([rdb(out) for rdb in self.rdbs], 1)
        out = self.conv_after_rdbs(concat_rdbs)
        out = self.final_conv(out)
        return out

# Custom loss function
class SumSquaredError(_Loss):
    def __init__(self, reduction='sum'):
        super(SumSquaredError, self).__init__(reduction=reduction)

    def forward(self, input, target):
        return torch.nn.functional.mse_loss(input, target, reduction=self.reduction).div_(2)