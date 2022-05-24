import torch
import torch.nn as nn

from .my_modules import UNetEncoder, ASPP


class PCL2D(nn.Module):
    def __init__(self, layers, atrous_rates, num_classes, num_groups):
        super(PCL2D, self).__init__()
        self.encoder = UNetEncoder(layers, replace_stride_with_dilation=(True, True, True))
        self.aspp = ASPP(512, atrous_rates)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(256, 256), nn.GroupNorm(num_groups, 256), nn.ReLU(True),
                                 nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.encoder(x)
        x = self.aspp(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)

        return x
