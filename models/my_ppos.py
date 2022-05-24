import torch.nn as nn

from .my_modules import UNetEncoder, ASPP


class PPos2D(nn.Module):
    def __init__(self, layers, atrous_rates, num_classes):
        super(PPos2D, self).__init__()
        self.encoder = UNetEncoder(layers, replace_stride_with_dilation=(True, True, True))
        self.aspp = ASPP(512, atrous_rates)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(256, 256 * num_classes, 1), nn.GroupNorm(num_classes, 256 * num_classes),
                                 nn.ReLU(True), nn.Conv2d(256 * num_classes, num_classes, 1, groups=num_classes),
                                 nn.Sigmoid())
        self.num_classes = num_classes

    def forward(self, x):
        x = self.encoder(x)
        x = self.aspp(x)
        x = self.avgpool(x)
        x = self.mlp(x)
        x = x.view(-1, self.num_classes)

        return x
