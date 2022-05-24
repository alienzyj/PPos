import torch
import torch.nn as nn

from .my_modules import ResidualUBlock, UNetEncoder, ASPP, SpatialAttention



class Genesis2D(nn.Module):
    def __init__(self, layers, heights, atrous_rates):
        super(Genesis2D, self).__init__()
        self.encoder = UNetEncoder(layers, replace_stride_with_dilation=(True, True, True))
        self.aspp = ASPP(512, atrous_rates)

        self.spatial_attention1 = SpatialAttention("2D")
        self.spatial_attention2 = SpatialAttention("2D")
        self.spatial_attention3 = SpatialAttention("2D")
        self.spatial_attention4 = SpatialAttention("2D")
        self.spatial_attention5 = SpatialAttention("2D")

        self.up1 = ResidualUBlock(768, 256, 128, heights[0], "2D")
        self.up2 = ResidualUBlock(512, 128, 64, heights[1], "2D")
        self.up3 = ResidualUBlock(256, 64, 32, heights[2], "2D")
        self.up4 = ResidualUBlock(128, 32, 32, heights[3], "2D")
        self.up5 = ResidualUBlock(64, 32, 32, heights[4], "2D")

        self.up_sample2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.voxel_classifier = nn.Sequential(nn.Conv2d(32, 14, 1), nn.Sigmoid())

    def forward(self, x):
        head = self.encoder.head(x)
        down1 = self.encoder.layer1(head)
        down2 = self.encoder.layer2(down1)
        down3 = self.encoder.layer3(down2)
        down4 = self.encoder.layer4(down3)
        x = self.aspp(down4)

        x = torch.cat([down4, x], 1)
        x = self.spatial_attention1(x)
        x = self.up1(x)

        x = torch.cat([down3, x], 1)
        x = self.spatial_attention2(x)
        x = self.up2(x)

        x = torch.cat([down2, x], 1)
        x = self.spatial_attention3(x)
        x = self.up3(x)

        x = torch.cat([down1, x], 1)
        x = self.spatial_attention4(x)
        x = self.up4(x)

        x = torch.cat([head, self.up_sample2x(x)], 1)
        x = self.spatial_attention5(x)
        x = self.up5(x)
        x = self.up_sample2x(x)
        out = self.voxel_classifier(x)

        return out
