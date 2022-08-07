import torch
import torch.nn as nn
from torch.nn import functional as F


class MyBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, outplanes, num_groups, stride=1, dilation=1):
        super(MyBottleneck, self).__init__()
        planes = inplanes * self.expansion
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1)
        self.gn1 = nn.GroupNorm(num_groups, planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation)
        self.gn2 = nn.GroupNorm(num_groups, planes)
        self.conv3 = nn.Conv3d(planes, outplanes, kernel_size=1)
        self.gn3 = nn.GroupNorm(num_groups, outplanes)
        self.relu = nn.ReLU(True)
        self.gc = GCBlock(outplanes, num_groups)

        if inplanes != outplanes:
            self.trans = nn.Sequential(nn.Conv3d(inplanes, outplanes, 1), nn.GroupNorm(num_groups, outplanes))

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.gn3(x)
        x = self.gc(x)

        if self.inplanes != self.outplanes:
            residual = self.trans(residual)

        x += residual
        x = self.relu(x)

        return x


class ResidualUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, internal_channels, num_groups, heights,dilation=1):
        super(ResidualUBlock, self).__init__()
        self.heights = heights
        self.block1 = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
                                    nn.GroupNorm(num_groups, out_channels), nn.ReLU(True))
        self.block2 = nn.Sequential(nn.Conv3d(out_channels, internal_channels, 3, padding=dilation, dilation=dilation),
                                    nn.GroupNorm(num_groups, internal_channels), nn.ReLU(True))
        self.block3 = nn.Sequential(nn.Conv3d(internal_channels, internal_channels, 3, padding=1),
                                    nn.GroupNorm(num_groups, internal_channels), nn.ReLU(True))
        self.block4 = nn.Sequential(
            nn.Conv3d(2 * internal_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels), nn.ReLU(True))
        self.block5 = nn.Sequential(nn.Conv3d(2 * out_channels, out_channels, 3, padding=1),
                                    nn.GroupNorm(num_groups, out_channels))
        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.maxpool = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.gc = GCBlock(out_channels, num_groups)
        self.spatial_attention = SpatialAttention()
        for i in range(heights):
            self.down_list.append(
                nn.Sequential(nn.Conv3d(internal_channels, internal_channels, 3, padding=1),
                              nn.GroupNorm(num_groups, internal_channels), nn.ReLU(True)))
            self.up_list.append(nn.Sequential(
                nn.Conv3d(2 * internal_channels, internal_channels, 3, padding=1),
                nn.GroupNorm(num_groups, internal_channels), nn.ReLU(True)))

        self.trans = nn.Sequential(nn.Conv3d(in_channels, out_channels, 1), nn.GroupNorm(num_groups, out_channels))

    def forward(self, x):
        internal_outputs = []
        res = self.trans(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x = x2
        for i, block in enumerate(self.down_list):
            x = self.maxpool(x)
            x = block(x)
            internal_outputs.append(x)
        x = self.block3(x)
        x = self.up_list[0](torch.cat((x, internal_outputs[-1]), 1))
        for i, block in enumerate(self.up_list[1:]):
            x = block(torch.cat((self.upsample(x), internal_outputs[-i - 2]), 1))
        x = self.block4(torch.cat((self.upsample(x), x2), 1))
        x = self.block5(torch.cat((x, x1), 1))
        x = self.gc(x)
        x = self.spatial_attention(x)
        x = x + res
        x = F.relu(x, True)

        return x


class GCBlock(nn.Module):
    def __init__(self, inplanes, num_groups, ratio=8):
        super(GCBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.channel_add_conv = nn.Sequential(
            nn.Conv3d(inplanes, inplanes // ratio, kernel_size=1),
            nn.GroupNorm(num_groups // ratio, inplanes // ratio),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplanes // ratio, inplanes, kernel_size=1))
        self.last_zero_init(self.channel_add_conv)

    def forward(self, x):
        context = self.avgpool(x)
        context = self.channel_add_conv(context)
        context = F.interpolate(context, x.size()[2:], mode='nearest')
        x = x + context

        return x

    @staticmethod
    def last_zero_init(model):
        if isinstance(model, nn.Sequential):
            for m in model.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.constant_(m.weight, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


class UNetClassifierEncoder(nn.Module):
    def __init__(self, heights, num_groups):
        super(UNetClassifierEncoder, self).__init__()

        self.head = nn.Sequential(nn.Conv3d(2, 32, 7, padding=3),
                                  nn.GroupNorm(num_groups, 32), nn.ReLU(True))

        self.down_dense1 = nn.Sequential(ResidualUBlock(32, 32, 32, num_groups, heights[0]),
                                         ResidualUBlock(32, 32, 32, num_groups, heights[0]))
        self.down_dense2 = nn.Sequential(ResidualUBlock(32, 32, 32, num_groups,  heights[1]),
                                         ResidualUBlock(32, 64, 32, num_groups,  heights[1]))
        self.down_dense3 = nn.Sequential(ResidualUBlock(64, 64, 64, num_groups, heights[2], 2),
                                         ResidualUBlock(64, 128, 64, num_groups, heights[2], 2))
        self.down_dense4 = nn.Sequential(ResidualUBlock(128, 128, 128, num_groups, heights[3], 4),
                                         ResidualUBlock(128, 256, 128, num_groups, heights[3], 4))

        self.down_sample = nn.MaxPool3d(2, 2)

    def forward(self, x):
        x = self.head(x)
        x = self.down_sample(x)

        x = self.down_dense1(x)
        x = self.down_sample(x)

        x = self.down_dense2(x)
        x = self.down_dense3(x)
        x = self.down_dense4(x)

        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, 7, padding=3)

    def forward(self, inputs):
        return inputs * torch.sigmoid(
            self.conv(torch.cat([torch.mean(inputs, 1, keepdim=True), torch.max(inputs, 1, keepdim=True)[0]], 1)))


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, num_groups):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, num_groups):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(True))

    def forward(self, x):
        size = x.shape[2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, axial_atrous_rates, depth_atrous_rates, num_groups, drop_rate):
        super(ASPP, self).__init__()
        modules = [nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(True))]

        rate = []
        for i in range(3):
            rate.append((depth_atrous_rates[i], axial_atrous_rates[i], axial_atrous_rates[i]))
        modules.append(ASPPConv(in_channels, out_channels, rate[0], num_groups))
        modules.append(ASPPConv(in_channels, out_channels, rate[1], num_groups))
        modules.append(ASPPConv(in_channels, out_channels, rate[2], num_groups))
        modules.append(ASPPPooling(in_channels, out_channels, num_groups))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(5 * out_channels, out_channels, 1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(True),
            nn.Dropout(drop_rate))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def make_bottleneck(inplanes, outplanes, blocks, num_groups,  stride=1, dilation=1):
    layers = [MyBottleneck(inplanes, outplanes, num_groups, stride, dilation)]
    for i in range(blocks - 1):
        layers.append(MyBottleneck(outplanes, outplanes, num_groups, stride, dilation))

    return nn.Sequential(*layers)


class UNetClassifier(nn.Module):
    def __init__(self, encoder_heights, unet_heights, classifier_blocks, num_groups, axial_atrous_rates,
                 depth_atrous_rates, drop_rate):
        super(UNetClassifier, self).__init__()

        self.encoder = UNetClassifierEncoder(encoder_heights, num_groups)
        self.aspp = ASPP(256, 256, axial_atrous_rates, depth_atrous_rates, num_groups, drop_rate)

        self.seg1 = nn.Conv3d(32, 32, 1)
        self.seg2 = nn.Conv3d(64, 64, 1)
        self.seg3 = nn.Conv3d(128, 128, 1)
        self.seg4 = nn.Conv3d(256, 256, 1)

        self.cls1 = nn.Conv3d(32, 32, 1)
        self.cls2 = nn.Conv3d(64, 64, 1)
        self.cls3 = nn.Conv3d(128, 128, 1)
        self.cls4 = nn.Conv3d(256, 256, 1)

        self.skip1 = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(64, 32, 1))
        self.skip2 = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(128, 64, 1))
        self.skip3 = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(256, 128, 1))
        self.skip4 = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(512, 256, 1))

        self.spatial_attention1 = SpatialAttention()
        self.spatial_attention2 = SpatialAttention()
        self.spatial_attention3 = SpatialAttention()
        self.spatial_attention4 = SpatialAttention()

        self.up_dense1 = nn.Sequential(ResidualUBlock(736, 256, 128, num_groups, unet_heights[0]),
                                       ResidualUBlock(256, 128, 128, num_groups,  unet_heights[0]))
        self.up_dense2 = nn.Sequential(ResidualUBlock(352, 128, 64, num_groups,  unet_heights[1]),
                                       ResidualUBlock(128, 64, 64, num_groups,  unet_heights[1]))
        self.up_dense3 = nn.Sequential(ResidualUBlock(288, 64, 32, num_groups,  unet_heights[2]),
                                       ResidualUBlock(64, 32, 32, num_groups, unet_heights[2]))
        self.up_dense4 = nn.Sequential(ResidualUBlock(256, 32, 32, num_groups,  unet_heights[3]),
                                       ResidualUBlock(32, 32, 32, num_groups, unet_heights[3]))

        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.voxel_classifier5 = nn.Sequential(nn.Conv3d(64, 64, 3, padding=1),
                                               nn.GroupNorm(num_groups, 64),
                                               nn.ReLU(True),
                                               nn.Conv3d(64, 1, 1),
                                               nn.Sigmoid())
        self.voxel_classifier4 = nn.Sequential(nn.Conv3d(32, 1, 3, padding=1), nn.Sigmoid())
        self.voxel_classifier3 = nn.Sequential(nn.Conv3d(32, 1, 3, padding=1), nn.Sigmoid())
        self.voxel_classifier2 = nn.Sequential(nn.Conv3d(64, 1, 3, padding=1), nn.Sigmoid())
        self.voxel_classifier1 = nn.Sequential(nn.Conv3d(128, 1, 3, padding=1), nn.Sigmoid())

        self.cls_layer1 = make_bottleneck(32, 64, classifier_blocks[0], num_groups)
        self.cls_layer2 = make_bottleneck(128, 128, classifier_blocks[1], num_groups)
        self.cls_layer3 = make_bottleneck(256, 256, classifier_blocks[2], num_groups)
        self.cls_layer4 = make_bottleneck(512, 512, classifier_blocks[3], num_groups)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool2x = nn.MaxPool3d(2, 2)
        self.maxpool4x = nn.MaxPool3d(4, 4)
        self.mlp = nn.Sequential(nn.Linear(256 * 3, 1), nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        head = self.encoder.head(x)
        x = self.encoder.down_sample(head)

        down_dense1 = self.encoder.down_dense1(x)
        x = self.encoder.down_sample(down_dense1)

        down_dense2 = self.encoder.down_dense2(x)
        down_dense3 = self.encoder.down_dense3(down_dense2)
        down_dense4 = self.encoder.down_dense4(down_dense3)

        aspp_out = self.aspp(down_dense4)

        x = self.cls1(down_dense1)
        x = self.cls_layer1(x)
        context1 = self.skip1(x)
        context1 = F.interpolate(context1, down_dense1.size()[2:], mode='nearest')
        x = self.maxpool2x(x)
        x = self.cls_layer2(torch.cat([x, self.cls2(down_dense2)], 1))
        context2 = self.skip2(x)
        context2 = F.interpolate(context2, down_dense2.size()[2:], mode='nearest')
        x = self.maxpool2x(x)
        x = self.cls_layer3(torch.cat([x, self.cls3(self.maxpool2x(down_dense3))], 1))
        context3 = self.skip3(x)
        context3 = F.interpolate(context3, down_dense3.size()[2:], mode='nearest')
        x = self.maxpool2x(x)
        x = self.cls_layer4(torch.cat([x, self.cls4(self.maxpool4x(down_dense4))], 1))
        context4 = self.skip4(x)
        context4 = F.interpolate(context4, down_dense4.size()[2:], mode='nearest')
        x = torch.cat([self.avgpool(aspp_out), self.avgpool(x)], 1)
        x = torch.flatten(x, 1)
        label = self.mlp(x)

        seg3 = self.seg3(down_dense3)
        seg2 = self.seg2(down_dense2)
        seg1 = self.seg1(down_dense1)

        x = torch.cat((self.seg4(down_dense4) + context4, aspp_out), 1)
        x = self.spatial_attention1(x)
        x = self.up_dense1(torch.cat((x, seg3, seg2, self.maxpool2x(seg1)), 1))
        up_dense1 = x
        out1 = self.voxel_classifier1(x)

        x = torch.cat((seg3 + context3, x), 1)
        x = self.spatial_attention2(x)
        x = self.up_dense2(torch.cat((x, seg2, self.maxpool2x(seg1)), 1))
        up_dense2 = x
        out2 = self.voxel_classifier2(x)

        x = torch.cat((seg2 + context2, x), 1)
        x = self.spatial_attention3(x)
        x = self.up_dense3(torch.cat((x, self.maxpool2x(seg1), up_dense1), 1))
        out3 = self.voxel_classifier3(x)

        x = torch.cat((seg1 + context1, self.up_sample(x)), 1)
        x = self.spatial_attention4(x)
        x = self.up_dense4(torch.cat((x, self.up_sample(up_dense1), self.up_sample(up_dense2)), 1))
        out4 = self.voxel_classifier4(x)

        x = self.voxel_classifier5(torch.cat((head, self.up_sample(x)), 1))

        return out1, out2, out3, out4, x, label
