import torch
import torch.nn as nn
from torch.nn import functional as F


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, padding=dilation, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.gc = GCBlock(planes * self.expansion, "2D")

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.gc(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out


class ResidualUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, internal_channels, heights, input_dimension, dilation=1):
        super(ResidualUBlock, self).__init__()
        self.heights = heights

        if input_dimension == "2D":
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
            self.maxpool = nn.MaxPool2d(2, 2)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        elif input_dimension == "3D":
            conv = nn.Conv3d
            norm = nn.GroupNorm
            self.maxpool = nn.MaxPool3d(2, 2)
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        else:
            raise ValueError("Please specify the correct input dimension")

        self.block1 = nn.Sequential(*CNR(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False,
                                         input_dimension=input_dimension))
        self.block2 = nn.Sequential(
            *CNR(out_channels, internal_channels, 3, padding=dilation, dilation=dilation, bias=False,
                 input_dimension=input_dimension))
        self.block3 = nn.Sequential(
            *CNR(internal_channels, internal_channels, 3, padding=1, bias=False, input_dimension=input_dimension))
        self.block4 = nn.Sequential(
            *CNR(2 * internal_channels, out_channels, 3, padding=1, bias=False, input_dimension=input_dimension))
        self.block5 = nn.Sequential(
            *CNR(2 * out_channels, out_channels, 3, padding=1, bias=False, input_dimension=input_dimension))
        self.down_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        self.gc = GCBlock(out_channels, input_dimension)
        self.spatial_attention = SpatialAttention(input_dimension)
        for i in range(heights):
            self.down_list.append(
                nn.Sequential(*CNR(internal_channels, internal_channels, 3, padding=1, bias=False,
                                   input_dimension=input_dimension)))
            self.up_list.append(nn.Sequential(*CNR(2 * internal_channels, internal_channels, 3, padding=1, bias=False,
                                                   input_dimension=input_dimension)))

        self.trans = nn.Sequential(*CNR(in_channels, out_channels, 1, bias=True, input_dimension=input_dimension)[:-1])

        for m in self.modules():
            if isinstance(m, conv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
    def __init__(self, inplanes, input_dimension, ratio=16):
        super(GCBlock, self).__init__()
        if input_dimension == "2D":
            conv = nn.Conv2d
            dim = (1, 1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        elif input_dimension == "3D":
            conv = nn.Conv3d
            dim = (1, 1, 1)
            self.avgpool = nn.AdaptiveAvgPool3d(1)
        else:
            raise ValueError("Please specify the correct input dimension")

        self.channel_add_conv = nn.Sequential(
            conv(inplanes, inplanes // ratio, kernel_size=1),
            nn.LayerNorm([inplanes // ratio, *dim]),
            nn.ReLU(inplace=True),
            conv(inplanes // ratio, inplanes, kernel_size=1))
        self.last_zero_init(self.channel_add_conv)

    def forward(self, x):
        context = self.avgpool(x)
        context = self.channel_add_conv(context)
        x = x + context

        return x

    @staticmethod
    def last_zero_init(model):
        if isinstance(model, nn.Sequential):
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    nn.init.constant_(m.weight, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


class UNetEncoder(nn.Module):
    def __init__(self, layers, block=Bottleneck2D, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(UNetEncoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.head = nn.Sequential(nn.Conv2d(14, self.inplanes, kernel_size=7, stride=2, padding=3,
                                            bias=False, groups=2), norm_layer(self.inplanes), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck2D):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, input_dimension):
        super(SpatialAttention, self).__init__()
        if input_dimension == "2D":
            self.conv = nn.Conv2d(2, 1, 7, padding=3)
        elif input_dimension == "3D":
            self.conv = nn.Conv3d(2, 1, 7, padding=3)
        else:
            raise ValueError("Please specify the correct input dimension")

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inputs):
        return inputs * torch.sigmoid(
            self.conv(torch.cat([torch.mean(inputs, 1, keepdim=True), torch.max(inputs, 1, keepdim=True)[0]], 1)))


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]

        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def CNR(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
        padding_mode='zeros', input_dimension="3D"):
    if input_dimension == "2D":
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                         padding_mode), nn.BatchNorm2d(out_channels), nn.ReLU(True)
    elif input_dimension == "3D":
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                         padding_mode), nn.GroupNorm(8, out_channels), nn.ReLU(True)
    else:
        raise ValueError("Please specify the correct input dimension")
