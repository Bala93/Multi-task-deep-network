from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class PsiNet(nn.Module):
    """
    Adapted from Vanilla UNet implementation - https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        input_channels: int = 3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample1 = nn.Upsample(scale_factor=2)
        upsample_bottom1 = nn.Upsample(scale_factor=bottom_s)
        upsample2 = nn.Upsample(scale_factor=2)
        upsample_bottom2 = nn.Upsample(scale_factor=bottom_s)
        upsample3 = nn.Upsample(scale_factor=2)
        upsample_bottom3 = nn.Upsample(scale_factor=bottom_s)

        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers1 = [upsample1] * len(self.up)
        self.upsamplers1[-1] = upsample_bottom1
        self.upsamplers2 = [upsample2] * len(self.up)
        self.upsamplers2[-1] = upsample_bottom2
        self.upsamplers3 = [upsample3] * len(self.up)
        self.upsamplers3[-1] = upsample_bottom3

        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        if add_output:
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        if add_output:
            self.conv_final3 = nn.Conv2d(up_filter_sizes[0], 1, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        x_out1 = x_out
        x_out2 = x_out
        x_out3 = x_out

        # Decoder mask segmentation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers1, self.up))
        ):
            x_out1 = upsample(x_out1)
            x_out1 = up(torch.cat([x_out1, x_skip], 1))

        # Decoder contour estimation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers2, self.up))
        ):
            x_out2 = upsample(x_out2)
            x_out2 = up(torch.cat([x_out2, x_skip], 1))

        # Regression
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers3, self.up))
        ):
            x_out3 = upsample(x_out3)
            x_out3 = up(torch.cat([x_out3, x_skip], 1))

        if self.add_output:
            x_out1 = self.conv_final1(x_out1)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)

        if self.add_output:
            x_out2 = self.conv_final2(x_out2)
            if self.num_classes > 1:
                x_out2 = F.log_softmax(x_out2, dim=1)

        if self.add_output:
            x_out3 = self.conv_final3(x_out3)
            x_out3 = F.sigmoid(x_out3)

        return [x_out1, x_out2, x_out3]


class UNet_DCAN(nn.Module):
    """
    Adapted from Vanilla UNet implementation - https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        input_channels: int = 3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample1 = nn.Upsample(scale_factor=2)
        upsample_bottom1 = nn.Upsample(scale_factor=bottom_s)
        upsample2 = nn.Upsample(scale_factor=2)
        upsample_bottom2 = nn.Upsample(scale_factor=bottom_s)

        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers1 = [upsample1] * len(self.up)
        self.upsamplers1[-1] = upsample_bottom1
        self.upsamplers2 = [upsample2] * len(self.up)
        self.upsamplers2[-1] = upsample_bottom2

        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        if add_output:
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        x_out1 = x_out
        x_out2 = x_out

        # Decoder mask segmentation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers1, self.up))
        ):
            x_out1 = upsample(x_out1)
            x_out1 = up(torch.cat([x_out1, x_skip], 1))

        # Decoder contour estimation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers2, self.up))
        ):
            x_out2 = upsample(x_out2)
            x_out2 = up(torch.cat([x_out2, x_skip], 1))

        if self.add_output:
            x_out1 = self.conv_final1(x_out1)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)

        if self.add_output:
            x_out2 = self.conv_final2(x_out2)
            if self.num_classes > 1:
                x_out2 = F.log_softmax(x_out2, dim=1)

        return [x_out1, x_out2]


class UNet_DMTN(nn.Module):
    """
    Adapted from Vanilla UNet implementation - https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        input_channels=3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample1 = nn.Upsample(scale_factor=2)
        upsample_bottom1 = nn.Upsample(scale_factor=bottom_s)
        upsample2 = nn.Upsample(scale_factor=2)
        upsample_bottom2 = nn.Upsample(scale_factor=bottom_s)

        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers1 = [upsample1] * len(self.up)
        self.upsamplers1[-1] = upsample_bottom1
        self.upsamplers2 = [upsample2] * len(self.up)
        self.upsamplers2[-1] = upsample_bottom2

        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
        if add_output:
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], 1, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        x_out1 = x_out
        x_out2 = x_out

        # Decoder mask segmentation
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers1, self.up))
        ):
            x_out1 = upsample(x_out1)
            x_out1 = up(torch.cat([x_out1, x_skip], 1))

        # Regression
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers2, self.up))
        ):
            x_out2 = upsample(x_out2)
            x_out2 = up(torch.cat([x_out2, x_skip], 1))

        if self.add_output:
            x_out1 = self.conv_final1(x_out1)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)

        if self.add_output:
            x_out2 = self.conv_final2(x_out2)
            x_out2 = F.sigmoid(x_out2)

        return [x_out1, x_out2]


class UNet(nn.Module):
    """
    Vanilla UNet.

    Implementation from https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        input_channels=3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        padding=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final = nn.Conv2d(up_filter_sizes[0], num_classes, padding)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)
            # print(x_out.shape)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers, self.up))
        ):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))
            # print(x_out.shape)

        if self.add_output:
            x_out = self.conv_final(x_out)
            # print(x_out.shape)
            if self.num_classes > 1:
                x_out = F.log_softmax(x_out, dim=1)

        return [x_out]


class UNet_ConvMCD(nn.Module):
    """
    Vanilla UNet.

    Implementation from https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """

    output_downscaled = 1
    module = UNetModule

    def __init__(
        self,
        input_channels: int = 3,
        filters_base: int = 32,
        down_filter_factors=(1, 2, 4, 8, 16),
        up_filter_factors=(1, 2, 4, 8, 16),
        bottom_s=4,
        num_classes=1,
        add_output=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(
                self.module(down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i])
            )
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
            self.conv_final2 = nn.Conv2d(up_filter_sizes[0], num_classes, 1)
            self.conv_final3 = nn.Conv2d(up_filter_sizes[0], 1, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
            list(zip(xs[:-1], self.upsamplers, self.up))
        ):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))

        if self.add_output:
            x_out1 = self.conv_final1(x_out)
            x_out2 = self.conv_final2(x_out)
            x_out3 = self.conv_final3(x_out)
            if self.num_classes > 1:
                x_out1 = F.log_softmax(x_out1, dim=1)
                x_out2 = F.log_softmax(x_out2, dim=1)
            x_out3 = F.sigmoid(x_out3)

        # return x_out,x_out1,x_out2,x_out3
        return [x_out1, x_out2, x_out3]
