import logging
from collections import OrderedDict

import torch
import torch.nn
import torch.nn.functional as F

from pytorch_helpers.initialization import init_model_weights


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ELU(inplace=True),
        )

    def forward(self, inputs):
        inputs = self.net(inputs)
        return inputs


class UNetModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = torch.nn.Sequential(
            ConvBlock(self.in_channels, self.out_channels),
            ConvBlock(self.out_channels, self.out_channels),
        )

    def forward(self, inputs):
        inputs = self.net(inputs)
        return inputs


class UpSampleModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.in_channels, self.out_channels, 4, stride=2, padding=1),
            torch.nn.ELU()
        )

    def forward(self, inputs):
        inputs = self.net(inputs)
        return inputs


class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 filter_base=16, filter_factors=(1, 2, 4, 8, 16), filter_bottom=1024):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_base = filter_base
        self.filter_factors = filter_factors
        self.filter_bottom = filter_bottom

        self.down_filter_sizes = [self.filter_base * f for f in self.filter_factors]
        self.up_filter_sizes = list(reversed(self.down_filter_sizes))

        self.down = torch.nn.ModuleList()
        self.down_samplers = torch.nn.ModuleList()
        for i, filter_size in enumerate(self.down_filter_sizes):
            f_in = in_channels if i == 0 else self.down_filter_sizes[i - 1]
            f_out = filter_size

            self.down.append(UNetModule(f_in, f_out))
            self.down_samplers.append(torch.nn.MaxPool2d(2, 2))

        self.bottom = UNetModule(self.down_filter_sizes[-1], self.filter_bottom)

        self.up = torch.nn.ModuleList()
        self.up_samplers = torch.nn.ModuleList()
        for i, filter_size in enumerate(self.up_filter_sizes):
            f_sampler_in = self.filter_bottom if i == 0 else self.up_filter_sizes[i - 1]
            f_sampler_out = f_sampler_in // 2
            f_in = f_sampler_out + self.down_filter_sizes[-(i + 1)]
            f_out = filter_size

            self.up_samplers.append(UpSampleModule(f_sampler_in, f_sampler_out))
            self.up.append(UNetModule(f_in, f_out))

        self.logits = torch.nn.Conv2d(self.up_filter_sizes[-1], self.out_channels, 1, padding=0)

        init_model_weights(self)

    def forward(self, inputs):
        net = inputs

        skips = []
        for i, (down, down_sampler) in enumerate(zip(self.down, self.down_samplers)):
            net = down(net)
            skips.append(net)
            net = down_sampler(net)

        # reverse skips to go from bottom to up using a 0-1-2 indexing
        skips = list(reversed(skips))

        net = self.bottom(net)

        for i, (up_sampler, up) in enumerate(zip(self.up_samplers, self.up)):
            net = up_sampler(net)
            net = torch.cat([net, skips[i]], dim=1)  # the shape is (batch_size, channels, h ,w ), so dim = 1
            net = up(net)

        logits = self.logits(net)

        return logits


class UNetDeepSmall(UNet):
    def __init__(self, in_channels, out_channels):
        super(UNetDeepSmall, self).__init__(in_channels, out_channels,
                                            filter_base=8, filter_factors=(2, 2, 4, 4, 8, 16), filter_bottom=256)
