from collections import OrderedDict

import logging
import numpy as np
import torch
import torch.nn

from pytorch_helpers.initialization import init_model_weights


class VGGBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, nb_convs, kernel_size=3, pool=True):
        super(VGGBlock, self).__init__()

        padding = int((kernel_size - 1) / 2)

        modules = OrderedDict()
        for i in range(nb_convs):
            f_in = in_channels if i == 0 else out_channels
            f_out = out_channels

            modules[f'conv_{i}'] = torch.nn.Conv2d(f_in, f_out, kernel_size, padding=padding)
            modules[f'conv_{i}_bn'] = torch.nn.BatchNorm2d(f_out)
            modules[f'conv_{i}_activation'] = torch.nn.ELU(inplace=True)

        if pool:
            modules['pool'] = torch.nn.MaxPool2d(2)

        self.net = torch.nn.Sequential(modules)

    def forward(self, inputs):
        inputs = self.net(inputs)

        return inputs


class FCN32(torch.nn.Module):
    """With a help of https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py"""

    def __init__(self, in_channels, out_channels):
        super(FCN32, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = VGGBlock(in_channels=self.in_channels, out_channels=64, nb_convs=2)
        self.conv2 = VGGBlock(in_channels=64, out_channels=128, nb_convs=2)
        self.conv3 = VGGBlock(in_channels=128, out_channels=256, nb_convs=3)
        self.conv4 = VGGBlock(in_channels=256, out_channels=512, nb_convs=3)
        self.conv5 = VGGBlock(in_channels=512, out_channels=512, nb_convs=3)

        self.conv6 = VGGBlock(in_channels=512, out_channels=512, nb_convs=1, kernel_size=1, pool=False)
        self.conv7 = VGGBlock(in_channels=512, out_channels=512, nb_convs=1, kernel_size=1, pool=False)

        self.upsample1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            torch.nn.ELU()
        )
        self.upsample1_scorer = torch.nn.Conv2d(512, 512, 1, padding=0)

        self.upsample2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            torch.nn.ELU()
        )
        self.upsample2_scorer = torch.nn.Conv2d(256, 256, 1, padding=0)

        self.upsample3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 32, 16, stride=8, padding=4),
            torch.nn.ELU()
        )

        self.conv8 = VGGBlock(in_channels=32, out_channels=32, nb_convs=2, pool=False)

        self.logits = torch.nn.Conv2d(32, self.out_channels, 1, padding=0)

        init_model_weights(self)

    def forward(self, inputs):
        net = self.conv1(inputs)
        net = self.conv2(net)
        net = self.conv3(net)
        pool3 = net

        net = self.conv4(net)
        pool4 = net

        net = self.conv5(net)

        net = self.conv6(net)
        net = self.conv7(net)

        # upsamplig
        net = self.upsample1(net)
        pool4_scored = self.upsample1_scorer(pool4)
        net = net + pool4_scored

        net = self.upsample2(net)
        pool3_scored = self.upsample2_scorer(pool3)
        net = net + pool3_scored

        net = self.upsample3(net)

        net = self.conv8(net)

        logits = self.logits(net)

        return logits
