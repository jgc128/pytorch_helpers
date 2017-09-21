import logging

import torch
import torch.nn
import torch.nn.functional as F

from pytorch_helpers.initialization import init_model_weights


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout2d(p=0.2),
        )

    def forward(self, inputs):
        inputs = self.net(inputs)
        return inputs


class TransitionDown(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionDown, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.MaxPool2d(2, 2)
        )

    def forward(self, inputs):
        inputs = self.net(inputs)
        return inputs


class TransitionUp(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.in_channels, self.out_channels, 4, stride=2, padding=1),
            # torch.nn.BatchNorm2d(self.out_channels),
            # torch.nn.ELU(inplace=True),
        )

    def forward(self, inputs):
        inputs = self.net(inputs)
        return inputs


class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, growth_rate, nb_layers):
        super(DenseBlock, self).__init__()

        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.nb_layers = nb_layers

        self.layers = torch.nn.ModuleList()
        for i in range(nb_layers):
            f_in = self.in_channels + i * growth_rate
            l = ConvLayer(f_in, growth_rate)
            self.layers.append(l)

    def forward(self, inputs):
        outputs = []
        for i in range(self.nb_layers):
            layer_output = self.layers[i](inputs)
            outputs.append(layer_output)
            inputs = torch.cat([layer_output, inputs], dim=1)

        output = torch.cat(outputs, dim=1)

        return output


class Tiramisu(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 nb_layers_per_block=(4, 5, 6, 7, 9), growth_rate=6, nb_layers_bottleneck=9, nb_filters_initial=48):
        super(Tiramisu, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_layers_per_block = nb_layers_per_block
        self.nb_layers_bottleneck = nb_layers_bottleneck
        self.nb_filters_initial = nb_filters_initial
        self.growth_rate = growth_rate

        # initial conv
        self.initial = ConvLayer(self.in_channels, nb_filters_initial)

        # transition down
        self.down = torch.nn.ModuleList()
        self.down_samplers = torch.nn.ModuleList()
        f_in = self.nb_filters_initial
        f_outs = []
        for i, nb_layers in enumerate(self.nb_layers_per_block):
            dense_block = DenseBlock(f_in, self.growth_rate, nb_layers)
            self.down.append(dense_block)

            f_in += self.growth_rate * nb_layers
            transition_down = TransitionDown(f_in, f_in)
            self.down_samplers.append(transition_down)

            f_outs.append(f_in)

        # bottleneck
        self.bottleneck = DenseBlock(f_in, self.growth_rate, self.nb_layers_bottleneck)
        f_in += self.growth_rate * self.nb_layers_bottleneck

        # transition up
        self.up = torch.nn.ModuleList()
        self.up_samplers = torch.nn.ModuleList()
        nb_layers_prev = self.nb_layers_bottleneck
        f_outs = list(reversed(f_outs))
        for i, nb_layers in enumerate(reversed(self.nb_layers_per_block)):
            transition_up = TransitionUp(f_in, nb_layers_prev * growth_rate)
            self.up_samplers.append(transition_up)

            f_in = nb_layers_prev * self.growth_rate
            dense_block = DenseBlock(f_in, self.growth_rate, nb_layers)
            self.up.append(dense_block)

            f_in = nb_layers_prev * self.growth_rate + nb_layers * self.growth_rate + f_outs[i]
            nb_layers_prev = nb_layers

        self.logits = torch.nn.Conv2d(f_in, self.out_channels, 1, padding=0)

        init_model_weights(self)

    def forward(self, inputs):
        # initial
        net = self.initial(inputs)

        # down
        skips = []
        for i, (down, down_sampler) in enumerate(zip(self.down, self.down_samplers)):
            db = down(net)
            net = torch.cat([db, net], dim=1)
            skips.append(net)

            net = down_sampler(net)

        # bottleneck
        db = self.bottleneck(net)
        net = torch.cat([db, net], dim=1)

        # up
        for i, (up_sampler, up, skip) in enumerate(zip(self.up_samplers, self.up, reversed(skips))):
            net = up_sampler(net)
            db = up(net)
            net = torch.cat([net, db, skip], dim=1)

        logits = self.logits(net)

        return logits


class TiramisuDeepSmall(Tiramisu):
    def __init__(self, in_channels, out_channels):
        super(TiramisuDeepSmall, self).__init__(in_channels, out_channels,
                                                nb_layers_per_block=(2, 3, 3, 4, 4), growth_rate=6,
                                                nb_layers_bottleneck=5, nb_filters_initial=48)
