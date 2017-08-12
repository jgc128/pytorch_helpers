import logging

import torch
import torch.nn


def init_model_weights(self):
    for m in self.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal(m.weight.data)
            m.bias.data.zero_()

            # if isinstance(m, torch.nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

            # if isinstance(m, torch.nn.ConvTranspose2d):
            #     assert m.kernel_size[0] == m.kernel_size[1]
            #     initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
            #     m.weight.data.copy_(initial_weight)

    logging.info('Weights initialized')


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    Make a 2D bilinear kernel suitable for upsampling
    https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/models/fcn32s.py
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[:, :, :, :] = filt
    return torch.from_numpy(weight).float()
