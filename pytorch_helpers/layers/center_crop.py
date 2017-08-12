import torch
import torch.nn


class CenterCrop2d(torch.nn.Module):
    def __init__(self, size):
        super(CenterCrop2d, self).__init__()

        self.size = size

    def forward(self, inputs):
        # TODO: optimize it
        nb_dims = len(inputs.size())
        if nb_dims == 4:
            inputs = inputs[:, :, self.size:-self.size, self.size:-self.size]
        elif nb_dims == 3:
            inputs = inputs[:, self.size:-self.size, self.size:-self.size]
        elif nb_dims == 2:
            inputs = inputs[self.size:-self.size, self.size:-self.size]
        else:
            raise ValueError(f'Unsupported number of dimensions: {nb_dims}')

        return inputs.contiguous()
