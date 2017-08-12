import abc
from enum import Enum

import numpy as np

import torch
import torch.nn
import torch.utils.data

from tqdm import tqdm

from pytorch_helpers.helpers import variable, cuda


class PyTorchTrainer(object):
    def __init__(self, model, loss, **kwargs):
        super(PyTorchTrainer, self).__init__()

        self.model = model
        self.loss = loss

    def fit(self, data_set_train, nb_epochs=10, batch_size=64, optimizer=None, data_set_val=None):
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters())

        data_loader_train = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True,
                                                        num_workers=1, pin_memory=torch.cuda.is_available())

        phases = ['train', ]
        data_loaders = [data_loader_train]
        if data_set_val is not None:
            data_loader_val = torch.utils.data.DataLoader(data_set_val, batch_size=batch_size, shuffle=False,
                                                          num_workers=1, pin_memory=torch.cuda.is_available())

            phases.append('val')
            data_loaders.append(data_loader_val)

        model = cuda(self.model)
        loss_fn = cuda(self.loss)

        j = 1
        for epoch in range(nb_epochs):
            for phase, data_loader in zip(phases, data_loaders):
                if phase == 'train':
                    model.train(True)
                else:
                    model.train(False)

                pbar_desc = f'Epoch {epoch}, {phase}'
                pbar = tqdm(total=len(data_loader.dataset), desc=pbar_desc, postfix={'loss': 0}, ncols=120)

                running_loss = 0.0
                for j, (inputs, targets) in enumerate(data_loader, 1):
                    inputs = variable(inputs)
                    targets = variable(targets)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs)

                    loss = loss_fn(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.data[0]

                    pbar.update(inputs.size(0))
                    pbar.set_postfix(loss=running_loss / j)

                    del loss
                    del outputs
                    del targets

                pbar.close()
                epoch_loss = running_loss / j

    def predict(self, data_loader):
        pass
