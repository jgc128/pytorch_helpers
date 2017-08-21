import abc
from enum import Enum

import numpy as np

import torch
import torch.nn
import torch.utils.data
import torch.optim.lr_scheduler

from tqdm import tqdm

from pytorch_helpers.crayon import get_crayon_experiment
from pytorch_helpers.helpers import variable, cuda, save_weights


class PyTorchTrainer(object):
    def __init__(self, model, loss, checkpoint_filename=None, crayon_exp_name=None, **kwargs):
        super(PyTorchTrainer, self).__init__()

        self.model = model
        self.loss = loss

        self.checkpoint_filename = checkpoint_filename
        self.crayon_exp_name = crayon_exp_name

        self._crayon_exp = None

    def fit(self, dataset_train, nb_epochs=10, batch_size=64, optimizer=None, lr=0.001, lr_step_size=0,
            dataset_val=None):
        if self._crayon_exp is None and self.crayon_exp_name is not None:
            self._crayon_exp = get_crayon_experiment(self.crayon_exp_name)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        lr_scheduler = None
        if lr_step_size != 0:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.5)

        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                                        num_workers=1, pin_memory=torch.cuda.is_available())

        phases = ['train', ]
        data_loaders = [data_loader_train]
        if dataset_val is not None:
            data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                                          num_workers=1, pin_memory=torch.cuda.is_available())

            phases.append('val')
            data_loaders.append(data_loader_val)

        model = cuda(self.model)
        loss_fn = cuda(self.loss)

        j = 1
        loss_best = np.inf
        for epoch in range(nb_epochs):
            for phase, data_loader in zip(phases, data_loaders):
                if phase == 'train':
                    model.train(True)
                else:
                    model.train(False)

                if phase == 'train' and lr_scheduler is not None:
                    lr_scheduler.step()
                    if self._crayon_exp is not None:
                        lr = optimizer.param_groups[0]['lr']
                        self._crayon_exp.add_scalar_value(f'learning_rate', lr)

                pbar_desc = f'Epoch {epoch}, {phase}'
                pbar = tqdm(total=len(data_loader.dataset), desc=pbar_desc, postfix={f'loss_{phase}': 0}, ncols=120)

                running_loss = 0.0
                for j, (inputs, targets) in enumerate(data_loader, 1):
                    volatile = phase == 'val'
                    inputs = variable(inputs, volatile=volatile)
                    targets = variable(targets, volatile=volatile)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = model(inputs)

                    loss = loss_fn(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    batch_loss = loss.data[0]
                    running_loss += batch_loss

                    pbar.update(inputs.size(0))

                    pbar.set_postfix(**{f'loss_{phase}': batch_loss})

                    if self._crayon_exp is not None:
                        self._crayon_exp.add_scalar_value(f'loss_batch/{phase}', batch_loss)

                    del loss
                    del outputs
                    del targets

                epoch_loss = running_loss / j
                pbar.set_postfix(**{f'loss_{phase}': epoch_loss})
                pbar.close()

                if self._crayon_exp is not None:
                    self._crayon_exp.add_scalar_value(f'loss_epoch/{phase}', epoch_loss)

                if phase == 'val' and epoch_loss < loss_best and self.checkpoint_filename is not None:
                    save_weights(model, self.checkpoint_filename)
                    loss_best = epoch_loss

    def predict(self, data_loader):
        pass
