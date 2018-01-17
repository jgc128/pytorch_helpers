import os
import logging

import numpy as np
import cv2

import torch
from torch.nn import Parameter
from torch.autograd import Variable


def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()

    return obj


def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj


def set_variable_repr():
    Variable.__repr__ = lambda x: f'Variable {tuple(x.size())}'
    Parameter.__repr__ = lambda x: f'Parameter {tuple(x.size())}'


def restore_weights(model, filename):
    if not isinstance(filename, str):
        filename = str(filename)

    map_location = None

    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        map_location = lambda storage, loc: storage

    state_dict = torch.load(filename, map_location=map_location)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.load_state_dict(state_dict)

    logging.info(f'Model restored: {os.path.basename(filename)}')


def save_weights(model, filename):
    if not isinstance(filename, str):
        filename = str(filename)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), filename)

    logging.info(f'Model saved: {os.path.basename(filename)}')


def load_image(filename, grayscale=False):
    if not grayscale:
        img = cv2.imread(str(filename))
    else:
        img = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)

    if not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
