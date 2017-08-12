import numpy as np

import torch
import torch.utils.data

from pytorch_helpers.helpers import load_image


class BaseImageSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, image_transform=None, mask_transform=None):
        super(BaseImageSegmentationDataset, self).__init__()

        self.images = []
        self.masks = None
        self.nb_images = 0

        self.transform = transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def _load_image_and_mask(self, index):
        image_filename = self.images[index]
        image = load_image(image_filename)

        if self.masks is not None:
            mask_filename = self.masks[index]
            mask = load_image(mask_filename, grayscale=True)
        else:
            mask = 0

        return image, mask

    def _apply_transforms(self, image, mask):
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None and isinstance(mask, np.ndarray):
            mask = self.mask_transform(mask)

        return image, mask

    def __getitem__(self, index):
        image, mask = self._load_image_and_mask(index)
        image, mask = self._apply_transforms(image, mask)

        return image, mask

    def __len__(self):
        return self.nb_images
