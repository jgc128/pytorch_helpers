import os
from pathlib import Path
import numpy as np

import torch
import torch.utils.data

from pytorch_helpers.helpers import load_image


class ImageFolderDataset(torch.utils.data.Dataset):
    IMG_EXTENSIONS = {'.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'}

    def __init__(self, images_dir, transform=None):
        super(ImageFolderDataset, self).__init__()

        if not isinstance(images_dir, Path):
            images_dir = Path(images_dir)

        self.images_dir = images_dir
        self.images = self._find_images(self.images_dir)

        self.transform = transform

    def _find_images(self, images_dir):
        images = [
            image_filename
            for image_filename in images_dir.iterdir()
            if image_filename.is_file() and image_filename.suffix in ImageFolderDataset.IMG_EXTENSIONS
        ]
        return images

    def __getitem__(self, index):
        image_filename = self.images[index]
        image = load_image(image_filename)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)
