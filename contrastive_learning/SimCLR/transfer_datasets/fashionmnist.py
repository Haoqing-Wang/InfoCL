import os
import copy
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

DATA_ROOTS = '../datasets'

class FashionMNIST(data.Dataset):
    def __init__(self, root=DATA_ROOTS, train=True, image_transforms=None):
        super().__init__()
        if not os.path.isdir(root):
            os.makedirs(root)
        self.image_transforms = image_transforms
        self.dataset = datasets.mnist.FashionMNIST(root, train=train, download=True)

    def __getitem__(self, index):
        img, target = self.dataset.data[index], int(self.dataset.targets[index])
        img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        if self.image_transforms is not None:
            img = self.image_transforms(img)
        return img, target

    def __len__(self):
        return len(self.dataset)