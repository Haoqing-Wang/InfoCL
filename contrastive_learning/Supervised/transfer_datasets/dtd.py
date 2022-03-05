import os
import copy
import numpy as np
from PIL import Image
from os.path import join
from itertools import chain
from collections import defaultdict

import torch
import torch.utils.data as data
from torchvision import transforms

DATA_ROOTS = '../datasets/DTD'

class DTD(data.Dataset):
    def __init__(self, root=DATA_ROOTS, train=True, image_transforms=None):
        super().__init__()
        self.root = root
        self.train = train
        self.image_transforms = image_transforms
        paths, labels = self.load_images()
        self.paths, self.labels = paths, labels

    def load_images(self):
        if self.train:
            train_info_path = os.path.join(self.root, 'labels', 'train1.txt')
            with open(train_info_path, 'r') as f:
                train_info = [line.split('\n')[0] for line in f.readlines()]

            val_info_path = os.path.join(self.root, 'labels', 'val1.txt')
            with open(val_info_path, 'r') as f:
                val_info = [line.split('\n')[0] for line in f.readlines()]
            split_info = train_info + val_info

        else:
            test_info_path = os.path.join(self.root, 'labels', 'test1.txt')
            with open(test_info_path, 'r') as f:
                split_info = [line.split('\n')[0] for line in f.readlines()]

        # pull out categoires from paths
        categories = []
        for row in split_info:
            image_path = row
            category = image_path.split('/')[0]
            categories.append(category)
        categories = sorted(list(set(categories)))

        all_paths, all_labels = [], []
        for row in split_info:
            image_path = row
            category = image_path.split('/')[0]
            label = categories.index(category)
            all_paths.append(join(self.root, 'images', image_path))
            all_labels.append(label)
        return all_paths, all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        image = Image.open(path).convert(mode='RGB')
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, label