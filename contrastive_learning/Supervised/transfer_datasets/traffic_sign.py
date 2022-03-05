import os
import copy
import json
import operator
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from itertools import chain
from scipy.io import loadmat
from collections import defaultdict

import torch
import torch.utils.data as data
from torchvision import transforms

DATA_ROOTS = '../datasets/TrafficSign'

class TrafficSign(data.Dataset):
    NUM_CLASSES = 43
    def __init__(self, root=DATA_ROOTS, train=True, image_transforms=None):
        super().__init__()
        self.root = root
        self.train = train
        self.image_transforms = image_transforms
        paths, labels = self.load_images()
        self.paths, self.labels = paths, labels

    def load_images(self):
        split = 'Final_Training'
        rs = np.random.RandomState(42)
        all_filepaths, all_labels = [], []
        for class_i in range(self.NUM_CLASSES):
            class_dir_i = join(self.root, split, 'Images', '{:05d}'.format(class_i))
            image_paths = glob(join(class_dir_i, "*.ppm"))
            # train test splitting
            image_paths = np.array(image_paths)
            num = len(image_paths)
            indexer = np.arange(num)
            rs.shuffle(indexer)
            image_paths = image_paths[indexer].tolist()
            if self.train:
                image_paths = image_paths[:int(0.8 * num)]
            else:
                image_paths  = image_paths[int(0.8 * num):]
            labels = [class_i] * len(image_paths)
            all_filepaths.extend(image_paths)
            all_labels.extend(labels)

        return all_filepaths, all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        image = Image.open(path).convert(mode='RGB')
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, label