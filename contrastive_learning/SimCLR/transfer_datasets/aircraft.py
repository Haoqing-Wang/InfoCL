import os
import numpy as np
from PIL import Image
from os.path import join
from collections import defaultdict
import torch.utils.data as data

DATA_ROOTS = '../datasets/Aircraft'

class Aircraft(data.Dataset):
    def __init__(self, root=DATA_ROOTS, train=True, image_transforms=None):
        super().__init__()
        self.root = root
        self.train = train
        self.image_transforms = image_transforms
        paths, bboxes, labels = self.load_images()
        self.paths = paths
        self.bboxes = bboxes
        self.labels = labels

    def load_images(self):
        split = 'trainval' if self.train else 'test'
        variant_path = os.path.join(self.root, 'data', 'images_variant_%s.txt'%split)
        with open(variant_path, 'r') as f:
            names_to_variants = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        names_to_variants = dict(names_to_variants)
        variants_to_names = defaultdict(list)
        for name, variant in names_to_variants.items():
            variants_to_names[variant].append(name)
        variants = sorted(list(set(variants_to_names.keys())))

        names_to_bboxes = self.get_bounding_boxes()
        split_files, split_labels, split_bboxes = [], [], []
        for variant_id, variant in enumerate(variants):
            class_files = [join(self.root, 'data', 'images', '%s.jpg'%filename) for filename in sorted(variants_to_names[variant])]
            bboxes = [names_to_bboxes[name] for name in sorted(variants_to_names[variant])]
            labels = list([variant_id] * len(class_files))
            split_files += class_files
            split_labels += labels
            split_bboxes += bboxes
        return split_files, split_bboxes, split_labels

    def get_bounding_boxes(self):
        bboxes_path = os.path.join(self.root, 'data', 'images_box.txt')
        with open(bboxes_path, 'r') as f:
            names_to_bboxes = [line.split('\n')[0].split(' ') for line in f.readlines()]
            names_to_bboxes = dict((name, list(map(int, (xmin, ymin, xmax, ymax)))) for name, xmin, ymin, xmax, ymax in names_to_bboxes)
        return names_to_bboxes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        bbox = tuple(self.bboxes[index])
        label = self.labels[index]

        image = Image.open(path).convert(mode='RGB')
        image = image.crop(bbox)

        if self.image_transforms:
            image = self.image_transforms(image)
        return image, label