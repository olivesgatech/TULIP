import glob
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from data.datasets.segmentation.common import custom_transforms as transforms
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from data.datasets.segmentation.cityscapes.dataobjects import CityscapesLoaderObject, CityscapesStructure
from data.datasets.segmentation.common.dataobjects import StatObject
from data.datasets.segmentation.seismic.dataset import SeismicLoader
from training.segmentation.segmentationtracker import SegmentationTracker
from uspecanalysis.utils.segmentationtracker import USPECTracker
from config import BaseConfig


class CityscapesLoader(Dataset):
    def __init__(self, data_config: CityscapesStructure, split: str, transform):
        if not data_config.is_configured:
            raise Exception('Dataset not configured yet!')
        if split == 'train':
            self._dataset = data_config.train_set
        elif split == 'val':
            self._dataset = data_config.val_set
        elif split == 'test':
            self._dataset = data_config.test_set
        else:
            raise Exception('Set not implemented yet!')

        # encode class maps correctly
        self._void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self._valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self._class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                             'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                             'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                             'motorcycle', 'bicycle']

        self._ignore_index = 255
        self.nclasses = 19
        self._class_map = dict(zip(self._valid_classes, range(self.nclasses)))
        self._transform = transform
        self._tensor = transforms.ToTensor()

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        # get images and labels from dataset
        img, label = self._dataset[index]

        # convert to np arrays
        sample = {'data': np.array(img), 'label': np.array(label, dtype=float)}

        # transform sample
        sample = self._transform(sample)

        # encode target and normalize image
        sample['label'] = self.encode_segmap(np.rint(sample['label']))
        sample = self._tensor(sample)

        # swap axis
        sample['data'] = sample['data'].permute(2, 0, 1)

        return sample

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self._void_classes:
            mask[mask == _voidc] = self._ignore_index
        for _validc in self._valid_classes:
            mask[mask == _validc] = self._class_map[_validc]
        return mask