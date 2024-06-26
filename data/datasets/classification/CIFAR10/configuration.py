import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.CIFAR10.dataset import LoaderCIFAR10
from data.datasets.classification.common.custom_transforms import Cutout
from config import BaseConfig


def create_calidation(data_config: ClassificationStructure):
    new_train_len = int(len(data_config.train_labels) * 0.9)
    val_len = len(data_config.train_labels) - new_train_len
    val_idxs = np.random.choice(len(data_config.train_labels), val_len, replace=False)
    data_config.train_labels = data_config.train_labels.cpu().numpy()
    data_config.val_set = data_config.train_set[val_idxs]
    data_config.val_labels = torch.from_numpy(data_config.train_labels[val_idxs])
    data_config.train_len = new_train_len
    data_config.train_set = np.delete(data_config.train_set, val_idxs, axis=0)
    data_config.train_labels = torch.from_numpy(np.delete(data_config.train_labels, val_idxs, axis=0))

    return data_config


def create_test_validation(data_config: ClassificationStructure):
    new_train_len = int(len(data_config.test_labels) * 0.9)
    val_len = len(data_config.test_labels) - new_train_len
    val_idxs = np.random.choice(len(data_config.test_labels), val_len, replace=False)
    # data_config.train_labels = data_config.train_labels.cpu().numpy()
    data_config.val_set = data_config.test_set[val_idxs]
    data_config.val_labels = data_config.test_labels[val_idxs]
    # data_config.train_len = new_train_len
    # data_config.train_set = np.delete(data_config.train_set, val_idxs, axis=0)
    # data_config.train_labels = torch.from_numpy(np.delete(data_config.train_labels, val_idxs, axis=0))

    return data_config


def get_cifar10(cfg: BaseConfig, idxs: np.ndarray = None, test_bs: bool = False):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/CIFAR10')
    raw_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=cfg.data.download)
    raw_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=cfg.data.download)

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = raw_tr.data
    data_config.train_labels = torch.from_numpy(np.array(raw_tr.targets))
    data_config.test_set = raw_te.data
    data_config.test_labels = torch.from_numpy(np.array(raw_te.targets))
    data_config.train_len = len(data_config.train_labels)
    data_config.test_len = len(data_config.test_labels)
    data_config.num_classes = 10
    data_config.img_size = 32

    data_config.is_configured = True

    if cfg.run_configs.scale_ics:
        # data_config = create_calidation(data_config)
        data_config = create_test_validation(data_config)

    if cfg.run_configs.create_validation:
        print('Creating Validation!')
        data_config = create_calidation(data_config)

    # add transforms
    train_transform = transforms.Compose([])

    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))

    # mandatory transforms
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # cutout requires tesnor inputs
    if cfg.data.augmentations.cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

    # test transforms
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    bs = cfg.classification.batch_size
    train_loader = DataLoader(LoaderCIFAR10(data_config=data_config,
                                            split='train',
                                            transform=train_transform, current_idxs=idxs),
                              batch_size=bs,
                              shuffle=True
                              )
    test_loader = DataLoader(LoaderCIFAR10(data_config=data_config,
                                           split='test',
                                           transform=test_transform),
                             batch_size=bs,
                             shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           data_configs=data_config)

    if cfg.run_configs.scale_ics:
        # val used for a loss and therefore, we use transforms
        val_loader = DataLoader(LoaderCIFAR10(data_config=data_config,
                                              split='val',
                                              transform=train_transform),
                                batch_size=bs,
                                shuffle=True)
        loaders.val_loader = val_loader

    return loaders
