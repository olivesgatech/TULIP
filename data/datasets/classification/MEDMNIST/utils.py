import os
import numpy as np
import torch
from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.MEDMNIST.dataset import LoaderMEDMNIST
from data.datasets.classification.common.custom_transforms import Cutout

from config import BaseConfig


def rm_uniform_data(imgs: np.ndarray, targets: np.ndarray, rm_data: float, num_classes: int) \
        -> Tuple[np.ndarray, np.ndarray]:
    if rm_data >= 1.0:
        raise ValueError(f'Cannot remove more than 100% of the dataset')
    print(f'Removing data points dataset....')
    for i in range(num_classes):
        im_idxs = np.argwhere(targets == i)
        num_rm = int(im_idxs.shape[0] * rm_data)
        remove_idxs = im_idxs[:num_rm]
        remove_idxs = np.squeeze(remove_idxs)

        imgs = np.delete(imgs, remove_idxs, axis=0)
        targets = np.delete(targets, remove_idxs)

        print(f'Class {i}: {len(imgs[targets == i])}  training')

    print(f'Total samples training {len(imgs)} test {len(targets)}')
    return imgs, targets


def unbalance(train_imgs: np.ndarray, train_labels: np.ndarray, test_imgs: np.ndarray, test_labels: np.ndarray,
              unbalance_pcent: float, num_classes: int,
              internal_pcent: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f'Unbalancing dataset....')

    num_unbalanced_total = int(unbalance_pcent * num_classes)
    num_unbalance = internal_pcent * num_unbalanced_total
    for i in range(num_unbalanced_total):
        if i < num_unbalance:
            im_idxs = np.argwhere(train_labels == i)
            rm_samples = int(len(im_idxs) * 0.9)
            remove_idxs = im_idxs[:rm_samples]
            remove_idxs = np.squeeze(remove_idxs)

            train_imgs = np.delete(train_imgs, remove_idxs, axis=0)
            train_labels = np.delete(train_labels, remove_idxs)
        else:
            im_idxs = np.argwhere(test_labels == i)
            rm_samples = int(len(im_idxs) * 0.9)
            remove_idxs = im_idxs[:rm_samples]
            remove_idxs = np.squeeze(remove_idxs)

            test_imgs = np.delete(test_imgs, remove_idxs, axis=0)
            test_labels = np.delete(test_labels, remove_idxs)

        print(f'Class {i}: {len(train_imgs[train_labels == i])}  training '
              f'{len(test_imgs[test_labels == i])} test')

    print(f'Total samples training {len(train_imgs)} test {len(test_imgs)}')
    return train_imgs, train_labels, test_imgs, test_labels


def get_med_mnist(cfg: BaseConfig, train_set: np.ndarray, train_targets: np.ndarray, val_set: np.ndarray,
                  val_targets: np.ndarray, test_set: np.ndarray, test_targets: np.ndarray, num_classes: int,
                  idxs: np.ndarray) -> LoaderObject:
    if cfg.data.rm_data > 0:
        train_set, train_targets = rm_uniform_data(train_set, train_targets, cfg.data.rm_data, num_classes)

    if cfg.data.unbalance:
        train_set, train_targets, test_set, test_targets = unbalance(train_imgs=train_set, train_labels=train_targets,
                                                                     test_imgs=test_set, test_labels=test_targets,
                                                                     unbalance_pcent=cfg.data.pcent_total_unbalance,
                                                                     internal_pcent=cfg.data.pcent_unbalance,
                                                                     num_classes=num_classes)

    # init data configs
    data_config = ClassificationStructure()
    data_config.train_set = torch.from_numpy(train_set)
    data_config.train_labels = torch.from_numpy(train_targets).squeeze()
    data_config.test_set = torch.from_numpy(test_set)
    data_config.test_labels = torch.from_numpy(test_targets).squeeze()
    data_config.train_len = len(train_set)
    data_config.test_len = len(test_set)
    data_config.val_set = torch.from_numpy(val_set)
    data_config.val_labels = torch.from_numpy(val_targets).squeeze()
    data_config.val_len = len(data_config.val_labels)
    data_config.num_classes = num_classes
    data_config.img_size = 28

    data_config.is_configured = True

    # add transforms
    train_transform = transforms.Compose([])

    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        train_transform.transforms.append(transforms.RandomCrop(28, padding=3))

    # mandatory transforms
    train_transform.transforms.append(transforms.Grayscale(num_output_channels=3))
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # cutout requires tesnor inputs
    if cfg.data.augmentations.cutout:
        train_transform.transforms.append(Cutout(n_holes=1, length=16))

    # test transforms
    # test_transform = transforms.Compose([transforms.RandomCrop(28, padding=4),
    test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    train_loader = DataLoader(LoaderMEDMNIST(data_config=data_config,
                                             split='train',
                                             transform=train_transform,
                                             current_idxs=idxs),
                              batch_size=cfg.classification.batch_size,
                              shuffle=True)
    test_loader = DataLoader(LoaderMEDMNIST(data_config=data_config,
                                            split='test',
                                            transform=test_transform),
                             batch_size=cfg.classification.batch_size,
                             shuffle=False)
    val_loader = None
    if cfg.data.create_validation:
        val_loader = DataLoader(LoaderMEDMNIST(data_config=data_config,
                                               split='val',
                                               transform=test_transform),
                                batch_size=cfg.classification.batch_size,
                                shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           val_loader=val_loader,
                           data_configs=data_config)

    return loaders


def makedirs(path: str):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        os.makedirs(path)


