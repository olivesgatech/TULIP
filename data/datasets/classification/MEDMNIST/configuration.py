import os
import numpy as np
import medmnist.dataset as mmnist
from .utils import get_med_mnist, makedirs
from config import BaseConfig


def get_med_mnist_loaders(cfg: BaseConfig, dataset: str, idxs: np.ndarray):
    if dataset == 'medmnist-organA':
        return get_organ_a_mnist(cfg, idxs)
    elif dataset == 'medmnist-organC':
        return get_organ_c_mnist(cfg, idxs)
    elif dataset == 'medmnist-organS':
        return get_organ_s_mnist(cfg, idxs)
    elif dataset == 'medmnist-blood':
        return get_blood_mnist(cfg, idxs)
    elif dataset == 'medmnist-tissue':
        return get_tissue_mnist(cfg, idxs)
    elif dataset == 'medmnist-pneumonia':
        return get_pneumonia_mnist(cfg, idxs)
    elif dataset == 'medmnist-oct':
        return get_oct_mnist(cfg, idxs)
    elif dataset == 'medmnist-path':
        return get_path_mnist(cfg, idxs)
    elif dataset == 'medmnist-derma':
        return get_derma_mnist(cfg, idxs)
    elif dataset == 'medmnist-breast':
        return get_breast_mnist(cfg, idxs)
    else:
        raise ValueError(f'{dataset} not implemented yet!')


def get_organ_a_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/MEDMNIST/organAMNIST')
    makedirs(os.path.expanduser(path) + '/MEDMNIST/organAMNIST')
    raw_tr = mmnist.OrganAMNIST(root=os.path.expanduser(path) + '/MEDMNIST/organAMNIST', split='train',
                                download=cfg.data.download)
    raw_te = mmnist.OrganAMNIST(root=os.path.expanduser(path) + '/MEDMNIST/organAMNIST', split='test',
                                download=cfg.data.download)
    raw_val = mmnist.OrganAMNIST(root=os.path.expanduser(path) + '/MEDMNIST/organAMNIST', split='val',
                                 download=cfg.data.download)

    train_set = raw_tr.imgs
    train_targets = raw_tr.labels
    test_set = raw_te.imgs
    test_targets = raw_te.labels
    val_set = raw_val.imgs
    val_targets = raw_val.labels

    num_classes = 11

    loaders = get_med_mnist(cfg, train_set, train_targets, val_set, val_targets, test_set, test_targets, num_classes,
                            idxs)

    return loaders


def get_organ_c_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    path = f'{os.path.expanduser(path)}/MEDMNIST/organCMNIST'
    print(path)
    makedirs(path)
    raw_tr = mmnist.OrganCMNIST(root=path, split='train',
                                download=cfg.data.download)
    raw_te = mmnist.OrganCMNIST(root=path, split='test',
                                download=cfg.data.download)
    raw_val = mmnist.OrganCMNIST(root=path, split='val',
                                 download=cfg.data.download)

    train_set = raw_tr.imgs
    train_targets = raw_tr.labels
    test_set = raw_te.imgs
    test_targets = raw_te.labels
    val_set = raw_val.imgs
    val_targets = raw_val.labels

    num_classes = 11

    for i in range(num_classes):
        train_lab = train_targets[train_targets == i]
        test_lab = test_targets[test_targets == i]
        print(f'Class {i}: training {train_lab.shape} test {test_lab.shape}')

    loaders = get_med_mnist(cfg, train_set, train_targets, val_set, val_targets, test_set, test_targets, num_classes,
                            idxs)

    return loaders


def get_organ_s_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    path = f'{os.path.expanduser(path)}/MEDMNIST/organSMNIST'
    print(path)
    makedirs(path)
    raw_tr = mmnist.OrganSMNIST(root=path, split='train',
                                download=cfg.data.download)
    raw_te = mmnist.OrganSMNIST(root=path, split='test',
                                download=cfg.data.download)
    raw_val = mmnist.OrganSMNIST(root=path, split='val',
                                 download=cfg.data.download)

    train_set = raw_tr.imgs
    train_targets = raw_tr.labels
    test_set = raw_te.imgs
    test_targets = raw_te.labels
    val_set = raw_val.imgs
    val_targets = raw_val.labels

    num_classes = 11

    loaders = get_med_mnist(cfg, train_set, train_targets, val_set, val_targets, test_set, test_targets, num_classes,
                            idxs)

    return loaders


def get_blood_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    path = f'{os.path.expanduser(path)}/MEDMNIST/BloodMNIST'
    print(path)
    makedirs(path)
    raw_tr = mmnist.BloodMNIST(root=path, split='train',
                                download=cfg.data.download)
    raw_te = mmnist.BloodMNIST(root=path, split='test',
                               download=cfg.data.download)
    raw_val = mmnist.BloodMNIST(root=path, split='val',
                                download=cfg.data.download)

    train_set = raw_tr.imgs
    train_targets = raw_tr.labels
    test_set = raw_te.imgs
    test_targets = raw_te.labels
    val_set = raw_val.imgs
    val_targets = raw_val.labels

    num_classes = 8

    loaders = get_med_mnist(cfg, train_set, train_targets, val_set, val_targets, test_set, test_targets, num_classes,
                            idxs)

    return loaders


def get_tissue_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    path = f'{os.path.expanduser(path)}/MEDMNIST/TissueMNIST'
    print(path)
    makedirs(path)
    raw_tr = mmnist.TissueMNIST(root=path, split='train',
                                download=cfg.data.download)
    raw_te = mmnist.TissueMNIST(root=path, split='test',
                                download=cfg.data.download)
    raw_val = mmnist.TissueMNIST(root=path, split='val',
                                 download=cfg.data.download)

    train_set = raw_tr.imgs
    train_targets = raw_tr.labels
    test_set = raw_te.imgs
    test_targets = raw_te.labels
    val_set = raw_val.imgs
    val_targets = raw_val.labels

    num_classes = 8

    loaders = get_med_mnist(cfg, train_set, train_targets, val_set, val_targets, test_set, test_targets, num_classes,
                            idxs)

    return loaders


def get_breast_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    path = f'{os.path.expanduser(path)}/MEDMNIST/BreastMNIST'
    print(path)
    makedirs(path)
    raw_tr = mmnist.BreastMNIST(root=path, split='train',
                                download=cfg.data.download)
    raw_te = mmnist.BreastMNIST(root=path, split='test',
                                download=cfg.data.download)
    raw_val = mmnist.BreastMNIST(root=path, split='val',
                                 download=cfg.data.download)

    train_set = raw_tr.imgs
    train_targets = raw_tr.labels
    test_set = raw_te.imgs
    test_targets = raw_te.labels
    val_set = raw_val.imgs
    val_targets = raw_val.labels

    num_classes = 2

    loaders = get_med_mnist(cfg, train_set, train_targets, val_set, val_targets, test_set, test_targets, num_classes,
                            idxs)

    return loaders


def get_pneumonia_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    path = f'{os.path.expanduser(path)}/MEDMNIST/PneumoniaMNIST'
    print(path)
    makedirs(path)
    raw_tr = mmnist.PneumoniaMNIST(root=path, split='train',
                                   download=cfg.data.download)
    raw_te = mmnist.PneumoniaMNIST(root=path, split='test',
                                   download=cfg.data.download)
    raw_val = mmnist.PneumoniaMNIST(root=path, split='val',
                                    download=cfg.data.download)

    train_set = raw_tr.imgs
    train_targets = raw_tr.labels
    test_set = raw_te.imgs
    test_targets = raw_te.labels
    val_set = raw_val.imgs
    val_targets = raw_val.labels

    num_classes = 2

    loaders = get_med_mnist(cfg, train_set, train_targets, val_set, val_targets, test_set, test_targets, num_classes,
                            idxs)

    return loaders


def get_oct_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    path = f'{os.path.expanduser(path)}/MEDMNIST/OCTMNIST'
    print(path)
    makedirs(path)
    raw_tr = mmnist.OCTMNIST(root=path, split='train',
                             download=cfg.data.download)
    raw_te = mmnist.OCTMNIST(root=path, split='test',
                             download=cfg.data.download)
    raw_val = mmnist.OCTMNIST(root=path, split='val',
                              download=cfg.data.download)

    train_set = raw_tr.imgs
    train_targets = raw_tr.labels
    test_set = raw_te.imgs
    test_targets = raw_te.labels
    val_set = raw_val.imgs
    val_targets = raw_val.labels

    num_classes = 4

    loaders = get_med_mnist(cfg, train_set, train_targets, val_set, val_targets, test_set, test_targets, num_classes,
                            idxs)

    return loaders


def get_derma_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    path = f'{os.path.expanduser(path)}/MEDMNIST/DermaMNIST'
    print(path)
    makedirs(path)
    raw_tr = mmnist.DermaMNIST(root=path, split='train',
                               download=cfg.data.download)
    raw_te = mmnist.DermaMNIST(root=path, split='test',
                               download=cfg.data.download)
    raw_val = mmnist.DermaMNIST(root=path, split='val',
                                download=cfg.data.download)

    train_set = raw_tr.imgs
    train_targets = raw_tr.labels
    test_set = raw_te.imgs
    test_targets = raw_te.labels
    val_set = raw_val.imgs
    val_targets = raw_val.labels

    num_classes = 7

    loaders = get_med_mnist(cfg, train_set, train_targets, val_set, val_targets, test_set, test_targets, num_classes,
                            idxs)

    return loaders


def get_path_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    path = f'{os.path.expanduser(path)}/MEDMNIST/PathMNIST'
    print(path)
    makedirs(path)
    raw_tr = mmnist.PathMNIST(root=path, split='train', download=cfg.data.download)
    raw_te = mmnist.PathMNIST(root=path, split='test', download=cfg.data.download)
    raw_val = mmnist.PathMNIST(root=path, split='val', download=cfg.data.download)

    train_set = raw_tr.imgs
    train_targets = raw_tr.labels
    test_set = raw_te.imgs
    test_targets = raw_te.labels
    val_set = raw_val.imgs
    val_targets = raw_val.labels

    num_classes = 9

    loaders = get_med_mnist(cfg, train_set, train_targets, val_set, val_targets, test_set, test_targets, num_classes,
                            idxs)

    return loaders