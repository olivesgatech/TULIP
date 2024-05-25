import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.datasets.classification.common.dataobjects import ClassificationStructure, LoaderObject
from data.datasets.classification.MNIST.dataset import LoaderMNIST


def get_mnist(cfg, idxs: np.ndarray = None):
    path = cfg.data.data_loc
    print(os.path.expanduser(path) + '/MNIST')
    raw_tr = datasets.MNIST(os.path.expanduser(path) + '/MNIST', train=True, download=cfg.data.download)
    raw_te = datasets.MNIST(os.path.expanduser(path) + '/MNIST', train=False, download=cfg.data.download)

    train_set = raw_tr.train_data.numpy()
    train_targets = raw_tr.train_labels.numpy()
    test_set = raw_te.test_data.numpy()
    test_targets = raw_te.test_labels.numpy()

    num_classes = 10
    if cfg.data.rm_data > 0:
        if cfg.data.rm_data >= 1.0:
            raise ValueError(f'Cannot remove more than 100% of the dataset')
        print(f'Removing data points dataset....')
        for i in range(num_classes):
            im_idxs = np.argwhere(train_targets == i)
            num_rm = int(im_idxs.shape[0] * cfg.data.rm_data)
            remove_idxs = im_idxs[:num_rm]
            remove_idxs = np.squeeze(remove_idxs)

            train_set = np.delete(train_set, remove_idxs, axis=0)
            train_targets = np.delete(train_targets, remove_idxs)

            print(f'Class {i}: {len(train_set[train_targets == i])}  training')

        print(f'Total samples training {len(train_set)} test {len(test_set)}')
    if cfg.data.unbalance:
        print(f'Unbalancing dataset....')

        num_unbalanced_total = int(cfg.data.pcent_total_unbalance * num_classes)
        num_unbalance = cfg.data.pcent_unbalance * num_unbalanced_total
        for i in range(num_unbalanced_total):
            if i < num_unbalance:
                im_idxs = np.argwhere(train_targets == i)
                rm_samples = int(len(im_idxs) * 0.9)
                remove_idxs = im_idxs[:rm_samples]
                remove_idxs = np.squeeze(remove_idxs)

                train_set = np.delete(train_set, remove_idxs, axis=0)
                train_targets = np.delete(train_targets, remove_idxs)
            else:
                im_idxs = np.argwhere(test_targets == i)
                rm_samples = int(len(im_idxs) * 0.9)
                remove_idxs = im_idxs[:rm_samples]
                remove_idxs = np.squeeze(remove_idxs)

                test_set = np.delete(test_set, remove_idxs, axis=0)
                test_targets = np.delete(test_targets, remove_idxs)

            print(f'Class {i}: {len(train_set[train_targets == i])}  training '
                  f'{len(test_set[test_targets == i])} test')

        print(f'Total samples training {len(train_set)} test {len(test_set)}')

    # init data configs
    data_config = ClassificationStructure()
    if cfg.data.create_validation:
        print(f'Creating validation with 10% of training set....')

        num_unbalanced_total = int(cfg.data.pcent_total_unbalance * num_classes)
        num_unbalance = cfg.data.pcent_unbalance * num_unbalanced_total
        val_set = None
        val_target = None
        for i in range(num_classes):
            im_idxs = np.argwhere(train_targets == i)
            num_val = int(im_idxs.shape[0] * 0.1)
            remove_idxs = im_idxs[:num_val]
            remove_idxs = np.squeeze(remove_idxs)
            val_set = np.concatenate((val_set, train_set[remove_idxs]), axis=0) if val_set is not None \
                else train_set[remove_idxs]
            val_target = np.concatenate((val_target, train_targets[remove_idxs]), axis=0) if val_target is not None \
                else train_targets[remove_idxs]

            train_set = np.delete(train_set, remove_idxs, axis=0)
            train_targets = np.delete(train_targets, remove_idxs)

        print(f'Total samples training {len(train_set)} validation {len(val_set)} test {len(test_set)}')
        data_config.val_set = torch.from_numpy(val_set)
        data_config.val_labels = torch.from_numpy(val_target)
        data_config.val_len = len(data_config.val_labels)

    # init data configs
    # data_config = ClassificationStructure()
    data_config.train_set = torch.from_numpy(train_set)
    data_config.train_labels = torch.from_numpy(train_targets)
    data_config.test_set = torch.from_numpy(test_set)
    data_config.test_labels = torch.from_numpy(test_targets)
    data_config.train_len = len(train_set)
    data_config.test_len = len(test_set)
    data_config.num_classes = 10
    data_config.img_size = 28

    data_config.is_configured = True

    # add transforms
    train_transform = transforms.Compose([])

    if cfg.data.augmentations.random_hflip:
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_crop:
        #train_transform.transforms.append(transforms.RandomCrop(28, padding=6))
        train_transform.transforms.append(transforms.RandomCrop(28, padding=3))

    # mandatory transforms
    train_transform.transforms.append(transforms.Grayscale(num_output_channels=3))
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))

    # test transforms
    #test_transform = transforms.Compose([transforms.RandomCrop(28, padding=4),
    test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    # create loaders
    train_loader = DataLoader(LoaderMNIST(data_config=data_config,
                                          split='train',
                                          transform=train_transform,
                                          current_idxs=idxs),
                              batch_size=cfg.classification.batch_size,
                              shuffle=True)
    test_loader = DataLoader(LoaderMNIST(data_config=data_config,
                                         split='test',
                                         transform=test_transform),
                              batch_size=cfg.classification.batch_size,
                              shuffle=False)
    val_loader = None
    if cfg.data.create_validation:
        val_loader = DataLoader(LoaderMNIST(data_config=data_config,
                                            split='val',
                                            transform=test_transform),
                                batch_size=cfg.classification.batch_size,
                                shuffle=False)
    loaders = LoaderObject(train_loader=train_loader,
                           test_loader=test_loader,
                           val_loader=val_loader,
                           data_configs=data_config)


    return loaders