import os
from torch.utils.data import DataLoader
from data.datasets.segmentation.common import custom_transforms as transforms
from torchvision import datasets
from data.datasets.segmentation.cityscapes.dataobjects import CityscapesLoaderObject, CityscapesStructure
from data.datasets.segmentation.common.dataobjects import StatObject
from data.datasets.segmentation.cityscapes.dataset import CityscapesLoader
from training.segmentation.segmentationtracker import SegmentationTracker
from uspecanalysis.utils.segmentationtracker import USPECTracker
from config import BaseConfig


def get_cityscapes_test_config(loaders: CityscapesLoaderObject, cfg: BaseConfig):
    out = []
    if cfg.run_configs.train:
        set_shape = (loaders.data_config.train_len, 1024, 2048)
        tracker = SegmentationTracker(set_shape)
        stats = StatObject(loaders.train_loader, tracker, 'training')
        out.append(stats)
    if cfg.run_configs.val:
        set_shape = (loaders.data_config.val_len, 1024, 2048)
        tracker = SegmentationTracker(set_shape)
        stats = StatObject(loaders.train_loader, tracker, 'val')
        out.append(stats)

    if cfg.run_configs.test:
        set_shape = (loaders.data_config.test_len, 1024, 2048)
        tracker = SegmentationTracker(set_shape, tracking_type='SE')
        stats = StatObject(loaders.test_loader, tracker, 'test')
        out.append(stats)

    return out


def get_cityscapes(cfg: BaseConfig):
    cityscapes_path = os.path.expanduser(cfg.data.data_loc + 'cityscapes/')

    data_configs = CityscapesStructure()

    data_configs.train_set = datasets.Cityscapes(root=cityscapes_path,
                                                 split='train',
                                                 mode='fine',
                                                 target_type='semantic')
    data_configs.train_len = len(data_configs.train_set)

    data_configs.val_set = datasets.Cityscapes(root=cityscapes_path,
                                               split='val',
                                               mode='fine',
                                               target_type='semantic')
    data_configs.val_len = len(data_configs.val_set)

    data_configs.test_set = datasets.Cityscapes(root=cityscapes_path,
                                                split='test',
                                                mode='fine',
                                                target_type='semantic')
    data_configs.test_len = len(data_configs.test_set)
    data_configs.num_classes = 19

    data_configs.is_configured = True

    # generate datasets
    # add transforms
    train_transform = transforms.Compose([])

    if cfg.data.augmentations.random_hflip:
        train_transform.append(transforms.RandomHorizontalFlip())
    if cfg.data.augmentations.random_rotate:
        train_transform.append(transforms.RandomRotate(10))
    if cfg.data.augmentations.random_crop:
        train_transform.append(transforms.RandomCrop(255))

    # mandatory transforms
    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    train_transform.append(transforms.Normalize(mean=mean, std=std))

    # test transforms
    test_transform = transforms.Compose([transforms.Normalize(mean=mean, std=std)])

    # create loaders
    train_loader = DataLoader(CityscapesLoader(data_config=data_configs,
                                            split='train',
                                            transform=train_transform),
                              batch_size=cfg.segmentation.batch_size,
                              shuffle=True)
    val_loader = DataLoader(CityscapesLoader(data_config=data_configs,
                                          split='val',
                                          transform=test_transform),
                            batch_size=1,
                            shuffle=False)
    test_loader = DataLoader(CityscapesLoader(data_config=data_configs,
                                           split='test',
                                           transform=test_transform),
                             batch_size=1,
                             shuffle=False)

    loaders = CityscapesLoaderObject(train_loader=train_loader,
                                  test_loader=test_loader,
                                  val_loader=val_loader,
                                  class_weights=None,
                                  data_configs=data_configs)

    return loaders