import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data.datasets.segmentation.common.dataobjects import SegmentationStructure, LoaderObject


class CityscapesStructure(SegmentationStructure):
    def __init__(self):
        super(SegmentationStructure, self).__init__()


class CityscapesLoaderObject(LoaderObject):
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                 class_weights: torch.tensor, data_configs: CityscapesStructure):
        super(CityscapesLoaderObject, self).__init__(train_loader=train_loader,
                                                     test_loader=test_loader,
                                                     val_loader=val_loader,
                                                     class_weights=class_weights,
                                                     data_configs=data_configs)