from data.datasets.segmentation.seismic.configuration import get_seismic, get_seismic_test_config
from data.datasets.segmentation.cityscapes.configuration import get_cityscapes, get_cityscapes_test_config
from data.datasets.segmentation.common.dataobjects import LoaderObject
from config import BaseConfig


def get_dataset(cfg: BaseConfig, override: str = None):
    if override is not None:
        dataset = override
    else:
        dataset = cfg.data.dataset

    if dataset == 'seismic':
        return get_seismic(cfg)
    elif dataset == 'cityscapes':
        return get_cityscapes(cfg)
    else:
        raise Exception('Dataset not implemented yet!')


def get_stat_config(loaders: LoaderObject, cfg: BaseConfig, uspec_analysis: bool = False):
    if cfg.data.dataset == 'seismic':
        out = get_seismic_test_config(loaders, cfg, uspec_analysis=uspec_analysis)
        return out
    if cfg.data.dataset == 'cityscapes':
        out = get_cityscapes_test_config(loaders, cfg)
        return out
    else:
        raise Exception('Dataset not implemented yet!')