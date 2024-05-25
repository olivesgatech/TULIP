from models.classification.resnet import ResNet18, ResNet34, ResNet82, ResNetn, MCResNet18, DMCResNet
from models.classification.sdnresnet_sngp import SNGPSDNResNet18, SNGPSDNResNet34, SNGPSDNResNet50, SNGPSDNResNet101, \
    SNGPSDNResNet152
from models.classification.spectralresnet import SpectralResNet18, SpectralResNet82
from models.classification.sngp import sngp_model
from models.segmentation.deeplabv3.deeplab import DeepLab
from data.datasets.shared.utils import DatasetStructure
from config import BaseConfig


def build_architecture(architecture: str, data_cfg: DatasetStructure, cfg: BaseConfig):
    sngp_override = False
    num_sdn_ics = -1
    gp = None
    if architecture == 'resnet-18':
        if cfg.classification.pretrained:
            out = ResNetn(num_classes=data_cfg.num_classes, type=18, pretrained=True)
        else:
            out = ResNet18(num_classes=data_cfg.num_classes)
    elif architecture == 'spectral-resnet-18':
        out = SpectralResNet18(num_classes=data_cfg.num_classes)
    elif architecture == 'spectral-resnet-82':
        out = SpectralResNet82(num_classes=data_cfg.num_classes)
    elif architecture == 'resnet-34':
        if cfg.classification.pretrained:
            out = ResNetn(num_classes=data_cfg.num_classes, type=34, pretrained=True)
        else:
            out = ResNet34(num_classes=data_cfg.num_classes)
    elif architecture == 'resnet-82':
        out = ResNet82(num_classes=data_cfg.num_classes)
    elif architecture == 'sngp-sdn-resnet-50':
        out, num_sdn_ics = SNGPSDNResNet50(data_cfg=data_cfg, batch_size=cfg.classification.batch_size,
                                           num_classes=data_cfg.num_classes, img_size=data_cfg.img_size)
    elif architecture == 'sngp-sdn-resnet-101':
        out, num_sdn_ics = SNGPSDNResNet101(data_cfg=data_cfg, batch_size=cfg.classification.batch_size,
                                            num_classes=data_cfg.num_classes, img_size=data_cfg.img_size)
    elif architecture == 'sngp-sdn-resnet-152':
        out, num_sdn_ics = SNGPSDNResNet152(data_cfg=data_cfg, batch_size=cfg.classification.batch_size,
                                            num_classes=data_cfg.num_classes, img_size=data_cfg.img_size)
    elif architecture == 'sngp-resnet-34':
        sngp_override = True
        out = sngp_model(cfg=cfg, data_cfg=data_cfg, num_classes=data_cfg.num_classes, extractor='resnet-34')
    elif architecture == 'sngp-resnet-82':
        sngp_override = True
        out = sngp_model(cfg=cfg, data_cfg=data_cfg, num_classes=data_cfg.num_classes, extractor='resnet-82')
    elif architecture == 'sngp-resnet-101':
        sngp_override = True
        out = sngp_model(cfg=cfg, data_cfg=data_cfg, num_classes=data_cfg.num_classes, extractor='resnet-101')
    elif architecture == 'sngp-resnet-152':
        sngp_override = True
        out = sngp_model(cfg=cfg, data_cfg=data_cfg, num_classes=data_cfg.num_classes, extractor='resnet-152')
    else:
        raise Exception('Architecture not implemented yet')
    output_dict = {
        'sngp_override': sngp_override,
        'num_sdn_ics': num_sdn_ics,
        'model': out,
        'gp': gp
    }
    return output_dict


def build_segmentation(architecture: str, data_cfg: DatasetStructure):
    mcd_override = -1
    if architecture == 'deeplab-v3':
        out = DeepLab(num_classes=data_cfg.num_classes)
    elif architecture == 'dropout-deeplab-v3':
        out = DeepLab(num_classes=data_cfg.num_classes, dropout=True)
    elif architecture == 'mcd-deeplab-v3':
        mcd_override = 15
        out = DeepLab(num_classes=data_cfg.num_classes, dropout=True)
    else:
        raise Exception('Architecture not implemented yet')

    output_dict = {
        'mcd_override': mcd_override,
        'model': out
    }
    return output_dict
