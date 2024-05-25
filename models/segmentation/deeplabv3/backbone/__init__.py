def build_backbone(backbone, output_stride, BatchNorm, dropout: bool = False):
    if dropout:
        from models.segmentation.deeplabv3.backbone import resnet2 as rn
    else:
        from models.segmentation.deeplabv3.backbone import resnet as rn
    if backbone == 'resnet_101':
        return rn.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet_18':
        return rn.resnet18(pretrained=False, progress=True)
    elif backbone == 'resnet_34':
        return rn.resnet34(pretrained=False, progress=True)
    else:
        raise NotImplementedError
