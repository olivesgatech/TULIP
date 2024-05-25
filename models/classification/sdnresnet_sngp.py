from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.classification.sngp import Laplace


# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1


# the internal classifier for all SDNs
class InternalFeatureExtractor(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, num_features=128):
        super(InternalFeatureExtractor, self).__init__()
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels
        self.bn1 = nn.BatchNorm1d(num_features)

        if red_kernel_size == -1:
            self.linear1 = nn.Linear(output_channels*input_size*input_size, num_features)
            self.forward = self.forward_wo_pooling
            self.features = self.features_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)

            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear1 = nn.Linear(output_channels*red_input_size*red_input_size, num_features)
            self.forward = self.forward_w_pooling
            self.features = self.features_w_pooling

    def features_w_pooling(self, x):
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        out = self.bn1(self.linear1(mixed.view(mixed.size(0), -1)))
        return out

    def features_wo_pooling(self, x):
        out = self.bn1(self.linear1(x.view(x.size(0), -1)))
        return out

    def forward_w_pooling(self, x):
        out = self.features_w_pooling(x)
        return out

    def forward_wo_pooling(self, x):
        out = self.features_wo_pooling(x)
        return out


class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, data_cfg, batch_size, num_features=128):
        super(InternalClassifier, self).__init__()
        feature_extractor = InternalFeatureExtractor(input_size, output_channels, num_classes, num_features)

        # Defaults from SNGP on CIFAR-10 in uncertainty-baselines
        num_deep_features = num_features
        num_gp_features = 64
        normalize_gp_features = True
        num_random_features = 512
        num_data = data_cfg.train_len
        mean_field_factor = 25
        ridge_penalty = 1
        feature_scale = 2

        self._model = Laplace(
            feature_extractor,
            num_deep_features,
            num_gp_features,
            normalize_gp_features,
            num_random_features,
            num_classes,
            num_data,
            batch_size,
            ridge_penalty,
            feature_scale,
            mean_field_factor,
        )

    def forward(self, x):
        out = self._model(x)
        return out


class BottleneckwOutput(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, params, data_cfg, batch_size, stride=1):
        super(BottleneckwOutput, self).__init__()
        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]

        self.depth = 2

        layers = nn.ModuleList()

        conv_layer = []
        # conv_layer.append(nn.BatchNorm2d(in_channels))
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU())
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU())
        # conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False))
        # conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.Conv2d(channels, channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels * self.expansion))

        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU())

        self.layers = layers

        if add_output:
            self.output = InternalClassifier(input_size, self.expansion * channels, num_classes, data_cfg, batch_size)
            self.no_output = False

        else:
            self.output = None
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x):
        layer0 = self.layers[0](x)  # conv layers
        fwd = layer0 + self.layers[1](x)  # shortcut
        # ic_input = layer0 - self.layers[1](x)
        ic_input = fwd
        ic_output = self.output(ic_input)
        return self.layers[2](fwd), 1, ic_output  # output layers for this module

    def only_output(self, x):
        layer0 = self.layers[0](x)  # conv layers
        ic_input = layer0 - self.layers[1](x)
        # fwd = fwd + self.layers[1](x)  # shortcut
        # fwd = self.layers[2](fwd)  # activation
        out = self.output(ic_input)  # output layers for this module
        return out

    def only_forward(self, x):
        fwd = self.layers[0](x)  # conv layers
        fwd = fwd + self.layers[1](x)  # shortcut
        return self.layers[2](fwd), 0, None  # activation


class BasicBlockWOutput(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, params, data_cfg, batch_size, stride=1):
        super(BasicBlockWOutput, self).__init__()
        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]

        self.depth = 2

        layers = nn.ModuleList()

        conv_layer = []
        # conv_layer.append(nn.BatchNorm2d(in_channels))
        conv_layer.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))
        conv_layer.append(nn.ReLU())
        conv_layer.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        conv_layer.append(nn.BatchNorm2d(channels))

        layers.append(nn.Sequential(*conv_layer))

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * channels)
            )

        layers.append(shortcut)
        layers.append(nn.ReLU())

        self.layers = layers

        if add_output:
            self.output = InternalClassifier(input_size, self.expansion * channels, num_classes, data_cfg, batch_size)
            self.no_output = False

        else:
            self.output = None
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x):
        layer0 = self.layers[0](x)  # conv layers
        fwd = layer0 + self.layers[1](x)  # shortcut
        ic_input = fwd
        # ic_input = layer0 - self.layers[1](x)
        ic_output = self.output(ic_input)
        return self.layers[2](fwd), 1, ic_output  # output layers for this module

    def only_output(self, x):
        layer0 = self.layers[0](x)  # conv layers
        ic_input = layer0 - self.layers[1](x)
        # fwd = layer0 + self.layers[1](x)  # shortcut
        # fwd = self.layers[2](fwd)  # activation
        out = self.output(ic_input)  # output layers for this module
        return out

    def only_forward(self, x):
        fwd = self.layers[0](x)  # conv layers
        fwd = fwd + self.layers[1](x)  # shortcut
        return self.layers[2](fwd), 0, None  # activation


# flatten the output of conv layers for fully connected layers
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def SNGPResNetSDN56(data_cfg, batch_size, num_classes: int = 10, img_size: int = 32):
    blocks = [9, 9, 9]
    internal_classifier_positions = [[0, 0, 0, 1, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0, 1, 0, 0, 0]] # original layout from paper
    internal_classifier_positions = [[0, 1, 0, 1, 0, 1, 0, 1, 0],
                                     [1, 0, 1, 0, 1, 0, 1, 0, 1],
                                     [0, 1, 0, 1, 0, 1, 0, 1, 0]]
    return ResNetSDN(num_classes=num_classes,
                     num_blocks=blocks,
                     add_ic=internal_classifier_positions,
                     img_size=img_size, data_cfg=data_cfg, batch_size=batch_size), 14
                     # img_size=img_size), 7


def SNGPSDNResNet18(data_cfg, batch_size, num_classes: int = 10, img_size: int = 32):
    blocks = [2, 2, 2, 2]
    internal_classifier_positions = [[0, 1],
                                     [1, 1],
                                     [1, 1],
                                     [1, 1]]
    return ResNetSDN(num_classes=num_classes,
                     num_blocks=blocks,
                     add_ic=internal_classifier_positions,
                     img_size=img_size, data_cfg=data_cfg, batch_size=batch_size), 9


def SNGPSDNResNet34(data_cfg, batch_size, num_classes: int = 10, img_size: int = 32):
    blocks = [3, 4, 6, 3]
    internal_classifier_positions = [[0, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 1, 0, 0, 1, 0],
                                     [0, 1, 0]]
    return ResNetSDN(num_classes=num_classes,
                     num_blocks=blocks,
                     add_ic=internal_classifier_positions,
                     img_size=img_size, data_cfg=data_cfg, batch_size=batch_size), 7


def SNGPSDNResNet50(data_cfg, batch_size, num_classes: int = 10, img_size: int = 32):
    blocks = [3, 4, 6, 3]
    internal_classifier_positions = [[0, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 1, 0, 0, 1, 0],
                                     [0, 1, 0]]
    return ResNetSDN(num_classes=num_classes,
                     num_blocks=blocks,
                     add_ic=internal_classifier_positions,
                     img_size=img_size,
                     block_type='bottleneck', data_cfg=data_cfg, batch_size=batch_size), 7


def SNGPSDNResNet82(data_cfg, batch_size, num_classes: int = 10, img_size: int = 32):
    blocks = [5, 15, 15, 5]
    internal_classifier_positions = [[0, 0, 1, 1, 0],
                                     [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
                                     [0, 0, 1, 1, 0]]
    return ResNetSDN(num_classes=num_classes,
                     num_blocks=blocks,
                     add_ic=internal_classifier_positions,
                     img_size=img_size, data_cfg=data_cfg, batch_size=batch_size), 22


def SNGPSDNResNet101(data_cfg, batch_size, num_classes: int = 10, img_size: int = 32):
    blocks = [3, 4, 23, 3]
    # v4 positions
    internal_classifier_positions = [[0, 1, 1],
                                     [0, 0, 1, 1],
                                     [0, 0, 1, 1, 0, 1, 1, 0, 1, 1,
                                      0, 1, 1, 0, 1, 1, 0, 1, 1, 0,
                                      1, 1, 0],
                                     [0, 1, 1]]
    # v5 positions 17 total
    internal_classifier_positions = [[0, 1, 0],
                                     [1, 0, 1, 0],
                                     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                      1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                      1, 0, 1],
                                     [0, 1, 0]]
    # v5 positions 13 total
    internal_classifier_positions = [[0, 0, 1],
                                     [0, 0, 1, 0],
                                     [0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                                      1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                                      0, 0, 1],
                                     [0, 0, 1]]
    return ResNetSDN(num_classes=num_classes,
                     num_blocks=blocks,
                     add_ic=internal_classifier_positions,
                     img_size=img_size,
                     block_type='bottleneck', data_cfg=data_cfg, batch_size=batch_size), 13


def SNGPSDNResNet152(data_cfg, batch_size, num_classes: int = 10, img_size: int = 32):
    blocks = [3, 8, 36, 3]
    internal_classifier_positions = [[0, 0, 1],
                                     [0, 0, 1, 0, 0, 1, 0, 0],
                                     [1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                                      0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
                                      0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                                      1, 0, 0, 1, 0, 0],
                                     [1, 0, 0]]
    return ResNetSDN(num_classes=num_classes,
                     num_blocks=blocks,
                     add_ic=internal_classifier_positions,
                     img_size=img_size,
                     block_type='bottleneck', data_cfg=data_cfg, batch_size=batch_size), 18


class ResNetSDN(nn.Module):
    def __init__(self, num_classes: int, num_blocks: List[int], add_ic: List[List[int]], img_size: int,  data_cfg,
                 batch_size,
                 block_type: str = 'basic'):
        super(ResNetSDN, self).__init__()
        num_features = 128
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.input_size = img_size
        self.block_type = block_type
        self.add_out_nonflat = add_ic
        self.add_output = [item for sublist in self.add_out_nonflat for item in sublist]
        self.init_weights = True
        self.in_channels = 16
        self.num_output = sum(self.add_output) + 1

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0
        self.data_cfg = data_cfg
        self.batch_size = batch_size

        if self.block_type == 'basic':
            self.block = BasicBlockWOutput
        elif self.block_type == 'bottleneck':
            self.block = BottleneckwOutput
        else:
            raise ValueError(f'Block type {self.block_type} not implemented yet!')

        init_conv = []

        if self.input_size == 32:  # cifar10
            self.cur_input_size = self.input_size
            # init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        else:  # tiny imagenet
            self.cur_input_size = int(self.input_size / 2)
            # self.cur_input_size = self.input_size
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False))

        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU())

        self.init_conv = nn.Sequential(*init_conv)
        self.ic_init_conv = InternalClassifier(input_size=self.cur_input_size, num_classes=num_classes,
                                               output_channels=self.in_channels, data_cfg=data_cfg,
                                               batch_size=batch_size)

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layer(64, block_id=0, stride=1))
        self.cur_input_size = np.ceil(self.cur_input_size / 2)
        self.layers.extend(self._make_layer(128, block_id=1, stride=2))

        self.cur_input_size = np.ceil(self.cur_input_size / 2)
        self.layers.extend(self._make_layer(256, block_id=2, stride=2))

        self.cur_input_size = np.ceil(self.cur_input_size / 2)
        # self.cur_input_size = 4
        self.layers.extend(self._make_layer(512, block_id=3, stride=2))

        end_layers = []

        # self.cur_input_size = np.ceil(self.cur_input_size / 2)

        # cifar10
        # end_layers.append(nn.AvgPool2d(kernel_size=4))

        # imagenet
        end_layers.append(nn.AvgPool2d(kernel_size=int(self.cur_input_size)))
        end_layers.append(Flatten())

        # cifar10
        # end_layers.append(nn.Linear(512 * self.block.expansion, self.num_classes))

        # imagenet
        end_layers.append(nn.Linear(512 * self.block.expansion, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def get_penultimate_dim(self):
        return 512 * self.block.expansion

    def _make_layer(self, channels, block_id, stride):
        num_blocks = int(self.num_blocks[block_id])
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for cur_block_id, stride in enumerate(strides):
            add_output = self.add_out_nonflat[block_id][cur_block_id]
            params = (add_output, self.num_classes, int(self.cur_input_size), self.cur_output_id)
            layers.append(self.block(self.in_channels, channels, params, self.data_cfg, self.batch_size, stride))
            self.in_channels = channels * self.block.expansion
            self.cur_output_id += add_output

        return layers

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []
        fwd = self.init_conv(x)
        outputs.append(self.ic_init_conv(fwd))
        for layer in self.layers:
            # print(fwd.shape)
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return outputs

    def ic_features(self, x):
        outputs = []
        fwd = self.init_conv(x)
        prev_features = self.ic_init_conv(fwd)
        outputs.append(prev_features)
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)

        return outputs

    def feature_list(self, x):
        outputs = []
        fwd = self.init_conv(x)
        prev_features = self.ic_init_conv(fwd)
        outputs.append(prev_features)
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        return fwd, outputs

    def intermediate_forward(self, x, layer_index):
        fwd, outputs = self.feature_list(x)
        if layer_index < 0 or layer_index >= len(outputs):
            raise ValueError(f'Index {layer_index} is not within SDN bounds!')
        return outputs[layer_index]

        # fwd = self.init_conv(x)
        # prev_features = self.ic_init_conv.features(fwd)
        # if layer_index == 0:
        #     return prev_features
        # ic_count = 1
        # no_out = True
        # for layer in self.layers:
        #     fwd, is_output, output, features = layer(fwd)
        #     if is_output:
        #         out = features
        #         if ic_count == layer_index:
        #             no_out = False
        #             break
        #         ic_count += 1
        # if no_out:
        #     raise ValueError(f'Index {layer_index} is not within SDN bounds!')
        # return out

    # takes a single input
    def early_exit(self, x):
        confidences = []
        outputs = []

        fwd = self.init_conv(x)
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)

            if is_output:
                outputs.append(output)
                softmax = nn.functional.softmax(output[0], dim=0)

                confidence = torch.max(softmax)
                confidences.append(confidence)

                if confidence >= self.confidence_threshold:
                    is_early = True
                    return output, output_id, is_early

                output_id += is_output

        output = self.end_layers(fwd)
        outputs.append(output)

        softmax = nn.functional.softmax(output[0], dim=0)
        confidence = torch.max(softmax)
        confidences.append(confidence)
        max_confidence_output = np.argmax(confidences)
        is_early = False
        return outputs[max_confidence_output], max_confidence_output, is_early