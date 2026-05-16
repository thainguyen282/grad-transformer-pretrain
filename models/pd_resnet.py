###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#   
# Copyright (c) 2017, Soumith Chintala. All rights reserved.
###############################################################################
'''
Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
Introduced partial convolutoins based padding for convolutional layers
'''

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from models.partialconv2d import PartialConv2d
import logging
import torch
logger = logging.getLogger(__name__)

import torch.nn.functional as F


__all__ = ['PDResNet', 'pdresnet18', 'pdresnet34', 'pdresnet50', 'pdresnet101',
           'pdresnet152']


# model_urls = {
#     'pdresnet18': '',
#     'pdresnet34': '',
#     'pdresnet50': '',
#     'pdresnet101': '',
#     'pdresnet152': '',
# }

model_urls = {
    'pdresnet18': '',
    'pdresnet34': '',
    'pdresnet50': '',
    'pdresnet101': '',
    'pdresnet152': '',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PartialConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, return_mask=True)

class DownsampleWithMask(nn.Module):
    """Wraps a downsample Sequential so it can accept and return a mask."""
    def __init__(self, conv, bn):
        super(DownsampleWithMask, self).__init__()
        self.conv = conv  # PartialConv2d, kernel=1
        self.bn = bn
 
    def forward(self, x, mask=None):
        # kernel=1 → no spatial padding needed
        out, mask_out = self.conv(x, mask)
        out = self.bn(out)
        return out, mask_out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change: add return_mask=True 
        self.conv1 = PartialConv2d(inplanes, planes, kernel_size=1, bias=False, return_mask=True)
        self.bn1 = nn.BatchNorm2d(planes)
        # change
        self.conv2 = PartialConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, return_mask=True)
        self.bn2 = nn.BatchNorm2d(planes)
        # change
        self.conv3 = PartialConv2d(planes, planes * self.expansion, kernel_size=1, bias=False, return_mask=True)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mask=None):
        print("=== Bottleneck forward start ===")

        residual = x
        input_mask = mask

        print("conv1")
        out, mask = self.conv1(x, mask)
        print("conv1_out")
        print("conv1_mask")

        out = self.bn1(out)
        print("bn1_out")

        out = self.relu(out)
        print("relu1_out")

        logger.info("conv2")
        out, mask = self.conv2(out, mask)
        print("conv2_out")
        print("conv2_mask")

        out = self.bn2(out)
        print("bn2_out")

        out = self.relu(out)
        print("relu2_out")

        out, mask = self.conv3(out, mask)
        print("conv3_out")
        print("conv3_mask")

        out = self.bn3(out)
        print("bn3_out")

        if self.downsample is not None:
            residual, _ = self.downsample(x, input_mask)
        out += residual
        out = self.relu(out)

        return out, mask


class PDResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(PDResNet, self).__init__()
        self.conv1 = PartialConv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, return_mask=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # change wrapper to allow mask
            downsample = DownsampleWithMask(
                PartialConv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride,
                              bias=False, return_mask=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # return nn.Sequential(*layers)
        # change from sequential to modellist to pass the mask
        return nn.ModuleList(layers) 
    def _run_layer(self, layer_list, x, mask):
        for idx, block in enumerate(layer_list):

            x, mask = block(x, mask)


        return x, mask

    def forward(self, x, mask=None):
        print("=== PDResNet forward start ===")

        x, mask = self.conv1(x, mask)
        print("after_conv1")

        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        print("")

        if mask is not None:
            mask = F.max_pool2d(mask, kernel_size=3, stride=2, padding=1)
            print("after_mask_pool", mask)

        print("layer1")
        x, mask = self._run_layer(self.layer1, x, mask)

        print("layer2")
        x, mask = self._run_layer(self.layer2, x, mask)

        print("layer3")
        x, mask = self._run_layer(self.layer3, x, mask)

        print("layer4")
        x, mask = self._run_layer(self.layer4, x, mask)

        x = self.avgpool(x)
        print("after_avgpool")

        x = x.view(x.size(0), -1)
        print("final_flatten")

        print("=== PDResNet forward end ===")

        return x


def pdresnet18(pretrained=False, **kwargs):
    """Constructs a PDResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet18']))
    return model


def pdresnet34(pretrained=False, **kwargs):
    """Constructs a PDResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet34']))
    return model


def pdresnet50(pretrained=False, **kwargs):
    """Constructs a PDResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet50']))
    return model


def pdresnet101(pretrained=False, **kwargs):
    """Constructs a PDResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet101']))
    return model


def pdresnet152(pretrained=False, **kwargs):
    """Constructs a PDResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PDResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['pdresnet152']))
    return model