import torch
import torch.nn as nn

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.datasets as datasets
from random import shuffle
from copy import deepcopy as dcp


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def downsmp(x, nres):
    lx = [x]
    for idx in range(nres - 1):
        xx = (F.max_pool2d(lx[idx], kernel_size=2))
        lx.append(xx)
    return lx


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, max_scales=4):
        super(BasicBlock, self).__init__()
        self.corr = [r for r in range(self.max_scales)]
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes, affine=False) for _ in range(max_scales)])
        self.relu = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(max_scales)])
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes, affine=False) for _ in range(max_scales)])
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        lx = downsmp(x, self.max_scales)
        residual = lx
        ly = []
        for r in self.corr:
            out = self.conv1(lx[r])
            out = self.bn1[r](out)
            out = self.relu[r](out)

            out = self.conv2(out)
            out = self.bn2[r](out)

            out += residual[r]
            out = self.relu[r](out)
            ly.append(self.interp(y, scale_factor=2 ** r, mode='nearest'))

        if self.downsample is not None:
            residual = self.downsample(x)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, channels, num_classes=10, max_scales=4, factor=1):
        # first block
        super(ResNet, self).__init__()
        self.lyr = lyr
        self.factor = factor
        # alinit disc
        self.alpha = nn.Parameter(torch.ones(self.max_scales))
        print('init alpha:', self.alpha.data)
        # self.athr disc
        self.corr = [r for r in range(self.max_scales)]
        self.nalpha = F.softmax(self.alpha*self.factor, 0).view(-1, 1)
        self.inplanes = channels

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(self.inplanes, affine=False) for _ in range(max_scales)])
        self.relu = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(max_scales)])
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.layer2 = self._make_layer(block, self.inplanes, layers[1], stride=1)  # for progressive self.inplanes*2
        self.layer3 = self._make_layer(block, self.inplanes, layers[2], stride=1)  # for progressive self.inplanes*2
        self.layer4 = self._make_layer(block, self.inplanes, layers[3], stride=1)  # for progressive self.inplanes*2
        self.avgpool = nn.AvgPool2d(32)
        self.fc = nn.Linear(self.inplanes, num_classes)

    def forward(self, x):
        lx = downsmp(x, self.max_scales)
        ly = []
        if lyr = 0:
            for r in self.corr:
                out = self.conv1(lx[r])  # 224x224
                # print('input size', x.size())
                out = self.bn1[r](out)
                out = self.relu[r](out)
                ly.append
        # x = self.maxpool(x)  # 112x112
        print('input size', x.size())
        x = self.layer1(x)  # 56x56
        print('input size', x.size())
        x = self.layer2(x)  # 28x28
        print('input size', x.size())
        x = self.layer3(x)  # 14x14
        print('input size', x.size())
        x = self.layer4(x)  # 7x7
        print('input size', x.size())

        x = self.avgpool(x)  # 1x1
        print('input size', x.size())
        x = x.view(x.size(0), -1)
        print('input size', x.size())
        x = self.fc(x)
        print('input size', x.size())

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes,affine=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model
net = resnet34()
# print(net)

def resnet18(**kwargs):

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

x = torch.rand(8,3,32,32)
output = net(x)

class sConv2d(nn.Module):
    def __init__(self, in_, out_, kernel_size, max_scales=4, usf=True, factor=1, alinit=0, athr=0, lf=0, prun=0,
                 alpha=0, pooling=False):
        super(sConv2d, self).__init__()
        self.interp = F.interpolate
        self.max_scales = max_scales
        self.athr = athr
        self.prun = prun
        self.usf = usf
        self.prun = prun
        # lf disc
        self.factor = factor
        # alinit disc
        self.alpha = nn.parameter(torch.ones(self.max_scales))
        print('init alpha:', self.alpha.data)
        # self.athr disc
        self.corr = [r for r in range(self.max_scales)]
        self.nalpha = F.softmax(self.alpha*self.factor, 0).view(-1, 1)
        # not self.usf disc
