# pylint: disable=missing-class-docstring,invalid-name
from typing import Type

import torch
import torch.nn.functional as F


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x): # pylint: disable=missing-function-docstring
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x): # pylint: disable=missing-function-docstring
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Cifar10ResNet(torch.nn.Module):
    def __init__(self, block: Type[torch.nn.Module] = BasicBlock, num_blocks=None, num_classes=10):
        super(Cifar10ResNet, self).__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x): # pylint: disable=missing-function-docstring
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet18(Cifar10ResNet):

    def __init__(self):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2])


class ResNet34(Cifar10ResNet):
    def __init__(self):
        super(ResNet34, self).__init__(BasicBlock, [3, 4, 6, 3])


class ResNet50(Cifar10ResNet):
    def __init__(self):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])


class ResNet101(Cifar10ResNet):
    def __init__(self):
        super(ResNet101, self).__init__(Bottleneck, [3, 4, 23, 3])


class ResNet152(Cifar10ResNet):
    def __init__(self):
        super(ResNet152, self).__init__(Bottleneck, [3, 8, 36, 3])
