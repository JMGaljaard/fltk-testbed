# pylint: disable=missing-class-docstring,invalid-name
from typing import Type

import torch

class BasicBlock(torch.nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = torch.nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x): # pylint: disable=missing-function-docstring
        return torch.nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Bottleneck(torch.nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_channels * Bottleneck.expansion),
        )

        self.shortcut = torch.nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, stride=stride, kernel_size=1, bias=False),
                torch.nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

    def forward(self, x): # pylint: disable=missing-function-docstring
        return torch.nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Cifar100ResNet(torch.nn.Module):
    def __init__(self, block: Type[torch.nn.Module] = BasicBlock, num_block=None, num_classes=100):
        super(Cifar100ResNet, self).__init__()
        if num_block is None:
            num_block = [2, 2, 2, 2]

        self.in_channels = 64

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Make resnet layers (I.e. not a singular (hidden) neuron layer), one layer may
        contain more than one residual blocks.
        Args:
            block: block type, basic block or bottleneck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return torch.nn.Sequential(*layers)

    def forward(self, x): # pylint: disable=missing-function-docstring
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class ResNet18(Cifar100ResNet):
    def __init__(self):
        super(ResNet18).__init__(BasicBlock, [2, 2, 2, 2])


class ResNet34(Cifar100ResNet):
    def __init__(self):
        super(ResNet34).__init__(BasicBlock, [3, 4, 6, 3])


class ResNet50(Cifar100ResNet):
    def __init__(self):
        super(ResNet50).__init__(Bottleneck, [3, 4, 6, 3])


class ResNet101(Cifar100ResNet):
    def __init__(self):
        super(ResNet101).__init__(Bottleneck, [3, 4, 23, 3])


class ResNet152(Cifar100ResNet):
    def __init__(self):
        super(ResNet152).__init__(Bottleneck, [3, 8, 36, 3])
