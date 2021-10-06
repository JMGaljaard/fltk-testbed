import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, in_channel, num_channel, use_conv1x1=False, strides=1):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channel, eps=1e-3)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=num_channel, kernel_size=3, padding=1,
                               stride=strides)
        self.bn2 = nn.BatchNorm2d(num_channel, eps=1e-3)
        self.conv2 = nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, padding=1)
        if use_conv1x1:
            self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=num_channel, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(self.relu(self.bn1(x)))
        y = self.conv2(self.relu(self.bn2(y)))
        # print (y.shape)
        if self.conv3:
            x = self.conv3(x)
        # print (x.shape)
        z = y + x
        return z


def ResNet_block(in_channels, num_channels, num_residuals, first_block=False):
    layers = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            layers += [Residual(in_channels, num_channels, use_conv1x1=True, strides=2)]
        elif i > 0 and not first_block:
            layers += [Residual(num_channels, num_channels)]
        else:
            layers += [Residual(in_channels, num_channels)]
    blk = nn.Sequential(*layers)
    return blk


class FashionMNISTResNet(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(FashionMNISTResNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block2 = nn.Sequential(ResNet_block(64, 64, 2, True),
                                    ResNet_block(64, 128, 2),
                                    ResNet_block(128, 256, 2),
                                    ResNet_block(256, 512, 2))
        # Set to adaptive as ResNet default average pool is not compatible with shape
        # (512, 1, 1) and kernel size=3 (... - 1 in size, so needs padding or AdaptiveAvgPool2d).
        self.block3 = nn.AdaptiveAvgPool2d((1, 1))
        # Instead of reshape/view use Flatten layer to perform flattening for 'Dense' layer for readability.
        self.flatten = nn.Flatten()
        self.Dense = nn.Linear(512, num_classes)

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)

        y = self.block3(y)
        y = self.Dense(self.flatten(y))
        return y
