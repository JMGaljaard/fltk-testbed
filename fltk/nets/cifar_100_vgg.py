# pylint: disable=missing-class-docstring,invalid-name,missing-function-docstring
import torch

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [torch.nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [torch.nn.BatchNorm2d(l)]

        layers += [torch.nn.ReLU(inplace=True)]
        input_channel = l

    return torch.nn.Sequential(*layers)


class Cifar100VGG(torch.nn.Module):

    def __init__(self, features=make_layers(cfg['D'], batch_norm=True), num_class=100):
        super(Cifar100VGG, self).__init__()
        self.features = features

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_class)
        )

    def forward(self, x): # pylint: disable=missing-function-docstring
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


def vgg11_bn():
    return Cifar100VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13_bn():
    return Cifar100VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16_bn():
    return Cifar100VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19_bn():
    return Cifar100VGG(make_layers(cfg['E'], batch_norm=True))
