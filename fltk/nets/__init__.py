from enum import Enum

from .cifar_10_cnn import Cifar10CNN
from .cifar_100_resnet import Cifar100ResNet
from .fashion_mnist_cnn import FashionMNISTCNN
from .fashion_mnist_resnet import FashionMNISTResNet
from .cifar_10_resnet import Cifar10ResNet
from .cifar_100_vgg import Cifar100VGG


class Nets(Enum):
    cifar100_resnet = "Cifar100ResNet"
    cifar100_vgg = "Cifar100VGG"
    cifar10_cnn = "Cifar10CNN"
    cifar10_resnet = "Cifar10ResNet"
    fashion_mnist_cnn = "FashionMNISTCNN"
    fashion_mnist_resnet = "FashionMNISTResNet"
    mnist_cnn = 'MNISTCNN'