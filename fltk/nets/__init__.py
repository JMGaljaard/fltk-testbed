from .cifar_10_cnn import Cifar10CNN
from .cifar_100_resnet import Cifar100ResNet
from .fashion_mnist_cnn import FashionMNISTCNN
from .fashion_mnist_resnet import FashionMNISTResNet
from .cifar_10_resnet import Cifar10ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .cifar_100_vgg import Cifar100VGG, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .reddit_lstm import RNNModel
from .mnist_cnn import MNIST_CNN
from .simple import SimpleMnist, SimpleNet
from ..util.definitions import Nets


def available_nets():
    return {
        Nets.cifar100_resnet: Cifar100ResNet,
        Nets.cifar100_vgg: Cifar100VGG,
        Nets.cifar10_cnn: Cifar10CNN,
        Nets.cifar10_resnet: Cifar10ResNet,
        Nets.fashion_mnist_cnn: FashionMNISTCNN,
        Nets.fashion_mnist_resnet: FashionMNISTResNet,
        Nets.mnist_cnn: MNIST_CNN,

    }

def get_net(name: Nets):
    return available_nets()[name]


def get_net_split_point(name: Nets):
    nets_split_point = {
        Nets.cifar100_resnet: 48,
        Nets.cifar100_vgg: 28,
        Nets.cifar10_cnn: 15,
        Nets.cifar10_resnet: 39,
        Nets.fashion_mnist_cnn: 7,
        Nets.fashion_mnist_resnet: 7,
        Nets.mnist_cnn: 2,
    }
    return nets_split_point[name]