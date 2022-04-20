from enum import unique, Enum


@unique
class Dataset(Enum):
    cifar10 = 'cifar10'
    cifar100 = 'cifar100'
    fashion_mnist = 'fashion-mnist'
    mnist = 'mnist'