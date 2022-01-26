from dataclasses import dataclass

# @dataclass
# class Aggregations:
#     avg = 'Avg'
#     fed_avg = 'FedAvg'
#     sum = 'Sum'

# @dataclass
# class  Samplers:
#     uniform = "uniform"
#     q_sampler = "q sampler"
#     limit_labels = "limit labels"
#     dirichlet = "dirichlet"
#     limit_labels_q = "limit labels q"
#     emd_sampler = 'emd sampler'

@dataclass
class Optimizations:
    sgd = 'SGD'
    fedprox = 'FedProx'
    fednova = 'FedNova'

# @dataclass
# class Datasets:
#     cifar10 = 'cifar10'
#     cifar100 = 'cifar100'
#     fashion_mnist = 'fashion-mnist'
#     mnist = 'mnist'

# @dataclass
# class Nets:
#     cifar100_resnet = "Cifar100ResNet"
#     cifar100_vgg = "Cifar100VGG"
#     cifar10_cnn = "Cifar10CNN"
#     cifar10_resnet = "Cifar10ResNet"
#     fashion_mnist_cnn = "FashionMNISTCNN"
#     fashion_mnist_resnet = "FashionMNISTResNet"
#     mnist_cnn = 'MNISTCNN'