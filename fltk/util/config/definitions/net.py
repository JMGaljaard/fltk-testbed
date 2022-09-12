from typing import Optional

from aenum import unique, Enum

@unique
class Nets(Enum):
    """ """
    cifar100_resnet = "Cifar100ResNet"
    cifar100_vgg = "Cifar100VGG"
    cifar10_cnn = "Cifar10CNN"
    cifar10_resnet = "Cifar10ResNet"
    fashion_mnist_cnn = "FashionMNISTCNN"
    fashion_mnist_resnet = "FashionMNISTResNet"
    mnist_cnn = 'MNISTCNN'

    @classmethod
    def _missing_name_(cls, name: str) -> Optional["Nets"]:
        """Helper function to get lower/higher-case configured network, to allow for case-insensitive lookup.

        Args:
          name (str): Name of network to lookup in case of missing lookup.

        Returns:
            Optional[Nets]: Of name is not part of defined networks, else reference to class implementing network.

        """
        for member in cls:
            if member.name.lower() == name.lower():
                return member
