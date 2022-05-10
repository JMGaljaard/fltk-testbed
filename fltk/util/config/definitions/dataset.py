from aenum import unique, Enum
from re import T


@unique
class Dataset(Enum):
    cifar10 = 'cifar10'
    cifar100 = 'cifar100'
    fashion_mnist = 'fashion-mnist'
    mnist = 'mnist'

    @classmethod
    def _missing_name_(cls, name: str) -> T:
        for member in cls:
            if member.name.lower() == name.lower():
                return member
