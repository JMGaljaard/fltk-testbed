from abc import abstractmethod

from fltk.util.arguments import Arguments


class DistDataset:
    train_sampler = None
    test_sampler = None
    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None

    def __init__(self, args: Arguments):
        self.args = args

    def get_args(self):
        """
        Returns the arguments during initialization of the method

        :return: Arguments
        """
        return self.args

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_train_sampler(self):
        return self.train_sampler

    def get_test_sampler(self):
        return self.test_sampler

    @abstractmethod
    def init_train_dataset(self):
        raise NotImplementedError("load_train_dataset() isn't implemented")

    @abstractmethod
    def init_test_dataset(self):
        raise NotImplementedError("load_train_dataset() isn't implemented")
