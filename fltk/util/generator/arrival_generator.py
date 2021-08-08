import abc
from abc import ABC


class ArrivalGenerator(ABC):
    """
    Abstract Base Class for generating arrivals in the system. These tasks must be run
    """
    @abc.abstractmethod
    def set_config(self):
        pass

    @abc.abstractmethod
    def generate_arrivals(self):
        pass



class ExperimentGenerator(ArrivalGenerator):
    def set_config(self):
        pass

    def generate_arrivals(self):
        pass


class EvaluationGenerator(ArrivalGenerator):
    def set_config(self):
        pass

    def generate_arrivals(self):
        pass