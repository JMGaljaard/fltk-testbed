import logging
from abc import abstractmethod, ABC
from logging import ERROR, WARNING, INFO
from math import floor
from typing import List, Dict

from numpy import random

from fltk.util.poison.poisonpill import FlipPill, PoisonPill


class Attack(ABC):

    def __init__(self, max_rounds: int, seed=42):
        self.logger = logging.getLogger()
        self.round = 1
        self.max_rounds = max_rounds
        self.seed = seed

    def advance_round(self):
        """
        Function to advance to the ne
        """
        self.round += 1
        if self.round > self.max_rounds:
            self.logger.log(WARNING, f'Advancing outside of preset number of rounds {self.round} / {self.max_rounds}')

    @abstractmethod
    def select_poisoned_workers(self, workers: List, ratio: float) -> List:
        pass

    @abstractmethod
    def build_attack(self):
        pass

    @abstractmethod
    def get_poison_pill(self, *ars, **kwargs) -> PoisonPill:
        pass


class LabelFlipAttack(Attack):

    def build_attack(self, flip_description=None) -> PoisonPill:
        """
        Build Label flip attack pill, default will flip the classes 0 and 9, assuming it is trained on Fasion MNIST.
        If a different train class is used, and has less than 10 classes, this will result in exceptions.
        """
        if flip_description is None:
            flip_description = {0: 9, 9: 0}
        return FlipPill(flip_description=flip_description)

    def __init__(self, max_rounds: int, ratio: float, label_shuffle: Dict, seed: int = 42, random=False):
        """

        """
        if 0 > ratio > 1:
            self.logger.log(ERROR, f'Cannot run with a ratio of {ratio}, needs to be in range [0, 1]')
            raise Exception("ratio is out of bounds")
        Attack.__init__(self, max_rounds, seed)
        self.ratio = ratio
        self.label_shuffle = label_shuffle
        self.random = random

    def select_poisoned_workers(self, workers: List, ratio: float):
        """
        Randomly select workers from a list of workers provided by the Federator.
        """
        self.logger.log(INFO, "Selecting workers to gather from")
        if not self.random:
            random.seed(self.seed)
        return random.choice(workers, floor(len(workers) * ratio), replace=False)

    def get_poison_pill(self):
        return FlipPill(self.label_shuffle)
