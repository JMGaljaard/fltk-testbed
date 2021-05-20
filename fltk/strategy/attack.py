import logging
from abc import abstractmethod, ABC
from logging import ERROR, WARNING, INFO
from math import floor
from typing import List, Dict

from fltk.util.poison.poisonpill import FlipPill
from numpy import random


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
    def get_poison_pill(self):
        pass

class LabelFlipAttack(Attack):

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
        self.logger.log(INFO)
        if not self.random:
            random.seed(self.seed)
        return random.choice(workers, floor(len(workers) * ratio), replace=False)

    def get_poison_pill(self):
        return FlipPill(self.label_shuffle)
