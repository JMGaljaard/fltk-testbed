import logging
from abc import abstractmethod, ABC
from logging import ERROR, WARNING, INFO
from math import floor, ceil
from typing import List, Dict

from numpy import random
from collections import ChainMap
from fltk.util.base_config import BareConfig
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
    def select_poisoned_workers(self, workers: List, ratio: float = None) -> List:
        pass

    @abstractmethod
    def build_attack(self):
        pass

    @abstractmethod
    def get_poison_pill(self) -> PoisonPill:
        pass


class LabelFlipAttack(Attack):

    def build_attack(self, flip_description=None) -> PoisonPill:
        """
        Build Label flip attack pill, default will flip the classes 0 and 9, assuming it is trained on Fasion MNIST.
        If a different train class is used, and has less than 10 classes, this will result in exceptions.
        """
        if flip_description is None:
            flip_description = {0: 9}
        return FlipPill(flip_description=flip_description)

    def __init__(self, max_rounds: int = 0, ratio: float = 0.0, label_shuffle: Dict = None, seed: int = 42, random=False,
                 cfg: BareConfig = None):
        """
        @param max_rounds:
        @type max_rounds: int
        @param ratio:
        @type ratio: float
        @param label_shuffle:
        @type label_shuffle: dict
        @param seed:
        @type seed: int
        @param random:
        @type random: bool
        @param cfg:
        @type cfg: dict
        """
        if cfg is None:
            if 0 > ratio > 1:
                self.logger.log(ERROR, f'Cannot run with a ratio of {ratio}, needs to be in range [0, 1]')
                raise Exception("ratio is out of bounds")
            Attack.__init__(self, cfg.epochs, cfg.get_poison_config().get('seed', None))
        else:
            Attack.__init__(self, max_rounds, seed)
            self.ratio = cfg.poison['ratio']
        self.label_shuffle = dict(ChainMap(*cfg.get_attack_config()['config']))
        self.random = random

    def select_poisoned_workers(self, workers: List, ratio: float = None):
        """
        Randomly select workers from a list of workers provided by the Federator.
        """
        self.logger.log(INFO, "Selecting workers to gather from")
        if not self.random:
            random.seed(self.seed)
        cloned_workers = workers.copy()
        random.shuffle(cloned_workers)
        return cloned_workers[0:ceil(len(workers) * ratio)]

    def get_poison_pill(self):
        return FlipPill(self.label_shuffle)


def create_attack(cfg: BareConfig) -> Attack:
    """
    Function to create Poison attack based on the configuration that was passed during execution.
    Exception gets thrown when the configuration file is not correct.
    """
    assert not cfg is None and not cfg.poison is None
    attack_mapper = {'flip': LabelFlipAttack}

    attack_class = attack_mapper.get(cfg.get_attack_type(), None)

    if not attack_class is None:
        attack = attack_class(cfg=cfg)
    else:
        raise Exception("Requested attack is not supported...")
    print(f'')
    return attack