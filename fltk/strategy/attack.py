import logging
import numpy as np
import random
from abc import abstractmethod, ABC
from logging import ERROR, WARNING, INFO
from math import floor, ceil
from typing import List, Dict

from collections import ChainMap

from fltk.strategy.client_selection import random_selection
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
    def select_poisoned_clients(self, workers: List, ratio: float = None) -> List:
        pass

    @abstractmethod
    def build_attack(self):
        pass

    @abstractmethod
    def get_poison_pill(self) -> PoisonPill:
        pass

    @abstractmethod
    def is_active(self, current_round: int = 0) -> bool:
        pass

    @abstractmethod
    def select_clients(self, poisoned_clients, healthy_workers, n):
        pass


class LabelFlipAttack(Attack):

    def is_active(self, current_round=0) -> bool:
        return True

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

    def select_poisoned_clients(self, workers: List, ratio: float = None):
        """
        Randomly select workers from a list of workers provided by the Federator.
        """
        self.logger.log(INFO, "Selecting workers to gather from")
        if not self.random:
            np.random.seed(self.seed)
        cloned_workers = workers.copy()
        np.random.shuffle(cloned_workers)
        return cloned_workers[0:ceil(len(workers) * ratio)]

    def get_poison_pill(self):
        return FlipPill(self.label_shuffle)

    def select_clients(self, poisoned_clients, healthy_clients, n):
        return random_selection(poisoned_clients + healthy_clients, n)

class TimedLabelFlipAttack(LabelFlipAttack):

    def __init__(self, start_round=0, end_round=0, availability=0, max_rounds: int = 0, ratio: float = 0, label_shuffle: Dict = None, seed: int = 42, random=False,
                 cfg: BareConfig = None, ):
        LabelFlipAttack.__init__(self, max_rounds, ratio, label_shuffle, seed, random, cfg)
        self.start_round = start_round
        self.end_round = end_round
        self.availability = availability

    def is_active(self, currentRound=0) -> bool:
        """
        Timed attack is only active when the current round is in between the start and end rounds of the attack.
        """
        return self.start_round <= currentRound <= self.end_round

    def select_clients(self, poisoned_clients: List, healthy_clients: List, n):
        """
        Select poisoned workers based on availability.
        When availability = 0.5, selecting a participant has a 50% chance of being a poisoned one.
        """
        poison_counter = 0
        healthy_counter = 0
        nr_poisoned_workers = len(poisoned_clients)
        for i in range(n):
            if random.random() <= self.availability and poison_counter < nr_poisoned_workers or healthy_counter >= len(healthy_clients):
                poison_counter += 1
            else:
                healthy_counter += 1
        return random.sample(poisoned_clients, poison_counter) + random.sample(healthy_clients, healthy_counter)


def create_attack(cfg: BareConfig, **kwargs) -> Attack:
    """
    Function to create Poison attack based on the configuration that was passed during execution.
    Exception gets thrown when the configuration file is not correct.
    TODO parse TimedFlipAttack from config
    """
    assert not cfg is None and not cfg.poison is None
    attack_mapper = {'flip': LabelFlipAttack, 'timed': TimedLabelFlipAttack}

    attack_class = attack_mapper.get(cfg.get_attack_type(), None)

    if not attack_class is None:
        attack = attack_class(cfg=cfg, **kwargs)
    else:
        raise Exception("Requested attack is not supported...")
    print(f'')
    return attack