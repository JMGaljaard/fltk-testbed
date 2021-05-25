import logging
from abc import abstractmethod, ABC
from logging import ERROR
from typing import Dict, List

import torch


class PoisonPill(ABC):

    def __init__(self):
        self.logger = logging.getLogger()

    @abstractmethod
    def poison_input(self, X: torch.Tensor, *args, **kwargs):
        """
        Poison the output according to the corresponding attack.
        """
        pass

    @abstractmethod
    def poison_output(self, X: torch.Tensor, Y: torch.Tensor, *args, **kwargs):
        """
        Poison the output according to the corresponding attack.
        """
        pass

    def poison_targets(self, targets):
        return targets


class FlipPill(PoisonPill):

    def poison_targets(self, targets: List[int]) -> List[int]:
        """
        Apply poison to the targets of a dataset. Note that this is a somewhat strange approach, as the pill ingest the
        targets, instead of the Dataset itself. However, this allows for a more efficient implementation.
        @param targets: Original targets of the dataset.
        @type targets: list
        @return: List of mapped targets according to self.flips.
        @rtype: list
        """
        # Apply mapping to the input, default value is the target itself!
        return list(map(lambda y: self.flips.get(y, y), targets))

    @staticmethod
    def check_consistency(flips) -> bool:
        for attack in flips.keys():
            if flips.get(flips.get(attack, -2), -1) != attack:
                # -1 because ONE_HOT encoding can never represent a negative number
                logging.getLogger().log(ERROR,
                                        f'Cyclic inconsistency, {attack} resolves back to {flips[flips[attack]]}')
                raise Exception('Inconsistent flip attack!')
        return True

    def __init__(self, flip_description: Dict[int, int]):
        """
            Implements the flip attack scenario, where one or multiple attacks are implemented
            """
        super().__init__()
        assert FlipPill.check_consistency(flip_description)
        self.flips = flip_description

    def poison_output(self, X: torch.Tensor, Y: torch.Tensor, *args, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        Flip attack does not affect output, rather the pill is taken by the dataset.
        """
        return X, Y

    def poison_input(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Flip attack does not change the input during training.
        """
        return X
