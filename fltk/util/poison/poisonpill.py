import logging
from abc import abstractmethod, ABC
from logging import WARNING, ERROR
from typing import Dict

import torch
from torch.nn.functional import one_hot


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


class FlipPill(PoisonPill):

    @staticmethod
    def check_consistency(flips) -> None:
        for attack in flips.keys():
            if flips.get(flips[attack], -1) != attack:
                # -1 because ONE_HOT encoding can never represent a negative number
                logging.getLogger().log(ERROR,
                                        f'Cyclic inconsistency, {attack} resolves back to {flips[flips[attack]]}')
                raise Exception('Inconsistent flip attack!')

    def __init__(self, flip_description: Dict[int, int]):
        """
            Implements the flip attack scenario, where one or multiple attacks are implemented
            """
        super().__init__()
        assert FlipPill.check_consistency(flip_description)
        self.flips = flip_description

    def poison_output(self, X: torch.Tensor, Y: torch.Tensor, *args, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        Apply flip attack, assumes a ONE_HOT encoded input (see torch.nn.functional.one_hot). The
        """
        if kwargs['classification']:
            decoded: torch.Tensor = Y.argmax(-1).cpu()
            # TODO: Figure out how to do this on GPU
            # TODO: Maybe implement on client in numpy in dataloader.
            updated_decoded = decoded.apply_(lambda x: self.flips.get(x, x)).to(Y.device)
            new_Y = torch.nn.functional.one_hot(updated_decoded)
        else:
            self.logger.log(WARNING, f'Label flip attack only support classification')
            new_Y = Y
        return X, new_Y

    def poison_input(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Flip attack does not change the input during training.
        """
        return X
