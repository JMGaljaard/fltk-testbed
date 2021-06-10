from abc import abstractmethod, ABC
import numpy as np
import torch

from fltk.nets.util.utils import flatten_params
from fltk.util.base_config import BareConfig
from fltk.util.fed_avg import average_nn_parameters


class Antidote(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def process_gradients(self, gradients):
        pass

class DummyAntidote(Antidote):

    def __init__(self, cfg: BareConfig):
        Antidote.__init__(self)
        pass

    def process_gradients(self, gradients):
        return average_nn_parameters(gradients)

class MultiKrumAntidote(Antidote):

    def __init__(self, cfg: BareConfig, **kwargs):
        Antidote.__init__(self)
        self.f = cfg.get_antidote_f_value()
        self.k = cfg.get_antidote_k_value()

    def process_gradients(self, gradients):
        """
        Function which returns the average of the k gradient with the lowest score.
        """
        # Initialize dict holding all distances.
        number_gradients = len(gradients)
        distance_matrix = np.array((number_gradients, number_gradients), dtype=float)

        # Fill distance matrix for every entry
        for i in range(number_gradients):
            for j in range(i + 1, number_gradients):
                # Calculate sum_squared_distance (ssd) of each pair.
                # Use flattened parameters to allow for simple calculation of SSD
                ssd = float(torch.sum(torch.pow(flatten_params(gradients[i]) - flatten_params(gradients[j]), 2)))
                distance_matrix[i][j] = ssd
                distance_matrix[j][i] = ssd

        # Calculate the score of each worker.
        score = np.zeros(number_gradients)
        for i in range(number_gradients):
            # Get the ones that are closeby
            closest_entries = np.sort(distance_matrix[i])[:(number_gradients - self.f - 2)]
            # And take the sum according to configuration
            score[i] = sum(closest_entries)

        # Now take k closest entries
        sorted_indices = np.argsort(score)[:self.k]
        top_gradients = [gradients[top_k_index] for top_k_index in sorted_indices]
        return average_nn_parameters(top_gradients)




def create_antidote(cfg: BareConfig, **kwargs) -> Antidote:
    assert cfg is not None
    if cfg.antidote is None:
        return DummyAntidote(cfg)
    antidote_mapper = {'dummy': DummyAntidote, 'multikrum': MultiKrumAntidote}

    antidote_class = antidote_mapper.get(cfg.get_antidote_type(), None)

    if not antidote_class is None:
        antidote = antidote_class(cfg=cfg, **kwargs)
    else:
        raise Exception("Requested antidote is not supported")
    print(f'')
    return antidote
