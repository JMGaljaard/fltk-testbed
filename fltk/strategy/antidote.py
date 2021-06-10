from abc import abstractmethod, ABC
import numpy as np

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
        n = len(gradients)
        dict = {}
        for i in range(n):
            dict[i] = []

        for i in range(n):
            for j in range(i + 1, n):
                # Calculate sum_squared_distance (ssd) of each pair.
                ssd = np.linalg.norm(gradients[i] - gradients[j]) ** 2
                dict[i].append(ssd)
                dict[j].append(ssd)

        # Calculate the score of each worker.
        score = []
        for i in range(n):
            dict[i].sort()
            closests = [x for index, x in enumerate(dict[i]) if index < (n - self.f - 2)]
            score.append(sum(closests))

        score = np.array(score)
        # Compute n vectors with lowest score
        idx = np.argpartition(score, self.k)
        selected = score[idx[:self.k]]

        # Return average of selected gradients
        avg = np.add.reduce(selected) / np.array(float(self.k) + 1)
        return avg

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
