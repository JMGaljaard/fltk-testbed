from abc import abstractmethod, ABC
import numpy as np
import torch

from fltk.nets.util.utils import flatten_params
from fltk.strategy.util.antidote import calc_krum_scores
from fltk.util.base_config import BareConfig
from fltk.util.fed_avg import average_nn_parameters

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        krum_scores = calc_krum_scores(gradients)
        # Now take k closest entries
        sorted_indices = np.argsort(krum_scores)[:self.k]
        top_gradients = [gradients[top_k_index] for top_k_index in sorted_indices]
        return average_nn_parameters(top_gradients)

class ClusterAntidote(Antidote):



    def __init__(self, cfg: BareConfig, **kwargs):
        Antidote.__init__(self)
        self.f = cfg.get_antidote_f_value()
        self.k = cfg.get_antidote_k_value()
        self.past_gradients = np.array([])

        # Rho for this round poisoned
        self.rho_1 = 0.5
        # Rho for this class poisoned
        self.rho_1 = 0.75
        self.max_epoch = 130

    def process_gradients(self, gradients):
        """
        Function which returns the average of the k gradient with the lowest score.
        """
        krum_scores = calc_krum_scores(gradients)
        most_likely_good = np.argmax(krum_scores)
        # Note gradients is a list of ordered dicts (pytorch state dicts)
        new_connected_grads = [next(reversed(gradient.values())).numpy() for gradient in gradients]
        self.past_gradients = np.stack([self.past_gradients] + new_connected_grads)

        # TODO: Decide when to allow for performing the analysis.
        # TODO: Decide how many runs you want to collect.

        # TODO: Decide on how to get the number of classes
        classes_ = 10
        for cls in range(classes_):
            # Slice to get only the rows corresponding the the output node.
            sub_sample = self.past_gradients[cls::classes_]
            clf = KMeans(2)
            scaler = StandardScaler()
            fitter = PCA(n_components=2)
            scaled_param_diff = scaler.fit_transform(self.past_gradients)
            dim_reduced_gradients = fitter.fit_transform(scaled_param_diff)
            classified = clf.fit_transform(dim_reduced_gradients)

            # Get the label assigned to the 'krum' vector.
            estimated_cluster = classified[-(len(gradients) - most_likely_good)]

            this_epoch = classified[-classes_:]
            # TODO: decide how to
            sum(this_epoch)

        if flagged_updates:

        # TODO: Do clustering

        # T


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
