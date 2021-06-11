from abc import abstractmethod, ABC
import numpy as np
import torch

from fltk.client import Client
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
    def process_gradients(self, gradients, **kwargs):
        pass

class DummyAntidote(Antidote):

    def __init__(self, cfg: BareConfig):
        Antidote.__init__(self)
        pass

    def process_gradients(self, gradients, **kwargs):
        return average_nn_parameters(gradients)

class MultiKrumAntidote(Antidote):

    def __init__(self, cfg: BareConfig, **kwargs):
        Antidote.__init__(self)
        self.f = cfg.get_antidote_f_value()
        self.k = cfg.get_antidote_k_value()

    def process_gradients(self, gradients, **kwargs):
        """
        Function which returns the average of the k gradient with the lowest score.
        """
        krum_scores = calc_krum_scores(gradients, self.f)

        # Now take k closest entries
        sorted_indices = np.argsort(krum_scores)[:self.k]
        top_gradients = [gradients[top_k_index] for top_k_index in sorted_indices]
        return average_nn_parameters(top_gradients)

class ClusterAntidote(Antidote):

    @staticmethod
    def ema(s_t_prev, value, t, rho, bias_correction = True):
        s_t = rho * s_t_prev + (1 - rho) * value
        s_t_hat = None
        if bias_correction:
            s_t_hat = s_t / (1.0 - rho**(t + 1))
        return s_t_hat if bias_correction else s_t

    def __init__(self, cfg: BareConfig, **kwargs):
        Antidote.__init__(self)
        self.f = cfg.get_antidote_f_value()
        self.k = cfg.get_antidote_k_value()
        self.past_gradients = np.array([])
        # TODO: Not hardcode this for cifar10
        self.class_targeted = np.zeros((10, cfg.epochs))
        # Rho for this round poisoned
        self.rho_1 = 0.5
        # Rho for this class poisoned
        self.rho_1 = 0.75
        self.max_epoch = 130
        self.num_classes = 10

    def process_gradients(self, gradients, **kwargs):
        """
        Function which returns the average of the k gradient with the lowest score.
        """
        epoch_indx = kwargs['epoch']
        # First 10 epochs we effectively don't do much
        if epoch_indx > 10:
            new_connected_grads = [next(reversed(gradient.values())).numpy() for gradient in gradients]
            self.past_gradients = np.stack([self.past_gradients] + new_connected_grads)
            # If collected enough data, we continue to the next round
            if epoch_indx > 20:
                trusty_indices = self.target_malicious(gradients, epoch_indx)
                return average_nn_parameters([gradients[indx] for indx in trusty_indices])
        return average_nn_parameters(gradients)

    def target_malicious(self, gradients, epoch_indx):
        truthy_gradient = np.zeros((self.num_classes, len(gradients)), dtype=bool)
        for cls in range(self.num_classes):
            # Slice to get only the rows corresponding the the output node.
            sub_sample = self.past_gradients[cls::self.num_classes]
            clf = KMeans(2)
            scaler = StandardScaler()
            fitter = PCA(n_components=2)
            scaled_param_diff = scaler.fit_transform(sub_sample)
            dim_reduced_gradients = fitter.fit_transform(scaled_param_diff)
            classified = clf.fit_transform(dim_reduced_gradients)

            # If total is roughly 50/50 then unlikely to be poisoned. Else likely to be poisoned
            cluster_split = np.average(classified)
            if 0.4 * epoch_indx * len(gradients) < cluster_split < 0.6 * len(gradients):
                # Roughly 50/50 divided, so we assume valid updates.
                # As such, we don't need to perform KRUM, as the distribution over the two clusters
                # is arbitrary. Hence, we cannot distill much information from the assignment to one of the
                # two clusters.
                truthy_gradient[cls] = True
            else:
                krum_scores = calc_krum_scores(gradients)
                most_likely_good = np.argmax(krum_scores)
                # Get the label assigned to the 'krum' vector, either 1/0
                estimated_cluster = classified[-(len(gradients) - most_likely_good)]
                # Boolean array to indicate which belong to the same cluster.
                truthy_gradient[cls] = classified[-len(gradients):] == estimated_cluster
        # Only select the gradients that we suspect that are unaffected
        # Take row-wise and, as such only a column that has only 'TRUE', will be selected using
        # the argwhere, because True evaluates to True.
        return np.argwhere(truthy_gradient)


def create_antidote(cfg: BareConfig, **kwargs) -> Antidote:
    assert cfg is not None
    if cfg.antidote is None:
        return DummyAntidote(cfg)
    medicine_cabinet = {'dummy': DummyAntidote, 'multikrum': MultiKrumAntidote, 'cluster': ClusterAntidote}

    antidote_class = medicine_cabinet.get(cfg.get_antidote_type(), None)

    if not antidote_class is None:
        antidote = antidote_class(cfg=cfg, **kwargs)
    else:
        raise Exception("Requested antidote is not supported")
    print(f'')
    return antidote
