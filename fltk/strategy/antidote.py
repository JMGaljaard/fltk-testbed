import logging
from abc import abstractmethod, ABC

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fltk.strategy.util.antidote import calc_krum_scores
from fltk.util.base_config import BareConfig
from fltk.util.fed_avg import average_nn_parameters


class Antidote(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def process_gradients(self, gradients, **kwargs):
        pass

    def save_data_and_reset(self, ratio, iteration=0):
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

    def ema(self, s_t_prev, value, t, bias_correction=True):
        """
        Exponential Moving Average, with bias correction by default.
        @param s_t_prev:
        @type s_t_prev:
        @param value:
        @type value:
        @param t:
        @type t:
        @param bias_correction:
        @type bias_correction:
        @return:
        @rtype:
        """
        s_t = self.rho_ * s_t_prev + (1 - self.rho_) * value
        s_t_hat = None
        if bias_correction:
            s_t_hat = s_t / (1.0 - self.rho_ ** (t + 1))
        return s_t_hat if bias_correction else s_t

    def __init__(self, cfg: BareConfig, **kwargs):
        super(ClusterAntidote, self).__init__()
        # Needed for KRUM/Multi-KRUM for testing purposes.
        self.f = cfg.get_antidote_f_value()
        self.k = cfg.get_antidote_k_value()
        self.past_gradients = np.array([])
        self.logger = logging.getLogger()
        # Rho for this class poisoned
        self.rho_ = 0.75
        self.max_epoch = 130
        self.num_classes = 10
        self.offset = 20

        # Logging information for testing only
        self.krum_proposed = list()
        self.selected_updates = list()
        self.cheating_client = dict()
        self.class_targeted = np.zeros((10, cfg.epochs + 2))


    def process_gradients(self, gradients, **kwargs):
        """
        Function which returns the average of the k gradient with the lowest score.
        """
        epoch_indx = kwargs['epoch']
        clients_round = kwargs['clients']
        model = kwargs['model']
        cur_last = list(model.values())[-2].numpy()
        # First 10 epochs we effectively don't do much
        if epoch_indx > 10:
            # Store gradients
            new_connected_grads = [list(gradient.values())[-2].numpy() - cur_last for gradient in gradients]
            # The array may be empty
            if self.past_gradients.size:
                self.past_gradients = np.vstack([self.past_gradients] + new_connected_grads)
            else:
                self.past_gradients = np.vstack(new_connected_grads)
            # If collected enough data, we continue to the next round
            if epoch_indx > self.offset:
                trusty_indices = self.target_malicious(gradients, epoch_indx, clients_round)
                return average_nn_parameters([gradients[indx] for indx in trusty_indices])
        return average_nn_parameters(gradients)

    def target_malicious(self, gradients, epoch_indx, clients_round):
        truthy_gradient = np.zeros((self.num_classes, len(gradients)), dtype=bool)
        krum_scores = calc_krum_scores(gradients, int(np.ceil(1/3 * len(gradients))))
        # Take the index of smallest client.
        most_likely_good_index = np.argmin(krum_scores)
        most_likely_good = clients_round[np.argmin(krum_scores)].name
        for cls in range(self.num_classes):
            # Slice to get only the rows corresponding the the output node.
            sub_sample = self.past_gradients[cls::self.num_classes]
            classified = self.unsupervised_classification(sub_sample)

            # If total is roughly 50/50 then unlikely to be poisoned. Else likely to be poisoned
            cluster_split = np.average(classified)
            self.logger.info(f"Cluster division: {cluster_split}")
            if 1 / 3 < cluster_split < 2 / 3:
                # Roughly 50/50 divided, so we assume valid updates.
                # As such, we don't need to perform KRUM, as the distribution over the two clusters
                # is arbitrary. Hence, we cannot distill much information from the assignment to one of the
                # two clusters.

                # Use 0 as estimate, because we suspect that the class is _not_ targeted.
                self.class_targeted[cls, epoch_indx + 1] = self.ema(self.class_targeted[cls, epoch_indx], 0,
                                                                    epoch_indx - self.offset)
                # Broadcast True ot the truthy_gradient matrix
                truthy_gradient[cls] = True
            else:
                # Use 0 as estimate, because we suspect that the class _is_ targeted.
                self.class_targeted[cls, epoch_indx + 1] = self.ema(self.class_targeted[cls, epoch_indx], 1,
                                                                    epoch_indx - self.offset)
                biggest_cluster = 0 if cluster_split < 0.5 else 1
                # Boolean array to indicate which belong to the same cluster.
                truthy_gradient[cls] = (classified[-len(gradients):] == classified[-len(gradients) + most_likely_good_index])
                self.logger.info(f"Biggest: {biggest_cluster}, KRUM cluster: {classified[-len(gradients) + most_likely_good_index]}")
        # Only select the gradients that we suspect that are unaffected
        # Take row-wise and, as such only a column that has only 'TRUE', will be selected using
        # the argwhere, because True evaluates to True.
        truthy_gradient_reduced = np.prod(truthy_gradient, axis=0)
        for indx, truthy in enumerate(truthy_gradient_reduced):
            suspect_cheating = clients_round[indx].name
            previous_array = self.cheating_client.get(suspect_cheating, [0])
            previous_array.append(self.ema(previous_array[-1], 1 if  truthy != 1 else 0, len(previous_array)))
            self.cheating_client[suspect_cheating] = previous_array
        selected_grads = np.argwhere(truthy_gradient_reduced == 1).reshape(-1)

        self.krum_proposed.append(most_likely_good)
        # Keep track of the gradients updates that we selected.
        self.selected_updates.append([clients_round[indx].name for indx in selected_grads])
        self.logger.info(f"KRUM: {most_likely_good}, clustered: {selected_grads}")
        self.logger.info(f"Suspicion classes: {self.class_targeted[:, epoch_indx + 1]}")
        self.logger.info(f"Suspicion clients: {[(k, v[-1]) for k, v in self.cheating_client.items()]}")
        return selected_grads

    def unsupervised_classification(self, sub_sample):
        clf = KMeans(2)
        scaler = StandardScaler()
        fitter = PCA(n_components=2)
        scaled_param_diff = scaler.fit_transform(sub_sample)
        dim_reduced_gradients = fitter.fit_transform(scaled_param_diff)
        classified = clf.fit_predict(dim_reduced_gradients)
        return classified

    def save_data_and_reset(self, ratio, iteration=0):
        data = {
            "selected_updates": self.selected_updates,
            "cheating_clients": self.cheating_client,
            "targeted_classes": self.class_targeted,
            "krum_proposal": self.krum_proposed,
            "gradients": self.past_gradients
        }
        np.save(f'./output/cluster_antidote_{ratio}_{iteration}.npy', data)
        self.selected_updates = list()
        self.cheating_client = dict()
        self.class_targeted = np.zeros_like(self.class_targeted)
        self.krum_proposed = list()
        self.past_gradients = np.array([])


def create_antidote(cfg: BareConfig, **kwargs) -> Antidote:
    assert cfg is not None
    if cfg.antidote is None:
        return DummyAntidote(cfg)
    medicine_cabinet = {'dummy': DummyAntidote, 'multikrum': MultiKrumAntidote, 'clustering': ClusterAntidote}

    antidote_class = medicine_cabinet.get(cfg.get_antidote_type(), None)

    if not antidote_class is None:
        antidote = antidote_class(cfg=cfg, **kwargs)
    else:
        raise Exception("Requested antidote is not supported")
    print(f'')
    return antidote
