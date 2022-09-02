from enum import unique, Enum


@unique
class DataSampler(Enum):
    uniform = "uniform"
    q_sampler = "q sampler"
    limit_labels = "limit labels"
    dirichlet = "dirichlet"
    limit_labels_q = "limit labels q"
    emd_sampler = 'emd sampler'
    limit_labels_flex = "limit labels flex"
    n_labels = "n labels"
