"""
Module providing an Enum based definition for different loss functions that are compatible with Pytorch.
In addition, it provides functionality to map a definition to an implementation.
"""

from enum import Enum, unique
from typing import Dict, Type

import torch
from torch.nn.modules.loss import _Loss


@unique
class Loss(Enum):
    l1_loss = 'L1Loss'
    mse_loss = 'MSELoss'
    cross_entropy_loss = 'CrossEntropyLoss'
    ctc_loss = 'CTCLoss'
    nll_loss = 'NLLLoss'
    poisson_nll_loss = 'PoissonNLLLoss'
    gaussian_nll_loss = 'GaussianNLLLoss'
    kldiv_loss = 'KLDivLoss'
    bce_loss = 'BCELoss'
    bce_with_logits_loss = 'BCEWithLogitsLoss'
    margin_ranking_loss = 'MarginRankingLoss'
    multi_label_margin_loss = 'MultiLabelMarginLoss'
    huber_loss = 'HuberLoss'
    smooth_l1_loss = 'SmoothL1Loss'
    soft_margin_loss = 'SoftMarginLoss'
    multi_label_soft_margin_loss = 'MultiLabelSoftMarginLoss'
    cosine_embedding_loss = 'CosineEmbeddingLoss'
    multi_margin_loss = 'MultiMarginLoss'
    triplet_margin_loss = 'TripletMarginLoss'
    triplet_margin_with_distance_loss = 'TripletMarginWithDistanceLoss'

