"""
Module providing an Enum based definition for different loss functions that are compatible with Pytorch.
In addition, it provides functionality to map a definition to an implementation.
"""
import logging
from enum import Enum, unique
from typing import Dict, Type, Union

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


def get_loss_function(request: Union[str, Loss]) -> Type[_Loss]:
    """
    Mapper function to map a request to a loss function. As fallback behavior the request is evaluated
    using the Python interpreter to try to load an existing implementation dynamically.
    """
    __lookup_dict: Dict[Loss, Type[_Loss]] = {
        Loss.l1_loss: torch.nn.L1Loss,
        Loss.mse_loss: torch.nn.MSELoss,
        Loss.cross_entropy_loss: torch.nn.CrossEntropyLoss,
        Loss.ctc_loss: torch.nn.CTCLoss,
        Loss.nll_loss: torch.nn.NLLLoss,
        Loss.poisson_nll_loss: torch.nn.PoissonNLLLoss,
        Loss.gaussian_nll_loss: torch.nn.GaussianNLLLoss,
        Loss.kldiv_loss: torch.nn.KLDivLoss,
        Loss.bce_loss: torch.nn.BCELoss,
        Loss.bce_with_logits_loss: torch.nn.BCEWithLogitsLoss,
        Loss.margin_ranking_loss: torch.nn.MarginRankingLoss,
        Loss.multi_label_margin_loss: torch.nn.MultiLabelMarginLoss,
        Loss.huber_loss: torch.nn.HuberLoss,
        Loss.smooth_l1_loss: torch.nn.SmoothL1Loss,
        Loss.soft_margin_loss: torch.nn.SoftMarginLoss,
        Loss.multi_label_soft_margin_loss: torch.nn.MultiLabelSoftMarginLoss,
        Loss.cosine_embedding_loss: torch.nn.CosineEmbeddingLoss,
        Loss.multi_margin_loss: torch.nn.MultiMarginLoss,
        Loss.triplet_margin_loss: torch.nn.TripletMarginLoss,
        Loss.triplet_margin_with_distance_loss: torch.nn.TripletMarginWithDistanceLoss}

    if isinstance(request, Loss):
        return __lookup_dict.get(request)
    else:
        logging.info(f"Loading non-predefined loss function {request}")
        return eval(request)
