"""
Module for declaring types and definitions, including helper functions that allow to retrieve
object (types) from a definition.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union, Type, Dict

import torch
from torch.nn.modules.loss import _Loss

from .data_sampler import DataSampler
from .optim import Optimizations
from .aggregate import Aggregations
from .dataset import Dataset
from .logging import LogLevel
from .net import Nets
from .optim import Optimizations
from .experiment_type import ExperimentType
from .loss import Loss
from fltk.util.config.definitions.orchestrator import OrchestratorType


if TYPE_CHECKING:
    from fltk.core.distributed import Orchestrator, BatchOrchestrator, SimulatedOrchestrator
    from fltk.util.config import DistributedConfig
    from fltk.util.cluster import ClusterManager
    from fltk.util.task.generator import ArrivalGenerator


def get_orchestrator(config: DistributedConfig, cluster_manager: ClusterManager, arrival_generator: ArrivalGenerator) -> Orchestrator:
    """
    Retrieve Orchestrator type given a Distributed (experiment) configuration. This allows for defining the
    type of experiment (Batch or Simulated arrivals) once, and letting the Orchestrator implementation
    make sure that the tasks are scheduled correctly.
    @param config: Distributed (cluster) configuration object for experiments.
    @type config: DistributedConfig
    @return: Type of Orchestrator as requested by configuration object.
    @rtype: Type[Orchestrator]
    """
    __lookup = {
        OrchestratorType.BATCH: BatchOrchestrator,
        OrchestratorType.SIMULATED: SimulatedOrchestrator
    }

    orchestrator_type = __lookup.get(config.cluster_config.orchestrator.orchestrator_type, None)
    return orchestrator_type(cluster_manager, arrival_generator, config)


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
