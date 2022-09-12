from __future__ import annotations

from enum import unique, Enum
from pathlib import Path

from fltk.core.distributed import Orchestrator, BatchOrchestrator, SimulatedOrchestrator
from fltk.util.task.generator import SequentialArrivalGenerator, SimulatedArrivalGenerator

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fltk.util.config import DistributedConfig
    from fltk.util.cluster import ClusterManager
    from fltk.util.task.generator import ArrivalGenerator


@unique
class OrchestratorType(Enum):
    """ """
    BATCH = 'batch'
    SIMULATED = 'simulated'


def get_orchestrator(config: DistributedConfig, cluster_manager: ClusterManager, arrival_generator: ArrivalGenerator) -> Orchestrator:
    """Retrieve Orchestrator type given a Distributed (experiment) configuration. This allows for defining the
    type of experiment (Batch or Simulated arrivals) once, and letting the Orchestrator implementation
    make sure that the tasks are scheduled correctly.

    Args:
      config (DistributedConfig): Distributed (cluster) configuration object for experiments.
      cluster_manager (ClusterManager):
      arrival_generator (ArrivalGenerator):

    Returns:
      Type[Orchestrator]: Type of Orchestrator as requested by configuration object.

    """
    __lookup = {
        OrchestratorType.BATCH: BatchOrchestrator,
        OrchestratorType.SIMULATED: SimulatedOrchestrator
    }

    orchestrator_type = __lookup.get(config.cluster_config.orchestrator.orchestrator_type, None)
    return orchestrator_type(cluster_manager, arrival_generator, config)


def get_arrival_generator(config: DistributedConfig, experiment: str) -> ArrivalGenerator:
    """Retrieval function to create generator functions

    Args:
      config (DistributedConfig): Distributed (cluster) configuration with general configuration.
      experiment (str): Experiment name.

    Returns:
      ArrivalGenerator: ArrivalGenerator object initialized with the experiment Path.

    """
    __lookup = {
        OrchestratorType.BATCH: SequentialArrivalGenerator,
        OrchestratorType.SIMULATED: SimulatedArrivalGenerator
    }

    return __lookup.get(config.cluster_config.orchestrator.orchestrator_type, None)(Path(experiment))
