from __future__ import annotations
from enum import unique, Enum
from pathlib import Path
from typing import Type

from fltk.core.distributed import BatchOrchestrator, SimulatedOrchestrator, Orchestrator

from typing import TYPE_CHECKING

from fltk.util.task.generator import ArrivalGenerator

if TYPE_CHECKING:
    from fltk.util.cluster import ClusterManager
    from fltk.util.config import DistributedConfig


@unique
class OrchestratorType(Enum):
    BATCH = 'batch'
    SIMULATED = 'simulated'


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