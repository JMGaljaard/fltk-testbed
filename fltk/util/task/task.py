from dataclasses import field, dataclass
from typing import OrderedDict, Dict
from uuid import UUID

from fltk.util.task.config.parameter import SystemParameters, HyperParameters


@dataclass
class ArrivalTask:
    id: UUID = field(compare=False)
    network: str = field(compare=False)
    dataset: str = field(compare=False)


@dataclass(order=True)
class DistributedArrivalTask(ArrivalTask):
    """
    Object to contain configuration of training task. It describes the following properties;
        * Number of machines
        * System-configuration
        * Network
        * Dataset
        * Hyper-parameters

    The tasks are by default sorted according to priority.
    """
    priority: int
    sys_conf: SystemParameters = field(compare=False)
    param_conf: HyperParameters = field(compare=False)


@dataclass
class FederatedArrivalTask(ArrivalTask):
    """

    """
    type_map: OrderedDict[str, int]
    sys_config_map: Dict[str, SystemParameters]
    param_config_map: Dict[str, HyperParameters]