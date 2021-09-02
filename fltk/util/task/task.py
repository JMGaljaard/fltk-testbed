import abc
from dataclasses import dataclass
from uuid import UUID

from fltk.util.task.config.parameter import SystemParameters, HyperParameters


@dataclass(order=True)
class ArrivalTask(abc):
    """
    Object to contain configuration of training task. It describes the following properties;
        * Number of machines
        * System-configuration
        * Network
        * Dataset
        * Hyper-parameters
    """
    id: UUID
    priority: int = field(st)
    network: str
    dataset: str
    sys_conf: SystemParameters
    param_conf: HyperParameters
