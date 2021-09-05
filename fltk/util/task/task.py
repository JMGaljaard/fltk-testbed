from dataclasses import field, dataclass
from uuid import UUID

from fltk.util.task.config.parameter import SystemParameters, HyperParameters


@dataclass(order=True)
class ArrivalTask:
    """
    Object to contain configuration of training task. It describes the following properties;
        * Number of machines
        * System-configuration
        * Network
        * Dataset
        * Hyper-parameters
    """
    priority: int
    id: UUID = field(compare=False)
    network: str = field(compare=False)
    dataset: str = field(compare=False)
    sys_conf: SystemParameters = field(compare=False)
    param_conf: HyperParameters = field(compare=False)
