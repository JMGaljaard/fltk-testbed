import abc
from dataclasses import dataclass

from fltk.util.cluster.task.config.parameter import SystemParameters, HyperParameters


@dataclass
class Task(abc):
    """
    Object to contain configuration of training task. It describes the following properties;
        * Number of machines
        * System-configuration
        * Network
        * Dataset
        * Hyper-parameters
    """
    network: str
    dataset: str
    system_config: SystemParameters
    parameter_config: HyperParameters
