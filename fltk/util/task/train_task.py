from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import config

from fltk.util.config.definitions import ExperimentType
from fltk.util.config.experiment_config import (NetworkConfiguration, SystemParameters, HyperParameters, Priority,
                                                LearningParameters, JobClassParameter)


@dataclass(order=True)
class TrainTask:
    """
    Training description used by the orchestrator to generate tasks. Contains 'transposed' information of the
    configuration file.

    Dataclass is ordered, to allow for ordering of arrived tasks in a PriorityQueue (can be used for scheduling).
    """
    network_configuration: NetworkConfiguration = field(compare=False)
    system_parameters: SystemParameters = field(compare=False)
    hyper_parameters: HyperParameters = field(compare=False)
    learning_parameters: Optional[LearningParameters] = field(compare=False)
    seed: int = field(compare=False)
    identifier: str = field(compare=False)
    replication: Optional[int] = field(compare=False, default=None)                     # Utilized for batch arrivals.
    priority: Optional[int] = None                                                      # Allow for sorting/priority.
    experiment_type: ExperimentType = field(compare=False, metadata=config(field_name="type"), default=None)

    def __init__(self,
                 identity: str,
                 job_parameters: JobClassParameter,
                 priority: Priority = None,
                 replication: int = None,
                 experiment_type: ExperimentType = None,
                 seed: int = None):
        """
        Overridden init method for dataclass, to allow for transposing a JobDescription to a flattened TrainTask, which
        contains the required information to schedule a task.
        @param job_parameters:
        @type job_parameters:
        @param priority:
        @type priority:
        """
        self.identifier = identity
        self.network_configuration = job_parameters.network_configuration
        self.system_parameters = job_parameters.system_parameters
        self.hyper_parameters = job_parameters.hyper_parameters
        if priority:
            self.priority = priority.priority
        self.learning_parameters = job_parameters.learning_parameters
        self.replication = replication
        self.experiment_type = experiment_type
        self.seed = seed
