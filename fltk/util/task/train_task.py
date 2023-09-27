from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import config

import fltk.util.config.definitions as defs
import fltk.util.config.experiment_config as exp_conf


@dataclass(order=True)
class TrainTask:
    """
    Training description used by the orchestrator to generate tasks. Contains 'transposed' information of the
    configuration file.

    Dataclass is ordered, to allow for ordering of arrived tasks in a PriorityQueue (can be used for scheduling).
    """
    network_configuration: exp_conf.NetworkConfiguration = field(compare=False)
    system_parameters: exp_conf.SystemParameters = field(compare=False)
    hyper_parameters: exp_conf.HyperParameters = field(compare=False)
    learning_parameters: Optional[exp_conf.LearningParameters] = field(compare=False)
    seed: int = field(compare=False)
    identifier: str = field(compare=False)
    replication: Optional[int] = field(compare=False, default=None)                     # Utilized for batch arrivals.
    priority: Optional[int] = None                                                      # Allow for sorting/priority.
    experiment_type: defs.ExperimentType = field(compare=False, metadata=config(field_name="type"), default=None)

    def __init__(self,
                 identity: str,
                 job_parameters: exp_conf.JobClassParameter,
                 priority: exp_conf.Priority = None,
                 replication: int = None,
                 experiment_type: defs.ExperimentType = None,
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
