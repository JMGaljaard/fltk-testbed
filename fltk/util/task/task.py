import abc
import collections
import uuid
from dataclasses import field, dataclass
from typing import OrderedDict, List, Optional
from uuid import UUID

from fltk.datasets.dataset import Dataset
from fltk.util.config.definitions import Nets
from fltk.util.task.config import SystemParameters, HyperParameters
from fltk.util.task.config.parameter import SystemResources, LearningParameters, \
    OptimizerConfig, SamplerConfiguration
from fltk.util.task.generator.arrival_generator import Arrival

MASTER_REPLICATION: int = 1  # Static master replication value, dictated by PytorchTrainingJobs


@dataclass
class ArrivalTask(abc.ABC):
    """
    DataClass representation of an ArrivalTask, representing all the information needed to spawn a new learning task.
    """
    id: UUID = field(compare=False)  # pylint: disable=invalid-name
    network: Nets = field(compare=False)
    dataset: Dataset = field(compare=False)
    seed: int = field(compare=False)
    replication: int = field(compare=False)
    type_map: Optional[OrderedDict[str, int]]
    system_parameters: SystemParameters = field(compare=False)
    hyper_parameters: HyperParameters = field(compare=False)
    learning_parameters: LearningParameters = field(compare=False)
    priority: Optional[int] = None

    def named_system_params(self) -> OrderedDict[str, SystemResources]:
        """
        Helper function to get system parameters by name.
        @param kwargs: kwargs for arguments.
        @type kwargs: dict
        @return: Dictionary corresponding to System resources per learner type.
        @rtype: OrderedDict[str, SystemResources]
        """
        sys_conf = self.system_parameters
        ret_dict = collections.OrderedDict(
                [(tpe, sys_conf.get(tpe)) for tpe in self.type_map.keys()])
        return ret_dict

    @abc.abstractmethod
    def typed_replica_count(self, replica_type: str) -> int:
        """
        Helper function to get replica cout per type of learner.
        @param replica_type: String representation of learner type.
        @type replica_type: str
        @return: Number of workers to spawn of a specific type.
        @rtype: int
        """


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

    def __init__(self, arrival: Arrival, u_id: uuid.UUID, repl: int):
        super(DistributedArrivalTask, self).__init__(
                id=u_id,
                network=arrival.get_network(),
                dataset=arrival.get_dataset(),
                seed=arrival.get_experiment_config().random_seed[repl],
                replication=repl,
                type_map={
                    'Master': MASTER_REPLICATION,
                    'Worker': arrival.task.system_parameters.data_parallelism - MASTER_REPLICATION
                },
                system_parameters=arrival.get_system_config(),
                hyper_parameters=arrival.get_parameter_config(),
                learning_parameters=arrival.get_learning_config())

    def typed_replica_count(self, replica_type):
        parallelism_dict = {'Master': MASTER_REPLICATION,
                            'Worker': self.system_parameters.data_parallelism - MASTER_REPLICATION}
        return parallelism_dict[replica_type]


@dataclass(order=True)
class FederatedArrivalTask(ArrivalTask):
    """
    Task describing configuration objects for running FederatedLearning experiments on K8s.
    """
    def __init__(self, arrival: Arrival, u_id: uuid.UUID, repl: int):
        super(FederatedArrivalTask, self).__init__(
                id=u_id,
                network=arrival.get_network(),
                dataset=arrival.get_dataset(),
                seed=arrival.get_experiment_config().random_seed[repl],
                replication=repl,
                type_map=arrival.get_experiment_config().worker_replication,
                system_parameters=arrival.get_system_config(),
                hyper_parameters=arrival.get_parameter_config(),
                priority=arrival.get_priority(),
                learning_parameters=arrival.get_learning_config())


    def typed_replica_count(self, replica_type):
        return self.type_map[replica_type]

    def get_hyper_param(self, tpe, parameter):
        """
        Helper function to acquire hyperparameters as-though the configuration is a flat configuration file.
        @param tpe:
        @type tpe:
        @param parameter:
        @type parameter:
        @return:
        @rtype:
        """
        return getattr(self.hyper_parameters.configurations[tpe], parameter)

    def get_learn_param(self, parameter):
        """
        Helper function to acquire federated learning parameters as-though the configuration is a flat configuration
        file.
        @param tpe:
        @type tpe:
        @param parameter:
        @type parameter:
        @return:
        @rtype:
        """
        return getattr(self.learning_parameters, parameter)

    def get_sampler_param(self, tpe, parameter):
        """
        Helper function to acquire federated datasampler parameters as-though the configuration is a flat configuration
        file.
        @param tpe:
        @type tpe:
        @param parameter:
        @type parameter:
        @return:
        @rtype:
        """
        return getattr(self.learning_parameters.data_sampler, parameter)

    def get_sampler_args(self, tpe: str):
        """
        Helper function to acquire federated datasampler arguments as-though the configuration is a flat configuration
        file.
        @param tpe:
        @type tpe:
        @param parameter:
        @type parameter:
        @return:
        @rtype:
        """
        sampler_conf: SamplerConfiguration = self.learning_parameters.data_sampler
        args = [sampler_conf.q_value, sampler_conf.seed]
        return args

    def get_optimizer_param(self, tpe, parameter):
        """
        Helper function to acquire optimizer parameters as-though the configuration is a flat configuration file.
        @param tpe: Type of learner to set the kwargs values for.
        @type tpe: str
        @param parameter: Which parameter to retrieve.
        @type parameter: str
        @return: Parameter corresponding to the requested field of the configuration.
        @rtype: Any
        """
        return getattr(self.hyper_parameters.configurations[tpe].optimizer_config, parameter)

    def get_optimizer_args(self, tpe: str):
        """
        Helper function to acquire optimizer arguments as-though the configuration is a flat configuration file.
        @note: This function requires a semantically correct configuration file to be provided, as otherwise
        arguments can be missing. For current version `lr` and `momentum` must be set in accordance to the type of
        learner.
        @param tpe: Type of learner to set the kwargs values for.
        @type tpe: str
        @return: Kwarg dict populated with optimizer configuration.
        @rtype: Dict[str, Any]
        """
        optimizer_conf: OptimizerConfig = self.hyper_parameters.configurations[tpe].optimizer_config
        kwargs = {
            'lr': optimizer_conf.lr,
            'momentum': optimizer_conf.momentum
        }
        return kwargs

    def get_scheduler_param(self, tpe, parameter):
        """
        Helper function to acquire learnign scheduler parameters as-though the configuration is a flat configuration
        file.
        @param tpe:
        @type tpe:
        @param parameter:
        @type parameter:
        @return:
        @rtype:
        """
        return getattr(self.hyper_parameters.configurations[tpe].scheduler_config, parameter)

    def get_net_param(self, parameter):
        """
        Helper function to acquire network parameters as-though the configuration is a flat configuration file.
        @param tpe:
        @type tpe:
        @param parameter:
        @type parameter:
        @return:
        @rtype:
        """
        return getattr(self, parameter)
