import abc
import collections
import random
import uuid
from dataclasses import field, dataclass
# noinspection PyUnresolvedReferences
from typing import Optional, T, List, OrderedDict
from uuid import UUID

from frozendict import FrozenOrderedDict

from fltk.datasets.dataset import Dataset
from fltk.util.config.definitions import Nets
from fltk.util.config.experiment_config import (OptimizerConfig, HyperParameters, SystemResources, SystemParameters,
                                                SamplerConfiguration, LearningParameters)
from fltk.util.task.generator.arrival_generator import Arrival

MASTER_REPLICATION: int = 1  # Static master replication value, dictated by PytorchTrainingJobs


@dataclass(frozen=True)
class _ArrivalTask(abc.ABC):
    """
    Private parent of ArrivalTasks, used internally for allowing to track
    """
    id: UUID = field(compare=False)  # pylint: disable=invalid-name

@dataclass(frozen=True)
class HistoricalArrivalTask(abc.ABC):
    """
    Dataclass to contain historical tasks, allowing for keeping track of tasks deployed in older deployments.
    """
    pass


@dataclass(frozen=True)
class ArrivalTask(_ArrivalTask):
    """
    DataClass representation of an ArrivalTask, representing all the information needed to spawn a new learning task.
    Allows for sorting by priority (integer) in case priority queues are needed.
    """
    network: Nets = field(compare=False)
    dataset: Dataset = field(compare=False)
    loss_function: str = field(compare=False)
    seed: int = field(compare=False)
    replication: int = field(compare=False)
    type_map: "Optional[FrozenOrderedDict[str, int]]" = field(compare=False)
    system_parameters: SystemParameters = field(compare=False)
    hyper_parameters: HyperParameters = field(compare=False)
    learning_parameters: LearningParameters = field(compare=False)
    priority: Optional[int] = None

    @staticmethod
    @abc.abstractmethod
    def build(arrival: Arrival, u_id: uuid.UUID, replication: int) -> T:
        """
        Function to build a specific type of ArrivalTask.
        @param arrival: Arrival object with configuration for an experiment (or Arrival).
        @type arrival: Arrival
        @param u_id: Unique identifier for an experiment to prevent collision in experiment names.
        @type u_id: UUID
        @param replication: Replication id (integer).
        @type replication: int
        @return: Type of (child) ArrivalTask with corresponding experiment configuration.
        @rtype: T
        """

    def named_system_params(self) -> OrderedDict[str, SystemResources]:
        """
        Helper function to get system parameters by name.
        @return: Dictionary corresponding to System resources per learner type.
        @rtype: OrderedDict[str, SystemResources]
        """
        sys_conf = self.system_parameters
        ret_dict = collections.OrderedDict([(tpe, sys_conf.get(tpe)) for tpe in self.type_map.keys()])
        return ret_dict

    def typed_replica_count(self, replica_type: str) -> int:
        """
        Helper function to get replica count per type of learner.
        @param replica_type: String representation of learner type.
        @type replica_type: str
        @return: Number of workers to spawn of a specific type.
        @rtype: int
        """
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
        Helper function to acquire federated learning parameters as-if the configuration is a flat configuration
        file.
        @param parameter:
        @type parameter:
        @return:
        @rtype:
        """
        return getattr(self.learning_parameters, parameter)

    def get_sampler_param(self, tpe: str, parameter: str):
        """
        Helper function to acquire federated data sampler parameters as-though the configuration is a flat configuration
        file.
        @param tpe: Type indication for a learner, future version with heterogenous deployment would require this.
        @type tpe: str
        @param parameter: Which parameter to get from the configuration object.
        @type parameter: str
        @return:
        @rtype:
        """
        return getattr(self.learning_parameters.data_sampler, parameter)

    def get_sampler_args(self, tpe: str) -> List[str]:
        """
        Helper function to acquire federated data sampler arguments as-though the configuration is a flat configuration
        file.
        @param tpe: Type indication for a learner, future version with heterogenous deployment would require this.
        @type tpe: str
        @return: Arguments for the sampler function.
        @rtype: List[str]
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
        }
        if optimizer_conf.momentum:
            kwargs['momentum'] = optimizer_conf.momentum
        if optimizer_conf.betas:
            kwargs['betas'] = optimizer_conf.betas
        return kwargs

    def get_scheduler_param(self, tpe, parameter):
        """
        Helper function to acquire learning scheduler parameters as-though the configuration is a flat configuration
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
        @param parameter:
        @type parameter:
        @return:
        @rtype:
        """
        return getattr(self, parameter)


@dataclass(order=True, frozen=True)
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

    @staticmethod
    def build(arrival: Arrival, u_id: uuid.UUID, replication: int) -> T:
        """
        Construct a DistributedArrivalTask from an Arrival object.

        Create DistributedArrivalTask from Arrival. This will create a task with random seed in
        [0, sys.maxsize] for randomness. It assumes that the random seed/performance of the model itself is not
        important on it own, but the reproducibility is. Word size is assumed to be equal to data_parallism (leader
        inclusive).
        Args:
            arrival (Arrival): Arrival to create FederatedArrivalTask from.
            u_id (UUID): Unique experiment ID to related back to the FedArv.task.
            replication (int): Replication index (for book-keeping).

        Returns: FederatedArrivalTask with pre-set seed, and parallelism.

        """
        task = DistributedArrivalTask(
                id=u_id,
                network=arrival.get_network(),
                priority=arrival.get_priority(),
                dataset=arrival.get_dataset(),
                loss_function=arrival.task.network_configuration.loss_function,
                seed=random.randint(0, 2**32 - 2),
                replication=replication,
                type_map=FrozenOrderedDict({
                    'Master': MASTER_REPLICATION,
                    'Worker': arrival.task.system_parameters.data_parallelism - MASTER_REPLICATION
                }),
                system_parameters=arrival.get_system_config(),
                hyper_parameters=arrival.get_parameter_config(),
                learning_parameters=arrival.get_learning_config())
        return task


@dataclass(order=True, frozen=True)
class FederatedArrivalTask(ArrivalTask):
    """
    Task describing configuration objects for running FederatedLearning experiments on K8s.
    """

    @staticmethod
    def build(arrival: Arrival, u_id: uuid.UUID, replication: int) -> "FederatedArrivalTask":
        """
        Create FederatedArrivalTask from Arrival, with pre-defined seed of Task (assuming replicable experiments),
        replication, and number of workers equal to data parallelism.
        Args:
            arrival (Arrival): Arrival to create FederatedArrivalTask from.
            u_id (UUID): Unique experiment ID to related back to the FedArv.task.
            replication (int): Replication index (for book-keeping).

        Returns: FederatedArrivalTask with pre-set seed, and parallelism.

        """
        task = FederatedArrivalTask(
                id=u_id,
                network=arrival.get_network(),
                dataset=arrival.get_dataset(),
                loss_function=arrival.task.network_configuration.loss_function,
                seed=arrival.task.seed,
                replication=replication,
                type_map=FrozenOrderedDict({
                    'Master': MASTER_REPLICATION,
                    'Worker': arrival.task.system_parameters.data_parallelism
                }),
                system_parameters=arrival.get_system_config(),
                hyper_parameters=arrival.get_parameter_config(),
                priority=arrival.get_priority(),
                learning_parameters=arrival.get_learning_config())
        return task
