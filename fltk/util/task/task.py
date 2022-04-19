import abc
from dataclasses import field, dataclass
from typing import OrderedDict, List, Optional
from uuid import UUID

from fltk.util.task.config import SystemParameters, HyperParameters
from fltk.util.task.config.parameter import SystemResources, LearningParameters, \
    OptimizerConfig, SamplerConfiguration


@dataclass
class ArrivalTask(abc.ABC):
    """
    DataClass representation of an ArrivalTask, representing all the information needed to spawn a new learning task.
    """
    id: UUID = field(compare=False) # pylint: disable=invalid-name
    network: str = field(compare=False)
    dataset: str = field(compare=False)

    @abc.abstractmethod
    def named_system_params(self, *args, **kwargs) -> OrderedDict[str, SystemResources]:
        """
        Helper function to get system parameters by name.
        @param kwargs: kwargs for arguments.
        @type kwargs: dict
        @return: Dictionary corresponding to System resources per learner type.
        @rtype: OrderedDict[str, SystemResources]
        """

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

    sys_conf: SystemParameters = field(compare=False)
    param_conf: HyperParameters = field(compare=False)
    priority: int

    def named_system_params(self, types: Optional[List[str]] = None) -> OrderedDict[str, SystemParameters]:
        """
        Helper function to get named system parameters for types. Default follows the naming convention of KubeFlow,
        where the first operator gets assigned the name 'Master' and subsequent compute units are assigned 'Worker'.
        @param types: List of types that need to be added to the dpeloyment, e.g. 'Worker' and 'Master'. Note
        that ordering matters, and first element in the list is assumed to be assigned IDX=0.
        @type types: Optional[List[str]]
        @return:
        @rtype:
        """
        if types is None:
            types = ['Master', 'Worker']
        ret_dict = OrderedDict[str, SystemParameters](
                [(tpe, self.sys_conf) for tpe in types])
        return ret_dict

    def typed_replica_count(self, replica_type):
        parallelism_dict = {'Master': 1,
                            'Worker': self.sys_conf.data_parallelism - 1}
        return parallelism_dict[replica_type]


@dataclass
class FederatedArrivalTask(ArrivalTask):
    """
    Task describing configuration objects for running FederatedLearning experiments on K8s.
    """

    type_map: OrderedDict[str, int]
    hyper_parameters: HyperParameters
    system_parameters: SystemParameters
    learning_parameters: LearningParameters

    def named_system_params(self) -> OrderedDict[str, SystemResources]:
        """
        Helper function to get named system parameters for types. Default follows the naming convention of KubeFlow,
        where the first operator gets assigned the name 'Master' and subsequent compute units are assigned 'Worker'.
        @return:
        @rtype:
        """
        ret_dict = OrderedDict[str, SystemResources](
                [(tpe, self.system_parameters.configurations[tpe]) for tpe in self.type_map.keys()])
        return ret_dict

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
