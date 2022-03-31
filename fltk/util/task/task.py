import abc
from dataclasses import field, dataclass
from typing import OrderedDict, Dict, List, Optional
from uuid import UUID

from fltk.util.task.config.parameter import SystemParameters, HyperParameters


@dataclass
class ArrivalTask(abc.ABC):
    id: UUID = field(compare=False)
    network: str = field(compare=False)
    dataset: str = field(compare=False)

    @abc.abstractmethod
    def named_system_params(self, **kwargs) -> OrderedDict[str, SystemParameters]:
        pass

    @abc.abstractmethod
    def typed_replica_count(self, replica_type):
        pass


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

    """

    type_map: OrderedDict[str, int]
    sys_config_map: Dict[str, SystemParameters]
    param_config_map: Dict[str, HyperParameters]

    def named_system_params(self) -> OrderedDict[str, SystemParameters]:
        """
        Helper function to get named system parameters for types. Default follows the naming convention of KubeFlow,
        where the first operator gets assigned the name 'Master' and subsequent compute units are assigned 'Worker'.
        @return:
        @rtype:
        """
        ret_dict = OrderedDict[str, SystemParameters](
                [(tpe, self.sys_config_map[tpe]) for tpe in self.type_map.keys()])
        return ret_dict

    def typed_replica_count(self, replica_type):
        return self.type_map[replica_type]
