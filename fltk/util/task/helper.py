from typing import Type

from fltk.util.config.definitions import ExperimentType
from fltk.util.task import DistributedArrivalTask, FederatedArrivalTask, ArrivalTask

__job_type_lookup = {
    ExperimentType.DISTRIBUTED: DistributedArrivalTask,
    ExperimentType.FEDERATED: FederatedArrivalTask
}


def get_job_arrival_class(job_type: ExperimentType) -> Type[ArrivalTask]:
    """
    Helper function to get the Arrival Task type based on an ExperimentType definition.
    @param job_type: ExperimentType definition (Enum). See also definitions.
    @type job_type: ExperimentType
    @return: Class reference for experiment type.
    @rtype: Type[ArrivalTask]
    """
    return __job_type_lookup[job_type]
