from typing import Type

from fltk.util.config.definitions import ExperimentType
from .task import DistributedArrivalTask, FederatedArrivalTask, ArrivalTask


__job_type_lookup = {
    ExperimentType.DISTRIBUTED: DistributedArrivalTask,
    ExperimentType.FEDERATED: FederatedArrivalTask
}


def get_job_arrival_class(job_type: ExperimentType) -> Type[ArrivalTask]:
    return __job_type_lookup[job_type]
