import logging
import time
import uuid
from queue import PriorityQueue
from typing import List

import kubernetes.config
from kubeflow.pytorchjob import PyTorchJobClient

from fltk.util.cluster.client import construct_job, ClusterManager
from fltk.util.config.base_config import BareConfig
from fltk.util.task.config.parameter import TrainTask
from fltk.util.task.generator.arrival_generator import ArrivalGenerator
from fltk.util.task.task import ArrivalTask


class Orchestrator(object):
    """
    Central component of the Federated Learning System: The Orchestrator

    The Orchestrator is in charge of the following tasks:
    - Running experiments
        - Creating and/or managing tasks
        - Keep track of progress (pending/started/failed/completed)
    - Keep track of timing

    Note that the Orchestrator does not function like a Federator, in the sense that it keeps a central model, performs
    aggregations and keeps track of Clients. For this, the KubeFlow PyTorch-Operator is used to deploy a train task as
    a V1PyTorchJob, which automatically generates the required setup in the cluster. In addition, this allows more Jobs
    to be scheduled, than that there are resources, as such, letting the Kubernetes Scheduler let decide when to run
    which containers where.
    """
    _alive = False
    # Priority queue, requires an orderable object, otherwise a Tuple[int, Any] can be used to insert.
    pending_tasks: "PriorityQueue[ArrivalTask]" = PriorityQueue()
    deployed_tasks: List[ArrivalTask] = []
    completed_tasks: List[str] = []

    def __init__(self, cluster_mgr: ClusterManager, arv_gen: ArrivalGenerator, config: BareConfig):
        self._logger = logging.getLogger('Orchestrator')
        self._logger.debug("Loading in-cluster configuration")
        kubernetes.config.load_incluster_config()

        self._cluster_mgr = cluster_mgr
        self.__arrival_generator = arv_gen
        self._config = config

        # API to interact with the cluster.
        self.__client = PyTorchJobClient()

    def stop(self) -> None:
        """
        Stop the Orchestrator.
        @return:
        @rtype:
        """
        self._logger.info("Received stop signal for the Orchestrator.")
        self._alive = False

    def run(self) -> None:
        """
        Main loop of the Orchestartor.
        :return:
        """
        self._alive = True
        start_time = time.time()
        while self._alive and time.time() - start_time < self._config.get_duration():
            # 1. Check arrivals
            # If new arrivals, store them in arrival list
            while not self.__arrival_generator.arrivals.empty():
                arrival: TrainTask = self.__arrival_generator.arrivals.get()
                unique_identifier = uuid.uuid4()
                task = ArrivalTask(id=unique_identifier,
                                   network=arrival.network_configuration.network,
                                   dataset=arrival.network_configuration.dataset,
                                   sys_conf=arrival.system_parameters,
                                   param_conf=arrival.hyper_parameters)

                self._logger.info(f"Arrival of: {task}")
                self.pending_tasks.put(task)
            while not self.pending_tasks.empty():
                # Do blocking request to priority queue
                curr_task = self.pending_tasks.get()
                self._logger.info(f"Scheduling arrival of Arrival: {curr_task}")
                job_to_start = construct_job(self._config, curr_task)

                self.__client.create(job_to_start, namespace=self._config.cluster_config.namespace)
                self.deployed_tasks.append(curr_task)
            # TODO: Keep track of Jobs that were started, but may have completed....
            # That would conclude the MVP.
            self._logger.debug("Still alive...")
            time.sleep(5)

        logging.info(f'Experiment completed, currently does not support waiting.')
