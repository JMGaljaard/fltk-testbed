import collections
import logging
import time
import uuid
from queue import PriorityQueue
from typing import List, OrderedDict, Dict

from jinja2 import Environment, FileSystemLoader
from kubeflow.pytorchjob import PyTorchJobClient
from kubeflow.pytorchjob.constants.constants import PYTORCHJOB_GROUP, PYTORCHJOB_VERSION, PYTORCHJOB_PLURAL
from kubernetes import client
from kubernetes.client import V1ConfigMap, V1ObjectMeta

from fltk.core.distributed.dist_node import DistNode
from fltk.util.cluster.client import construct_job, ClusterManager
from fltk.util.config import DistributedConfig
from fltk.util.task.generator.arrival_generator import ArrivalGenerator, Arrival
from fltk.util.task.task import DistributedArrivalTask, FederatedArrivalTask, ArrivalTask

EXPERIMENT_DIR = 'experiments'
__ENV = Environment(loader=FileSystemLoader(EXPERIMENT_DIR))


def _prepare_experiment_maps(task: FederatedArrivalTask, u_id, replication: int = 1) -> \
        (OrderedDict[str, V1ConfigMap], OrderedDict[str, str]):
    template = __ENV.get_template('node.jinja.yaml')
    type_dict = collections.OrderedDict()
    name_dict = collections.OrderedDict()
    for tpe in task.type_map.keys():
        name = str(f'{tpe}-{u_id}-{replication}').lower()
        meta = V1ObjectMeta(name=name,
                            labels={'app.kubernetes.io/name': f"fltk.node.config.{tpe}"})
        # TODO: Replication / seed information
        filled_template = template.render(task=task, tpe=tpe, replication=replication)
        type_dict[tpe] = V1ConfigMap(data={'node.config.yaml': filled_template}, metadata=meta)
        name_dict[tpe] = name
    return type_dict, name_dict


class Orchestrator(DistNode):
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
    deployed_tasks: List[DistributedArrivalTask] = []
    completed_tasks: List[str] = []

    def __init__(self, cluster_mgr: ClusterManager, arv_gen: ArrivalGenerator, config: DistributedConfig):
        self.__logger = logging.getLogger('Orchestrator')
        self.__logger.debug("Loading in-cluster configuration")
        self.__cluster_mgr = cluster_mgr
        self.__arrival_generator = arv_gen
        self._config = config

        # API to interact with the cluster.
        self._client = PyTorchJobClient()
        self._v1 = client.CoreV1Api()

    def stop(self) -> None:
        """
        Stop the Orchestrator.
        @return:
        @rtype:
        """
        self.__logger.info("Received stop signal for the Orchestrator.")
        self._alive = False

    def run(self, clear: bool = True) -> None:
        """
        Main loop of the Orchestrator.
        @param clear: Boolean indicating whether a previous deployment needs to be cleaned up (i.e. lingering jobs that
        were deployed by the previous run).

        @type clear: bool
        @return: None
        @rtype: None
        """
        self._alive = True
        start_time = time.time()
        if clear:
            self.__clear_jobs()
        while self._alive and time.time() - start_time < self._config.get_duration():
            # 1. Check arrivals
            # If new arrivals, store them in arrival list
            while not self.__arrival_generator.arrivals.empty():
                arrival: Arrival = self.__arrival_generator.arrivals.get()
                unique_identifier: uuid.UUID = uuid.uuid4()
                task = DistributedArrivalTask(priority=arrival.get_priority(),
                                              id=unique_identifier,
                                              network=arrival.get_network(),
                                              dataset=arrival.get_dataset(),
                                              sys_conf=arrival.get_system_config(),
                                              param_conf=arrival.get_parameter_config())

                self.__logger.debug(f"Arrival of: {task}")
                self.pending_tasks.put(task)

            while not self.pending_tasks.empty():
                # Do blocking request to priority queue
                curr_task = self.pending_tasks.get()
                self.__logger.info(f"Scheduling arrival of Arrival: {curr_task.id}")
                job_to_start = construct_job(self._config, curr_task)

                # Hack to overcome limitation of KubeFlow version (Made for older version of Kubernetes)
                self.__logger.info(f"Deploying on cluster: {curr_task.id}")
                self._client.create(job_to_start, namespace=self._config.cluster_config.namespace)
                self.deployed_tasks.append(curr_task)

                # TODO: Extend this logic in your real project, this is only meant for demo purposes
                # For now we exit the thread after scheduling a single task.

                self.stop()
                return

            self.__logger.debug("Still alive...")
            time.sleep(5)

        logging.info('Experiment completed, currently does not support waiting.')

    def run_federated(self, clear: bool = True) -> None:
        """
        Main loop of the Orchestrator.
        @param clear: Boolean indicating whether a previous deployment needs to be cleaned up (i.e. lingering jobs that
        were deployed by the previous run).

        @type clear: bool
        @return: None
        @rtype: None
        """
        self._alive = True
        start_time = time.time()
        if clear:
            self.__clear_jobs()
        # TODO: Set duration correctly/till everything is done
        while self._alive and time.time() - start_time < self._config.get_duration():
            # 1. Check arrivals
            # If new arrivals, store them in arrival list
            while not self.__arrival_generator.arrivals.empty():
                arrival: Arrival = self.__arrival_generator.arrivals.get()
                unique_identifier: uuid.UUID = uuid.uuid4()
                repl = arrival.task.replication
                seed = arrival.task.experiment_configuration.random_seed[repl]
                task = FederatedArrivalTask(id=unique_identifier,
                                            network=arrival.get_network(),
                                            dataset=arrival.get_dataset(),
                                            seed=seed,
                                            replication=repl,
                                            type_map=arrival.get_experiment_config().worker_replication,
                                            system_parameters=arrival.get_system_config(),
                                            hyper_parameters=arrival.get_parameter_config(),
                                            learning_parameters=arrival.get_learning_config())

                self.__logger.debug(f"Arrival of: {task}")
                self.pending_tasks.put(task)

            while not self.pending_tasks.empty():
                # Do blocking request to priority queue
                curr_task = self.pending_tasks.get()
                self.__logger.info(f"Scheduling arrival of Arrival: {curr_task.id}")
                config_dict, configmap_name_dict = _prepare_experiment_maps(curr_task, curr_task.id, 1)
                job_to_start = construct_job(self._config, curr_task, configmap_name_dict)

                self.__create_config_maps(config_dict)
                # Hack to overcome limitation of KubeFlow version (Made for older version of Kubernetes)
                self.__logger.info(f"Deploying on cluster: {curr_task.id}")
                self._client.create(job_to_start, namespace=self._config.cluster_config.namespace)
                self.deployed_tasks.append(curr_task)

                # TODO: Extend this logic in your real project, this is only meant for demo purposes
                # For now we exit the thread after scheduling a single task.

                self.stop()

            self.__logger.debug("Still alive...")
            time.sleep(5)

        logging.info('Experiment completed, currently does not support waiting.')

    def __clear_jobs(self):
        """
        Function to clear existing jobs in the environment (i.e. old experiments/tests)
        @return: None
        @rtype: None
        """
        namespace = self._config.cluster_config.namespace
        self.__logger.info(f'Clearing old jobs in current namespace: {namespace}')

        for job in self._client.get(namespace=self._config.cluster_config.namespace)['items']:
            job_name = job['metadata']['name']
            self.__logger.info(f'Deleting: {job_name}')
            try:
                self._client.custom_api.delete_namespaced_custom_object(
                        PYTORCHJOB_GROUP,
                        PYTORCHJOB_VERSION,
                        namespace,
                        PYTORCHJOB_PLURAL,
                        job_name)
            except Exception as excp:
                self.__logger.warning(f'Could not delete: {job_name}. Reason: {excp}')

    def __create_config_maps(self, config_maps: Dict[str, V1ConfigMap]):
        for _, config_map in config_maps.items():
            self._v1.create_namespaced_config_map(self._config.cluster_config.namespace,
                                                  config_map)
