from __future__ import annotations

import abc
import collections
import logging
import threading
import time
import uuid
from functools import total_ordering
from queue import PriorityQueue
from typing import OrderedDict, Dict, Type, Union, List
from typing import TYPE_CHECKING

import dateutil.parser
from google.cloud import container_v1
from google.oauth2 import service_account
from jinja2 import Environment, FileSystemLoader
from kubeflow.training import PyTorchJobClient
from kubeflow.training.constants.constants import PYTORCHJOB_GROUP, PYTORCHJOB_VERSION, PYTORCHJOB_PLURAL
from kubernetes import client
from kubernetes.client import V1ConfigMap, V1ObjectMeta

from fltk.core.distributed.dist_node import DistNode
from fltk.util.cluster.client import construct_job, ClusterManager
from fltk.util.task import get_job_arrival_class, DistributedArrivalTask, FederatedArrivalTask, ArrivalTask
from fltk.util.task.generator import ArrivalGenerator

if TYPE_CHECKING:
    from fltk.util.config import DistributedConfig

# Setup required variables for Jinja templates.
EXPERIMENT_DIR = 'experiments'
__ENV = Environment(loader=FileSystemLoader(EXPERIMENT_DIR))


def _get_running_average(curr, delta):
    return (curr * 0.9 + 0.1 * delta) if curr is not None else delta


def _generate_experiment_path_name(task: ArrivalTask, u_id: Union[uuid.UUID,
                                                                  str],
                                   config: DistributedConfig):
    """
    Helper function to generate experiment name for logging without conflicts
    @param task: Arrival task for Task related information.
    @type task: ArrivalTask
    @param u_id: Unique identifier string corresponding to the experiment.
    @type u_id: str
    @param config: Distributed configuration for logging directory configuration.
    @type config: DistributedConfig
    @return: String representation of the logging path for a specific experiment.
    @rtype: str
    """
    log_dir = config.execution_config.log_path
    experiment_path = config.execution_config.experiment_prefix
    experiment_name = f"{task.dataset}_{task.network}_{u_id}_{task.replication}"
    full_path = f"{log_dir}/{experiment_path}/{experiment_name}"
    return full_path


def render_template(task: ArrivalTask, tpe: str, replication: int,
                    experiment_path: str) -> str:
    """
    Helper function to render jinja templates with necessary arguments for experiment (types). These templates are
    used for generating ConfigMaps used by the Pods that perform the learning experiments.
    @param task: Arrival description of Experiment/Deployment.
    @type task: ArrivalTask
    @param tpe: Indicator to distinct between 'learner' and 'parameter server'/'federator'
    @type tpe: str
    @param replication: Count for the replication of an experiment.
    @type replication: int
    @param experiment_path: Path where the experiment folder resides.
    @type experiment_path: str
    @return: Rendered template containing the content of a ConfigMap for a learner of `tpe` for the provided task.
    @rtype: str
    """
    if isinstance(task, FederatedArrivalTask):
        template = __ENV.get_template('node.jinja.yaml')
    elif isinstance(task, DistributedArrivalTask):
        template = __ENV.get_template('dist_node.jinja.yaml')
    else:
        raise Exception(f"Cannot handle type of task: {task}")
    filled_template = template.render(task=task,
                                      tpe=tpe,
                                      replication=replication,
                                      experiment_path=experiment_path)
    return filled_template


def _prepare_experiment_maps(task: ArrivalTask, config: DistributedConfig, u_id: uuid.UUID, replication: int = 1) -> \
        (OrderedDict[str, V1ConfigMap], OrderedDict[str, str]):
    """
    Helper private function to create ConfigMap descriptions for a deployment of learners.
    @param task: Task description object.
    @type task: ArrivalTask
    @param config:
    @type config:
    @param u_id:
    @type u_id:
    @param replication:
    @type replication:
    @return:
    @rtype:
    """
    type_dict = collections.OrderedDict()
    name_dict = collections.OrderedDict()
    for tpe in task.type_map.keys():
        name = str(f'{tpe}-{u_id}-{replication}').lower()
        meta = V1ObjectMeta(
            name=name,
            labels={'app.kubernetes.io/name': f"fltk.node.config.{tpe}"})
        exp_path = _generate_experiment_path_name(task, u_id, config)
        filled_template = render_template(task=task,
                                          tpe=tpe,
                                          replication=replication,
                                          experiment_path=exp_path)
        type_dict[tpe] = V1ConfigMap(
            data={'node.config.yaml': filled_template}, metadata=meta)
        name_dict[tpe] = name
    return type_dict, name_dict


def _generate_task(arrival) -> ArrivalTask:
    """
    Function to generate a task from an Arrival.
    @param arrival: Arrival to create a (runnable) Task from.
    @type arrival: Arrival
    @return: Mapped ArrivalTask for the given task.
    @rtype: ArrivalTask
    """
    unique_identifier: uuid.UUID = uuid.uuid4()
    task_type: Type[ArrivalTask] = get_job_arrival_class(
        arrival.task.experiment_type)
    task = task_type.build(arrival=arrival,
                           u_id=unique_identifier,
                           replication=arrival.task.replication)
    return task


class Orchestrator(DistNode, abc.ABC):
    """
    Central component of the Federated Learning System: The Orchestrator.

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
    # deployed_tasks: Set[_ArrivalTask] = set()
    # completed_tasks: Set[_ArrivalTask] = set()
    SLEEP_TIME = 5

    def __init__(self, cluster_mgr: ClusterManager, arv_gen: ArrivalGenerator,
                 config: DistributedConfig):
        self._logger = logging.getLogger('Orchestrator')
        self._logger.debug("Loading in-cluster configuration")
        self._cluster_mgr = cluster_mgr
        self._arrival_generator = arv_gen
        self._config = config

        # API to interact with the cluster.
        self._client = PyTorchJobClient()
        self._v1 = client.CoreV1Api()

    def stop(self) -> None:
        """
        Stop the Orchestrator.
        @return: None
        @rtype: None
        """
        self._logger.info("Received stop signal for the Orchestrator.")
        self._alive = False

        self._cluster_mgr.stop()

    @abc.abstractmethod
    def run(self,
            clear: bool = False,
            experiment_replication: int = -1) -> None:
        """
        Main loop of the Orchestrator for simulated arrivals. By default, previous deployments are not stopped (i.e.
        PytorchTrainingJobs) on the cluster, which may interfere with utilization statistics of your cluster.
        Make sure to check if you want previous results to be removed.
        @param clear: Boolean indicating whether a previous deployment needs to be cleaned up (i.e. lingering jobs that
        were deployed by the previous run).
        @type clear: bool
        @param experiment_replication: Replication index (integer) to allow for the logging to experiment specific
        directories for experiments.
        @type experiment_replication: int
        @return: None
        @rtype: None
        """

    def _clear_jobs(self):
        """
        Function to clear existing jobs in the environment (i.e. old experiments/tests). This will will, currently,
        not remove configuration map objects. A later version will allow for removing these autmatically as well.
        @return: None
        @rtype: None
        """
        namespace = self._config.cluster_config.namespace
        self._logger.info(
            f'Clearing old jobs in current namespace: {namespace}')

        for job in self._client.get(
                namespace=self._config.cluster_config.namespace)['items']:
            job_name = job['metadata']['name']
            self._logger.info(f'Deleting: {job_name}')
            try:
                self._client.custom_api.delete_namespaced_custom_object(
                    PYTORCHJOB_GROUP, PYTORCHJOB_VERSION, namespace,
                    PYTORCHJOB_PLURAL, job_name)
            except Exception as excp:
                self._logger.warning(
                    f'Could not delete: {job_name}. Reason: {excp}')

    def _create_config_maps(self, config_maps: Dict[str, V1ConfigMap]) -> None:
        """
        Private helper function to generate V1ConfigMap resources that are to be attached to the different trainers.
        This allows for dynamic deployment with generated configuration files.
        """
        for _, config_map in config_maps.items():
            self._v1.create_namespaced_config_map(
                self._config.cluster_config.namespace, config_map)

    # def wait_for_jobs_to_complete(self, others: Optional[List[str]] = None):
    #     """
    #     Function to wait for all tasks to complete. This allows to wait for all the resources to free-up after running
    #     an experiment. Thereby allowing for running multiple experiments on a single cluster, without letting
    #     experiments interfere with each other.
    #     """
    #     if others:
    #         uuid_regex = re.compile(
    #             "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    #         )
    #
    #         ids = {
    #             uuid_regex.search(task).group()
    #             for task in others if uuid_regex.search(task) is not None
    #         }
    #         historical_tasks = map(HistoricalArrivalTask, ids)
    #         self.deployed_tasks.update(historical_tasks)
    #     while len(self.deployed_tasks) > 0:
    #         task_to_move = set()
    #         for task in self.deployed_tasks:
    #             try:
    #                 job_status = self._client.get_job_status(
    #                     name=f"trainjob-{task.id}", namespace='test')
    #             except Exception:
    #                 logging.debug(
    #                     msg=f"Could not retrieve job_status for {task.id}")
    #                 job_status = None
    #
    #             if job_status and job_status in {
    #                 'Completed', 'Failed', 'Succeeded'
    #             }:
    #                 logging.info(
    #                     f"{task.id} was completed with status: {job_status}, moving to completed"
    #                 )
    #                 task_to_move.add(task)
    #             else:
    #                 logging.info(
    #                     f"Waiting for {task.id} to complete, {self.pending_tasks.qsize()} pending, "
    #                     f"{self._arrival_generator.arrivals.qsize()} arrivals"
    #                 )
    #         self.completed_tasks.update(task_to_move)
    #         self.deployed_tasks.difference_update(task_to_move)
    #         time.sleep(self.SLEEP_TIME)


@total_ordering
class SkyScrapeJob:

    def __init__(self, uuid: uuid.UUID):
        self.uuid = uuid
        self.deploy_time = time.time()
        self.start_time = -1.0

    def __lt__(self, other):
        # We can sort by expected duration or longest time as well
        return self.deploy_time < other.deploy_time

    def __eq__(self, other):
        if type(other) == uuid.UUID:
            return self.uuid == other

        return self.deploy_time == other.deploy_time and self.start_time == other.start_time

    def __str__(self):
        return f"{self.uuid}: {self.start_time}"

    def __hash__(self):
        return id(self)


class SimulatedOrchestrator(Orchestrator):
    """
    Orchestrator implementation for Simulated arrivals. Currently, supports only Poisson inter-arrival times.
    """

    def __init__(self, cluster_mgr: ClusterManager,
                 arrival_generator: ArrivalGenerator,
                 config: DistributedConfig):
        super().__init__(cluster_mgr, arrival_generator, config)

        self._unclaimed_jobs: "PriorityQueue[SkyScrapeJob]" = PriorityQueue()
        self._deployed_jobs: List[SkyScrapeJob] = list()
        self._completed_jobs: Dict[uuid.UUID, float] = dict()
        self._arrival_times: Dict[uuid.UUID, float] = dict()

        self.nodes_running = 0

        # We calculate this value ourselves to simulate real arrivals instead of simulated
        self._average_interarrival_time = None
        self._average_service_time = None
        self._average_response_time = None
        self._time_of_last_job_arrival = None
        self._average_resize_time = None

        credentials = service_account.Credentials.from_service_account_file(
            "configs/key.json", scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self.cluster_manager_client = container_v1.ClusterManagerClient(credentials=credentials)

        # Make sure we start at 0 or 1 node depending on config
        self.resize_cluster_async(1 if not self._config.cluster_config.naive else 0)
        # Reset average resize time because jobs may have had to be terminated, inducing a longer resize time
        self._average_resize_time = None
        self.resizing = False

    def get_earliest_unclaimed_task(self) -> Union[SkyScrapeJob, None]:
        if self._unclaimed_jobs.empty():
            return None
        return self._unclaimed_jobs.get_nowait()

    def resize_cluster_async(self, nodes) -> None:
        time.sleep(self._config.execution_config.additional_resize_time)

        response = self.cluster_manager_client.set_node_pool_size(
            {"node_count": nodes,
             "node_pool_id": "medium-fltk-pool-1",
             "zone": "us-central1-c",
             "project_id": "qpe-k3z6awuymv44",  # todo make this a variable
             "cluster_id": "fltk-testbed-cluster"})

        self._logger.info("Resizing cluster to %d nodes", nodes)
        while not response.end_time:
            request = container_v1.GetOperationRequest(name=response.self_link.split("/v1/")[1])
            response = self.cluster_manager_client.get_operation(request=request)
            time.sleep(0.05)
        self._logger.info("Cluster resized to %d nodes", nodes)

        delta = (dateutil.parser.parse(response.end_time) - dateutil.parser.parse(response.start_time)).total_seconds()
        self._average_resize_time = _get_running_average(self._average_resize_time, delta)
        self.resizing = False

    def deploy(self, curr_task: ArrivalTask, replication: int):
        # Create persistent logging information. A these will not be deleted by the Orchestrator, as such, they
        # allow you to retrieve information of experiments after removing the PytorchJob after completion.
        config_dict, configmap_name_dict = _prepare_experiment_maps(
            curr_task,
            config=self._config,
            u_id=curr_task.id,
            replication=replication)
        self._create_config_maps(config_dict)

        sky_scrape_job = SkyScrapeJob(curr_task.id)
        self._deployed_jobs.append(sky_scrape_job)

        job_to_start = construct_job(self._config, curr_task,
                                     self._arrival_times[curr_task.id],
                                     configmap_name_dict)

        self._client.create(job_to_start, namespace=self._config.cluster_config.namespace)
        self._logger.info(f"Deployed on cluster: {curr_task.id}")

    def _update_service_time(self, task: SkyScrapeJob):
        delta = time.time() - task.start_time
        self._average_service_time = _get_running_average(self._average_service_time, delta)

    def _update_interarrival_time(self):
        delta = time.time() - self._time_of_last_job_arrival
        self._average_interarrival_time = _get_running_average(self._average_interarrival_time, delta)

    def check_if_jobs_finished(self):
        task_to_move = list()
        for task in self._deployed_jobs:
            try:
                job_status = self._client.get_job_status(name=f"trainjob-{task.uuid}", namespace='test')
            except IndexError:
                # Job not found. This means that it is still loading
                continue

            if task.start_time == -1.0:
                self._logger.info(f"Job {task.uuid} started after {time.time() - task.deploy_time} seconds")
                # Task has started actual execution. Note: this may mean: 'Container creating'
                task.start_time = time.time()
                self._unclaimed_jobs.put(task)

            if job_status != "Succeeded":
                continue
            task_to_move.append(task)
            self._update_service_time(task)

            # Google has unpredictable scheduling times, so we need to wait for the job to start being detectable
            # before being able to measure the service time
            time_google_scheduling = task.start_time - task.deploy_time
            self._completed_jobs[task.uuid] = time.time() - self._arrival_times[task.uuid] - time_google_scheduling
            self._average_response_time = _get_running_average(self._average_response_time,
                                                               self._completed_jobs[task.uuid])

            self._logger.info(f"{task.uuid} was completed with response time {self._completed_jobs[task.uuid]} seconds."
                              f" In reality it took {time.time() - self._arrival_times[task.uuid]} seconds, but spent "
                              f" {time_google_scheduling} seconds to Google scheduling.")

            if task in self._unclaimed_jobs.queue:
                self._unclaimed_jobs.queue.remove(task)

            # self._client.delete(name=f"trainjob-{task.uuid}", namespace='test')

        for task in task_to_move:
            self._deployed_jobs.remove(task)

    def _scale_up_and_deploy(self, task, replication):
        # Wait for previous resize to finish
        while self.resizing:
            time.sleep(1)
        self.resizing = True

        self.nodes_running += 1
        # Ensure that we always have an extra job running
        nodes = self.nodes_running + (1 if not self._config.cluster_config.naive else 0)
        # Run resize on background
        threading.Thread(target=self.resize_cluster_async, args=(nodes,)).start()
        self.deploy(task, replication)

    def run(self,
            clear: bool = False,
            experiment_replication: int = -1) -> None:
        self._alive = True
        start_time = time.time()
        if clear:
            self._clear_jobs()

        self._logger.info("SkyScraper started")

        while self._alive and (
                (time.time() - start_time) < self._config.get_duration()):
            # 1. Check arrivals
            # If new arrivals, store them in arrival list
            if not self._arrival_generator.arrivals.empty():
                if self._time_of_last_job_arrival is not None:
                    self._update_interarrival_time()
                self._time_of_last_job_arrival = time.time()

                while not self._arrival_generator.arrivals.empty():
                    arrival = self._arrival_generator.arrivals.get()
                    task = _generate_task(arrival)
                    self._arrival_times[task.id] = time.time()
                    self._logger.debug(f"Arrival of: {task.id}")
                    self.pending_tasks.put(task)

            # Deploy all pending tasks using SkyScraper
            while not self.pending_tasks.empty():
                task = self.pending_tasks.get()

                pods = self.nodes_running * self._config.cluster_config.max_pods_per_node
                if len(self._deployed_jobs) < pods:
                    self._logger.info("Enough capacity. Deploying task " + str(task.id))
                    # We have enough capacity for job: directly deploy
                    self.deploy(task, experiment_replication)
                    continue
                self._logger.info(f"Not enough capacity. {pods} pod positions, {len(self._deployed_jobs)} jobs running.")

                if None in [self._average_interarrival_time, self._average_service_time,
                            self._average_resize_time] or self._config.cluster_config.naive:
                    # We don't know the inter-arrival time or service time yet, so we cannot make a decision
                    # Scale up
                    self._logger.info("Not enough data yet. Resizing and deploying task " + str(task.id))
                    self._scale_up_and_deploy(task, experiment_replication)
                    continue

                # Get earliest started task, which position has not yet been claimed by a pending job
                earliest_task = self.get_earliest_unclaimed_task()
                if earliest_task is None:
                    self._logger.info("All tasks claimed. Resizing and deploying task  " + str(task.id))
                    # All tasks have been claimed
                    # Scale up
                    self._scale_up_and_deploy(task, experiment_replication)
                    continue

                # Calculate expected amount of time until that job will be finished
                time_since_start = time.time() - earliest_task.start_time
                expected_remaining_time = self._average_service_time - time_since_start

                if expected_remaining_time > self._average_resize_time:
                    # It is faster to scale up and deploy the new job than to wait for the earliest job to finish
                    self._logger.info(
                        "Expected remaining time is larger than resize time. Resizing and deploying task " + str(
                            task.id))
                    # Re-add the task to the PriorityQueue
                    self._unclaimed_jobs.put(earliest_task)
                    self._scale_up_and_deploy(task, experiment_replication)
                    continue

                # Claim the position of the earliest unclaimed job
                self._logger.info("Claiming position of task " + str(earliest_task.uuid) + " for task " + str(task.id) +
                                  " because expected time is " + str(expected_remaining_time) + " and resize time is " +
                                  str(self._average_resize_time))
                # Directly deploy task that has claimed other task, such that it may start already on the redundant node
                # (if possible) or will start as soon as the claimed node finishes.
                self.deploy(task, experiment_replication)

            self.check_if_jobs_finished()

            time.sleep(self.SLEEP_TIME)
        self.stop()
        self._logger.info("SkyScraper stopped")
        self._logger.info(f"Average service time: {self._average_service_time}")
        self._logger.info(f"Average interarrival time: {self._average_interarrival_time}")
        self._logger.info(f"Average resize time: {self._average_resize_time}")
        self._logger.info(f"Average response time: {self._average_response_time}")
        self._logger.info(f"Completed jobs: {self._completed_jobs}")
        self.resize_cluster_async(0)
        self._logger.info('Experiment completed.')


class BatchOrchestrator(Orchestrator):
    """
    Orchestrator implementation to allow for running all experiments that were defined in one go.
    """

    def __init__(self, cluster_mgr: ClusterManager,
                 arrival_generator: ArrivalGenerator,
                 config: DistributedConfig):
        super().__init__(cluster_mgr, arrival_generator, config)

    def run(self,
            clear: bool = False,
            experiment_replication: int = 1,
            wait_historical: bool = True) -> None:
        """
        Main loop of the Orchestrator for processing a configuration as a batch, i.e. deploy all-at-once (batch)
        without any scheduling or simulation applied. This will make use of Kubeflow Training-operators to ensure that
        pods are created with sufficient resources (depending on resources available on your cluster).
        @param clear: Boolean indicating whether a previous deployment needs to be cleaned up (i.e. lingering jobs that
        were deployed by the previous run).
        @param experiment_replication: Number of times the experiment needs to be replicated.
        @param wait_historical: Boolean indicating whether the orchestrator should wait for historical jobs to complete
        @return: None
        @rtype: None
        """
        raise NotImplementedError("SkyScraper does not support batch mode")
