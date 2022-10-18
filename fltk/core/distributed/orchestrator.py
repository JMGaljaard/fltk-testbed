from __future__ import annotations

import abc
import collections
import logging
import threading
import time
import uuid
import tomli

from functools import total_ordering
from queue import PriorityQueue
from typing import OrderedDict, Dict, Type, Union, List, Tuple
from typing import TYPE_CHECKING

import numpy as np
from google.cloud import container_v1
from google.oauth2 import service_account
from jinja2 import Environment, FileSystemLoader
from kubeflow.training import PyTorchJobClient
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

with open("configs/experiment.toml", "rb") as f:
    toml_config = tomli.load(f)

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

    @param config: Distributed configuration for logging directory
    configuration.
    @type config: DistributedConfig

    @return: String representation of the logging path for a specific
    experiment.
    @rtype: str
    """
    log_dir = config.execution_config.log_path
    experiment_path = config.execution_config.experiment_prefix
    experiment_name = f"{task.dataset}_{task.network}_{u_id}_{task.replication}"
    full_path = f"{log_dir}/{experiment_path}/{experiment_name}"
    return full_path


def render_template(task: ArrivalTask, tpe: str, replication: int,
                    experiment_path: str) -> str:
    """Helper function to render jinja templates with necessary
    arguments for experiment (types). These templates are used for
    generating ConfigMaps used by the Pods that perform the learning
    experiments.

    @param task: Arrival description of Experiment/Deployment.
    @type task: ArrivalTask
    @param tpe: Indicator to distinct between 'learner' and 'parameter server'/'federator'
    @type tpe: str
    @param replication: Count for the replication of an experiment.
    @type replication: int
    @param experiment_path: Path where the experiment folder resides.
    @type experiment_path: str
    @return: Rendered template containing the content of a ConfigMap
             for a learner of `tpe` for the provided task.
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


def _prepare_experiment_maps(task: ArrivalTask, config: DistributedConfig,
                             u_id: uuid.UUID, replication: int = 1) -> \
        Tuple[OrderedDict[str, V1ConfigMap], OrderedDict[str, str]]:
    """Helper private function to create ConfigMap descriptions for a
    deployment of learners.

    @param task: Task description object.
    @type task: ArrivalTask
    @param config:
    @type config:
    @param u_id:
    @type u_id:
    @param replication:
    @type replication:
    @return: None
    @rtype: None
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


class MockJob:

    def __init__(self, name, epochs):
        self._name = name
        self._duration = 0
        config = toml_config['job']
        for _ in range(epochs):
            self._duration += max(0, np.random.normal(config['centre'], config['std']))
        self._status = "Pending"
        threading.Thread(target=self._thread_target).start()

    def _thread_target(self):
        time.sleep(0.5)
        self._status = "Running"
        time.sleep(self._duration)
        self._status = "Succeeded"


class ClientMocker:

    def __init__(self):
        self._jobs = {}
        self._uuid = ""

    def delete_job(self, job: str) -> None:
        print("delete: " + job)

    def create(self, job, namespace=None):
        name = job.metadata.name
        self._jobs[name] = MockJob(name, toml_config['job']['epochs'])  # todo change amount of epochs here
        self._uuid = name.split("trainjob-")[1]

    def get_job_status(self, name="", namespace=None):
        return self._jobs[name]._status


class Orchestrator(DistNode, abc.ABC):
    """Central component of the Federated Learning System: The Orchestrator.

    The Orchestrator is in charge of the following tasks:
    - Running experiments
        - Creating and/or managing tasks
        - Keep track of progress (pending/started/failed/completed)
    - Keep track of timing

    Note that the Orchestrator does not function like a Federator, in
    the sense that it keeps a central model, performs aggregations and
    keeps track of Clients. For this, the KubeFlow PyTorch-Operator is
    used to deploy a train task as a V1PyTorchJob, which automatically
    generates the required setup in the cluster. In addition, this
    allows more Jobs to be scheduled, than that there are resources,
    as such, letting the Kubernetes Scheduler let decide when to run
    which containers where.
    """
    _alive = False
    # Priority queue, requires an orderable object, otherwise a Tuple[int, Any] can be used to insert.
    pending_tasks: "PriorityQueue[ArrivalTask]" = PriorityQueue()
    # deployed_tasks: Set[_ArrivalTask] = set()
    # completed_tasks: Set[_ArrivalTask] = set()
    SLEEP_TIME = toml_config['orchestrator']['sleep']

    def __init__(self, cluster_mgr: ClusterManager, arv_gen: ArrivalGenerator,
                 config: DistributedConfig):
        self._logger = logging.getLogger('Orchestrator')
        self._logger.debug("Loading in-cluster configuration")
        self._cluster_mgr = cluster_mgr
        self._arrival_generator = arv_gen
        self._config = config

        # API to interact with the cluster.
        if config.execution_config.local:
            self._client = ClientMocker()
        else:
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
        Main loop of the Orchestrator for simulated arrivals. By
        default, previous deployments are not stopped (i.e.
        PytorchTrainingJobs) on the cluster, which may interfere with
        utilization statistics of your cluster.  Make sure to check if
        you want previous results to be removed.

        @param clear: Boolean indicating whether a previous deployment
        needs to be cleaned up (i.e. lingering jobs that were deployed
        by the previous run).
        @type clear: bool

        @param experiment_replication: Replication index (integer) to
        allow for the logging to experiment specific directories for
        experiments.  @type experiment_replication: int

        @return: None
        @rtype: None
        """

    def _create_config_maps(self, config_maps: Dict[str, V1ConfigMap]) -> None:
        """
        Private helper function to generate V1ConfigMap resources that
        are to be attached to the different trainers.  This allows for
        dynamic deployment with generated configuration files.
        """
        for _, config_map in config_maps.items():
            self._v1.create_namespaced_config_map(
                self._config.cluster_config.namespace, config_map)

# Used for naming virtual nodes
NODE_COUNT = 0


class SkyScrapeNode:
    """Mocked version of a node which stores metadata about a node and
    keeps track of the jobs that are running on it."""
    def __init__(self, jobs: List[SkyScrapeJob]):
        global NODE_COUNT
        self._watt_usage = toml_config['node']['watt_usage']  # The amount of watt a node uses when idle
        self._watt_delta = toml_config['node']['watt_delta']  # The amount of watt a node uses per job
        self._jobs = jobs
        self._type = toml_config['node']['type']
        self._node_count = NODE_COUNT
        NODE_COUNT += 1

    def get_expected_usage(self, remaining_time):
        """
        get expected energy usage if a new job gets added to this node
        """
        baseline = self._watt_usage
        jobs_length = len(self._jobs)

        return self._watt_delta * (jobs_length + 1) * remaining_time + baseline * remaining_time

    def __str__(self):
        return f"Node {self._node_count}"

    def add_job(self, job: SkyScrapeJob):
        self._jobs.append(job)

    def remove_job(self, job):
        print(self._jobs.index(job))


@total_ordering
class SkyScrapeJob:

    def __init__(self, _uuid: uuid.UUID, node: SkyScrapeNode):
        self.uuid = _uuid
        self.deploy_time = time.time()
        self.start_time = -1.0
        self.node = node

    def __lt__(self, other):
        # We can sort by expected duration or longest time as well
        return self.deploy_time < other.deploy_time

    def __eq__(self, other):
        if type(other) == uuid.UUID:
            return self.uuid == other

        return self.deploy_time == other.deploy_time and \
            self.start_time == other.start_time

    def __str__(self):
        return f"{self.uuid}: {self.start_time}"

    def __hash__(self):
        return id(self)


class SimulatedOrchestrator(Orchestrator):
    """Orchestrator implementation for Simulated arrivals. Currently,
    supports only Poisson inter-arrival times."""

    def __init__(self, cluster_mgr: ClusterManager,
                 arrival_generator: ArrivalGenerator,
                 config: DistributedConfig):
        super().__init__(cluster_mgr, arrival_generator, config)
        
        self._unclaimed_jobs: "PriorityQueue[SkyScrapeJob]" = PriorityQueue()
        self._deployed_jobs: List[SkyScrapeJob] = []
        self._completed_jobs: List[uuid.UUID] = []
        self._resource_claims: Dict[uuid.UUID, ArrivalTask] = dict()
        self._nodes: List[SkyScrapeNode] = []

        # We calculate this value ourselves to simulate real arrivals
        self._average_interarrival_time = None
        self._average_service_time = None
        self._time_of_last_job_arrival = None
        self._average_resize_time = None
        # We use total average, as it is our resulting statistic
        self.amount_of_nodes = []

        if self._config.execution_config.local:
            credentials = service_account.Credentials.from_service_account_file(
                "configs/key.json",
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self.cluster_manager_client = container_v1.ClusterManagerClient(
                credentials=credentials)

        # Make sure we start at 0 or 1 node depending on config
        self.resize_cluster()
        
        # Reset average resize time because jobs may have had to be terminated,
        # inducing a longer resize time
        self._average_resize_time = None

    def get_earliest_unclaimed_task(self) -> Union[SkyScrapeJob, None]:
        """Gets the earliest task in the queue or returns None if
        there is nothing left."""
        if self._unclaimed_jobs.empty():
            return None
        return self._unclaimed_jobs.get_nowait()

    def watch_amount_of_nodes(self):
        """Watches the amount of nodes in the cluster and logs it"""
        # Note that we can not use the main orchestrator loop, as it may be hanging on a resize
        while self._alive:
            time.sleep(self.SLEEP_TIME)
            self.amount_of_nodes.append(len(self._nodes))

    def resize_cluster(self) -> None:
        """Resizes the cluster guarded by a semaphore."""
        start_time = time.time()

        # Resize to len(self._nodes)
        if self._config.execution_config.local:
            # Simulate a resize
            config = toml_config['resize']
            time.sleep(max(0, np.random.normal(config['centre'], config['std'])))
        else:
            time.sleep(self._config.execution_config.additional_resize_time)

            response = self.cluster_manager_client.set_node_pool_size(
                {"node_count": len(self._nodes),
                 "node_pool_id": "medium-fltk-pool-1",
                 "zone": "us-central1-c",
                 "project_id": "qpe-k3z6awuymv44",  # todo make this a variable
                 "cluster_id": "fltk-testbed-cluster"})

            self._logger.info("Resizing cluster to %d nodes", len(self._nodes))
            while not response.end_time:
                request = container_v1.GetOperationRequest(name=response.self_link.split("/v1/")[1])
                response = self.cluster_manager_client.get_operation(request=request)
                time.sleep(0.05)
            self._logger.info("Cluster resized to %d nodes", len(self._nodes))

        end_time = time.time()
        self._average_resize_time = _get_running_average(self._average_resize_time, end_time - start_time)

    def deploy(self, curr_task: ArrivalTask, replication: int, node):
        """Adds a task to the internal queue and deploys it onto Kubernetes"""
        # Create persistent logging information. A these will not be
        # deleted by the Orchestrator, as such, they allow you to
        # retrieve information of experiments after removing the
        # PytorchJob after completion.
        config_dict, configmap_name_dict = _prepare_experiment_maps(
            curr_task,
            config=self._config,
            u_id=curr_task.id,
            replication=replication)
        self._create_config_maps(config_dict)

        sky_scrape_job = SkyScrapeJob(curr_task.id, node)
        self._deployed_jobs.append(sky_scrape_job)
        node.add_job(sky_scrape_job)

        job_to_start = construct_job(self._config, curr_task,
                                     configmap_name_dict)
        self._client.create(job_to_start,
                            namespace=self._config.cluster_config.namespace)

        self._logger.info(
            f"Deployed on cluster: {curr_task.id} on node {node}")

    def _update_service_time(self, task: SkyScrapeJob):
        delta = time.time() - task.start_time
        self._average_service_time = \
            _get_running_average(self._average_service_time, delta)

    def _update_interarrival_time(self):
        delta = time.time() - self._time_of_last_job_arrival
        self._average_interarrival_time = \
            _get_running_average(self._average_interarrival_time, delta)

    def scale_down_if_possible(self, node) -> bool:
        #   - resize costs = resize time * baseline usage * 2
        #   - costs to keep job running = time until arrival of job that needs the node * baseline

        # If we do not count the current node, how much jobs can we fit?
        max_amount_of_pods = toml_config['orchestrator']['max_pods_per_node'] * (len(self._nodes) - 1)
        capacity = max_amount_of_pods - len(self._deployed_jobs)
        assert capacity >= 0

        # The cost of scaling
        time_to_next_arrival = self._average_interarrival_time - (time.time() - self._time_of_last_job_arrival)
        time_until_node_is_necessary = time_to_next_arrival + self._average_interarrival_time * capacity

        costs_to_keep_node_running = node._watt_usage * time_until_node_is_necessary

        self._logger.info(f"Time until node is necessary: {time_until_node_is_necessary}. Time to next arrival {time_to_next_arrival}. Capacity {capacity}. Average interarrival time {self._average_interarrival_time}")

        # The cost to resize the system twice in case a job arrives
        resize_costs = 2 * self._get_average_baseline_usage() * self._average_resize_time

        if resize_costs > costs_to_keep_node_running:
            # It is cheaper to keep the node
            self._logger.info(f"Not scaling down, because resize costs are {resize_costs} and costs to keep node running {costs_to_keep_node_running}")
            return False

        self._logger.info("Scaling down")
        # It is cheaper to resize
        return True

    def check_if_jobs_finished(self, replication: int):
        """Checks each job currently deployed to ensure that the
        internal program state remains consistent"""
        task_to_move = []
        for task in self._deployed_jobs:
            try:
                job_status = self._client.get_job_status(
                    name=f"trainjob-{task.uuid}", namespace='test')
            except IndexError:
                # Job not found. This means that it is still loading
                continue

            if task.start_time == -1.0:
                self._logger.info(f"Job {task.uuid} started after " +
                                  f"{time.time() - task.deploy_time} seconds")

                # Task has started actual execution.
                # Note: this may mean: 'Container creating'
                task.start_time = time.time()
                self._unclaimed_jobs.put(task)

            if job_status != "Succeeded":
                continue
            self._logger.info(f"Job has completed: {task.uuid}")
            task_to_move.append(task)
            self._update_service_time(task)

            self._completed_jobs.append(task.uuid)

            if task in self._resource_claims:
                self.deploy(self._resource_claims.pop(task.uuid), replication, task.node)

            if task in self._unclaimed_jobs.queue:
                self._unclaimed_jobs.queue.remove(task)

        for task in task_to_move:
            self._deployed_jobs.remove(task)
            task.node._jobs.remove(task)

        nodes_to_remove = []
        for node in self._nodes:
            if len(node._jobs) == 0:
                if self.scale_down_if_possible(node):
                    nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self._nodes.remove(node)
            self.resize_cluster()

    def _scale_up_and_deploy(self, task, replication):
        new_node = SkyScrapeNode([])
        self._nodes.append(new_node)

        # Run resize on background
        self.resize_cluster()
        self.deploy(task, replication, new_node)

    def _check_arrivals(self):
        if not self._arrival_generator.arrivals.empty():
            if self._time_of_last_job_arrival is not None:
                self._update_interarrival_time()
            self._time_of_last_job_arrival = time.time()

        while not self._arrival_generator.arrivals.empty():
            arrival = self._arrival_generator.arrivals.get()
            task = _generate_task(arrival)
            self._logger.debug(f"Arrival of: {task.id}")
            self.pending_tasks.put(task)

    def _get_node_with_space(self) -> Union[SkyScrapeNode, None]:
        for node in self._nodes:
            if len(node._jobs) <= toml_config['orchestrator']['max_pods_per_node']:
                return node
        return None

    def _get_average_baseline_usage(self):
        return sum([node._watt_usage for node in self._nodes]) / len(self._nodes)

    # def _calculate_energy_usage(self, remaining_time, node) -> bool:
    #     expected_usage_if_job_added_to_node = node.get_expected_usage(remaining_time)
    #
    #     # Calculate based on the other nodes what the expected usage
    #     # of the job and the base use of the node will be. This could
    #     # be done in the future by using linear regresion or n nearest
    #     # neighbors, provided that information on node and task type
    #     # is known.
    #     average_baseline_usage = self._get_average_baseline_usage()
    #     average_usage_per_job = sum([node._watt_delta for node in self._nodes]) / len(self._nodes)
    #     # no need to include running job energy usage because it is consistent if you resize or not
    #     resize_energy = (average_baseline_usage * self._average_resize_time) + \
    #         self._average_service_time * (average_usage_per_job + average_baseline_usage)
    #     no_resize_energy = average_baseline_usage * (remaining_time + self._average_service_time) + \
    #                        average_usage_per_job * self._average_service_time
    #
    #     self._logger.info("energy: %d - %d", resize_energy, no_resize_energy)
    #     # If true, we should resize
    #     return self._average_resize_time < no_resize_energy

    def _should_we_scale(self, task) -> Tuple[bool, Union[SkyScrapeNode, None]]:
        """Decides whether a node can be reused or whether the cluster is scaled"""
        pods = len(self._nodes) * toml_config['orchestrator']['max_pods_per_node']

        if len(self._deployed_jobs) < pods:
            self._logger.info("Enough capacity. Deploying task %s", task.id)
            return False, self._get_node_with_space()

        self._logger.info("Not enough capacity. %d pod positions, %d jobs running.",
                          pods, len(self._deployed_jobs))

        # Get earliest started task, which position has not
        # yet been claimed by a pending job.
        earliest_task = self.get_earliest_unclaimed_task()
        averages = [self._average_interarrival_time,
                    self._average_service_time,
                    self._average_resize_time]

        if None in averages or earliest_task is None or \
           self._config.cluster_config.naive:
            # We don't know the inter-arrival time or service
            # time yet, so we cannot make a decision Scale up
            self._logger.info("Not enough data or no task left to claim. Resizing and deploying task %s", str(task.id))
            return True, None

        # Calculate expected amount of time until that job
        # will be finished.
        time_since_start = time.time() - earliest_task.start_time
        expected_remaining_time = self._average_service_time - time_since_start

        if self._average_resize_time < expected_remaining_time:
            self._logger.info(f"Should resize, because resize time {self._average_resize_time} is less than expected remaining time {expected_remaining_time}")
            
            self._unclaimed_jobs.put(earliest_task)
            return True, None

        # Claim the position of the earliest unclaimed job
        self._logger.info(
            "Claiming position of task %s for task %s because time is %s and resize time is %s",
            earliest_task.uuid, task.id, expected_remaining_time,
            self._average_resize_time)

        self._resource_claims[earliest_task.uuid] = task
        return False, None


    def _deploy_task(self, task, experiment_replication) -> None:
        """Deploys the task to a pre-existing node or scales the cluster up."""
        (scale, node) = self._should_we_scale(task)
        if not scale:
            if node:
                # Directly deploy task that has claimed other task,
                # such that it may start already on the redundant node
                # (if possible) or will start as soon as the claimed
                # node finishes.
                self.deploy(task, experiment_replication, node)
            else:
                # Task has claimed the position of other task, will run eventually
                pass
        else:
            self._scale_up_and_deploy(task, experiment_replication)

    def run(self,
            clear: bool = False,
            experiment_replication: int = -1) -> None:
        self._alive = True
        start_time = time.time()

        self._logger.info("SkyScraper started")
        threading.Thread(target=self.watch_amount_of_nodes).start()

        while self._alive and (time.time() - start_time < self._config.get_duration()):
            # 1. Check arrivals
            # If new arrivals, store them in arrival list
            self._check_arrivals()

            # Deploy all pending tasks using SkyScraper
            while not self.pending_tasks.empty():
                self._deploy_task(self.pending_tasks.get(),
                                  experiment_replication)

            self.check_if_jobs_finished(experiment_replication)
            time.sleep(self.SLEEP_TIME)

        self.stop()
        self._logger.info("SkyScraper stopped")
        self._logger.info(
            f"Average service time: {self._average_service_time}")
        self._logger.info(
            f"Average interarrival time: {self._average_interarrival_time}")
        self._logger.info(f"Average resize time: {self._average_resize_time}")
        self._logger.info(f"Average amount of nodes: {sum(self.amount_of_nodes) / len(self.amount_of_nodes)}")
        self._logger.info(f"Total amount of jobs: {len(self._completed_jobs)}")
        # todo result = total_time * baseline_usage * average amount of nodes + total number of jobs * average job usage * service time
        # self._logger.info(f"result: {}")
        self.resize_cluster()
        self._logger.info('Experiment completed.')


class BatchOrchestrator(Orchestrator):
    """Orchestrator implementation to allow for running all
    experiments that were defined in one go."""

    def __init__(self, cluster_mgr: ClusterManager,
                 arrival_generator: ArrivalGenerator,
                 config: DistributedConfig):
        super().__init__(cluster_mgr, arrival_generator, config)

    def run(self,
            clear: bool = False,
            experiment_replication: int = 1,
            wait_historical: bool = True) -> None:
        """Main loop of the Orchestrator for processing a
        configuration as a batch, i.e. deploy all-at-once (batch)
        without any scheduling or simulation applied. This will make
        use of Kubeflow Training-operators to ensure that pods are
        created with sufficient resources (depending on resources
        available on your cluster).

        @param clear: Boolean indicating whether a previous deployment
        needs to be cleaned up (i.e. lingering jobs that were deployed
        by the previous run).

        @param experiment_replication: Number of times the experiment
        needs to be replicated.

        @param wait_historical: Boolean indicating whether the
        orchestrator should wait for historical jobs to complete

        @return: None
        @rtype: None
        """
        raise NotImplementedError("SkyScraper does not support batch mode")
