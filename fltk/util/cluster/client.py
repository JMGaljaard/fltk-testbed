import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Dict

import schedule
from kubernetes import client, config
from torch.utils.tensorboard import SummaryWriter

from fltk.util.cluster.conversion import Convert
from fltk.util.singleton import Singleton
from fltk.util.task.config.parameter import TrainTask


@dataclass
class ClientRef:
    name: str
    ref: str
    tb_writer: SummaryWriter
    data_size = 0

    def __repr__(self):
        return self.name


@dataclass
class Resource:
    node_name: str
    cpu_allocatable: int
    memory_allocatable: int
    cpu_requested: int
    memory_requested: int
    cpu_limit: int
    memory_limit: int


class BuildDescription:
    resources: client.V1ResourceRequirements
    identifier: str
    container: client.V1Container
    template: client.V1PodTemplateSpec
    spec: client.V1JobSpec


class ResourceWatchDog:
    """
    Class to be used to monitor the resources available within the cluster. For this the resource API is not needed, but
    can be used to extend/speedup/prettify the implementation. The implementation is based on the work by @gorenje found
    on GithHub:

    https://gist.github.com/gorenje/dff508489c3c8a460433ad709f14b7db
    """
    _alive: False
    _time: float = -1
    _node_lookup: Dict[str, client.V1Node] = dict()
    _resource_lookup: Dict[str, Resource] = dict()

    def __init__(self):
        """
        Work should be based on the details listed here:
        https://github.com/scylladb/scylla-cluster-tests/blob/a7b09e69f0152a4d70bfb25ded3d75b7e7328acc/sdcm/cluster_k8s/__init__.py#L216-L223
        """
        self._v1: client.CoreV1Api = None
        self._logger = logging.getLogger('ResourceWatchDog')
        self._Q = Convert().convert

    def stop(self) -> None:
        """
        Function to stop execution. The runner thread _should_ merge back to the thread pool after calling this function
        to the thread pool.
        @return: None
        @rtype: None
        """
        self._logger.info("[WatchDog] Received request to stop execution")
        self._alive = False

    def start(self) -> None:
        """
        Function to start the resource watch dog. Currently, it only monitors the per-node memory and cpu availability.
        This does not handle event scheudling.
        @return: None
        @rtype: None
        """
        self._logger.info("Starting resource watchdog")
        self._alive = True
        self._v1 = client.CoreV1Api()
        self.__monitor_nodes()

        # Every 10 seconds we check the nodes with all the pods.
        schedule.every(10).seconds.do(self.__monitor_pods).tag('node-monitoring')
        # Every 1 minutes we check all the pods (in case the topology changes).
        schedule.every(1).minutes.do(self.__monitor_pods).tag('pod-monitoring')

        self._logger.info("Starting with watching resources")
        while self._alive:
            schedule.run_pending()
            time.sleep(1)

    def __monitor_nodes(self) -> None:
        """
        Watchdog function that watches the Cluster resources in a K8s cluster. Requires the config to be set and loaded
        prior to calling.
        @return: None
        @rtype: None
        """
        self._logger.info("Fetching node information of cluster...")
        try:
            node_list: client.V1NodeList = self._v1.list_node(watch=False)
            self._node_lookup = {node.metadata.name: node for node in node_list.items}
            if not self._alive:
                self._logger.info("Instructed to stop, stopping list_node watch on Kubernetes.")
                return
        except Exception as e:
            self._logger.error(e)
            raise e

    def __monitor_pods(self) -> None:
        """
        Function to monitor pod activity of currently listed pods. The available pods themselves are to be fetched
        prior to calling this function. Stale pod information will result in incomplete update, as pods will be missed.
        @return: None
        @rtype: None
        """
        node: client.V1Node
        new_resource_mapper = {}

        self._logger.info("Fetching pod information of cluster...")
        try:
            for node_name, node in self._node_lookup.items():
                # Create field selector to only get active pods that 'request' memory
                selector = f'status.phase!=Succeeded,status.phase!=Failed,spec.nodeName={node_name}'
                # Select pods from all namespaces on specific Kubernetes node
                # try:
                pod_list: client.V1PodList = self._v1.list_pod_for_all_namespaces(watch=False, field_selector=selector)
                # Retrieve allocatable memory of node
                alloc_cpu, alloc_mem = (self._Q(node.status.allocatable[item]) for item in ['cpu', 'memory'])
                core_req, core_lim, mem_req, mem_lim = 0, 0, 0, 0
                for pod in pod_list.items:
                    for container in pod.spec.containers:
                        response = container.resources
                        reqs = defaultdict(lambda: 0, response.requests or {})
                        lmts = defaultdict(lambda: 0, response.limits or {})
                        core_req += self._Q(reqs["cpu"])
                        mem_req += self._Q(reqs["memory"])
                        core_lim += self._Q(lmts["cpu"])
                        mem_lim += self._Q(lmts["memory"])
                resource = Resource(node_name, alloc_cpu, alloc_mem, core_req, mem_req, core_lim, mem_lim)
                new_resource_mapper[node_name] = resource
        except Exception as e:
            self._logger.error(f'Namespace lookup for {node_name} failed. Reason: {e}')

        self._resource_lookup = new_resource_mapper
        self._logger.debug(self._resource_lookup)


class ClusterManager(metaclass=Singleton):
    _alive = False
    __threadpool: ThreadPool = None

    def __init__(self):
        # When executing in a pod, load the incluster configuration according to
        # https://github.com/kubernetes-client/python/blob/master/examples/in_cluster_config.py#L21
        self._v1 = client.CoreV1Api()
        self._logger = logging.getLogger('ClusterManager')
        self._config = config.load_incluster_config()
        self._watchdog = ResourceWatchDog()
        self._client_handler = ClientHandler()

    def start(self):
        self._logger.info("Spinning up cluster manager...")
        # Set debugging to WARNING only, as otherwise DEBUG statements will flood the logs.
        client.rest.logger.setLevel(logging.WARNING)
        self._alive = True
        self.__thread_pool = ThreadPool(processes=2)
        self.__thread_pool.apply_async(self._watchdog.start)
        self.__thread_pool.apply_async(self._run)

    def _stop(self):
        self._logger.info("Stopping execution of ClusterManager, halting components...")
        self._watchdog.stop()
        self.__thread_pool.join()
        self._logger.info("Successfully stopped execution of ClusterManager")

    def _run(self):
        while self._alive:
            self._logger.info("Still alive...")
            time.sleep(10)

        self._stop()

    def _schedulable_task(self, train_task: TrainTask):
        current_resources = self._watchdog._resource_lookup

    def deploy_task(self):
        train_task: TrainTask = None


class DeploymentBuilder:
    _buildDescription = BuildDescription()

    def reset(self) -> None:
        self._buildDescription = BuildDescription()

    def create_identifier(self, identifier) -> None:
        # TODO: Move, or create identifier here.
        self._buildDescription.identifier = identifier

    def build_resources(self, mem_req, cpu_req, mem_lim, cpu_lim) -> None:
        def builder(memory, cpu) -> Dict[str, str]:
            return {'memory': memory, 'cpu': cpu}

        req_dict = builder(mem_req, cpu_req)
        lim_dict = builder(mem_lim, cpu_lim)
        self._buildDescription.container = client.V1ResourceRequirements(requests=req_dict,
                                                                         limits=lim_dict)

    def build_container(self, identifier: str = None) -> None:
        self._buildDescription.container = client.V1Container(
            name=f'client-{identifier or self._buildDescription.identifier}',
            image='localhost:5000/fltk',
            # TODO: Generate a means to start-up a
            command=["python3", "fltk/launch.py", "single",
                     "configs/cloud_experiment.yaml"],
            # TODO: Decide how to give client identifier.
            args=['hello world'],
            image_pull_policy='Always',
        )

    def build_template(self, restart_policy='Never') -> None:
        self._buildDescription.template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": "fltk-client"}),
            spec=client.V1PodSpec(restart_policy=restart_policy,
                                  containers=[self._buildDescription.container]))

    def build_spec(self, back_off_limit: int = 3, ttl_seconds: int = 30) -> None:
        self._buildDescription.spec = client.V1JobSpec(
            template=self._buildDescription.template,
            backoff_limit=back_off_limit,
            ttl_seconds_after_finished=ttl_seconds)

    def construct(self) -> client.V1Job:
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            # TODO: Decide whether to use this part of the functionality.
            metadata=client.V1ObjectMeta(name='helloworld'),
            spec=self._buildDescription.spec)
        return job


class ClientHandler(object):
    def __init__(self):
        self._v1 = client.CoreV1Api()

    def deploy_client(self, description):
        # API to exec with
        k8s_apps_v1 = client.AppsV1Api()
