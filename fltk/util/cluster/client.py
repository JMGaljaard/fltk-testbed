import logging
import time
from multiprocessing.pool import ThreadPool
from typing import Dict

from kubernetes import client, watch, config
from kubernetes.client import V1NodeList

from fltk.util.cluster.conversion import Convert


class buildDescription:
    resources = client.V1ResourceRequirements
    container: client.V1Container
    template: client.V1PodTemplateSpec
    spec: client.V1JobSpec


class ResourceWatchDog:
    """
    Class to be used to monitor the resources in a cluster.
    """
    _alive: False
    _time: float = -1
    resource_lookup: Dict[str, Dict[str, int]] = dict()

    def __init__(self):
        """
        Work should be based on the details listed here:
        https://github.com/scylladb/scylla-cluster-tests/blob/a7b09e69f0152a4d70bfb25ded3d75b7e7328acc/sdcm/cluster_k8s/__init__.py#L216-L223
        """
        self._v1: client.CoreV1Api

    def stop(self) -> None:
        """
        Function to stop execution. The runner thread _should_ merge back to the thread pool after calling this function
        to the thread pool.
        @return: None
        @rtype: None
        """
        logging.info("[WatchDog] Received request to stop execution")
        self._alive = False

    def start(self) -> None:
        """
        Function to start the resource watch dog. Currently, it only monitors the per-node memory and cpu availability.
        This does not handle event scheudling.
        @return: None
        @rtype: None
        """
        logging.info("Starting resource watchdog")
        self._alive = True
        self._v1 = client.CoreV1Api()
        self.__monitor_allocatable_resources()

    def __monitor_allocatable_resources(self) -> None:
        """
        Watchdog function that watches the Cluster resources in a K8s cluster. Requires the config to be set and loaded
        prior to calling.
        @return: None
        @rtype: None
        """

        try:
            # TODO: See how fine grained this is. Otherwise, we miss a lot of events.
            # Alternative is to regularly poll this, or only when is needed. (Alternative).
            logging.info("[WatchDog] Monitoring resources while alive...")
            while self._alive:
                node_list: client.V1NodeList = self._v1.list_node(watch=False)
                self.resource_lookup = {node.metadata.uid: {
                    "memory": Convert.memory(node.status.allocatable['memory']),
                    "cpu": Convert.cpu(node.status.allocatable['cpu'])} for node in node_list.items}
                logging.debug(f'[WatchDog] {self.resource_lookup}')
                if not self._alive:
                    logging.info("Instructed to stop, stopping list_node watch on Kubernetes.")
                    return
                time.sleep(1)

        except Exception as e:
            logging.error(e)
            raise e


class ClusterManager:
    _alive = False

    def __init__(self):
        # When executing in a pod, load the incluster configuration according to
        # https://github.com/kubernetes-client/python/blob/master/examples/in_cluster_config.py#L21
        self._v1 = client.CoreV1Api()
        self._config = config.load_incluster_config()
        self._watchdog = ResourceWatchDog()
        self._client_handler = ClientHandler()

    def start(self):
        logging.info("[ClusterManager] Spinning up cluster manager...")
        self._alive = True
        _thread_pool = ThreadPool(processes=2)
        _thread_pool.apply(self._watchdog.start)
        _thread_pool.apply(self._run)

        _thread_pool.join()

    def _stop(self):
        logging.info("[WatchDog] Stopping execution of ClusterManager")
        self._watchdog.stop()

    def _run(self):
        while self._alive:
            logging.info("Still alive...")
            time.sleep(10)

        self._stop()


class DeploymentBuilder:

    # TODO: build deployment configuration compatible with the minimal working example.
    def __init__(self):
        self._buildDescription = buildDescription()

    def reset(self):
        self._buildDescription = buildDescription()

    def create_identifier(self):
        # TODO: Move, or create identifier here.
        self._buildDescription.identifier = None

    def build_resources(self):
        self._buildDescription.container = client.V1ResourceRequirements(requests={},
                                                                         limits={})

    def build_container(self, identifier):
        self._buildDescription.container = client.V1Container(
            name=f'client-{identifier}',
            image='fltk',
            # TODO: Generate a means to start-up a
            command=["python3", "fltk/launch.py", "single",
                     "configs/cloud_experiment.yaml"],
            # TODO: Decide how to give client identifier.
            args=['hello world'],
            image_pull_policy='Always',
        )

    def build_template(self, restart_policy='Never'):
        self._buildDescription.template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"app": "fltk-client"}),
            spec=client.V1PodSpec(restart_policy=restart_policy,
                                  containers=[self._buildDescription.container]))

    def build_spec(self, back_off_limit=3):
        self._buildDescription.spec = client.V1JobSpec(
            template=self._buildDescription.template,
            backoff_limit=back_off_limit,
            ttl_seconds_after_finished=60)

    def construct(self):
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
