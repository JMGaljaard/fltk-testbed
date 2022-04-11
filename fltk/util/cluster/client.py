import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Dict, List, Tuple, Optional, OrderedDict, Union
from uuid import UUID

import schedule
from kubeflow.pytorchjob import V1PyTorchJob, V1ReplicaSpec, V1PyTorchJobSpec
from kubernetes import client
from kubernetes.client import V1ObjectMeta, V1ResourceRequirements, V1Container, V1PodTemplateSpec, \
    V1VolumeMount, V1Toleration, V1Volume, V1PersistentVolumeClaimVolumeSource, V1ConfigMapVolumeSource, V1KeyToPath

from fltk.util.cluster.conversion import Convert
from fltk.util.config import DistributedConfig
from fltk.util.singleton import Singleton
from fltk.util.task.config.parameter import SystemResources
from fltk.util.task.task import DistributedArrivalTask, ArrivalTask


@dataclass
class Resource:
    node_name: str
    cpu_allocatable: int
    memory_allocatable: int
    cpu_requested: int
    memory_requested: int
    cpu_limit: int
    memory_limit: int


@dataclass
class BuildDescription:
    resources = OrderedDict[str, V1ResourceRequirements]()
    typed_containers = OrderedDict[str, V1Container]()
    typed_templates = OrderedDict[str, V1PodTemplateSpec]()
    id: Optional[UUID] = None
    spec: Optional[V1PyTorchJobSpec] = None
    tolerations: Optional[List[V1Toleration]] = None


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
        self._v1: client.CoreV1Api
        self._logger = logging.getLogger('ResourceWatchDog')
        self._Q = Convert()

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
        Watchdog function that watches the Cluster resources in a K8s cluster. Requires the conf to be set and loaded
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
        for node_name, node in self._node_lookup.items():
            try:

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
    """
    Object to potentially further extend. This shows how the information of different Pods in a cluster can be
    requested and parsed. Currently, it mainly exists to start the ResourceWatchDog, which now only logs the amount of
    resources...
    """
    __alive = False
    __threadpool: ThreadPool = None

    def __init__(self):
        # When executing in a pod, load the incluster configuration according to
        # https://github.com/kubernetes-client/python/blob/master/examples/in_cluster_config.py#L21
        self._v1 = client.CoreV1Api()
        self._logger = logging.getLogger('ClusterManager')
        self._watchdog = ResourceWatchDog()

    def start(self):
        self._logger.info("Spinning up cluster manager...")
        # Set debugging to WARNING only, as otherwise DEBUG statements will flood the logs.
        client.rest.logger.setLevel(logging.WARNING)
        self.__alive = True
        self.__thread_pool = ThreadPool(processes=2)
        self.__thread_pool.apply_async(self._watchdog.start)
        self.__thread_pool.apply_async(self._run)

    def _stop(self):
        self._logger.info("Stopping execution of ClusterManager, halting components...")
        self._watchdog.stop()
        self.__alive = False
        self.__thread_pool.join()
        self._logger.info("Successfully stopped execution of ClusterManager")

    def _run(self):
        while self.__alive:
            self._logger.info("Still alive...")
            time.sleep(10)

        self._stop()


def _generate_command(config: DistributedConfig, task: ArrivalTask, federated=True) -> List[str]:
    """
    Function to generate commands for containers to start working with. Either a federated learnign command
    will be realized, or a distributed learning command. Note that distributed learning commands will be revised
    in an upcomming version of KFLTK.
    @param config:
    @type config:
    @param task:
    @type task:
    @param federated:
    @type federated:
    @return:
    @rtype:
    """
    if not federated:
        command = (f'python3 -m fltk client {config.config_path} {task.id} '
                   f'--model {task.network} --dataset {task.dataset} '
                   f'--optimizer Adam --max_epoch {task.param_conf.max_epoch} '
                   f'--batch_size {task.param_conf.bs} --learning_rate {task.param_conf.lr} '
                   f'--decay {task.param_conf.lr_decay} --loss CrossEntropy '
                   f'--backend gloo')
    else:
        command = (f'python3 -m fltk remote experiments/node.config.yaml')
    return command.split(' ')


def _build_typed_container(conf: DistributedConfig, cmd: List[str], resources: V1ResourceRequirements,
                           name: str = "pytorch", requires_mount: bool = False, experiment_name: str = None) -> V1Container:
    """
    Function to build the Master worker container. This requires the LOG PV to be mounted on the expected
    logging directory. Make sure that any changes in the Helm charts are also reflected here.
    @param name:
    @type name:
    @return:
    @rtype:
    """
    mount_list: Optional[List[V1VolumeMount]] = []
    if requires_mount:
        mount_list.append(V1VolumeMount(
                mount_path=f'/opt/federation-lab/{conf.get_log_dir()}',
                name='fl-log-claim',
                read_only=False
        ))
    # TODO: Mount volume

    # mount_list.append(V1VolumeMount(
    #             mount_path=f'/opt/federation-lab/experiments',
    #             name='experiment',
    #             read_only=True
    # ))
    # Create mount for configuration
    container = V1Container(name=name,
                            image=conf.cluster_config.image,
                            command=cmd,
                            image_pull_policy='Always',
                            # Set the resources to the pre-generated resources
                            resources=resources,
                            volume_mounts=mount_list)
    return container


def _resource_dict(mem: Union[str, int], cpu: Union[str, int]) -> Dict[str, str]:
    """
    Private helper function to create a resource dictionary for deployments. Currently only supports the creation
    of the requests/limits directory that is needed for a V1ResoruceRequirements object.
    @param mem: Memory Request/Limit for a Container's ResoruceRequirement
    @type mem: str
    @param cpu: CPU Request/Limit for a Container's ResourceRequirement.
    @type cpu: int

    @return:
    @rtype:
    """
    return {'memory': f'{mem}', 'cpu': f'{cpu}'}


class DeploymentBuilder:
    _buildDescription = BuildDescription()

    def reset(self) -> None:
        del self._buildDescription
        self._buildDescription = BuildDescription()

    def build_resources(self, arrival_task: ArrivalTask) -> None:
        system_reqs: Dict[str, SystemResources] = arrival_task.named_system_params()
        for tpe, sys_reqs in system_reqs.items():
            typed_req_dict = _resource_dict(mem=sys_reqs.memory,
                                            cpu=sys_reqs.cores)
            # Currently the request is set to the limits. You may want to change this.
            self._buildDescription.resources[tpe] = client.V1ResourceRequirements(requests=typed_req_dict,
                                                                                  limits=typed_req_dict)

    def build_container(self, task: ArrivalTask, conf: DistributedConfig, config_name_dict: Optional[Dict[str, str]]):
        """
        Function to build container descriptions for deploying from within an Orchestrator pod.
        @param task:
        @type task:
        @param conf:
        @type conf:
        @return:
        @rtype:
        """
        # TODO: Implement cmd / config reference.
        cmd = _generate_command(conf, task)
        for indx, (tpe, curr_resource) in enumerate(self._buildDescription.resources.items()):
            self._buildDescription.typed_containers[tpe] = _build_typed_container(conf, cmd, curr_resource,
                                                                                  requires_mount=not indx,
                                                                                  experiment_name=config_name_dict[tpe])

    def build_tolerations(self, tols: List[Tuple[str, Optional[str], str, str]] = None):
        if not tols:
            self._buildDescription.tolerations = [
                V1Toleration(key="fltk.node",
                             operator="Exists",
                             effect="NoSchedule")]
        else:
            self._buildDescription.tolerations = \
                [V1Toleration(key=key, value=vl, operator=op, effect=effect) for key, vl, op, effect in tols]

    def build_template(self, config_name_dict: Optional[Dict[str, str]]) -> None:
        """

        @return:
        @rtype:
        """
        # TODO: Add support for tolerations to use only affinity nodes to deploy to...
        # Ensure with taints that
        # https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/

        volumes = \
            [V1Volume(name="fl-log-claim",
                      persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name='fl-log-claim'))
             ]
        # if config_name_dict:
        #     for tpe, tpe_config_map_name in config_name_dict.items():
        #         volumes.append(V1Volume(name='experiment',
        #                  config_map=V1ConfigMapVolumeSource(tpe_config_map_name)))
        for tpe, container in self._buildDescription.typed_containers.items():
            # TODO: Make this less hardcody
            self._buildDescription.typed_templates[tpe] = \
                client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(labels={"app": "fltk-worker"}),
                        spec=client.V1PodSpec(containers=[container],
                                              volumes=volumes,
                                              tolerations=self._buildDescription.tolerations))

    def build_spec(self, task: ArrivalTask, restart_policy: str = 'OnFailure') -> None:
        pt_rep_spec = OrderedDict[str, V1ReplicaSpec]()
        for tpe, tpe_template in self._buildDescription.typed_templates.items():
            typed_replica_spec = V1ReplicaSpec(
                    replicas=task.typed_replica_count(tpe),
                    restart_policy=restart_policy,
                    template=tpe_template)
            typed_replica_spec.openapi_types = typed_replica_spec.swagger_types
            pt_rep_spec[tpe] = typed_replica_spec

        job_spec = V1PyTorchJobSpec(pytorch_replica_specs=pt_rep_spec)
        job_spec.openapi_types = job_spec.swagger_types
        self._buildDescription.spec = job_spec

    def construct(self) -> V1PyTorchJob:
        """
        Contruct V1PyTorch object following the description of the building process. Note that V1PyTorchJob differs
        slightly from a V1Job object in Kubernetes. Refer to the kubeflow documentation for more information on the
        PV1PyTorchJob object.
        @return: V1PyTorchJob object that was dynamically constructed.
        @rtype: V1PyTorchJob
        """
        job = V1PyTorchJob(
                api_version="kubeflow.org/v1",
                kind="PyTorchJob",
                metadata=V1ObjectMeta(name=f'trainjob-{self._buildDescription.id}', namespace='test'),
                spec=self._buildDescription.spec)
        return job

    def create_identifier(self, task: DistributedArrivalTask):
        self._buildDescription.id = task.id


def construct_job(conf: DistributedConfig, task: DistributedArrivalTask,
                  config_name_dict: Optional[Dict[str, str]] = None) -> V1PyTorchJob:
    """
    Function to build a Job, based on the specifications of an ArrivalTask, and the general configuration of the
    BareConfig.
    @param conf: configuration object that contains specifics to properly start a client.
    @type conf: DistributedConfig
    @param task: Learning task for which a job description must be made.
    @type task: DistributedArrivalTask
    @return: KubeFlow compatible PyTorchJob description to create a Job with the requested system and hyper parameters.
    @rtype: V1PyTorchJob
    """
    dp_builder = DeploymentBuilder()
    dp_builder.create_identifier(task)
    dp_builder.build_resources(task)
    dp_builder.build_container(task, conf, config_name_dict)
    dp_builder.build_tolerations()
    dp_builder.build_template(config_name_dict)
    dp_builder.build_spec(task)
    job = dp_builder.construct()
    # Fix to deploy on more up-to-date Kubernetes clusters. See if needed for KubeFlow operator release.
    job.openapi_types = job.swagger_types
    return job
