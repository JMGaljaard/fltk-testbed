from __future__ import annotations
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Dict, List, Tuple, Optional, OrderedDict, Union
from uuid import UUID

import schedule
from kubeflow.training import V1ReplicaSpec, KubeflowOrgV1PyTorchJob, KubeflowOrgV1PyTorchJobSpec, V1RunPolicy
from kubernetes import client
from kubernetes.client import V1ObjectMeta, V1ResourceRequirements, V1Container, V1PodTemplateSpec, \
    V1VolumeMount, V1Toleration, V1Volume, V1PersistentVolumeClaimVolumeSource, V1ConfigMapVolumeSource

from fltk.util.cluster.conversion import Convert
from fltk.util.singleton import Singleton
from fltk.util.task.config.parameter import SystemResources
from fltk.util.task.task import DistributedArrivalTask, ArrivalTask, FederatedArrivalTask

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fltk.util.config import DistributedConfig

@dataclass
class Resource:
    """
    Dataclass describing a Kubernetes Pod's resources.
    """
    node_name: str
    cpu_allocatable: int
    memory_allocatable: int
    cpu_requested: int
    memory_requested: int
    cpu_limit: int
    memory_limit: int


@dataclass
class BuildDescription:
    """
    Dataclass containing intermediate step objects for creating a PytorchV1JobDescription. Used by the builder function.
    """
    resources = OrderedDict[str, V1ResourceRequirements]()
    typed_containers = OrderedDict[str, V1Container]()
    typed_templates = OrderedDict[str, V1PodTemplateSpec]()
    id: Optional[UUID] = None  # pylint: disable=invalid-name
    spec: Optional[KubeflowOrgV1PyTorchJobSpec] = None
    tolerations: Optional[List[V1Toleration]] = None


class ResourceWatchDog:
    """
    Class to be used to monitor the resources available within the cluster. For this the resource API is not needed, but
    can be used to extend/speedup/prettify the implementation. The implementation is based on the work by @gorenje found
    on GithHub:

    https://gist.github.com/gorenje/dff508489c3c8a460433ad709f14b7db

    N.B. this class acts as a starting point for scheduling based on resource availability.
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
        self._Q = Convert()  # pylint: disable=invalid-name

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
            time.sleep(5)

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
        except Exception as excep:
            self._logger.error(excep)
            raise excep

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
            except Exception as excep:  # pylint: disable=broad-except
                self._logger.error(f'Namespace lookup for {node_name} failed. Reason: {excep}')

        self._resource_lookup = new_resource_mapper
        self._logger.debug(self._resource_lookup)


class ClusterManager(metaclass=Singleton):
    """
    Object with basic monitoring functionality. This shows how the information of different Pods in a cluster can be
    requested and parsed. Currently, it mainly exists to start the ResourceWatchDog, which now only keeps track of the
    amount of resources requested/used in the cluster.
    """

    def __init__(self):
        # When executing in a pod, load the incluster configuration according to
        # https://github.com/kubernetes-client/python/blob/master/examples/in_cluster_config.py#L21
        self.__alive = False
        self.__thread_pool: ThreadPool = ThreadPool(processes=2)
        self._v1 = client.CoreV1Api()
        self._logger = logging.getLogger('ClusterManager')
        self._watchdog = ResourceWatchDog()

    def start(self):
        """
        Function to start ClusterManager to start keeping track of resources.
        @return: None
        @rtype: None
        """
        self._logger.info("Spinning up cluster manager...")
        # Set debugging to WARNING only, as otherwise DEBUG statements will flood the logs.
        client.rest.logger.setLevel(logging.WARNING)
        self.__alive = True
        self.__thread_pool.apply_async(self._watchdog.start)
        self.__thread_pool.apply_async(self._run)

    def stop(self):
        self._logger.info("Stopping execution of ClusterManager, halting components...")
        self._watchdog.stop()
        self.__alive = False
        self._logger.info("Successfully stopped execution of ClusterManager")

    def _run(self):
        while self.__alive:
            self._logger.debug("Still alive...")
            time.sleep(5)
        self._logger.info("Exiting ClusterManager loop.")



def _generate_command(config: DistributedConfig, task: ArrivalTask) -> List[str]:
    """
    Function to generate commands for containers to start working with. Either a federated learnign command
    will be realized, or a distributed learning command. Note that distributed learning commands will be revised
    in an upcomming version of KFLTK.
    @param config: DistributedConfiguration map with share
    @type config:
    @param task:
    @type task:
    @return:
    @rtype:
    """
    federated = isinstance(task, FederatedArrivalTask)
    if federated:
        command = 'python3 -m fltk remote experiments/node.config.yaml'
    else:
        command = (f'python3 -m fltk client experiments/node.config.yaml {task.id} '
                   f'{config.config_path} --backend gloo')
    return command.split(' ')


def _build_typed_container(conf: DistributedConfig, cmd: List[str], resources: V1ResourceRequirements,
                           name: str = "pytorch", requires_mount: bool = False,
                           experiment_name: str = None) -> V1Container:
    """
    Function to build the Master worker container. This requires the LOG PV to be mounted on the expected
    logging directory. Make sure that any changes in the Helm charts are also reflected here.
    @param conf: configuration object that contains specifics to properly start a client.
    @type conf: DistributedConfig
    @param name: Required name for deployment. Don't change unless you really need to.
    @type name: str
    @return: Unique experiment name, this will not be checked during deployment for uniqueness.
    @rtype: str
    """
    volume_mounts: Optional[List[V1VolumeMount]] = []
    if requires_mount:
        volume_mounts.append(V1VolumeMount(
                mount_path=f'/opt/federation-lab/{conf.get_log_dir()}',
                name='fl-log-claim',
                read_only=False
        ))

    volume_mounts.append(V1VolumeMount(
            mount_path='/opt/federation-lab/experiments',
            name=experiment_name,
            read_only=True
    ))
    # Mount the runtime configuration configmap as a volume to use shared configuration.
    volume_mounts.append(V1VolumeMount(
            mount_path='/opt/federation-lab/config',
            name='fltk-orchestrator-config-volume',
            read_only=True
    ))
    # Create mount for configuration
    container = V1Container(name=name,
                            image=conf.cluster_config.image,
                            command=cmd,
                            image_pull_policy='Always',
                            # Set the resources to the pre-generated resources
                            resources=resources,
                            volume_mounts=volume_mounts)
    return container


def _resource_dict(mem: Union[str, int], cpu: Union[str, int]) -> Dict[str, str]:
    """
    Private helper function to create a resource dictionary for deployments. Currently, only supports the creation
    of the requests/limits directory that is needed for a `V1ResourceRequirements` object.
    @param mem: Memory Request/Limit for a Container's ResourceRequirement.
    @type mem: str
    @param cpu: CPU Request/Limit for a Container's ResourceRequirement.
    @type cpu: int

    @return: Dictionary with resource requirements for a container (i.e. memory and cpu requests).
    @rtype: Dict[str, str]
    """
    return {'memory': f'{mem}', 'cpu': f'{cpu}'}


class DeploymentBuilder:
    """
    Builder class to build a V1PytorchJob. This is to make construction of the complex required object easier.
    Currently, it requires the object to be built in order. As such, re-calling a specific function will require all
    following functions to be called. This is due to the bottom-to-top construction of the configuration objects.

    A later version might refactor to a jinja templating approach to be more flexible for re-construction.
    """
    _build_description = BuildDescription()

    def reset(self) -> None:
        """
        Helper function to reset Build description, to start creating a new V1PytorchJob object.
        @return: None
        @rtype: None
        """
        del self._build_description
        self._build_description = BuildDescription()

    def build_resources(self, arrival_task: ArrivalTask) -> None:
        """
        Build resources for a V1PytorchJob, specificing the amount of RAM and CPU (cores) for the Pods to be spawned
        in the training job deployment.
        @param arrival_task: Arrival task containing the description of the Job that is being built.
        @type arrival_task: ArrivalTask
        @return: None
        @rtype: None
        """
        system_reqs: Dict[str, SystemResources] = arrival_task.named_system_params()
        for tpe, sys_reqs in system_reqs.items():
            typed_req_dict = _resource_dict(mem=sys_reqs.memory,
                                            cpu=sys_reqs.cores)
            # Note: currently, the request is set to the limits. You may want to change this.
            self._build_description.resources[tpe] = client.V1ResourceRequirements(requests=typed_req_dict,
                                                                                   limits=typed_req_dict)

    def build_container(self, task: ArrivalTask, conf: DistributedConfig,
                        configmap_name_dict: Optional[Dict[str, str]]):
        """
        Function to build container descriptions for deploying from within an Orchestrator pod.
        @param conf: configuration object that contains specifics to properly start a client.
        @type conf: DistributedConfig
        @param task: Learning task for which a job description must be made.
        @type task: DistributedArrivalTask
        @param configmap_name_dict: Mapping of pod names to their respective K8s configMap names.
        @type configmap_name_dict: Optional[Dict[str, str]]
        @return:
        @rtype:
        """
        # TODO: Implement cmd / config reference.
        for indx, (tpe, curr_resource) in enumerate(self._build_description.resources.items()):
            cmd = _generate_command(conf, task)
            container = _build_typed_container(conf, cmd, curr_resource,
                                               requires_mount=not indx,
                                               experiment_name=configmap_name_dict[tpe])
            self._build_description.typed_containers[tpe] = container

    def build_tolerations(self, tols: Optional[List[Tuple[str, Optional[str], str, str]]] = None,
                          specific_nodes: bool = True) -> None:
        """
        Function to set the V1Tolerations in the job. This allows for scheduling pods on specific Kubernetes Nodes that
        have specific Taints. Setting tols to an empyt list of `specific_nodes` to False, a Pod becomes schedulable on
        any Node that has enough resources for the Pod to be scheduled.
        @param tols: Toleration list.
        @type tols: Optional[List]
        @param specific_nodes: Boolean incidating whether specific nodes (i.e. Nodes with Taints) should be considered.
        Seting to False allows for running on any availabe Node with enough resources.
        @type specific_nodes: bool
        @return: None
        @rtype: None
        """
        if not tols:
            if specific_nodes:
                self._build_description.tolerations = [
                    V1Toleration(key="fltk.node",
                                 operator="Exists",
                                 effect="NoSchedule")]
            else:
                self._build_description.tolerations = []
        else:
            self._build_description.tolerations = \
                [V1Toleration(key=key, value=vl, operator=op, effect=effect) for key, vl, op, effect in tols]

    def build_template(self, configmap_name_dict: Optional[Dict[str, str]]) -> None:
        """
        Build Pod Template specs needed for PytorchJob object. This function will create the V1TemplateSpec
        with the required V1PodSpec  (including mounts fo volumes and containers, etc.). Requires other parameters to be
        set appropriately.
        @param configmap_name_dict: Mapping of pod names to their respective K8s configMap names.
        @type configmap_name_dict: Optional[Dict[str, str]]
        @return: None
        @rtype: None
        """
        # TODO: Add support for tolerations to use only affinity nodes to deploy to...
        # Ensure with taints that
        # https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/

        volumes = \
            [V1Volume(name="fl-log-claim",
                      persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(claim_name='fl-log-claim'))
             ]
        if configmap_name_dict:
            for tpe, tpe_config_map_name in configmap_name_dict.items():
                # Use default file permission 0644
                conf_map = V1ConfigMapVolumeSource(name=tpe_config_map_name)
                volumes.append(V1Volume(name=tpe_config_map_name,
                                        config_map=conf_map))
        volumes.append(V1Volume(name='fltk-orchestrator-config-volume',
                                config_map=V1ConfigMapVolumeSource(name='fltk-orchestrator-config')))
        for tpe, container in self._build_description.typed_containers.items():
            # TODO: Make this less hardcody
            self._build_description.typed_templates[tpe] = \
                client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(labels={"app": "fltk-worker"}),
                        spec=client.V1PodSpec(containers=[container],
                                              volumes=volumes,
                                              tolerations=self._build_description.tolerations))

    def build_spec(self, task: ArrivalTask, restart_policy: str = 'Never', clean_policy:  Optional[str] = None) -> None:
        """
        Function to build V1JobSpec object that contains the specifications of the Pods to be spawned in Kubernetes.
        Effectively this function creates the replica counts for the `Master` and `Worker` nodes in the train job
        that is being constructed.
        @param task: Arrival task containing the specifications of the replica counts.
        @type task: ArrivalTask
        @param restart_policy: Optional parameter to set the restart policy. Default behavior is to not restart pods by
        setting their restartPolicy to the option "Never".
        @type restart_policy: str
        @return: None
        @rtype: None
        """
        pt_rep_spec = OrderedDict[str, V1ReplicaSpec]()
        for tpe, tpe_template in self._build_description.typed_templates.items():
            typed_replica_spec = V1ReplicaSpec(
                    replicas=task.typed_replica_count(tpe),
                    restart_policy=restart_policy,
                    template=tpe_template)
            pt_rep_spec[tpe] = typed_replica_spec

        # N.B. This will not result in deleting pods.
        job_spec = KubeflowOrgV1PyTorchJobSpec(pytorch_replica_specs=pt_rep_spec,
                                               run_policy=V1RunPolicy(clean_pod_policy=clean_policy))
        self._build_description.spec = job_spec

    def construct(self) -> KubeflowOrgV1PyTorchJob:
        """
        Contruct V1PyTorch object following the description of the building process. Note that V1PyTorchJob differs
        slightly from a V1Job object in Kubernetes. Refer to the kubeflow documentation for more information on the
        PV1PyTorchJob object.
        @return: V1PyTorchJob object that was dynamically constructed.
        @rtype: V1PyTorchJob
        """
        job = KubeflowOrgV1PyTorchJob(
                api_version="kubeflow.org/v1",
                kind="PyTorchJob",
                metadata=V1ObjectMeta(name=f'trainjob-{self._build_description.id}', namespace='test'),
                spec=self._build_description.spec)
        return job

    def create_identifier(self, task: ArrivalTask):
        """
        Function to set the task identifier.
        @param task: Learning task for which a job description must be made.
        @type task: DistributedArrivalTask
        @return: None
        @rtype: None
        """
        self._build_description.id = task.id


def construct_job(conf: DistributedConfig, task: ArrivalTask,
                  configmap_name_dict: Optional[Dict[str, str]] = None) -> KubeflowOrgV1PyTorchJob:
    """
    Function to build a Job, based on the specifications of an ArrivalTask, and the general configuration of the
    DistributedConfig.
    @param conf: configuration object that contains specifics to properly start a client.
    @type conf: DistributedConfig
    @param task: Learning task for which a job description must be made.
    @type task: DistributedArrivalTask
    @param configmap_name_dict: Mapping of pod names to their respective K8s configMap names.
    @type configmap_name_dict: Optional[Dict[str, str]]
    @return: KubeFlow compatible KubeflowOrgV1PyTorchJob description to create a Job with the requested system and hyper-parameters.
    @rtype: KubeflowOrgV1PyTorchJob
    """
    dp_builder = DeploymentBuilder()
    dp_builder.create_identifier(task)
    dp_builder.build_resources(task)
    dp_builder.build_container(task, conf, configmap_name_dict)
    dp_builder.build_tolerations()
    dp_builder.build_template(configmap_name_dict)
    dp_builder.build_spec(task)
    job = dp_builder.construct()
    return job
