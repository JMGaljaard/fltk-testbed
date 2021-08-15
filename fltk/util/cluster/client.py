import time
from typing import Dict

from kubernetes import client, watch

from fltk.util.cluster.conversion import Convert

watch.Watch()


class buildDescription:
    resources = client.V1ResourceRequirements
    container: client.V1Container
    template: client.V1PodTemplateSpec
    spec: client.V1JobSpec

class ResourceWatchDog:
    """
    Class to be used t
    """
    _alive: False
    _time: float
    resource_lookup: Dict[str, Dict[str, int]] = dict()

    def __init__(self):
        """
        Work should be based on the details listed here:
        https://github.com/scylladb/scylla-cluster-tests/blob/a7b09e69f0152a4d70bfb25ded3d75b7e7328acc/sdcm/cluster_k8s/__init__.py#L216-L223
        """
        self._v1 = client.CoreV1Api()
        self._w = watch.Watch()

    def start(self):
        """
        Function to start the resource watch dog. Currently, it only monitors the per-node memory and cpu availability.
        This does not handle event scheudling.
        @return:
        @rtype:
        """
        self._alive = True
        self.__monitor_allocatable_resources()

    def __monitor_allocatable_resources(self):
        """
        Watchdog function that streams the node
        @return:
        @rtype:
        """
        try:
            w = watch.Watch()
        except Exception as e:
            print(e)

        # TODO: See how fine grained this is. Otherwise, we miss a lot of events.
        # Alternative is to regularly poll this, or only when is needed. (Alternative).
        for event in w.stream(self._v1.list_node):
            print("Gettig")
            with event.get('object', None) as node_list:
                self.resource_lookup = {node.metadata.uid: {
                    "memory": Convert.memory(node.status.allocatable['memory']),
                    "cpu": Convert.cpu(node.status.allocatable['cpu'])} for node in node_list.items}
            self._time = time.time()
            if not self._alive:
                w.stop()


    def stale_timestamp(self):
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

    def __init__(self, cluster_config):
        self.config = cluster_config

    def deploy_client(self, description):
        # API to exec with
        k8s_apps_v1 = client.AppsV1Api()
