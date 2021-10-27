import subprocess
import itertools
import json
from kubernetes import client, config
import time
from google.cloud import container_v1

CLUSTER_NAMES = ['fltk-cluster']
PROJECT_NAME = 'qpefcs-course-project'
CONFIG_FILE = 'configs/tasks/example_arrival_config.json'

configs_set = [
    {
        "parallelism": 1, "executorCores": "3000m", "executorMemory": "4Gi", "network": "FashionMNISTCNN",
        "dataset": "FashionMNIST", "elastic_index": "rep_lazy_high_perform_fashion_no_paral", "lambda": 15, "maxEpoch": "5"
    },
    {
        "parallelism": 3, "executorCores": "3000m", "executorMemory": "4Gi", "network": "CIFAR10CNN",
        "dataset": "CIFAR10", "elastic_index": "rep_lazy_high_perform_cifar_paral", "lambda": 30, "maxEpoch": "1"
    }, {
        "parallelism": 3, "executorCores": "1000m", "executorMemory": "2Gi", "network": "FashionMNISTCNN",
        "dataset": "FashionMNIST", "elastic_index": "rep_lazy_low_perform_fashion_paral", "lambda": 15, "maxEpoch": "5"
    }, {
        "parallelism": 1, "executorCores": "1000m", "executorMemory": "2Gi", "network": "CIFAR10CNN",
        "dataset": "CIFAR10", "elastic_index": "rep_lazy_low_perform_cifar_no_paral", "lambda": 30, "maxEpoch": "1"
    },  {
        "parallelism": 3, "executorCores": "3000m", "executorMemory": "4Gi", "network": "FashionMNISTCNN",
        "dataset": "FashionMNIST", "elastic_index": "rep_predictive_high_perform_fashion_paral", "lambda": 30, "maxEpoch": "5"
    },     {
        "parallelism": 1, "executorCores": "3000m", "executorMemory": "4Gi", "network": "CIFAR10CNN",
        "dataset": "CIFAR10", "elastic_index": "rep_predictive_high_perform_cifar_no_paral", "lambda": 30, "maxEpoch": "1"
    },     {
        "parallelism": 1, "executorCores": "1000m", "executorMemory": "2Gi", "network": "FashionMNISTCNN",
        "dataset": "FashionMNIST", "elastic_index": "rep_predictive_low_perform_fashion_no_paral", "lambda": 15, "maxEpoch": "5"
    }, {
        "parallelism": 3, "executorCores": "1000m", "executorMemory": "2Gi", "network": "CIFAR10CNN",
        "dataset": "CIFAR10", "elastic_index": "rep_predictive_low_perform_cifar_paral", "lambda": 30, "maxEpoch": "1"
    }
]


for config_set in configs_set:

    with open(CONFIG_FILE, 'r') as json_file:
        conf = json.load(json_file)
        conf[0]['jobClassParameters'][0]['networkConfiguration']['network'] = config_set['network']
        conf[0]['jobClassParameters'][0]['networkConfiguration']['dataset'] = config_set['dataset']
        conf[0]['elasticsearchIndex'] = config_set['elastic_index']
        conf[0]['lambda'] = config_set['lambda']
        conf[0]['jobClassParameters'][0]['systemParameters']['dataParallelism'] = config_set['parallelism']
        conf[0]['jobClassParameters'][0]['systemParameters']['executorCores'] = config_set['executorCores']
        conf[0]['jobClassParameters'][0]['systemParameters']['executorMemory'] = config_set['executorMemory']
        conf[0]['jobClassParameters'][0]['hyperParameters']['maxEpoch'] = config_set['maxEpoch']

    with open(CONFIG_FILE, 'w') as json_file:
        json.dump(conf, json_file, indent=4)

    start = time.time()

    # building the docker container
    cmd = 'DOCKER_BUILDKIT=1 docker build . --tag gcr.io/{}/fltk'.format(PROJECT_NAME)
    subprocess.run(cmd, shell=True)

    # pushing the docker container
    cmd = 'docker push gcr.io/{}/fltk'.format(PROJECT_NAME)
    subprocess.run(cmd, shell=True)

    # cd into charts and install the extractor
    cmd = 'helm uninstall extractor -n test'
    subprocess.run(cmd, shell=True, cwd='charts/')

    # cd into charts and install the extractor
    cmd = 'helm uninstall orchestrator -n test'
    subprocess.run(cmd, shell=True, cwd='charts/')

    cmd = r"""kubectl patch pvc fl-server-claim --namespace test -p '{"metadata":{"finalizers": []}}' --type=merge"""
    subprocess.run(cmd, shell=True, cwd='charts/')

    cmd = r"""kubectl patch pvc fl-log-claim --namespace test -p '{"metadata":{"finalizers": []}}' --type=merge"""
    subprocess.run(cmd, shell=True, cwd='charts/')

    cmd = 'helm upgrade --install extractor ./extractor -f fltk-values.yaml --namespace test'
    subprocess.run(cmd, shell=True, cwd='charts/')

    cmd = 'helm upgrade --install orchestrator ./orchestrator --namespace test -f fltk-values.yaml'
    subprocess.run(cmd, shell=True, cwd='charts/')

    config.load_kube_config()
    v1 = client.CoreV1Api()
    print("Listing pods with their IPs:")
    pod_status_failed = False
    time.sleep(100)

    while not pod_status_failed:
        pod_list = v1.list_namespaced_pod("test")
        fl_server_pod = \
        list(filter(lambda item: item.metadata.labels.get('fltk.service') == 'fl-server', pod_list.items))[0]
        pod_status_failed = fl_server_pod.status.phase == 'Failed'
        end = time.time()
        if end - start >= 60 * 45:
            break
        if not pod_status_failed:
            time.sleep(100)

    cmd = 'helm uninstall extractor -n test'
    subprocess.run(cmd, shell=True, cwd='charts/')

    # cd into charts and install the extractor
    cmd = 'helm uninstall orchestrator -n test'
    subprocess.run(cmd, shell=True, cwd='charts/')

    cluster_manager_client = container_v1.ClusterManagerClient()
    cluster_manager_client.set_node_pool_size({"node_pool_id": "pool-1", "node_count": 4, "zone": "us-central1-c",
                                               "project_id": "qpefcs-course-project", "cluster_id": "flfk-cluster"})

    time.sleep(300)
    break
