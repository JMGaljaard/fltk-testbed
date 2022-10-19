from plumbum.cmd import minikube, kubectl, helm, docker
from plumbum import local
from itertools import product
from time import sleep

import tomli_w
import os

def gen_job_config():
    epochs = [1, 5]
    std = [10]
    centre = [80]
    values = product(epochs, std, centre)
    names = ["epochs", "std", "centre"]
    return [dict(zip(names, value)) for value in values]


def gen_orch_config():
    sleep = [5]
    max_pods_per_node = [3, 10]
    values = product(sleep, max_pods_per_node)
    names = ["sleep", "max_pods_per_node"]
    return [dict(zip(names, value)) for value in values]

def gen_node_config():
    watt_usage = [40]
    watt_delta = [15]
    type_ = ["baremetal"]
    names = ["watt_usage", "watt_delta", "type"]
    values = product(watt_usage, watt_delta, type_)
    return [dict(zip(names, value)) for value in values]

def gen_resize_config():
    std = [10]
    centre = [40, 80]
    values = product(std, centre)
    names = ["std", "centre"]
    return [dict(zip(names, value)) for value in values]

def gen_configs():
    jobs = gen_job_config()
    orchs = gen_orch_config()
    nodes = gen_node_config()
    resizes = gen_resize_config()
    options = product(jobs, orchs, nodes, resizes)
    names = ["job", "orchestrator", "node", "resize"]
    return [dict(zip(names, option)) for option in options]


ITERATIONS = 1
DURATION = 80  # Time that the experiment is set to
INSTALL_CMD = """install flearner charts/orchestrator --namespace test -f charts/fltk-values.yaml --set-file orchestrator.experiment=./configs/distributed_tasks/example_arrival_config.json,orchestrator.configuration=./configs/example_cloud_experiment.json""".split(" ")

REGISTRY = "registry.gitlab.com/valentijn/fltk-testbed-qpe"

if __name__ == "__main__":
    configs = gen_configs()
    print(f"Total experiments to be run: {len(configs)}")

    print(minikube("start", "--driver=podman", "--container-runtime=cri-o"))

    import random
    for config in random.sample(configs, 10):
        for _ in range(ITERATIONS):
            toml_config = tomli_w.dumps(config)
            print("Running the following iteration:\n===========")
            print(toml_config)

            with open("configs/experiment.toml", "wb") as f:
                tomli_w.dump(config, f)

            print("Updating and pushing container")
            with local.env(DOCKER_BUILDKIT="1"):
                print(docker("build", "--platform", "linux/amd64", "-t", REGISTRY, "."))
                print(docker("push", REGISTRY))

            print("Uninstalling flearner")
            try:
                helm("uninstall", "-n", "test", "flearner")
            except:
                # Helm uninstall can fail whenever it is not already installed
                # this can safely be ignored
                pass
            print("Reinstalling flearner")
            helm(*INSTALL_CMD)
            print("experiment running")
            sleep(DURATION)
            print("result")
            print(kubectl("logs", "-n", "test", "fl-server"))

    print(minikube("stop"))
    
