# FLTK - Federation Learning Toolkit
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)
[![Python 3.6](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.6](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

This toolkit is can be used to run Federated Learning experiments.
Pytorch Distributed ([docs](https://pytorch.org/tutorials/beginner/dist_overview.html)) is used in this project.
The goal if this project is to launch Federated Learning nodes in truly distribution fashion.

This project is tested with Ubuntu 20.04 and python {3.7, 3.8}.



### Building locally

To build locally, run the following command in the project root directory.

```bash 
DOCKER_BUILDKIT=1 docker build . --tag fltk
```
### Global idea
Pytorch distributed works based on a world_size and ranks. The ranks should be between 0 and world_size-1.
Generally, the federator has rank 0 and the clients have ranks between 1 and world_size-1.

General protocol:

1. Client selection by the federator
2. The selected clients download the model.
2. Local training on the clients for X number of epochs
3. Weights/gradients of the trained model are send to the federator
4. Federator aggregates the weights/gradients to create a new and improved model
5. Updated model is shared to the clients
6. Repeat step 1 to 5 until convergence

Important notes:

* Data between clients is not shared to each other
* The data is non-IID
* Hardware can heterogeneous
* The location of devices matters (network latency and bandwidth)
* Communication can be costly

## Project structure
Structure with important folders and files explained:
```
project
├── configs
│     └── experiment.yaml                     # Example of an experiment configuration
├── deploy                                    # Templates for automatic deployment  
│     └── templates
│          ├── client_stub_default.yml
│          ├── client_stub_medium.yml
│          ├── client_stub_slow.yml
│          └── system_stub.yml                # Describes the federator and the network
├── fltk                                      # Source code
│     ├── datasets                            # Different dataset definitions
│     │    ├── data_distribution              # Datasets with distributed sampler
│     │    └── distributed                    # "regular" datasets for centralized use
│     ├── nets                                # Available networks
│     ├── schedulers                          # Learning Rate Schedulers
│     ├── strategy                            # Client selection and model aggregation algorithms
│     └── util
│          └── generate_docker_compose.py     # Generates a docker-compose.yml for a containerized run
├── Dockerfile                                # Dockerfile to run in containers
├── LICENSE
├── README.md
└── setup.py
```

## Models

* Cifar10-CNN
* Cifar10-ResNet
* Cifar100-ResNet
* Cifar100-VGG
* Fashion-MNIST-CNN
* Fashion-MNIST-ResNet
* Reddit-LSTM

## Datasets

* Cifar10
* Cifar100
* Fashion-MNIST

## Prerequisites

When running in docker containers the following dependencies need to be installed:

* Docker
* Docker-compose

## Install
```bash
python3 setup.py install
```

### Load models
```bash
python3 fltk/util/default_models.py
```

## Examples
<details><summary>Show Examples</summary>

<p>

### Single machine (Native)

#### Launch single client
Launch Federator
```bash
python3 -m fltk single configs/experiment.yaml --rank=0
```
Launch Client
```bash
python3 -m fltk single configs/experiment.yaml --rank=1
```

#### Spawn FL system
```bash
python3 -m fltk spawn configs/experiment.yaml
```

### Two machines (Native)
To start a cross-machine FL system you have to configure the network interface connected to your network.
For example, if your machine is connected to the network via the wifi interface (for example with the name `wlo1`) this has to be configured as shown below:
```bash
os.environ['GLOO_SOCKET_IFNAME'] = 'wlo1'
os.environ['TP_SOCKET_IFNAME'] = 'wlo1'
```
Use `ifconfig` to find the name of the interface name on your machine.

### Docker Compose
1. Make sure docker and docker-compose are installed.
2. Generate a `docker-compose.yml` file for your experiment. You can use the script `generate_docker_compose.py` for this.
   From the root folder: ```python3 fltk/util/generate_docker_compose.py 4``` to generate a system with 4 clients.
   Feel free to change/extend `generate_docker_compose.py` for your own need.
   A `docker-compose.yml` file is created in the root folder.
3. Run docker-compose to start the system:
    ```bash
    docker-compose up
    ```
### Google Cloud Platform
See Manual on brightspace

</p>
</details>


### Deploying on Kubernetes


#### Setting up environment

 * Kustomize 3.9.0
 * Helm
 * Kubectl
 * gcloud sdk

#### Setting up cluster


#### MiniKube

#### GKE

### Installing KubeFlow
Kubeflow is a ML toolkit that allows to perform a multitude of machine and deep learning operations no 
Kubernetes clusters. For this we will make use of the 1.3 release.


```bash
git clone ... --branch=v1.3-branch
cd manifitesst
```

You might want to read the `README` file for more information. We will make use of the Kustomize files provided
by KubeFlow to install a basic KubeFlow instance on the cluster. If you have already worked with KubeFlow on GKE
you might want to follow the GKE deployment on the official KubeFlow documentation.

```bash
kustomize build common/cert-manager/cert-manager/base | kubectl apply -f -
# Wait before executing the following command, as 
kustomize build common/cert-manager/kubeflow-issuer/base | kubectl apply -f -
```


```bash
kustomize build common/istio-1-9/istio-crds/base | kubectl apply -f -
kustomize build common/istio-1-9/istio-namespace/base | kubectl apply -f -
kustomize build common/istio-1-9/istio-install/base | kubectl apply -f -
```

```bash
kustomize build common/dex/overlays/istio | kubectl apply -f -
```

```bash
kustomize build common/oidc-authservice/base | kubectl apply -f -
```

```bash

kustomize build common/knative/knative-serving/base | kubectl apply -f -
kustomize build common/istio-1-9/cluster-local-gateway/base | kubectl apply -f -
```


```bash
kustomize build common/kubeflow-namespace/base | kubectl apply -f -
```

```bash
kustomize build common/kubeflow-roles/base | kubectl apply -f -
kustomize build common/istio-1-9/kubeflow-istio-resources/base | kubectl apply -f -
```

```bash
kustomize build apps/pytorch-job/upstream/overlays/kubeflow | kubectl apply -f -
```
#### 
## Known issues

* Currently, there is no GPU support docker containers (or docker compose)
* First epoch only can be slow (6x - 8x slower)