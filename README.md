# FLTK - Federation Learning Toolkit
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)
[![Python 3.6](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.6](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

This toolkit is can be used to run Federated Learning experiments.
Pytorch Distributed ([docs](https://pytorch.org/tutorials/beginner/dist_overview.html)) is used in this project.
The goal if this project is to launch Federated Learning nodes in truly distribution fashion.

This project is tested with Ubuntu 20.04 and python {3.7, 3.8}.
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
TBD

</p>
</details>

## Known issues

* Currently, there is no GPU support docker containers (or docker compose)
* First epoch only can be slow (6x - 8x slower)