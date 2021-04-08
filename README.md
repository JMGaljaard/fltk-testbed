# FLTK - Federation Learning Toolkit
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

This toolkit is can be used to run Federated Learning experiments.
Pytorch Distributed ([docs](https://pytorch.org/tutorials/beginner/dist_overview.html)) is used in this project.
The goal if this project is to launch Federated Learning nodes in truly distribution fashion.

## Project structure

TBD

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

[comment]: <> (```bash)

[comment]: <> (pip3 install -r ./requirements.txt)

[comment]: <> (```)

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
python3 -m fedsim spawn configs/experiment.yaml
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

```bash
docker-compose up
```

TBD

### Google Cloud Platform
TBD

</p>
</details>

## Known issues

* Currently, there is no GPU support. Not for native nor for docker compose
* First epoch only can be slow (6x - 8x slower)