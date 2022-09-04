# Kubernetes - Federation Learning Toolkit ((K)FLTK)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

This toolkit can be used to run Distributed and Federated experiments. This project makes use of
Pytorch Distributed (Data Parallel) ([docs](https://pytorch.org/tutorials/beginner/dist_overview.html))
as well as Kubernetes, KubeFlow (Pytorch-Operator) ([docs](https://www.kubeflow.org/docs/)) and Helm ([docs](https://helm.sh/docs)) for deployment.
The goal of this project is to launch Federated Learning nodes in a true distribution fashion, with simple deployments 
using proven technology.

This project builds on the work by Bart Cox, on the Federated Learning toolkit developed to run with Docker and 
Docker Compose ([repo](https://github.com/bacox/fltk))


This project is tested with Ubuntu 20.04, Arch Linux, MacOS, with Python {3.7, 3.8, 3.9}. Python 3.9 is recommended.

## Global idea
Pytorch Distributed works based on a `world_size` and `rank`s. The ranks should be between `0` and `world_size-1`.
Generally, the process leading the learning process has rank `0` and the clients have ranks `[1,..., world_size-1]`.

Currently, it is assumed that Distributed Learning is performed (and *not* Federated Learning), however, future 
extension of the project is planned to implement a `FederatedClient` that allows for a more realistic simulation of 
*Federated* Learning experiments.

### (Distributed Learning)

**General protocol:**

1. Client creation and spawning by the Orchestrator (using KubeFlows Pytorch-Operator)
2. Clients prepare needed data and model and synchronize using PyTorch Distributed.
   1. `WORLD_SIZE = 1`: Client performs training locally. 
   2. `WORLD_SIZE > 1`: Clients run epochs with DistributedDataParallel together.
3. Client logs/reports progress during and after training.

**Important notes:**

* Data between clients (`WORLD_SIZE > 1`) is not shared
* Hardware can heterogeneous
* The location of devices matters (network latency and bandwidth)
* Communication is performed through RPC, aggregation is performed with `AllReduce`.

### Federated Learning
**General protocol:**

1. Client selection by the Federator.
2. The selected clients download the model.
3. Local training on the clients for X number of epochs
4. Weights/gradients of the trained model are send to the Federator
5. Federator aggregates the weights/gradients to create a new and improved model
6. Updated model is shared to the clients
7. Repeat step 1 to 6 until convergence/stopping condition.

**Important notes:**

* Data between clients is not shared to each other
* The data is non-IID
* Hardware can heterogeneous
* The location of devices matters (network latency and bandwidth)
* Communication can be costly




### Overview of deployed project
When deploying the system, the following diagram shows how the system operates. `PyTorchJob`s are launched by the 
Orchestrator (see the [Orchestrator charts](./charts/orchestrator)). The Extractor keeps track of progress (see the 
[Extractor charts](./charts/extractor)).

The `PyTorchJob`s can consist on a variable number of machines, with different hardware for the Master/Leader node and the
Client nodes. KubeFlow (not depicted) orchestrates the deployment of the `PyTorchJob`s.

![Overview of deployment](https://lucid.app/publicSegments/view/027793d8-a059-4c45-a030-660a492a4c0a/image.png)
## Something is broken/missing

It might be that something is missing, please open a pull request/issue).

## Project structure
Structure with important folders and files explained:

```
project
├── terraform                    # Contains terraform charts for deployment on GKE
├── jupyter                      # Contains jupyter notebook files for setup and loading tensorboard files 
├── charts                       # Templates for deploying projects with Helm 
│   ├── extractor                   - Template for 'extractor' for centralized logging (using NFS)
│   └── orchestrator                - Template for 'orchestrator' for launching distributed experiments 
├── configs                      # General configuration files
│   ├── quantities                  - Files for (Kubernetes) quantity conversion
│   └── tasks                       - Files for experiment description
├── data                         # Directory for default datasets (for a reduced load on hosting providers)
├── fltk                         # Source code
│   ├── datasets                    - Datasets (by default provide Pytorchs Distributed Data-Parallel)
│   ├── nets                        - Default models
│   ├── schedulers                  - Learningrate schedulers
│   ├── strategy                    - (Future) Basic strategies for Federated Learning experiments
│   └── util                        - Helper utilities
│       ├── cluster                    * Cluster interaction (Job Creation and monitoring) 
│       ├── config                     * Configuration file loading
│       └── task                       * Arrival/TrainTask generation
└── logging                      # Default logging location
```

## Execution modes
Federated Learning experiments can be set up in various ways (Simulation, Emulation, or fully distributed). Not all have the same requirements and thus some setup are more suited then others depending on the experiment.

### Simulation
With the method as single machine is used to execute all the different nodes in the system.
The execution is done in a sequential manner, i.e. first node 1 is executed, then node 2, and so on. One of the upsides of this option is the ability to use GPU acceleration for the computations.

### Docker-Compose (Emulation)
With systems like docker we can emulate a federated learning system on a single machine. Each node is allocated to one or more CPU cores and executed in an isolated container. This allows for real-time experiments where timing is important and where the execution of clients have effect on eachother. Docker also allows for containers to be limited by CPU speed, RAM, and network properties.

### Real distributed (Google Cloud)
In this case, the code is deployed natively on a machine, for example a cluster. 
The is the closest real-world approximation when experimenting with Federated Learning systems. This allows for real-time experiments where timing is important and where the execution of clients have effect on eachother. A downside of this method is the shear number of machines needed to run an experiment. Additionally the compute speed and other hardware spcifications are more difficult to limit.

### Hybrid
The Docker (Compose) and real-distributed method can be mixed in a hybrid system. For example two servers can run a set of docker containers that are linked to each other. Similarly, a set of docker images on a server can participate in a system with real distributed machines. 

## Models

* Cifar10-CNN (CIFAR10CNN)
* Cifar10-ResNet
* Cifar100-ResNet
* Cifar100-VGG
* Fashion-MNIST-CNN
* Fashion-MNIST-ResNet
* Reddit-LSTM

## Datasets

* CIFAR10
* Cifar100
* Fashion-MNIST
* MNIST

## Pre-requisites

The following tools need to be set up in your development environment before working with the (Kubernetes) FLTK.

* Hard requirements
  * Docker ([docs](https://www.docker.com/get-started)) (with support for BuildKit [docs](https://docs.docker.com/develop/develop-images/build_enhancements/))
  * Kubectl ([docs](https://kubernetes.io/docs/setup/))
  * Helm ([docs](https://helm.sh/docs/chart_template_guide/getting_started/))
  * (Terraform installation) Terraform
  * (Manual installation) Kustomize (3.2.0) ([docs](https://kubectl.docs.kubernetes.io/installation/kustomize/))
* Local execution (single machine):
  * MiniKube ([docs](https://minikube.sigs.k8s.io/docs/start/))
    * It must be noted that certain functionality might require additional steps to work on MiniKube. This is currently untested.
* Google Cloud Environment (GKE) execution:
  * GCloud SDK ([docs](https://cloud.google.com/sdk/docs/quickstart))
* Your own cluster provider:
  * A Kubernetes cluster supporting Kubernetes `>1.15,<=1.22`.

## Getting started

Before continuing a deployment, first, the used datasets need to be downloaded. This is done to prevent the need for
downloading each dataset for each container. Per default, these models are included in the Docker container that gets
deployed on a Kubernetes Cluster.

### Download datasets
To download the models, execute the following command from the [project root](.).

```bash
python3 -m fltk extractor ./configs/example_cloud_experiment.json  
```

## Deployment (Terraform)
To setup the the test-bed using Terraform, the following setup needs to be done. This can be achieved through following
the steps described in [`jupyter/terraform_notebook.ipynb`](jupyter/terraform_notebook.ipynb).

### Prerequisites

Before starting the jupyter notebook server locally, make sure to have the following dependencies installed.
We will create a virtual environment capable of running a jupyter notebook server with a `bash_kernel`.

For Windows users, make sure to run the following commands in a `bash` capable terminal, e.g. using 
Windows Subsystem for Linux (WSL).


```bash
python3 -m venv venv-jupyter
source venv-jupyter/bin/activate

# Install python dependencies for running the notebook
pip3 install jupyter ipython bash_kernel
# Install bash kernel to use for the notebook
python3 -m bash_kernel.install
```

When running the notebook (through an IDE or browser), make sure to set the kernel to the freshly installed
`bash_kernel`. Otherwise, the cells will be ran as Python code...

### Deploying cluster and dependencies

To start working in the notebook, run the following command in a bash shell, and follow the steps in the notebook.

```bash
cd jupyter
jupyter notebook
```

Click on the link that is displayed in the output, default is `localhost:8888`, and open the terraform notebook.

### Running the Extractor and Experiments

#### Creating and uploading Docker container 


The following commands will all (unless specified otherwise) be executed in the project root of the git repo.
Before we do so, first we need to setup a Python interpreter/environmen, this can also be used for development.


-   First we will create and active a Python venv.

    ``` bash
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    ```

-   Then we will download the datasets using a Python script in the same
    terminal (or another terminal with the `venv` activated).

    ``` bash
    python3 -m fltk extractor ./configs/example_cloud_experiment.json
    ```

Afterwards, we can run the following commands to build the Docker
container. With the use of BuildKit, consecutive builds allow to use cached requirements. Speeding
up your builds when adding Python dependencies to your project.

``` bash
DOCKER_BUILDKIT=1 docker build . --tag gcr.io/<project-id>/fltk
docker push gcr.io/<project-id>/fltk
```

#### Setting up the Extractor 

This section only needs to be run once, as this will set up the
TensorBoard service, as well as create the Volumes needed for the
deployment of the `Orchestrator`'s chart. It does, however, require you
to have pushed the docker container to a registry that can be accessed
from your Cluster.

N.B. that removing the `Extractor` chart will result in the deletion of
the Persistent Volumes once all Claims are released. This will remove
the data that is stored on these volumes. Make sure to copy the contents
of these directories to your local file system before uninstalling the
`Extractor` Helm chart. The following commands deploy the `Extractor`
Helm chart, under the name `extractor` in the `test` Namespace.\
Make sure to update the project name in the `chart` of the extractor in case you have changed
the default `PROJECT_ID`.

``` bash
cd charts
helm install extractor ./extractor -f fltk-values.yaml --namespace test
```

And wait for it to deploy. (Check with `helm ls –namespace test`)

N.B. To download data from the `Extractor` node (which mounts the
logging director), the following `kubectl` command can be used. This
will download the data in the logging directory to your file system.
Note that downloading many small files is slow (as they will be
compressed individually). The command assumes that the default name is
used `fl-extractor`.

``` bash
kubectl cp --namespace test fl-extractor:/opt/federation-lab/logging ./logging
```

Which will copy the data to a directory logging (you may have to create
this directory using `mkdir logging`).

#### Launching an experiment 

We have now completed the setup of the project and can continue by
running actual experiments.


##### Federated Experiment

```bash
helm install flearner charts/orchestrator --namespace test -f charts/fltk-values.yaml\
  --set-file orchestrator.experiment=./configs/federated_tasks/example_arrival_config.json,\
  orchestrator.configuration=./configs/example_cloud_experiment.json
```

##### Distributed Experiment

```bash
helm install flearner charts/orchestrator --namespace test -f charts/fltk-values.yaml\
  --set-file orchestrator.experiment=./configs/distributed_tasks/example_arrival_config.json,\
  orchestrator.configuration=./configs/example_cloud_experiment.json
```

## Deployment (manual)
Please refer to the wiki pages to see how to deploy manually. This has been replaced to rely on terraform in the future
(see above).

## Running tests
In addition to the FLTK framework implementation, some tests are available to prevent regression of bugs. Currently, only a limited subset of features 
is tested of FLTK. All current tests are deterministic, flaky tests indicate that something is likely broken.

To run the test included, run the following command to run the tests in a terminal:


```bash
python3 -m pytest tests
```

These tests should all pass (currently, mainly providing smoke tests). Warnings can be safely ignored.
### Prerequisites
Setup a `development` virtual environment, using the [`requirements-dev.txt`](requirements-dev.txt) requirements file.
This will install the same requirements as the [`requirements.txt`](requirements.txt), with some additional packages needed to run the tests.

```bash
python3 -m venv venv-dev
source venv-dev/bin/activate
pip install -r requirements.txt
```

### Executing tests
Make sure to run in a shell with the `venv-dev` virtual environment. With the environment enabled, we can run using:

```bash
python3 -m pytest -v
```

Which will collect and run all the tests in the repository, and show in `verbose` which tests passed. 


## Known issues / Limitations

* Currently, there is no GPU support in the Docker containers, for this the `Dockerfile` will need to be updated to
accommodate for this.
