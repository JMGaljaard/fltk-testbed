import dataclasses
import unittest
from pathlib import Path
from typing import Union
from unittest.mock import patch

from parameterized import parameterized


from fltk.core.client import Client
from fltk.core.distributed import DistClient
from fltk.util.config import DistributedConfig, get_distributed_config, get_learning_param_config, FedLearningConfig, \
    DistLearningConfig

from fltk.datasets.dataset import Dataset as DS
from fltk.util.config.definitions import Nets, Dataset

TEST_DIST_CONF = './configs/test/test_experiment.json'
TEST_PARAM_CONF_PARALLEL = './experiments/test/data_parallel.yaml'
TEST_PARAM_CONF_FEDERATED = './experiments/test/federated.yaml'
MODEL_SET_PAIRING = [
    [Nets.cifar100_resnet, Dataset.cifar100],
    [Nets.cifar100_vgg, Dataset.cifar100],
    [Nets.cifar10_cnn, Dataset.cifar10],
    [Nets.cifar10_resnet, Dataset.cifar10],
    [Nets.fashion_mnist_cnn, Dataset.fashion_mnist],
    # [Nets.fashion_mnist_resnet, Dataset.fashion_mnist], # Fixme
    [Nets.mnist_cnn, Dataset.mnist]
]


def _limit_dataset(dist_client: Union[DistClient, Client]):
    dataset = dist_client.dataset
    dataset.test_loader = [next(iter(dataset.test_loader))]
    dataset.train_loader = [next(iter(dataset.train_loader))]


class TestLocalDistLearnerSmoke(unittest.TestCase):
    test_dist_config: DistributedConfig = None
    test_dist_learn_param: DistLearningConfig = None

    def setUp(self):
        self.test_dist_config = get_distributed_config(None, TEST_DIST_CONF)
        self.learning_params = get_learning_param_config(None, TEST_PARAM_CONF_PARALLEL)

    @parameterized.expand(
        [[f"{x.value}-{y.value}", x, y] for x, y in MODEL_SET_PAIRING]
    )
    def test_parallel_client(self, name, net: Nets, dataset: Dataset):
        self.learning_params = dataclasses.replace(self.learning_params, model=net, dataset=dataset)
        dist_client = DistClient(0, 'test-id', 1, self.test_dist_config, self.learning_params)
        dist_client.prepare_learner(distributed=False)

        _limit_dataset(dist_client)
        self.assertTrue(dist_client.run_epochs())


class TestFederatedLearnerSmoke(unittest.TestCase):
    learning_config: FedLearningConfig = None

    def setUp(self):
        self.learning_config = FedLearningConfig.from_yaml(Path(TEST_PARAM_CONF_FEDERATED))

    @parameterized.expand(
        [[f"{x.value}-{y.value}", x, y] for x, y in MODEL_SET_PAIRING]
    )
    def test_fed_client(self, name, net: Nets, dataset: Dataset):
        self.learning_config.net_name = net
        self.learning_config.dataset_name = dataset
        fed_client = Client('test-id', 1, 60000, self.learning_config)

        fed_client.init_dataloader()
        self.assertTrue(fed_client.is_ready())

        with patch.object(DS, 'get_train_dataset', fed_client.dataset):
            self.assertTrue(fed_client.exec_round(1, 0))

