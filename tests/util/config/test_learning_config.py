import random
import unittest
from dataclasses import fields
from pathlib import Path

import yaml

from fltk.core.distributed.orchestrator import render_template
from fltk.util.config import DistLearningConfig, FedLearningConfig, get_safe_loader
from fltk.util.config.definitions import Optimizations
from fltk.util.task import FederatedArrivalTask, DistributedArrivalTask
from fltk.util.task.config import TrainTask, ExperimentParser
from fltk.util.task.generator.arrival_generator import Arrival

TEST_FED_CONF = './configs/test/fed_non_default.json'
TEST_DIST_CONF = './configs/test/dist_non_default.json'

TEST_PARAM_CONF_FEDERATED = './experiments/test/federated_non_default.yaml'
TEST_PARAM_CONF_DISTRIBUT = './experiments/test/data_parallel_non_default.yaml'

TEST_PARSED_CONF_FED = 'experiments/test/parsing/federated_parsed.yaml'
TEST_PARSED_CONF_DIST = 'experiments/test/parsing/data_parallel_parsed.yaml'


class FedLearningConfigTest(unittest.TestCase):

    test_dist_learn_param: DistLearningConfig = None

    default = FedLearningConfig(batch_size=128,
                                test_batch_size=128,
                                cuda=False,
                                scheduler_step_size=50,
                                scheduler_gamma=0.5,
                                min_lr=1e-10,
                                optimizer=Optimizations.sgd,
                                replication=-1)
    def setUp(self):
        self.learning_params = FedLearningConfig.from_yaml(Path(TEST_PARAM_CONF_FEDERATED))

    def test_excluded_non_defaults(self):
        exclude_set = {'log_level', 'num_clients', 'default_model_folder_path', 'data_path', 'rank', 'world_size', 'experiment_prefix'}
        for field in fields(self.default):
            if field.name not in exclude_set:
                self.assertNotEqual(getattr(self.default, field.name), getattr(self.learning_params, field.name), msg=field.name)

    def test_parsed_equals(self):
        description = ExperimentParser(config_path=Path(TEST_FED_CONF)).parse()
        job_description = description.train_tasks[0]
        train_task = TrainTask(identity='test_fed',
                               job_parameters=job_description.job_class_parameters[0],
                               priority=None,
                               replication=-1,
                               experiment_type=job_description.experiment_type,
                               seed=431)
        arrival_task = FederatedArrivalTask.build(Arrival(None, train_task, 'test_fed'), train_task.identifier, -1)
        template = render_template(arrival_task, 'Master', -1, TEST_FED_CONF)

        self.assertEquals(FedLearningConfig.from_yaml(Path(TEST_PARSED_CONF_FED)),
                          FedLearningConfig.from_dict(yaml.load(template, Loader=get_safe_loader())))


class DistLearningConfigTest(unittest.TestCase):

    test_dist_learn_param: DistLearningConfig = None


    def setUp(self):
        random.seed(42)
        self.learning_params = FedLearningConfig.from_yaml(Path(TEST_PARAM_CONF_DISTRIBUT))


    def test_parsed_equals(self):
        description = ExperimentParser(config_path=Path(TEST_DIST_CONF)).parse().train_tasks[0]
        train_task = TrainTask(identity='test_fed',
                  job_parameters=description.job_class_parameters[0],
                  priority=None,
                  replication=-1,
                  experiment_type=description.experiment_type,
                  seed=2053695854357871005)
        arrival_task = DistributedArrivalTask.build(Arrival(None, train_task, 'test_fed'), train_task.identifier, -1)
        template = render_template(arrival_task, 'Master', -1, TEST_FED_CONF)
        self.assertEquals(DistLearningConfig.from_yaml(Path(TEST_PARSED_CONF_DIST)),
                          DistLearningConfig.from_dict(yaml.load(template, Loader=get_safe_loader())))