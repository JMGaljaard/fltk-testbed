import abc
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List

from iteration_utilities import deepflatten

from fltk.util.config.parameter import ExperimentParser, TrainTask


@dataclass
class ArrivalGenerator(ABC):
    """
    Abstract Base Class for generating arrivals in the system. These tasks must be run
    """
    configuration_path: Path

    @abc.abstractmethod
    def load_config(self, config_path: Path):
        pass

    @abc.abstractmethod
    def generate_arrivals(self):
        pass


class ExperimentGenerator(ArrivalGenerator):
    start_time: int
    stop_time: int
    train_tasks: List[TrainTask]

    def load_config(self):
        """
        Generate
        """
        parser = ExperimentParser(config_path=self.configuration_path)
        experiment_descriptions = parser.parse()
        jobs = [[[TrainTask(params, description, priority) for priority in params.priorities]
                 for params in description.job_class_parameters] for description in experiment_descriptions]
        self.train_tasks = list(deepflatten(jobs))

    def generate_arrivals(self):
        pass


@dataclass
class EvaluationGenerator(ArrivalGenerator):
    def load_config(self, config_path: Path):
        pass

    def generate_arrivals(self):
        pass


if __name__ == '__main__':
    experiment_path = '/home/jeroen/Documents/CSE/MSc/work/fltk/fltk-testbed-gr-30/configs/tasks/example_arrival_config.yaml'
    conf_path = Path(experiment_path)
    experiment_generator = ExperimentGenerator(conf_path)
    experiment_generator.load_config()
    for train_task in experiment_generator.train_tasks:
        print(train_task)

    print(sum(map(lambda x: x.probability, experiment_generator.train_tasks)))
