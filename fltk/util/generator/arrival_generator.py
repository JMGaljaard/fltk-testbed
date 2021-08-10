from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from random import choices
from typing import List, Union

import numpy as np
from iteration_utilities import deepflatten

from fltk.util.config.parameter import ExperimentParser, TrainTask


@dataclass
class ArrivalGenerator(ABC):
    """
    Abstract Base Class for generating arrivals in the system. These tasks must be run
    """
    configuration_path: Path

    @abstractmethod
    def load_config(self):
        pass

    @abstractmethod
    def generate_arrivals(self):
        pass


class ExperimentGenerator(ArrivalGenerator):
    start_time: int
    stop_time: int
    train_tasks: List[TrainTask]
    inter_arrival: Union[float]


    def load_config(self):
        """
        Generate
        """
        parser = ExperimentParser(config_path=self.configuration_path)
        experiment_descriptions = parser.parse()
        jobs = [[[TrainTask(params, description, priority) for priority in params.priorities]
                 for params in description.job_class_parameters] for description in experiment_descriptions]
        self.train_tasks = list(deepflatten(jobs))

    def generate_arrivals(self, tasks: int = 1) -> List[TrainTask]:
        self.inter_arrival = np.random.poisson(lam=self.train_tasks[0].arrival_statistic, size=tasks)
        return choices(population=self.train_tasks, weights=list(map(lambda x: x.probability, self.train_tasks)),
                       k=tasks)

@dataclass
class EvaluationGenerator(ArrivalGenerator):
    def load_config(self):
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
    experiment_generator.generate_arrivals(100)
