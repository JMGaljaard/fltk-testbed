from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class EpochData:
    epoch_id: int
    num_epochs: int
    duration_train: float
    duration_test: float
    loss_train: float
    accuracy: float
    loss: float
    class_precision: np.array
    class_recall: np.array
    confusion_mat: np.array
    training_process: int
    client_id: str = None
    client_wall_time: float = 0
    global_wall_time: float = 0
    global_epoch_id: int = 0

    def to_csv_line(self):
        delimeter = ','
        values = self.__dict__.values()
        values = [str(x) for x in values]
        return delimeter.join(values)
