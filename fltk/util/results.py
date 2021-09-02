from dataclasses import dataclass

import numpy as np


@dataclass
class EpochData:
    duration_train: int
    duration_test: int
    loss_train: float
    accuracy: float
    loss: float
    class_precision: np.array
    class_recall: np.array
    confusion_mat: np.array
    client_id: str = None

    def to_csv_line(self):
        delimeter = ','
        values = self.__dict__.values()
        values = [str(x) for x in values]
        return delimeter.join(values)
