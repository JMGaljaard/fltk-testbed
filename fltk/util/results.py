from dataclasses import dataclass
import numpy as np


@dataclass
class EpochData:
    """
    Dataclass describing statistic corresponding to a learning epoch.
    """
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


    def to_csv_line(self): # pylint: disable=missing-function-docstring
        delimeter = ','
        values = self.__dict__.values()
        values = [str(x) for x in values]
        return delimeter.join(values)
