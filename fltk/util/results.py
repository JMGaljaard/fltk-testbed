from dataclasses import dataclass
from typing import Any

@dataclass
class EpochData:
    epoch_id: int
    num_epochs: int
    duration_train: float
    duration_test: float
    loss_train: float
    accuracy: float
    loss: float
    class_precision: Any
    class_recall: Any
    training_process: int
    client_id: str = None

    def to_csv_line(self):
        delimeter = ','
        values = self.__dict__.values()
        values = [str(x) for x in values]
        return delimeter.join(values)
