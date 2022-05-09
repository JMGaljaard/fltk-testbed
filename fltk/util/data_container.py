import csv

import numpy as np
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Type, TextIO


@dataclass
class DataRecord:
    pass


@dataclass
class FederatorRecord(DataRecord):
    num_selected_clients: int
    round_id: int
    round_duration: int
    test_loss: float
    test_accuracy: float
    # Accuracy per class?
    timestamp: float = time.time()
    node_name: str = ''
    confusion_matrix: np.array = None

@dataclass
class ClientRecord(DataRecord):
    round_id: int
    train_duration: float
    test_duration: float
    round_duration: float
    num_epochs: int
    trained_items: int
    accuracy: float
    train_loss: float
    test_loss: float
    # Accuracy per class?
    timestamp: float = time.time()
    node_name: str = ''
    confusion_matrix: np.array = None


class DataContainer:
    """
    Datacontainer class for collecting experiment data. By default, an 'Excel' compatible format is used by numpy and
    the csv library. As such, it is advised to use a library such as `pandas` to load data for analysis purposes.
    """
    records: List[DataRecord]
    file_name: str
    file_handle: TextIO
    file_path: Path
    append_mode: bool
    record_type: Type[DataRecord]
    delimiter = ','
    name: str

    def __init__(self, name: str, output_location: Path, record_type: Type[DataRecord], append_mode: bool = False):
        # print(f'Creating new Data container for client {name}')
        self.records = []
        self.file_name = f'{name}.csv'
        self.name = name
        output_location = Path(output_location)
        output_location.mkdir(parents=True, exist_ok=True)
        self.file_path = output_location / self.file_name
        self.append_mode = append_mode
        file_flag = 'a' if append_mode else 'w'
        self.file_handle = open(self.file_path, file_flag)
        print(f'[<=========>] Creating data container at {self.file_path}')
        self.record_type = record_type
        if self.append_mode:
            open(self.file_path, 'w').close()
            dw = csv.DictWriter(self.file_handle, self.record_type.__annotations__)
            dw.writeheader()
            self.file_handle.flush()

    def append(self, record: DataRecord):
        record.node_name = self.name
        self.records.append(record)
        if self.append_mode:
            dw = csv.DictWriter(self.file_handle, self.record_type.__annotations__)
            dw.writerow(record.__dict__)
            self.file_handle.flush()

    def save(self):
        """
        Function to save the encapsulated data to the experiment file. The format is 'excel' compatible,
        resulting in the capability of loading complex objects such as ndarrays as a field.
        @return: None
        @rtype: None
        """
        if self.append_mode:
            return
        import numpy as np
        np.set_printoptions(linewidth=10**6)
        dw = csv.DictWriter(self.file_handle, self.record_type.__annotations__)
        dw.writeheader()
        # print(f'Saving {len(self.records)} for node {self.name}')
        for record in self.records:
            record.node_name = self.name
            dw.writerow(record.__dict__)
        self.file_handle.flush()
