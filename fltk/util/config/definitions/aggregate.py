from enum import unique, Enum


@unique
class Aggregations(Enum):
    """Enum for Provided aggregation Types."""
    avg = 'Avg'
    fedavg = 'FedAvg'
    sum = 'Sum'
