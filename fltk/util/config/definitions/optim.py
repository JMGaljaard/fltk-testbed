from enum import unique, Enum


@unique
class Optimizations(Enum):
    adam = 'Adam'
    adam_w = 'AdamW'
    sgd = 'SGD'
    fedprox = 'FedProx'
    fednova = 'FedNova'