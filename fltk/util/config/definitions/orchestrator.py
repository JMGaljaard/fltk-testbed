from enum import unique, Enum


@unique
class OrchestratorType(Enum):
    BATCH = 'batch'
    SIMULATED = 'simulated'
