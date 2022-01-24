from enum import Enum


class OffloadingStrategy(Enum):
    VANILLA = 1
    DEADLINE = 2
    SWYH = 3
    FREEZE = 4
    MODEL_OFFLOAD = 5,
    TIFL_BASIC = 6,
    TIFL_ADAPTIVE = 7

    @classmethod
    def Parse(cls, string_value):
        if string_value == 'vanilla':
            return OffloadingStrategy.VANILLA
        if string_value == 'deadline':
            return OffloadingStrategy.DEADLINE
        if string_value == 'swyh':
            return OffloadingStrategy.SWYH
        if string_value == 'freeze':
            return OffloadingStrategy.FREEZE
        if string_value == 'offload':
            return OffloadingStrategy.MODEL_OFFLOAD
        if string_value == 'tifl-basic':
            return OffloadingStrategy.TIFL_BASIC
        if string_value == 'tifl-adaptive':
            return OffloadingStrategy.TIFL_ADAPTIVE