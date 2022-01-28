from enum import Enum


class OffloadingStrategy(Enum):
    VANILLA = 1
    DEADLINE = 2
    SWYH = 3
    FREEZE = 4
    MODEL_OFFLOAD = 5,
    TIFL_BASIC = 6,
    TIFL_ADAPTIVE = 7,
    DYN_TERMINATE = 8,
    DYN_TERMINATE_SWYH = 9,
    MODEL_OFFLOAD_STRICT = 10,
    MODEL_OFFLOAD_STRICT_SWYH = 11

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
        if string_value == 'dynamic-terminate':
            return OffloadingStrategy.DYN_TERMINATE
        if string_value == 'dynamic-terminate-swyh':
            return OffloadingStrategy.DYN_TERMINATE_SWYH
        if string_value == 'offload-strict':
            return OffloadingStrategy.MODEL_OFFLOAD_STRICT
        if string_value == 'offload-strict-swyh':
            return OffloadingStrategy.MODEL_OFFLOAD_STRICT_SWYH


def parse_strategy(strategy: OffloadingStrategy):
    deadline_enabled = False
    swyh_enabled = False
    freeze_layers_enabled = False
    offload_enabled = False
    dyn_terminate = False
    dyn_terminate_swyh = False
    if strategy == OffloadingStrategy.VANILLA:
        deadline_enabled = False
        swyh_enabled = False
        freeze_layers_enabled = False
        offload_enabled = False
    if strategy == OffloadingStrategy.DEADLINE:
        deadline_enabled = True
        swyh_enabled = False
        freeze_layers_enabled = False
        offload_enabled = False
    if strategy == OffloadingStrategy.SWYH:
        deadline_enabled = True
        swyh_enabled = True
        freeze_layers_enabled = False
        offload_enabled = False
    if strategy == OffloadingStrategy.FREEZE:
        deadline_enabled = True
        swyh_enabled = False
        freeze_layers_enabled = True
        offload_enabled = False
    if strategy == OffloadingStrategy.MODEL_OFFLOAD:
        deadline_enabled = True
        swyh_enabled = False
        freeze_layers_enabled = True
        offload_enabled = True
    if strategy == OffloadingStrategy.DYN_TERMINATE:
        deadline_enabled = False
        swyh_enabled = False
        freeze_layers_enabled = False
        offload_enabled = False
        dyn_terminate = True
    if strategy == OffloadingStrategy.DYN_TERMINATE_SWYH:
        deadline_enabled = False
        swyh_enabled = False
        freeze_layers_enabled = False
        offload_enabled = False
        dyn_terminate_swyh = True
    if strategy == OffloadingStrategy.MODEL_OFFLOAD_STRICT:
        deadline_enabled = True
        swyh_enabled = True
        freeze_layers_enabled = True
        offload_enabled = True
    if strategy == OffloadingStrategy.MODEL_OFFLOAD_STRICT_SWYH:
        deadline_enabled = True
        swyh_enabled = True
        freeze_layers_enabled = True
        offload_enabled = True
    return deadline_enabled, swyh_enabled, freeze_layers_enabled, offload_enabled, dyn_terminate, dyn_terminate_swyh
