import numpy as np


def tifl_select_tier(tiers):
    print([x[3] for x in tiers])
    return np.random.choice([x[0] for x in tiers], 1, p=[x[3] for x in tiers])[0]


def tifl_update_probs(tiers):
    n = len([x for x in tiers if x[2] > 0])
    D = n * (n +1) / 2
    tiers.sort(key=lambda x:x[1])
    idx_decr = 0
    for idx, tier in enumerate(tiers):
        if tier[2] > 0:
            tier[3] = (n - (idx - idx_decr)) / D
        else:
            tier[3] = 0
            idx_decr += 1


def tifl_select_tier_and_decrement(tiers):
    selected_tier = tifl_select_tier(tiers)
    for tier in tiers:
        if tier[0] == selected_tier:
            tier[2] -= 1
    return selected_tier


def tifl_can_select_tier(tiers):
    return len([x for x in tiers if x[2] > 0])