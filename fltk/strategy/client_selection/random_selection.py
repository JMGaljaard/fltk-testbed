import numpy as np


def random_selection(clients, n):
    return np.random.choice(clients, n, replace=False)