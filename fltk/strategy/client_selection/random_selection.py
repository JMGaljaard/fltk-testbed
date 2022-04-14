from typing import List

import numpy as np

from fltk.util.remote import ClientRef


def random_selection(clients: List[ClientRef], size: int) -> List[ClientRef]:
    """
    Function to uniformly sample random clients without replacment from a pool of unique clients.
    @param clients: List of clients to sample from.
    @type clients: List[ClientRef]
    @param size: Number of clients to sample.
    @type size: int
    @return: List of client sampled from the provided pool of clients.
    @rtype: List[ClientRef]
    """
    return np.random.choice(clients, size, replace=False)
