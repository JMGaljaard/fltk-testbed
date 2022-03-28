import numpy as np


def calculate_class_precision(conf_mat: np.array) -> np.array:
    """
    Calculates the precision for each class from a confusion matrix.
    """
    return np.diagonal(conf_mat) / np.sum(conf_mat, axis=0)


def calculate_class_recall(conf_mat: np.array) -> np.array:
    """
    Calculates the recall for each class from a confusion matrix.
    """
    return np.diagonal(conf_mat) / np.sum(conf_mat, axis=1)
