import os
import numpy as np
from visualization.uncertainty.common import get_test_targets


def check_file(file_name: str):
    if not os.path.exists(file_name):
        raise Exception('File: ' + file_name + ' does not exist. CHeck spelling or rerun experiment!')


def entropy(probabilities: np.ndarray) -> np.ndarray:
    logs = np.log2(probabilities)
    mult = logs * probabilities
    entropy = -1*np.sum(mult, axis=1)
    return np.squeeze(entropy)


def get_entropy_scores(path: str, dataset: str, seed: int = 0) -> np.ndarray:
    path = os.path.expanduser(path)
    target_file = f'{path}/probabilities/{dataset}/probabilities_seed{seed}.npy'
    check_file(target_file)
    out = entropy(np.load(target_file))
    return out
