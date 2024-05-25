import os
import numpy as np

from sklearn.linear_model import LogisticRegressionCV


def entropy(probabilities: np.ndarray, axis: int = 1) -> np.ndarray:
    logs = np.log2(probabilities)
    mult = logs * probabilities
    entropy = -1*np.sum(mult, axis=axis)
    return np.squeeze(entropy)


def logits_entropy(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    probabilities = np.nan_to_num(np.exp(logits)/np.sum(np.exp(logits), axis=axis, keepdims=True))
    logs = np.nan_to_num(np.log2(probabilities))
    mult = logs * probabilities
    entropy = -1*np.sum(mult, axis=axis)
    return np.squeeze(np.nan_to_num(entropy))


def check_file(file_name: str):
    if not os.path.exists(file_name):
        raise Exception('File: ' + file_name + ' does not exist. Check spelling or rerun experiment!')


def single_log_regression(scores_in: np.ndarray, scores_test: np.ndarray, switches_in: np.ndarray):
    np.random.seed(0)
    in_train = scores_in
    average = np.mean(np.sum(switches_in, axis=1))
    # print(f'Average Switches {average}')
    y_in = np.sum(switches_in, axis=1).squeeze() > 1

    total_x_train = in_train
    total_y_train = y_in

    regression = LogisticRegressionCV(n_jobs=-1, max_iter=500).fit(total_x_train, total_y_train)
    weights = regression.coef_
    out = np.sum(scores_test * weights, axis=1)

    return out, weights


def get_sngp_score(logits: np.ndarray, axis: int):
    # sngp uncertainties
    num_classes = logits.shape[axis]
    belief_mass = np.sum(np.exp(logits), axis=axis)
    # belief_mass = logits.exp().sum(1)
    scores = num_classes / (belief_mass + num_classes)
    return scores


def get_sdn_scores(path: str, dataset: str, seed: int = 0):
    path = os.path.expanduser(path)
    norm_file = f'{path}/logit_mahalanobis/{dataset}/scores_all_layers_out_seed{seed}.npy'
    norm_test_file = f'{path}/logit_mahalanobis/{dataset}/scores_all_layers_out_seed{seed}.npy'

    in_norm_file = f'{path}/logit_mahalanobis/{dataset}/scores_all_layers_in_seed{seed}.npy'

    in_switch_file = f'{path}/in_layer_switches/{dataset}/scores_seed{seed}.npy'

    check_file(norm_file)
    check_file(in_switch_file)

    in_norms = np.load(in_norm_file)
    test_norms = np.load(norm_test_file)
    in_switches = np.load(in_switch_file)

    probs, w = single_log_regression(scores_in=in_norms, switches_in=in_switches, scores_test=test_norms)

    return probs
