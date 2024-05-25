import os
import numpy as np
from visualization.uncertainty.common import get_test_targets


def check_file(file_name: str):
    if not os.path.exists(file_name):
        raise Exception('File: ' + file_name + ' does not exist. CHeck spelling or rerun experiment!')


def get_sngp_scores(path: str, dataset: str, seed: int = 0):
    path = os.path.expanduser(path)
    target_file = f'{path}/sngp_uncertainties/{dataset}/scores_seed{seed}.npy'
    check_file(target_file)
    return np.load(target_file)


def sngp_curve(target_folder: str, in_distr_data: str, ood_data: str, percent_remove_list: np.ndarray,
              ref_pred_path: str, seed: int = 0, rm_ood_acc: bool = True):
    # get path
    path = os.path.expanduser(target_folder)
    ref_path = os.path.expanduser(ref_pred_path)
    combination_list = [in_distr_data, ood_data]

    for i in range(2):
        combination = combination_list[i]
        pred_file = f'{ref_path}/predictions/{combination}/predictions_seed{seed}.npy'
        check_file(pred_file)

        # get target labels
        targets, num_classes = get_test_targets(combination)

        # load files
        scores = get_sngp_scores(path, combination)
        predictions = np.load(pred_file)
        if i != 0 and rm_ood_acc:
            acc = np.zeros(predictions.shape, dtype=bool)
            ideal_acc = np.zeros(predictions.shape, dtype=bool)
        else:
            acc = predictions == targets
            ideal_acc = np.ones(predictions.shape, dtype=bool)

        if i == 0:
            total_acc = acc
            total_ideal_acc = ideal_acc
            total_scores = scores
        else:
            total_acc = np.append(total_acc, acc)
            total_ideal_acc = np.append(total_ideal_acc, ideal_acc)
            total_scores = np.append(total_scores, scores)

    score_order = np.argsort(total_scores)
    ordered_acc = total_acc[score_order]

    # generate curve
    for i in range(len(percent_remove_list)):
        pcent = percent_remove_list[i]
        considered_samples = int((1.0 - pcent) * total_acc.shape[0])
        event_acc = np.sum(ordered_acc[:considered_samples]) / considered_samples
        th_max = np.sum(total_ideal_acc[:considered_samples]) / considered_samples

        if i == 0:
            event_curve = np.array([event_acc])
            maximum_curve = np.array([th_max])
        else:
            event_curve = np.append(event_curve, [event_acc])
            maximum_curve = np.append(maximum_curve, [th_max])

    return event_curve, maximum_curve