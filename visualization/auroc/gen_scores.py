import glob
import os
import re
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from visualization.uncertainty.sngp import get_sngp_scores
from visualization.uncertainty.entropy import get_entropy_scores
from visualization.uncertainty.sdn import get_sdn_scores
from visualization.uncertainty.common import get_test_targets
from data.datasets.classification.CIFAR10C.configuration import CORRUPTIONS


def get_uncertainty_scores(path: str, type: str, combination: str, seed: int, ood: str = 'none') -> np.ndarray:
    if type == 'sngp':
        scores = get_sngp_scores(path, combination, seed)
        return scores
    elif type == 'sdn':
        scores = get_sdn_scores(path, combination, seed)
        return scores
    elif type == 'entropy':
        scores = get_entropy_scores(path, combination, seed)
        return scores
    else:
        raise ValueError(f'Uncertainty type {type} not implemented yet!')


def load_prediction_file(path: str, combination: str, seed: int):
    path = os.path.expanduser(path)
    target_path = f'{path}/predictions/{combination}/predictions_seed{seed}.npy'
    if not os.path.exists(target_path):
        raise FileExistsError(f'File {path} does not exist!')
    return np.load(target_path)


def get_in_dist_auroc_scores(target_folder: str, type: str, target_distr: str,
                             seed: int = 0, imbalance: int = None, window: int = 100):
    # get path
    path = os.path.expanduser(target_folder)
    gts = get_test_targets(target_distr, imbalance=imbalance)[0]
    preds = load_prediction_file(path, target_distr, seed)

    scores = get_uncertainty_scores(path, type, target_distr, seed, ood=target_distr)
    mean_correct_scores = np.mean(scores[preds == gts])
    mean_incorrect_scores = np.mean(scores[preds != gts])
    mean_scores = np.mean(scores)

    targets = np.ones(scores.shape, dtype=int) - np.equal(preds, gts)

    auc = roc_auc_score(targets, scores, average=None)
    pr_auc = average_precision_score(targets, scores)
    fpr, tpr, _ = roc_curve(targets, scores)
    # print(f'AUROC: {auc}')
    out = {
        'auc': auc,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'mean cs': mean_correct_scores,
        'mean is': mean_incorrect_scores,
        'mean s': mean_scores
    }
    return out


def get_auroc_scores(target_folder: str, type: str, in_distr_data: str, ood_data: str,
                     seed: int = 0, gts: List[np.ndarray] = None):
    # get path
    path = os.path.expanduser(target_folder)
    combination_list = [in_distr_data, ood_data]

    total_scores = None
    total_targets = None
    for i in range(len(combination_list)):
        combination = combination_list[i]
        scores = get_uncertainty_scores(path, type, combination, seed, ood=ood_data)

        if gts is not None:
            preds = load_prediction_file(path, combination, seed)
            test_gt = gts[i]
            targets = np.ones(scores.shape, dtype=int) - np.equal(preds, test_gt)
        else:
            targets = np.zeros(scores.shape) if i == 0 else np.ones(scores.shape)
        total_scores = np.concatenate((total_scores, scores)) if total_scores is not None else scores
        total_targets = np.concatenate((total_targets, targets)) if total_targets is not None else targets

    auc = roc_auc_score(total_targets, total_scores, average=None)
    pr_auc = average_precision_score(total_targets, total_scores)
    fpr, tpr, _ = roc_curve(total_targets, total_scores)
    return auc, pr_auc, fpr, tpr


def get_num_seeds(path: str, combination: str):
    path = os.path.expanduser(path)
    target_path = f'{path}/predictions/{combination}/'
    if not os.path.exists(target_path):
        raise FileExistsError(f'Path {target_path} does not exist!')
    files = glob.glob(os.path.join(target_path, 'predictions_seed*.npy'))
    # TODO: only works for linux and macos
    seeds = [int(re.findall(r'\d+', file.split('/')[-1])[0]) for file in files]
    return seeds


def flatten_list(cur_list):
    arr = np.array(cur_list)
    return list(arr.flatten())


def compare_arurocs(curve_types: list, folder_list: list, in_distr: str, ood: List[str],
                    misprediction_detection: bool = False):
    if len(curve_types) != len(folder_list):
        raise Exception('All lists must have the same length!')
    visual_dict = defaultdict(list)
    numeric_dict = defaultdict(list)

    if misprediction_detection:
        gts = [flatten_list([get_test_targets(in_distr)[0]])]
        for dataset in range(len(ood)):
            gts.append(flatten_list(get_test_targets(ood[dataset])[0]))
    else:
        gts = None

    for i in range(len(curve_types)):
        seeds = get_num_seeds(folder_list[i], in_distr)
        aurocs = []
        auprs = []
        tprs = []
        fprs = []
        for j in seeds:
            ood_aurocs = []
            ood_auprs = []
            for corr in ood:
                auroc, aupr, fpr, tpr = get_auroc_scores(target_folder=folder_list[i], type=curve_types[i],
                                                         in_distr_data=in_distr, ood_data=corr, seed=j, gts=gts)
                visual_dict['Algorithm'].extend([f'{curve_types[i]} {corr}' for k in range(len(fpr))])
                visual_dict['Seed'].extend([j for k in range(len(fpr))])
                visual_dict['FPR'].extend(list(fpr))
                visual_dict['TPR'].extend(list(tpr))
                ood_aurocs.append(auroc)
                ood_auprs.append(aupr)
            aurocs.append(np.mean(ood_aurocs))
            auprs.append(np.mean(ood_auprs))
            tprs.append(tpr)
            fprs.append(fpr)
        avg_auroc = np.mean(aurocs)
        std_aurocs = np.std(aurocs)
        avg_aupr = np.mean(auprs)
        std_aupr = np.std(auprs)
        numeric_dict['Algorithm'].append(curve_types[i])
        numeric_dict['Avg. AUROC'].append(avg_auroc)
        numeric_dict['Std. AUROC'].append(std_aurocs)
        numeric_dict['Avg. AUPR'].append(avg_aupr)
        numeric_dict['Std. AUPR'].append(std_aupr)
        numeric_dict['In-Distr.'].append(in_distr)
        numeric_dict['Out-of-Distr.'].append(corr)
        plt.plot(fpr, tpr, label=f'{curve_types[i]} {corr}')
        print(f'Type {curve_types[i]} AUROC: {avg_auroc:.3f} ± {std_aurocs:.3f} AUPR: {avg_aupr:.3f} ± {std_aupr:.3f}')
    num_df = pd.DataFrame(data=numeric_dict)
    num_df.to_excel('area_results.xlsx')

    visual_df = pd.DataFrame(data=visual_dict)
    # sns.lineplot(data=visual_df, x='FPR', y='TPR', hue='Algorithm')
    plt.legend(loc="upper left", prop={'size': 10})
    plt.show()


if __name__ == '__main__':
    path = os.path.expanduser('~/redo/LDUv1/CIFAR10/rcrop-rhflip-cutout/resnet50/paramsv0/')
    curves = [
        'sdn',
    ]
    addons = [
        'sdn_sngp_resnet_org50v0.2.0_remaining/',
    ]
    paths = [path + addons[i] for i in range(len(addons))]
    in_dist = 'test'
    ds = 'CIFAR10'
    misprediction = False
    levels = [1, 2, 3, 4, 5]
    ood_dist = []
    for corruption in CORRUPTIONS.keys():
        ood_dist.extend(f'CIFAR100C{corruption}{level}' for level in levels)
        # ood_dist.extend(f'{corruption}{level}' for level in levels)
    # ood_dist = ['CIFAR100']

    compare_arurocs(curve_types=curves, folder_list=paths, in_distr=in_dist, ood=ood_dist,
                    misprediction_detection=misprediction)

