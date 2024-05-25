import glob
import os
import re
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import scipy
from visualization.uncertainty.common import get_test_targets
from visualization.auroc.gen_scores import get_uncertainty_scores, get_num_seeds, load_prediction_file


def get_surface(target_folder: str, type: str, target_distr: str,
                seed: int = 0, imbalance: int = None, num_points: int = 10):
    # get path
    path = os.path.expanduser(target_folder)
    gts = get_test_targets(target_distr, imbalance=imbalance)[0]
    preds = load_prediction_file(path, target_distr, seed)

    scores = get_uncertainty_scores(path, type, target_distr, seed, ood=target_distr)
    score_window = (scores.max() - scores.min()) / num_points
    sorted_idxs = scores.argsort()
    sorted_gts = gts[sorted_idxs]
    sorted_preds = preds[sorted_idxs]

    window = scores.shape[0] // num_points
    uncertainty_scores = []
    acc = []
    num_samples = []

    for i in range(num_points):
        start_score = i * score_window
        end_score = (i + 1) * score_window

        acc_arr = np.equal(preds[(scores >= start_score) & (scores <= end_score)],
                           gts[(scores >= start_score) & (scores <= end_score)])
        acc.append(np.sum(acc_arr) / acc_arr.shape[0])
        num_samples.append(acc_arr.shape[0])
        uncertainty_scores.append(np.mean(scores[(scores >= start_score) & (scores <= end_score)]))

    out = {
        'acc': np.array(acc),
        'num_samples': np.array(num_samples),
        'uncertainty': np.array(uncertainty_scores)
    }
    return out


def plot_surface(curve_type: str, folder_list: list, idx_list: List[int], in_distr: str,
                 imbalance: bool = False, num_points: int = 10, plot_col: str = 'acc'):
    if len(folder_list) != len(idx_list):
        raise Exception('All lists must have the same length!')
    imbalance_idx = []
    acc = np.zeros((len(idx_list), num_points), dtype=float)
    uncertainty = np.zeros((len(idx_list), num_points), dtype=float)
    for i in range(len(folder_list)):
        seeds = get_num_seeds(folder_list[i], in_distr)
        imbalance_idx.append([idx_list[i] for k in range(num_points)])
        for j in seeds:
            i_pcent = idx_list[i] if imbalance else None

            stats = get_surface(target_folder=folder_list[i], type=curve_type,
                                target_distr=in_distr, seed=j, imbalance=i_pcent, num_points=num_points)
            acc[i, :] += stats[plot_col] / len(seeds)
            uncertainty[i, :] += stats['uncertainty'] / len(seeds)

    x = uncertainty
    y = np.array(imbalance_idx)
    z = acc

    # regular grid covering the domain of the data
    plot_points = 40
    X, Y = np.meshgrid(np.arange(uncertainty.min(), uncertainty.max(), (uncertainty.max() - uncertainty.min()) /
                                 plot_points),
                       np.arange(acc.min(), acc.max(), (acc.max() - acc.min()) /
                                 plot_points))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.scatter(x.flatten(), y.flatten(), z.flatten())

    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    # surf = ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), cmap=cm.jet, linewidth=0.1)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()


if __name__ == '__main__':
    path = os.path.expanduser('~/results/LDU/CIFAR100/rcrop-rhflip-cutout/100classes/unbalanced50pcent/with_scores/')

    curves = [
        'Original',
        'Original',
        'Original',
        'Original',
        'Original',
    ]
    addons = [
        # '100pcent/paramsv0/resnet18/resnet18v0/',
        # '80pcent/paramsv0/resnet18/resnet18v0/',
        # '50pcent/paramsv0/resnet18/resnet18v0/',
        # '20pcent/paramsv0/resnet18/resnet18v0/',
        # '0pcent/paramsv0/resnet18/resnet18v0/',

        '100pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '80pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '50pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '20pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '0pcent/paramsv0/resnet18/spectral_resnet18v0/',
    ]
    index_list = [100, 80, 50, 20, 0]
    paths = [path + addons[i] for i in range(len(addons))]
    ct = 'entropy'
    in_dist = 'CIFAR100'
    plot = 'AUROC'
    # plot = 'mCS'
    misprediction = True
    imbalance = True
    pc = 'num_samples'
    # for corruption in CORRUPTIONS.keys():
    #     ood_dist.extend(f'{corruption}{level}' for level in levels)
    # ood_dist = ['CINIC10']

    plot_surface(curve_type=ct, folder_list=paths, idx_list=index_list, in_distr=in_dist,
                 imbalance=imbalance, plot_col=pc)
