import glob
import os
import re
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization.uncertainty.common import get_test_targets
from visualization.auroc.gen_scores import get_uncertainty_scores, get_in_dist_auroc_scores, get_num_seeds, CORRUPTIONS


def auroc_curves(curve_types: list, folder_list: list, names: List[str], idx_list: List[int], in_distr: str,
                 plot_col: str = 'AUROC', imbalance: bool = False):
    if len(curve_types) != len(folder_list) != len(idx_list):
        raise Exception('All lists must have the same length!')
    visual_dict = defaultdict(list)

    for i in range(len(curve_types)):
        seeds = get_num_seeds(folder_list[i], in_distr)
        for j in seeds:
            i_pcent = idx_list[i] if imbalance else None

            stats = get_in_dist_auroc_scores(target_folder=folder_list[i], type=curve_types[i],
                                             target_distr=in_distr, seed=j, imbalance=i_pcent)
            visual_dict['Algorithm'].append(f'{names[i]}')
            visual_dict['AUROC'].append(stats['auc'])
            visual_dict['mCS'].append(stats['mean cs'])
            visual_dict['mIS'].append(stats['mean is'])
            visual_dict['mS'].append(stats['mean s'])
            visual_dict['Seed'].append(j)
            visual_dict['x-axis'].append(idx_list[i])

    visual_df = pd.DataFrame(data=visual_dict)
    sns.lineplot(data=visual_df, x='x-axis', y=plot_col, hue='Algorithm')
    plt.legend(loc="upper left", prop={'size': 10})
    plt.show()


if __name__ == '__main__':
    path = os.path.expanduser('~/results/LDU/CIFAR100/rcrop-rhflip-cutout/100classes/unbalanced50pcent/with_scores/')

    curves = [
        'Original',
        'Original',
        'Original',
        'Original',
        'Original',

        'Spectral Normalization',
        'Spectral Normalization',
        'Spectral Normalization',
        'Spectral Normalization',
        'Spectral Normalization',
    ]
    addons = [
        '100pcent/paramsv0/resnet18/resnet18v0/',
        '80pcent/paramsv0/resnet18/resnet18v0/',
        '50pcent/paramsv0/resnet18/resnet18v0/',
        '20pcent/paramsv0/resnet18/resnet18v0/',
        '0pcent/paramsv0/resnet18/resnet18v0/',

        '100pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '80pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '50pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '20pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '0pcent/paramsv0/resnet18/spectral_resnet18v0/',
    ]
    index_list = [100, 80, 50, 20, 0]
    index_list.extend(index_list)
    paths = [path + addons[i] for i in range(len(addons))]
    ct = ['entropy' for i in range(len(addons))]
    in_dist = 'CIFAR100'
    plot = 'AUROC'
    # plot = 'mCS'
    misprediction = True
    imbalance = True
    # for corruption in CORRUPTIONS.keys():
    #     ood_dist.extend(f'{corruption}{level}' for level in levels)
    # ood_dist = ['CINIC10']

    auroc_curves(curve_types=ct, folder_list=paths, names=curves, idx_list=index_list, in_distr=in_dist,
                 imbalance=imbalance, plot_col=plot)
