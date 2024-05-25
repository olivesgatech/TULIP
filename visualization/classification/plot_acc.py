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
from visualization.auroc.gen_scores import get_uncertainty_scores, get_auroc_scores, get_num_seeds, CORRUPTIONS


def plot_curves(curve_types: list, folder_list: list, idx_list: List[int], plot_col: str = 'test'):
    if len(curve_types) != len(folder_list) != len(idx_list):
        raise Exception('All lists must have the same length!')
    visual_dict = defaultdict(list)
    x_label = '% Data Removed'

    for i in range(len(curve_types)):
        acc_df = pd.read_excel(os.path.join(folder_list[i], 'accuracy.xlsx'))
        accs = acc_df[plot_col].tolist()
        visual_dict['Algorithm'].extend([f'{curve_types[i]}' for k in range(len(accs))])
        visual_dict[x_label].extend([idx_list[i] for k in range(len(accs))])
        visual_dict['Seed'].extend([k for k in range(len(accs))])
        visual_dict['Accuracy'].extend(accs)

    visual_df = pd.DataFrame(data=visual_dict)
    sns.lineplot(data=visual_df, x=x_label, y='Accuracy', hue='Algorithm')
    plt.legend(loc="upper right", prop={'size': 10})
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
    # index_list.reverse()

    # index_list.reverse()
    index_list.extend(index_list)
    paths = [path + addons[i] for i in range(len(addons))]
    plot = 'test'

    plot_curves(curve_types=curves, folder_list=paths, idx_list=index_list, plot_col=plot)

'''
    path = os.path.expanduser('~/results/LDU/CIFAR100/rcrop-rhflip-cutout/100classes/unbalanced50pcent/')
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
        '../100pcent/paramsv0/resnet18/resnet18v0/',

        '100pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '80pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '50pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '20pcent/paramsv0/resnet18/spectral_resnet18v0/',
        '../100pcent/paramsv0/resnet18/spectral_resnet18v0/',
    ]
    index_list = [1.0, 0.8, 0.5, 0.2, 0]
'''