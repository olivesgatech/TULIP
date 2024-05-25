import glob
import os
import re
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from npeet import entropy_estimators as ee

from visualization.uncertainty.common import get_test_targets
from visualization.auroc.gen_scores import get_uncertainty_scores, get_auroc_scores, get_num_seeds, CORRUPTIONS


def plot_curves(curve_types: list, folder_list: list, idx_list: List[int], plot_col: str = 'test'):
    if len(curve_types) != len(folder_list) != len(idx_list):
        raise Exception('All lists must have the same length!')
    visual_dict = defaultdict(list)
    seeds = get_num_seeds(folder_list[0], combination='val')
    x_label = 'I(Y; Y_ID)'
    y_label = 'I(h_first; h_last)'

    for i in range(len(curve_types)):
        for seed in seeds:
            val_embeddings = np.load(os.path.join(folder_list[i], 'first_layer_embeddings', plot_col,
                                                  f'embeddings_seed{seed}.npy'))
            target_embeddings = np.load(os.path.join(folder_list[i], 'penultimate_embeddings', plot_col,
                                                     f'embeddings_seed{seed}.npy'))
            idxs = np.random.choice(target_embeddings.shape[0], size=val_embeddings.shape[0], replace=False)

            mi = ee.mi(val_embeddings, target_embeddings)
            print(f'MI seed {seed}: {mi}')
            visual_dict['Algorithm'].append(f'{curve_types[i]}')
            visual_dict[x_label].append(idx_list[i])
            visual_dict['Seed'].append(seed)
            visual_dict[y_label].append(mi)

    visual_df = pd.DataFrame(data=visual_dict)
    sns.lineplot(data=visual_df, x=x_label, y=y_label, hue='Algorithm')
    plt.legend(loc="upper left", prop={'size': 10})
    plt.show()


if __name__ == '__main__':
    path = os.path.expanduser('~/results/LDU/SVHN/no-augmentations/unbalanced50pcent/')
    curves = [
        'Original',
        'Original',
        'Original',

        'Spectral',
        'Spectral',
        'Spectral',
    ]
    addons = [
        '100pcent/paramsv0/skip-mimlp/skip_mimlpv0/',
        '50pcent/paramsv0/skip-mimlp/skip_mimlpv0/',
        '0pcent/paramsv0/skip-mimlp/skip_mimlpv0/',

        '100pcent/paramsv0/skip-mimlp/spectral_skip_mimlpv0/',
        '50pcent/paramsv0/skip-mimlp/spectral_skip_mimlpv0/',
        '0pcent/paramsv0/skip-mimlp/spectral_skip_mimlpv0/',
    ]
    index_list = [100, 50, 0]
    # index_list.reverse()
    index_list.extend(index_list)
    paths = [path + addons[i] for i in range(len(addons))]
    plot = 'val'

    plot_curves(curve_types=curves, folder_list=paths, idx_list=index_list, plot_col=plot)