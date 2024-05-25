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


def auroc_curves(curve_types: list, folder_list: list, idx_list: List[int], in_distr: str, ood: List[str],
                 misprediction_detection: bool = False, plot_col: str = 'AUROC'):
    if len(curve_types) != len(folder_list):
        raise Exception('All lists must have the same length!')
    visual_dict = defaultdict(list)

    if misprediction_detection:
        gts = [get_test_targets(in_distr)[0]]
        for dataset in range(len(ood)):
            gts.extend(get_test_targets(ood[dataset])[0])
    else:
        gts = None

    for i in range(len(curve_types)):
        seeds = get_num_seeds(folder_list[i], in_distr)
        for j in seeds:
            for corr in ood:
                auroc, aupr, fpr, tpr = get_auroc_scores(target_folder=folder_list[i], type=curve_types[i],
                                                         in_distr_data=in_distr, ood_data=corr, seed=j, gts=gts)
                visual_dict['Algorithm'].append(f'{curve_types[i]}')
                visual_dict['AUROC'].append(auroc)
                visual_dict['AUPR'].append(auroc)
                visual_dict['Seed'].append(j)
                visual_dict['Num Layers'].append(idx_list[i])

    visual_df = pd.DataFrame(data=visual_dict)
    sns.lineplot(data=visual_df, x='Num Layers', y=plot_col, hue='Algorithm')
    plt.legend(loc="upper left", prop={'size': 10})
    plt.show()