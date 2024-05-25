import argparse
import toml
import shutil
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegressionCV
from collections import defaultdict
from config import BaseConfig
from training.classification.trainer import ClassificationTrainer
from uspecanalysis.utils.classificationtracker import get_uspec_inputs


def save_scores(path: str, score_name: str, dataset: str, scores: np.ndarray, seed: str = '', ood: str = 'none'):
    path = os.path.expanduser(os.path.join(path, score_name, dataset))
    if not os.path.exists(path):
        os.makedirs(path)
    if ood == 'none':
        np.save(os.path.join(path, f'scores_seed{seed}.npy'), np.squeeze(scores))
    else:
        np.save(os.path.join(path, f'scores_{ood}_seed{seed}.npy'), np.squeeze(scores))


def main(cfg: BaseConfig):
    torch.backends.cudnn.benchmark = True
    accuracies = defaultdict(list)
    for seed in range(cfg.run_configs.start_seed, cfg.run_configs.end_seed):
        epochs = cfg.classification.epochs
        if cfg.classification.fix_seed:
            init_seed = cfg.classification.init_seed
            print(f'Fixing seed to {init_seed}')
            random.seed(init_seed)
            torch.manual_seed(init_seed)
            np.random.seed(init_seed)
        else:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        print(f'Seed: {seed}')
        trainer = ClassificationTrainer(cfg)
        stat_inputs = get_uspec_inputs(cfg)
        for i in range(epochs):
            trainer.training(i, save_checkpoint=True, track_summaries=True)
            if cfg.classification.track_statistics:
                for elem in stat_inputs:
                    print(elem[1])
                    print(f'Seed: {seed}')
                    _, _, _ = trainer.testing(i, alternative_loader_struct=elem[2])
        if cfg.run_configs.scale_ics:
            for i in range(200):
                print(f'Epoch {i}')
                trainer.scale_ics()
        # gather train events
        if cfg.classification.track_statistics:
            path = cfg.run_configs.ld_folder_name + '/events/train/'
            if not os.path.exists(path):
                os.makedirs(path)
            for stat in range(len(trainer.train_statistics)):
                ext = '' if stat == len(trainer.train_statistics) - 1 else f'_ic{stat}'
                trainer.train_statistics[stat] .save_statistics(path, f'/seed{str(seed)}/', extenstion=ext)

        cur_row = []
        names = []

        # derive training stats
        if trainer.sdn:
            print(f'Evaluating in distribution samples')
            in_distances, _, _ = trainer.testing(0, name='val')
            in_test = in_distances['logit_mahalanobis']

        for elem in stat_inputs:
            print(f'Dataset name: {elem[1]}')
            preds, cur_acc, tracker = trainer.testing(0, alternative_loader_struct=elem[2],
                                                      name=elem[1])
            accuracies[elem[1]].append(preds['accuracy'])

            # sngp uncertainties if applicable
            if trainer.checksngp():
                path = cfg.run_configs.ld_folder_name + '/sngp_uncertainties/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/scores_seed' + str(seed) + '.npy', np.squeeze(preds['sngp_uncertainties']))

            # predictions
            if cfg.classification.track_predictions:
                cur_row.append(cur_acc)
                names.append(elem[1])
                path = cfg.run_configs.ld_folder_name + '/predictions/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/predictions_seed' + str(seed) + '.npy', np.squeeze(preds['predictions']))

            # probabilities
            if cfg.classification.track_probs:
                cur_row.append(cur_acc)
                names.append(elem[1])
                path = cfg.run_configs.ld_folder_name + '/probabilities/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/probabilities_seed' + str(seed) + '.npy', np.squeeze(preds['probs']))

            # sdn switches
            if trainer.sdn:
                path = cfg.run_configs.ld_folder_name + '/sdn_uncertainties/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + f'/scores_seed{seed}.npy', np.squeeze(preds['switch_uncertainties']))
                weights_path = os.path.expanduser(os.path.join(cfg.run_configs.ld_folder_name, 'logit_mahalanobis',
                                                               elem[1]))
                if not os.path.exists(weights_path):
                    os.makedirs(weights_path)

                np.save(os.path.join(weights_path, f'scores_all_layers_out_seed{seed}.npy'), preds['logit_mahalanobis'])
                np.save(os.path.join(weights_path, f'scores_all_layers_in_seed{seed}.npy'), in_test)
                save_scores(cfg.run_configs.ld_folder_name, score_name='in_layer_switches', dataset=elem[1],
                            scores=in_distances['layer_switches'], seed=str(seed))

    # save accuracies
    acc_df = pd.DataFrame(data=accuracies)
    acc_df.to_excel(os.path.expanduser(os.path.join(cfg.run_configs.ld_folder_name, 'accuracy.xlsx')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/USPEC-LD/example_config.toml')

    args = parser.parse_args()
    # set random seeds deterministicly to 0

    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    main(configs)
    shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.ld_folder_name + 'parameters.toml')
