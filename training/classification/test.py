import argparse
import toml
import shutil
import os
import random
import numpy as np
import torch
from config import BaseConfig
from training.classification.trainer import ClassificationTrainer
from uspecanalysis.utils.classificationtracker import get_uspec_inputs


def main(cfg: BaseConfig):
    trainer = ClassificationTrainer(cfg)
    stat_inputs = get_uspec_inputs(cfg)

    cur_row = []
    names = []
    for elem in stat_inputs:
        print(elem[1])
        preds, cur_acc, tracker = trainer.testing(0, alternative_loader_struct=elem[2],
                                                  track_embeddings=cfg.classification.track_embeddings,
                                                  name=elem[1])
        # duq uncertainties if applicable
        if trainer.checkduq():
            path = cfg.run_configs.ld_folder_name + '/duq_uncertainties/' + elem[1]
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + '/scores.npy', np.squeeze(preds['duq_uncertainties']))

        # sngp uncertainties if applicable
        if trainer.checksngp():
            path = cfg.run_configs.ld_folder_name + '/sngp_uncertainties/' + elem[1]
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + '/scores_seed0.npy', np.squeeze(preds['sngp_uncertainties']))

        # predictions
        if cfg.classification.track_predictions:
            cur_row.append(cur_acc)
            names.append(elem[1])
            path = cfg.run_configs.ld_folder_name + '/predictions/' + elem[1]
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + '/predictions.npy', np.squeeze(preds['predictions']))

        # embeddings
        if cfg.classification.track_embeddings:
            cur_row.append(cur_acc)
            names.append(elem[1])

            # non grad embeddings
            path = cfg.run_configs.ld_folder_name + '/nongradembeddings/' + elem[1]
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + '/embeddings_seed.npy', np.squeeze(preds['nongrad_embeddings']))

        # probabilities
        if cfg.classification.track_probs:
            cur_row.append(cur_acc)
            names.append(elem[1])
            path = cfg.run_configs.ld_folder_name + '/probabilities/' + elem[1]
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + '/probabilities.npy', np.squeeze(preds['probs']))

        # sdn switches
        if trainer.sdn:
            path = cfg.run_configs.ld_folder_name + '/sdn_uncertainties/' + elem[1]
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + '/scores.npy', np.squeeze(preds['switch_uncertainties']))


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
