import os
import tqdm
import torch
import torch.nn as nn
import gpytorch

from typing import List
from models.architectures import build_architecture
from training.common.trainutils import determine_multilr_milestones
from data.datasets.classification.common.aquisition import get_dataset
from training.classification.classificationtracker import ClassifcationTracker
from training.common.saver import Saver
from training.common.summaries import TensorboardSummary
from config import BaseConfig


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def reset_precision(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Laplace'):
            m.reset_precision_matrix()


def get_sngp_score(logits: torch.Tensor):
    # sngp uncertainties
    num_classes = logits.shape[1]
    belief_mass = logits.exp().sum(1)
    scores = num_classes / (belief_mass.float() + num_classes)
    return scores

def disable_grads_except_ics(model):
    """ Function to enable the dropout layers during test-time """

    for param in model.module.init_conv.parameters():
        param.requires_grad = False

    for layer in model.module.layers:
        for param in layer.layers.parameters():
            param.requires_grad = False

    for param in model.module.end_layers.parameters():
        param.requires_grad = False


def enable_grads(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        for p in m.parameters():
            p.requires_grad = True


class SDNLoss:
    def __init__(self, num_ics: int, coeffs: List[int] = None):
        self._num_ics = num_ics
        self._loss = nn.CrossEntropyLoss()
        self._mse = nn.MSELoss()

        if coeffs is not None:
            if len(coeffs) != num_ics:
                raise ValueError(f'Number of coefficients must be equal to number of ics!')
            self._coeffs = coeffs
        else:
            self._coeffs = [float(1/num_ics) for _ in range(num_ics)]

    def __call__(self, outputs: List[torch.Tensor], target: torch.Tensor):
        loss = 0.0
        for i in range(self._num_ics):
            loss += self._coeffs[i]*self._loss(outputs[i], target)

        return loss


def update_trackers(trackers: List[ClassifcationTracker], outputs: List[torch.Tensor], target: torch.Tensor,
                    idxs: torch.Tensor):
    if len(trackers) < 1:
        raise ValueError(f'At least one tracker/output pair must be provided!')
    if len(trackers) != len(outputs):
        raise ValueError(f'Number of trackers and outputs must match!')

    for i in range(len(trackers)):
        _, pred = torch.max(outputs[i].data, 1)

        # collect forgetting events
        acc = pred.eq(target.data)

        # check if prediction has changed
        predicted = pred.cpu().numpy()
        trackers[i].update(acc.cpu().numpy(), predicted, idxs.cpu().numpy())
    # return accuracy of last layer
    return trackers, acc


class ClassificationTrainer:
    def __init__(self, cfg: BaseConfig):
        self._cfg = cfg
        self._epochs = cfg.classification.epochs
        self._device = cfg.run_configs.gpu_id

        # Define Saver
        self._saver = Saver(cfg)

        # Define Tensorboard Summary
        self._summary = TensorboardSummary(self._saver.experiment_dir)
        self._writer = self._summary.create_summary()

        self._loaders = get_dataset(cfg)
        model_dict = build_architecture(
            self._cfg.classification.model,
            self._loaders.data_config, cfg
        )

        self._model = model_dict['model']
        self._sngp_override, self._num_sdn_ics= (
            model_dict['sngp_override'],
            model_dict['num_sdn_ics'],
        )

        self._softmax = nn.Softmax(dim=1)
        self._num_classes = self._loaders.data_config.num_classes
        self.sdn = False
        self._train_mahala_stats = {'initialized': False}
        if self._num_sdn_ics > 0:
            self._loss = SDNLoss(self._num_sdn_ics)
            self.sdn = True
        elif self._cfg.classification.loss == 'ce':
            self._loss = nn.CrossEntropyLoss()
        else:
            raise Exception('Loss not implemented yet!')

        if self._cfg.classification.optimization.optimizer == 'adam':
            self._optimizer = torch.optim.Adam(self._model.parameters(),
                                               lr=self._cfg.classification.optimization.lr)
        elif self._cfg.classification.optimization.optimizer == 'sgd':
            self._optimizer = torch.optim.SGD(self._model.parameters(),
                                              lr=self._cfg.classification.optimization.lr,
                                              momentum=0.9,
                                              nesterov=True,
                                              weight_decay=5e-4)
        else:
            raise Exception('Optimizer not implemented yet!')

        if self._cfg.classification.optimization.scheduler == 'multiLR':
            milestones = determine_multilr_milestones(self._epochs, self._cfg.classification.optimization.multiLR_steps)
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   milestones=milestones,
                                                                   gamma=self._cfg.classification.optimization.gamma)
        elif self._cfg.classification.optimization.scheduler == 'none':
            pass
        else:
            raise Exception('Scheduler not implemented yet!')

        # Using cuda
        if self._cfg.run_configs.cuda:
            # use multiple GPUs if available
            self._model = torch.nn.DataParallel(self._model,
                                                device_ids=[self._cfg.run_configs.gpu_id])
        else:
            self._device = torch.device('cpu')

        # LD stats
        if self._num_sdn_ics > 0:
            self.train_statistics = [ClassifcationTracker(self._loaders.data_config.train_len)
                                     for _ in range(self._num_sdn_ics)]
            self.test_statistics = [ClassifcationTracker(self._loaders.data_config.test_len)
                                    for _ in range(self._num_sdn_ics)]
        else:
            self.train_statistics = [ClassifcationTracker(self._loaders.data_config.train_len)]
            self.test_statistics = [ClassifcationTracker(self._loaders.data_config.test_len)]

        if self._cfg.run_configs.resume != 'none':
            resume_file = self._cfg.run_configs.resume
            # we have a checkpoint
            if not os.path.isfile(resume_file):
                raise RuntimeError("=> no checkpoint found at '{}'".format(resume_file))
            # load checkpoint
            checkpoint = torch.load(resume_file)
            # minor difference if working with cuda
            if self._cfg.run_configs.cuda:
                self._model.load_state_dict(checkpoint['state_dict'])
            else:
                self._model.load_state_dict(checkpoint['state_dict'])
            # self._optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))

    def checksngp(self):
        return self._sngp_override

    def training(self, epoch, save_checkpoint=False, track_summaries=False):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        if self._sngp_override:
            if self._cfg.run_configs.cuda:
                self._model.module.reset_precision_matrix()
            else:
                self._model.reset_precision_matrix()
        else:
            reset_precision(self._model)

        # initializes cool bar for visualization
        tbar = tqdm.tqdm(self._loaders.train_loader)
        num_img_tr = len(self._loaders.train_loader)

        # init statistics parameters
        train_loss = 0.0
        train_kl = 0.0
        correct_samples = 0
        total = 0

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            # sets model into training mode -> important for dropout batchnorm. etc.
            self._model.train()
            image, target, idxs = sample['data'], sample['label'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)

            # update model
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            # computes output of our model
            output = self._model(image)

            if self._num_sdn_ics <= 0:
                output = [output]

            self.train_statistics, acc = update_trackers(self.train_statistics, output, target, idxs)

            total += target.size(0)

            # Perform model update
            # calculate loss
            if self._num_sdn_ics <= 0:
                loss = self._loss(output[-1], target.long())
            else:
                loss = self._loss(output, target.long())

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            # perform backpropagation
            loss.backward()

            # update params with gradients
            self._optimizer.step()

            if track_summaries:
                self._writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            correct_samples += acc.cpu().sum()

        # Update optimizer step
        if self._cfg.classification.optimization.scheduler != 'none':
            self._scheduler.step(epoch)

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        if track_summaries:
            self._writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.classification.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Training Accuracy: %.3f' % acc)

        # save checkpoint
        if save_checkpoint:
            self._saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            })
        return acc

    def testing(self, epoch, alternative_loader_struct=None, name: str = None):
        return self.dnn_testing(epoch, alternative_loader_struct, name)

    def get_test_loader(self, alternative_loader_struct = None, name: str = None):
        if alternative_loader_struct is None:
            num_samples = self._loaders.data_config.test_len
            num_classes = self._loaders.data_config.num_classes
            tbar = tqdm.tqdm(self._loaders.test_loader)
            num_img_tr = len(self._loaders.test_loader)
            tracker = self.test_statistics
        else:
            alternative_loader = alternative_loader_struct[0]
            if self._num_sdn_ics > 0:
                tracker = [alternative_loader_struct[1] for _ in range(self._num_sdn_ics)]
            else:
                tracker = [alternative_loader_struct[1]]
            num_classes = alternative_loader.data_config.num_classes
            if name == 'train':
                num_samples = alternative_loader.data_config.train_len
                tbar = tqdm.tqdm(alternative_loader.train_loader)
                num_img_tr = len(alternative_loader.train_loader)
            elif name == 'val':
                num_samples = alternative_loader.data_config.val_len
                tbar = tqdm.tqdm(alternative_loader.val_loader)
                num_img_tr = len(alternative_loader.val_loader)
            else:
                num_samples = alternative_loader.data_config.test_len
                tbar = tqdm.tqdm(alternative_loader.test_loader)
                num_img_tr = len(alternative_loader.test_loader)

        loader_dict = {
            'num_samples': num_samples,
            'num_classes': num_classes,
            'tqdm_loader': tbar,
            'tracker': tracker,
            'num_imgs': num_img_tr
        }
        return loader_dict


    def dnn_testing(self, epoch, alternative_loader_struct = None, name: str = None):
        """
        tests the model on a given holdout set. Provide an alterantive loader structure if you do not want to test on
        the test set.
        :param epoch:
        :param alternative_loader_struct:
        :return:
        """
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()

        # init loader
        loader_dict = self.get_test_loader(alternative_loader_struct, name)
        num_samples = loader_dict['num_samples']
        tbar = loader_dict['tqdm_loader']
        tracker = loader_dict['tracker']

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        # get embed dim
        if self._cfg.run_configs.cuda:
            embedDim = self._model.module.get_penultimate_dim()
        else:
            embedDim = self._model.get_penultimate_dim()

        # iterate over all samples in each batch i
        predictions = torch.zeros(num_samples, dtype=torch.long, device=self._device)
        sngp_uncertainties = torch.zeros(num_samples, dtype=torch.float32, device=self._device)
        probabilites = torch.zeros((num_samples, self._num_classes), dtype=torch.float, device=self._device)
        switch_uncertainties = torch.zeros(num_samples, dtype=torch.float64, device=self._device)
        num_layers = -1

        for i, sample in enumerate(tbar):
            image, target, idxs = sample['data'], sample['label'], sample['idx']
            # assign each image and target to GPU
            if self._cfg.run_configs.cuda:
                image, target = image.to(self._device), target.to(self._device)

            # update model
            self._optimizer.zero_grad()

            # convert image to suitable dims
            image = image.float()

            with torch.no_grad():

                # computes output of our model
                output = self._model(image)

                if self._num_sdn_ics <= 0:
                    output = [output]

                tracker, acc = update_trackers(tracker, output, target, idxs)

                # compute ic uncertainty TODO: Make more modular!
                if self._num_sdn_ics > 0:
                    if num_layers < 0:
                        if self._cfg.run_configs.cuda:
                            _, out_list = self._model.module.feature_list(image)
                        else:
                            _, out_list = self._model.feature_list(image)
                        num_layers = len(out_list)
                        logit_mahalanobis = torch.zeros((num_samples, num_layers), dtype=torch.float,
                                                        device=self._device)
                        sdn_switches = torch.zeros((num_samples, num_layers), dtype=torch.long, device=self._device)
                    preds = []

                    for layer in range(num_layers):
                        layer_score = get_sngp_score(output[layer])
                        logit_mahalanobis[idxs, layer] = layer_score

                        _, layer_pred = torch.max(output[layer].data, 1)

                        # calculate sdn switches
                        if len(preds) > 0:
                            for k in range(len(preds)):
                                sdn_switches[idxs, layer] += preds[-1].ne(layer_pred).long() if len(preds) > 0 else 0.0
                        preds.append(layer_pred)

                    preds = []
                    for l in range(len(output)):

                        _, pred = torch.max(output[l].data, 1)

                        # collect forgetting events
                        acc = pred.eq(target.data)

                        if len(preds) > 0:
                            for k in range(len(preds)):
                                prev_pred = preds[k]
                                change = prev_pred != pred
                                switch_uncertainties[idxs] += change
                        preds.append(pred)

                # take final layer as total output
                output = output[-1]

                # probabilities
                probs_output = self._softmax(output)
                probabilites[idxs] = probs_output
                # sngp uncertainties
                num_classes = output.shape[1]
                belief_mass = output.exp().sum(1)
                sngp_uncertainties[idxs] = num_classes / (belief_mass.float() + num_classes)

                total += target.size(0)

            correct_samples += acc.cpu().sum()

        # Update optimizer step
        if self._cfg.classification.optimization.scheduler != 'none':
            self._scheduler.step()

        # calculate accuracy
        acc = 100.0 * correct_samples.item() / total
        self._writer.add_scalar('test/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.classification.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        print('Test Accuracy: %.3f' % acc)
        out = {'predictions': predictions.cpu().numpy(),
               'probs': probabilites.cpu().numpy(),
               'accuracy': acc}

        if self._num_sdn_ics > 0:
            out['switch_uncertainties'] = switch_uncertainties.cpu().numpy()
            out['logit_mahalanobis'] = logit_mahalanobis.cpu().numpy()
            out['layer_switches'] = sdn_switches.cpu().numpy()
        if self._sngp_override:
            out['sngp_uncertainties'] = sngp_uncertainties.cpu().numpy()

        return out, acc, tracker

    def get_loaders(self, mode: str):
        if mode == 'test' or mode == 'train':
            return self._loaders
        else:
            raise Exception('Test mode not implemented yet')
