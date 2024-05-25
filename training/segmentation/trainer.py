import os
import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Union
from torch.utils.data import DataLoader
from models.architectures import build_segmentation
from data.datasets.segmentation.common.acquisition import get_dataset
from training.segmentation.segmentationtracker import SegmentationTracker
from training.segmentation.evaluator import Evaluator
from training.segmentation.utils import TestOutput, enable_dropout, gen_folder
from training.common.trainutils import determine_multilr_milestones
from training.common.saver import Saver
from training.common.summaries import TensorboardSummary
from config import BaseConfig


class SegmentationTrainer:
    def __init__(self, cfg: BaseConfig):
        self._cfg = cfg
        self._epochs = cfg.segmentation.epochs
        self._device = cfg.run_configs.gpu_id

        # Define Saver
        self._saver = Saver(cfg)

        # Define Tensorboard Summary
        self._summary = TensorboardSummary(self._saver.experiment_dir)
        self._writer = self._summary.create_summary()

        self._loaders = get_dataset(cfg)

        model_dict = build_segmentation(self._cfg.segmentation.model, self._loaders.data_config)
        self._model, self._mcd_iterations = model_dict['model'], model_dict['mcd_override']

        self._softmax = nn.Softmax(dim=1)

        # evaluator
        self._evaluator = Evaluator(self._loaders.data_config.num_classes)
        self._num_classes = self._loaders.data_config.num_classes

        # create cols for output df
        self._cols = ['CA', 'MCA', 'mIoU']
        for i in range(self._num_classes):
            self._cols.append('class' + str(i))

        # Using cuda
        if self._cfg.run_configs.cuda:
            # use multiple GPUs if available
            self._model = torch.nn.DataParallel(self._model,
                                                device_ids=[self._cfg.run_configs.gpu_id])
        else:
            self._device = torch.device('cpu')

        if self._cfg.segmentation.loss == 'ce':
            if self._loaders.class_weights is not None:
                weight = self._loaders.class_weights.to(self._device)
                self._loss = nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
            else:
                self._loss = nn.CrossEntropyLoss(ignore_index=-1)

        else:
            raise Exception('Loss not implemented yet!')

        if self._cfg.segmentation.optimization.optimizer == 'adam':
            self._optimizer = torch.optim.Adam(self._model.parameters(),
                                               lr=self._cfg.segmentation.optimization.lr)
        elif self._cfg.segmentation.optimization.optimizer == 'sgd':
            self._optimizer = torch.optim.SGD(self._model.parameters(),
                                              lr=self._cfg.segmentation.optimization.lr,
                                              momentum=0.9,
                                              nesterov=True,
                                              weight_decay=5e-4)
        else:
            raise Exception('Optimizer not implemented yet!')

        if self._cfg.segmentation.optimization.scheduler == 'multiLR':
            milestones = determine_multilr_milestones(self._epochs, self._cfg.segmentation.optimization.multiLR_steps)
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   milestones=milestones,
                                                                   gamma=self._cfg.segmentation.optimization.gamma)
        elif self._cfg.segmentation.optimization.scheduler == 'none':
            pass
        else:
            raise Exception('Scheduler not implemented yet!')

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
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))

    def training(self, epoch, save_checkpoint=False):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.train()
        # initializes cool bar for visualization
        tbar = tqdm.tqdm(self._loaders.train_loader)
        num_img_tr = len(self._loaders.train_loader)

        # init statistics parameters
        train_loss = 0.0
        correct_samples = 0
        total = 0

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
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

            _, pred = torch.max(output.data, 1)
            total += target.size(0)

            # collect forgetting events
            acc = pred.eq(target.data)

            # check if prediction has changed
            predicted = pred.cpu().numpy()
            # self._train_statistics.update(acc.cpu().numpy(), predicted, idxs.cpu().numpy())

            # Perform model update
            # calculate loss
            loss = self._loss(output, target.long())
            # perform backpropagation
            loss.backward()

            # update params with gradients
            self._optimizer.step()

            # extract loss value as float and add to train_loss
            train_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self._writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            correct_samples += acc.cpu().sum()

        # Update optimizer step
        if self._cfg.segmentation.optimization.scheduler != 'none':
            self._scheduler.step()

        # calculate accuracy
        self._writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.segmentation.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        # save checkpoint
        if save_checkpoint:
            self._saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            })
        return acc

    def test_sample(self, sample: Dict[str, torch.Tensor], tracker: SegmentationTracker, loader: DataLoader,
                    usp_predictions: np.ndarray = None, mc_iterations: int = -1) -> Dict[str,
                                                                                         Union[torch.Tensor, np.ndarray,
                                                                                               SegmentationTracker]]:
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
            if mc_iterations <= 0:
                output = self._model(image)
            else:
                for k in range(mc_iterations):
                    cur_output = self._model(image)
                    if k == 0:
                        output = self._softmax(cur_output)
                    else:
                        output += self._softmax(cur_output)
                output /= mc_iterations

            probs_output = self._softmax(output)
            pred = torch.argmax(output, dim=1)

            logs = torch.log2(probs_output)
            mult = logs * probs_output
            entropy = -1 * torch.sum(mult, dim=1)

            # collect forgetting events
            acc = pred.eq(target.data)

            # check if prediction has changed
            predicted = pred.cpu().numpy()
            if tracker is not None:
                tracker.update(acc=acc.cpu().numpy(), pred=predicted, idxs=idxs.cpu().numpy())

            # Perform model update
            # calculate loss
            loss = self._loss(output, target.long())

        # track predictions
        if usp_predictions is None:
            usp_predictions = np.zeros((len(loader.dataset),) + target.shape[1:], dtype=int)

        usp_predictions[idxs] = predicted

        self._evaluator.add_batch(target.cpu().numpy(), predicted)

        output = {
            'loss': loss,
            'tracker': tracker,
            'usp_predictions': usp_predictions,
            'entropy': entropy.cpu().numpy(),
            'predicted': pred.cpu().numpy(),
            'target': target.cpu().numpy()
        }

        return output

    def testing(self, epoch: int, loader: DataLoader, name: str, tracker: SegmentationTracker = None,
                track_images: bool = False):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        self._model.eval()

        if self._mcd_iterations > 0:
            enable_dropout(self._model)

        # initializes cool bar for visualization
        tbar = tqdm.tqdm(loader)
        num_img_tr = len(loader)

        # init statistics parameters
        test_loss = 0.0
        total = 0

        # reset evaluator
        self._evaluator.reset()

        # init predictions array
        usp_predictions = None

        # generate folder
        folder = os.path.expanduser(self._cfg.run_configs.ld_folder_name + '/images/entropy/' + name)
        pred_folder = os.path.expanduser(self._cfg.run_configs.ld_folder_name + '/predictions/' + name)
        gt_folder = os.path.expanduser(self._cfg.run_configs.ld_folder_name + '/gt/' + name)
        if track_images:
            gen_folder(folder)
            gen_folder(pred_folder)
            gen_folder(gt_folder)

        # iterate over all samples in each batch i
        for i, sample in enumerate(tbar):
            total += sample['label'].size(0)

            test_output = self.test_sample(sample, tracker, loader, usp_predictions,
                                           mc_iterations=self._mcd_iterations)

            tracker, usp_predictions, loss = test_output['tracker'], test_output['usp_predictions'], test_output['loss']

            if track_images:
                for b in range(sample['label'].size(0)):
                    number = sample['idx'][b]
                    file_name = folder + f'/{number}.npy'
                    pred_file_name = pred_folder + f'/{number}.npy'
                    gt_file_name = gt_folder + f'/{number}.npy'
                    np.save(file_name, test_output['entropy'][b, ...])
                    np.save(pred_file_name, test_output['predicted'][b, ...])
                    np.save(gt_file_name, test_output['target'][b, ...])
                    tbar.set_description(f'Save: {b}')

            test_loss += loss.item()

            # Do fun bar stuff
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            self._writer.add_scalar('test/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        # calculate accuracy
        self._writer.add_scalar('test/total_loss_epoch', test_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.segmentation.batch_size +
                                               sample['label'].data.shape[0]))
        print('Loss: %.3f' % test_loss)
        acc = self._evaluator.Pixel_Accuracy()
        acc_class, acc_all_class = self._evaluator.Pixel_Accuracy_Class()
        mIoU = self._evaluator.Mean_Intersection_over_Union()
        df = pd.DataFrame(columns=self._cols)
        df.loc[0, 'CA'] = acc
        df.loc[0, 'MCA'] = acc_class
        df.loc[0, 'mIoU'] = mIoU
        for i in range(self._num_classes):
            key = 'class' + str(i)
            df.loc[0, key] = acc_all_class[i]

        print(df)

        # define output
        ret = TestOutput(out_df=df, out_tracker=tracker, prediction=usp_predictions)

        return ret

    def mcd_uncertainties(self, epoch: int, loader: DataLoader, name: str, tracker: SegmentationTracker = None):
        '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
        for one epoch'''
        # sets model into training mode -> important for dropout batchnorm. etc.
        mc_iterations = 15
        self._model.eval()
        enable_dropout(self._model)
        # initializes cool bar for visualization
        tbar = tqdm.tqdm(loader)
        num_img_tr = len(loader)

        # init statistics parameters
        test_loss = 0.0
        total = 0

        # reset evaluator
        self._evaluator.reset()

        # init predictions array
        usp_predictions = None

        # generate folder
        folder = os.path.expanduser(self._cfg.run_configs.ld_folder_name + '/images/entropy/' + name)
        pred_folder = os.path.expanduser(self._cfg.run_configs.ld_folder_name + '/predictions/' + name)
        gt_folder = os.path.expanduser(self._cfg.run_configs.ld_folder_name + '/gt/' + name)
        gen_folder(folder)
        gen_folder(pred_folder)
        gen_folder(gt_folder)

        # iterate over all samples in each batch i
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


                probs_output = (1 / mc_iterations) * final
                pred = torch.argmax(probs_output, dim=1)
                total += target.size(0)
                logs = torch.log2(probs_output)
                mult = logs * probs_output
                entropy = -1 * torch.sum(mult, dim=1)
                entropy = entropy.cpu().numpy()
                predicted = pred.cpu().numpy()
                targets = target.cpu().numpy()

                for b in range(target.size(0)):
                    number = sample['idx'][b]
                    file_name = folder + f'/{number}.npy'
                    pred_file_name = pred_folder + f'/{number}.npy'
                    gt_file_name = gt_folder + f'/{number}.npy'
                    np.save(file_name, entropy[b, ...])
                    np.save(pred_file_name, predicted[b, ...])
                    np.save(gt_file_name, targets[b, ...])
                    tbar.set_description(f'MCD: {k} Save: {b}')

                # collect forgetting events
                acc = pred.eq(target.data)

                # check if prediction has changed
                if tracker is not None:
                    tracker.update(acc=acc.cpu().numpy(), pred=predicted, idxs=idxs.cpu().numpy())

                # Perform model update
                # calculate loss
                loss = self._loss(output, target.long())

            # track predictions
            if usp_predictions is None:
                usp_predictions = np.zeros((len(loader.dataset),) + target.shape[1:], dtype=int)

            usp_predictions[idxs] = predicted

            self._evaluator.add_batch(target.cpu().numpy(), predicted)
            # extract loss value as float and add to train_loss
            test_loss += loss.item()

            # Do fun bar stuff
            self._writer.add_scalar('test/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        # calculate accuracy
        self._writer.add_scalar('test/total_loss_epoch', test_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self._cfg.segmentation.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % test_loss)
        acc = self._evaluator.Pixel_Accuracy()
        acc_class, acc_all_class = self._evaluator.Pixel_Accuracy_Class()
        mIoU = self._evaluator.Mean_Intersection_over_Union()
        df = pd.DataFrame(columns=self._cols)
        df.loc[0, 'CA'] = acc
        df.loc[0, 'MCA'] = acc_class
        df.loc[0, 'mIoU'] = mIoU
        for i in range(self._num_classes):
            key = 'class' + str(i)
            df.loc[0, key] = acc_all_class[i]

        print(df)

    def save_statistics(self, directory: str, tracker: SegmentationTracker, ld_type: str):
        tracker.save_statistics(directory, ld_type=ld_type)

    def get_loaders(self):
        return self._loaders


