import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from data.datasets.classification.CINIC10.dataset import LoaderCINIC10
from data.datasets.classification.CIFAR10C.configuration import CORRUPTIONS
from data.datasets.classification.common.dataobjects import ClassificationStructure


def get_test_targets(dataset: str, imbalance: int = None):
    path = os.path.expanduser('~/datasets/')
    path = os.path.expanduser('~/data/')
    if dataset == 'CIFAR10' or dataset[:-1] in CORRUPTIONS.keys() or dataset == 'test':
        target = np.array(datasets.CIFAR10(path + '/CIFAR10', train=False, download=False).targets)
        num_classes = 10
    elif dataset == 'CIFAR100':
        target = np.array(datasets.CIFAR100(path + '/CIFAR100', train=False, download=False).targets)
        num_classes = 100
    elif dataset == 'MNIST':
        target = datasets.MNIST(os.path.expanduser(path) + '/MNIST', train=False, download=False).train_labels
        num_classes = 10
    elif dataset == 'STL10':
        target = datasets.STL10(path + '/STL10', split='test', download=False).labels
        num_classes = 10
    elif dataset == 'SVHN':
        target = datasets.SVHN(os.path.expanduser(path) + '/SVHN', split='test', download=False).labels
        num_classes = 10
    elif dataset == 'CINIC10':
        raw_te = datasets.ImageFolder(os.path.expanduser(path) + '/CINIC10/test')
        data_config = ClassificationStructure()
        data_config.test_set = raw_te
        data_config.test_len = len(raw_te)
        test_tr = transforms.Compose([transforms.ToTensor()])
        cinic_loader = DataLoader(LoaderCINIC10(data_config=data_config,
                                                split='test',
                                                transform=test_tr),
                                  batch_size=len(raw_te),
                                  shuffle=False)
        sample = next(iter(cinic_loader))
        target = sample['label'].cpu().numpy()
        num_classes = 10
    else:
        raise Exception('Dataset not implemented yet!')

    if imbalance is not None:
        if imbalance < 0 or imbalance > num_classes:
            raise ValueError(f'imbalance value must be in % and within the range [0, 100]')

        if imbalance != 0:
            num_unbalanced_total = int(imbalance / 100 * num_classes)
            num_unbalance = 0.5 * num_unbalanced_total
            for i in range(num_unbalanced_total):
                if i >= num_unbalance:
                    im_idxs = np.argwhere(target == i)
                    remove_idxs = im_idxs[:90]
                    remove_idxs = np.squeeze(remove_idxs)

                    target = np.delete(target, remove_idxs)

    return target, num_classes