import numpy as np
import itertools
import os
import random
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)

def get_current_consistency_weight(consistency_weight, epoch, consistency_rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency_weight * sigmoid_rampup(epoch, consistency_rampup)



def get_label_idx(dataset):
    labeled_idxs, unlabeled_idxs = [],[]
    sup_keys, unsup_keys = dataset.sup_keys, dataset.unsup_keys
    for idx in range(len(dataset)):
        path = dataset.X_train.iloc[idx]
        eye_x  = dataset.y_train.iloc[idx]['eye_x']
        filename = os.path.basename(path)
        if (path, eye_x) in sup_keys:
            labeled_idxs.append(idx)
        elif (path, eye_x) in unsup_keys:
            unlabeled_idxs.append(idx)
        else:
            print(f"{(path, eye_x)} not in either key!")

    labeled_idxs = sorted(set(range(len(dataset.X_train))) - set(unlabeled_idxs))
    print("Num labeled: {}, num unlabeled: {}".format(len(labeled_idxs), len(unlabeled_idxs)))

    return labeled_idxs, unlabeled_idxs


def get_label_idx_fulldata(dataset):
    labeled_idxs, unlabeled_idxs = [],[]
    sup_keys, unsup_keys = dataset.sup_keys, dataset.unsup_keys
    for idx in range(len(dataset)):
        path = dataset.X_train.iloc[idx]
        eye_x  = dataset.y_train.iloc[idx]['eye_x']
        eye_y  = dataset.y_train.iloc[idx]['eye_y']
        filename = os.path.basename(path)
        if (path, eye_x, eye_y) in sup_keys:
            labeled_idxs.append(idx)
        elif (path, eye_x, eye_y) in unsup_keys:
            unlabeled_idxs.append(idx)
        else:
            print(f"{(path, eye_x, eye_y)} not in either key!")

    labeled_idxs = sorted(set(range(len(dataset.X_train))) - set(unlabeled_idxs))
    print("Num labeled: {}, num unlabeled: {}".format(len(labeled_idxs), len(unlabeled_idxs)))

    return labeled_idxs, unlabeled_idxs


def get_label_idx_videoatt(dataset):
    labeled_idxs, unlabeled_idxs = [],[]
    sup_labels = dataset.sup_labels_person
    for idx in range(len(dataset)):
        if sup_labels[idx]:
            labeled_idxs.append(idx)
        else:
            unlabeled_idxs.append(idx)

    print("Num labeled: {}, num unlabeled: {}".format(len(labeled_idxs), len(unlabeled_idxs)))

    return labeled_idxs, unlabeled_idxs



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        
        return (
            #tuple(random.shuffle(list(primary_batch + secondary_batch)))
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



class TwoStreamBatchSampler_Distributed(DistributedSampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        indices = list(super().__iter__())
        primary_indices = list(set(self.primary_indices).union(set(indices)))
        primary_iter = iterate_once(primary_indices)
        secondary_indices = list(set(self.secondary_indices).union(set(indices)))
        secondary_iter = iterate_eternally(secondary_indices)
        
        return (
            #tuple(random.shuffle(list(primary_batch + secondary_batch)))
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)