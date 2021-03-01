import random
import time
from contextlib import contextmanager
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Lambda

from datasets.exemplars_dataset import ExemplarsDataset
from networks.network import LLL_Net


class ExemplarsSelector:
    """Exemplar selector for approaches with an interface of Dataset"""

    def __init__(self, exemplars_dataset: ExemplarsDataset):
        self.exemplars_dataset = exemplars_dataset

    def __call__(self, model: LLL_Net, trn_loader: DataLoader, transform):
        clock0 = time.time()
        exemplars_per_class = self._exemplars_per_class_num(model)
        with override_dataset_transform(trn_loader.dataset, transform) as ds_for_selection:
            # change loader and fix to go sequentially (shuffle=False), keeps same order for later, eval transforms
            sel_loader = DataLoader(ds_for_selection, batch_size=trn_loader.batch_size, shuffle=False,
                                    num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            selected_indices = self._select_indices(model, sel_loader, exemplars_per_class, transform)
        with override_dataset_transform(trn_loader.dataset, Lambda(lambda x: np.array(x))) as ds_for_raw:
            x, y = zip(*(ds_for_raw[idx] for idx in selected_indices))
        clock1 = time.time()
        print('| Selected {:d} train exemplars, time={:5.1f}s'.format(len(x), clock1 - clock0))
        return x, y

    def _exemplars_per_class_num(self, model: LLL_Net):
        if self.exemplars_dataset.max_num_exemplars_per_class:
            return self.exemplars_dataset.max_num_exemplars_per_class

        num_cls = model.task_cls.sum().item()
        num_exemplars = self.exemplars_dataset.max_num_exemplars
        exemplars_per_class = int(np.ceil(num_exemplars / num_cls))
        assert exemplars_per_class > 0, \
            "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. " \
            "Limit of exemplars: {}".format(num_cls,
                                            num_exemplars)
        return exemplars_per_class

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform) -> Iterable:
        pass


class RandomExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on random selection, which produces a random list of samples."""

    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform) -> Iterable:
        num_cls = sum(model.task_cls)
        result = []
        labels = self._get_labels(sel_loader)
        for curr_cls in range(num_cls):
            # get all indices from current class -- check if there are exemplars from previous task in the loader
            cls_ind = np.where(labels == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # select the exemplars randomly
            result.extend(random.sample(list(cls_ind), exemplars_per_class))
        return result

    def _get_labels(self, sel_loader):
        if hasattr(sel_loader.dataset, 'labels'):  # BaseDataset, MemoryDataset
            labels = np.asarray(sel_loader.dataset.labels)
        elif isinstance(sel_loader.dataset, ConcatDataset):
            labels = []
            for ds in sel_loader.dataset.datasets:
                labels.extend(ds.labels)
            labels = np.array(labels)
        else:
            raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))
        return labels


class HerdingExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on herding selection, which produces a sorted list of samples of one
    class based on the distance to the mean sample of that class. From iCaRL algorithm 4 and 5:
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
    """
    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform) -> Iterable:
        model_device = next(model.parameters()).device  # we assume here that whole model is on a single device

        # extract outputs from the model for all train samples
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            model.eval()
            for images, targets in sel_loader:
                feats = model(images.to(model_device), return_features=True)[1]
                feats = feats / feats.norm(dim=1).view(-1, 1)  # Feature normalization
                extracted_features.append(feats)
                extracted_targets.extend(targets)
        extracted_features = (torch.cat(extracted_features)).cpu()
        extracted_targets = np.array(extracted_targets)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_feats = extracted_features[cls_ind]
            # calculate the mean
            cls_mu = cls_feats.mean(0)
            # select the exemplars closer to the mean of each class
            selected = []
            selected_feat = []
            for k in range(exemplars_per_class):
                # fix this to the dimension of the model features
                sum_others = torch.zeros(cls_feats.shape[1])
                for j in selected_feat:
                    sum_others += j / (k + 1)
                dist_min = np.inf
                # choose the closest to the mean of the current class
                for item in cls_ind:
                    if item not in selected:
                        feat = extracted_features[item]
                        dist = torch.norm(cls_mu - feat / (k + 1) - sum_others)
                        if dist < dist_min:
                            dist_min = dist
                            newone = item
                            newonefeat = feat
                selected_feat.append(newonefeat)
                selected.append(newone)
            result.extend(selected)
        return result


class EntropyExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on entropy selection, which produces a sorted list of samples of one
    class based on entropy of each sample. From RWalk http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112
    """
    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform) -> Iterable:
        model_device = next(model.parameters()).device  # we assume here that whole model is on a single device

        # extract outputs from the model for all train samples
        extracted_logits = []
        extracted_targets = []
        with torch.no_grad():
            model.eval()
            for images, targets in sel_loader:
                extracted_logits.append(torch.cat(model(images.to(model_device)), dim=1))
                extracted_targets.extend(targets)
        extracted_logits = (torch.cat(extracted_logits)).cpu()
        extracted_targets = np.array(extracted_targets)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_logits = extracted_logits[cls_ind]
            # select the exemplars with higher entropy (lower: -entropy)
            probs = torch.softmax(cls_logits, dim=1)
            log_probs = torch.log(probs)
            minus_entropy = (probs * log_probs).sum(1)  # change sign of this variable for inverse order
            selected = cls_ind[minus_entropy.sort()[1][:exemplars_per_class]]
            result.extend(selected)
        return result
    
    
class DistanceExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on distance-based selection, which produces a sorted list of samples of
    one class based on closeness to decision boundary of each sample. From RWalk
    http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112
    """
    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int,
                        transform) -> Iterable:
        model_device = next(model.parameters()).device  # we assume here that whole model is on a single device

        # extract outputs from the model for all train samples
        extracted_logits = []
        extracted_targets = []
        with torch.no_grad():
            model.eval()
            for images, targets in sel_loader:
                extracted_logits.append(torch.cat(model(images.to(model_device)), dim=1))
                extracted_targets.extend(targets)
        extracted_logits = (torch.cat(extracted_logits)).cpu()
        extracted_targets = np.array(extracted_targets)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_logits = extracted_logits[cls_ind]
            # select the exemplars closer to boundary
            distance = cls_logits[:, curr_cls]  # change sign of this variable for inverse order
            selected = cls_ind[distance.sort()[1][:exemplars_per_class]]
            result.extend(selected)
        return result


def dataset_transforms(dataset, transform_to_change):
    if isinstance(dataset, ConcatDataset):
        r = []
        for ds in dataset.datasets:
            r += dataset_transforms(ds, transform_to_change)
        return r
    else:
        old_transform = dataset.transform
        dataset.transform = transform_to_change
        return [(dataset, old_transform)]


@contextmanager
def override_dataset_transform(dataset, transform):
    try:
        datasets_with_orig_transform = dataset_transforms(dataset, transform)
        yield dataset
    finally:
        # get bac original transformations
        for ds, orig_transform in datasets_with_orig_transform:
            ds.transform = orig_transform
