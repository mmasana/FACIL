import torch
import numpy as np

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Class Incremental Learning With Dual Memory (IL2M) approach described in
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf
    """

    def __init__(self, model, device, nepochs=100, lr=0.1, lr_min=1e-4, lr_factor=3, lr_patience=10, clipgrad=10000,
                 momentum=0.9, wd=0.0001, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.init_classes_means = []
        self.current_classes_means = []
        self.models_confidence = []
        # FLAG to not do scores rectification while finetuning training
        self.ft_train = False

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        assert (have_exemplars > 0), 'Error: IL2M needs exemplars.'

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def il2m(self, t, trn_loader):
        """Compute and store statistics for score rectification"""
        old_classes_number = sum(self.model.task_cls[:t])
        classes_counts = [0 for _ in range(sum(self.model.task_cls))]
        models_counts = 0

        # to store statistics for the classes as learned in the current incremental state
        self.current_classes_means = [0 for _ in range(old_classes_number)]
        # to store statistics for past classes as learned in their initial states
        for cls in range(old_classes_number, old_classes_number + self.model.task_cls[t]):
            self.init_classes_means.append(0)
        # to store statistics for model confidence in different states (i.e. avg top-1 pred scores)
        self.models_confidence.append(0)

        # compute the mean prediction scores that will be used to rectify scores in subsequent tasks
        with torch.no_grad():
            self.model.eval()
            for images, targets in trn_loader:
                outputs = self.model(images.to(self.device))
                scores = np.array(torch.cat(outputs, dim=1).data.cpu().numpy(), dtype=np.float)
                for m in range(len(targets)):
                    if targets[m] < old_classes_number:
                        # computation of class means for past classes of the current state.
                        self.current_classes_means[targets[m]] += scores[m, targets[m]]
                        classes_counts[targets[m]] += 1
                    else:
                        # compute the mean prediction scores for the new classes of the current state
                        self.init_classes_means[targets[m]] += scores[m, targets[m]]
                        classes_counts[targets[m]] += 1
                        # compute the mean top scores for the new classes of the current state
                        self.models_confidence[t] += np.max(scores[m, ])
                        models_counts += 1
        # Normalize by corresponding number of images
        for cls in range(old_classes_number):
            self.current_classes_means[cls] /= classes_counts[cls]
        for cls in range(old_classes_number, old_classes_number + self.model.task_cls[t]):
            self.init_classes_means[cls] /= classes_counts[cls]
        self.models_confidence[t] /= models_counts

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        self.ft_train = True
        super().train_loop(t, trn_loader, val_loader)
        self.ft_train = False

        # IL2M outputs rectification
        self.il2m(t, trn_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        if self.ft_train:
            # no score rectification while training
            hits_taw, hits_tag = super().calculate_metrics(outputs, targets)
        else:
            # Task-Aware Multi-Head
            pred = torch.zeros_like(targets.to(self.device))
            for m in range(len(pred)):
                this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
            hits_taw = (pred == targets.to(self.device)).float()
            # Task-Agnostic Multi-Head
            if self.multi_softmax:
                outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            # Eq. 1: rectify predicted scores
            old_classes_number = sum(self.model.task_cls[:-1])
            for m in range(len(targets)):
                rectified_outputs = torch.cat(outputs, dim=1)
                pred[m] = rectified_outputs[m].argmax()
                if old_classes_number:
                    # if the top-1 class predicted by the network is a new one, rectify the score
                    if int(pred[m]) >= old_classes_number:
                        for o in range(old_classes_number):
                            o_task = int((self.model.task_cls.cumsum(0) <= o).sum())
                            rectified_outputs[m, o] *= (self.init_classes_means[o] / self.current_classes_means[o]) * \
                                                       (self.models_confidence[-1] / self.models_confidence[o_task])
                        pred[m] = rectified_outputs[m].argmax()
                    # otherwise, rectification is not done because an old class is directly predicted
            hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
