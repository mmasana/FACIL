import torch
from torch import nn
from copy import deepcopy
from argparse import ArgumentParser

from datasets.data_loader import get_loaders
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """ Class implementing the Deep Model Consolidation (DMC) approach
    described in https://arxiv.org/abs/1903.07864
    Original code available at https://github.com/juntingzh/incremental-learning-baselines
    """

    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, aux_dataset='imagenet_32', aux_batch_size=128):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.model_new = None
        self.aux_dataset = aux_dataset
        self.aux_batch_size = aux_batch_size
        # get dataloader for auxiliar dataset
        aux_trn_ldr, _, aux_val_ldr, _ = get_loaders([self.aux_dataset], num_tasks=1, nc_first_task=None, validation=0,
                                                     batch_size=self.aux_batch_size, num_workers=4, pin_memory=False)
        self.aux_trn_loader = aux_trn_ldr[0]
        self.aux_val_loader = aux_val_ldr[0]
        # Since an auxiliary dataset is available, using exemplars could be redundant
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        assert (have_exemplars == 0), 'Warning: DMC does not use exemplars. Comment this line to force it.'

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4.2.1 "We use ImageNet32x32 dataset as the source for auxiliary data in the model consolidation stage."
        parser.add_argument('--aux-dataset', default='imagenet_32_reduced', type=str, required=False,
                            help='Auxiliary dataset (default=%(default)s)')
        parser.add_argument('--aux-batch-size', default=128, type=int, required=False,
                            help='Batch size for auxiliary dataset (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        if t > 0:
            # Re-initialize model
            for m in self.model.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    m.reset_parameters()
            # Get new model
            self.model_new = deepcopy(self.model)
            for h in self.model_new.heads[:-1]:
                with torch.no_grad():
                    h.weight.zero_()
                    h.bias.zero_()
                for p in h.parameters():
                    p.requires_grad = False
        else:
            self.model_new = self.model

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        if t > 0:
            # Args for the new data trainer and for the student trainer are the same
            dmc_args = dict(nepochs=self.nepochs, lr=self.lr, lr_min=self.lr_min, lr_factor=self.lr_factor,
                            lr_patience=self.lr_patience, clipgrad=self.clipgrad, momentum=self.momentum,
                            wd=self.wd, multi_softmax=self.multi_softmax, wu_nepochs=self.warmup_epochs,
                            wu_lr_factor=self.warmup_lr, fix_bn=self.fix_bn, logger=self.logger)
            # Train new model in new data
            new_trainer = NewTaskTrainer(self.model_new, self.device, **dmc_args)
            new_trainer.train_loop(t, trn_loader, val_loader)
            self.model_new.eval()
            self.model_new.freeze_all()
            print('=' * 108)
            print("Training of student")
            print('=' * 108)
            # Train student model using both old and new model
            student_trainer = StudentTrainer(self.model, self.model_new, self.model_old, self.device, **dmc_args)
            student_trainer.train_loop(t, self.aux_trn_loader, self.aux_val_loader)
        else:
            # FINETUNING TRAINING -- contains the epochs loop
            super().train_loop(t, trn_loader, val_loader)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()


class NewTaskTrainer(Inc_Learning_Appr):
    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None):
        super(NewTaskTrainer, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad,
                                             momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor, fix_bn,
                                             eval_on_train, logger)


class StudentTrainer(Inc_Learning_Appr):
    def __init__(self, model, model_new, model_old, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10,
                 lr_patience=8, clipgrad=10000, momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0,
                 wu_lr_factor=1, fix_bn=False, eval_on_train=False, logger=None):
        super(StudentTrainer, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad,
                                             momentum, wd, multi_softmax, wu_nepochs, wu_lr_factor, fix_bn,
                                             eval_on_train, logger)

        self.model_old = model_old
        self.model_new = model_new

    # Runs a single epoch of student's training
    def train_epoch(self, t, trn_loader):
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.cuda(), targets.cuda()
            # Forward old and new model
            targets_old = self.model_old(images)
            targets_new = self.model_new(images)
            # Forward current model
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets_old, targets_new)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    # Contains the evaluation code for evaluating the student
    def eval(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                images = images.cuda()
                # Forward old and new model
                targets_old = self.model_old(images)
                targets_new = self.model_new(images)
                # Forward current model
                outputs = self.model(images)
                loss = self.criterion(t, outputs, targets_old, targets_new)
                # Log
                total_loss += loss.item() * len(targets)
                total_num += len(targets)
        return total_loss / total_num, -1, -1

    # Returns the loss value for the student
    def criterion(self, t, outputs, targets_old, targets_new=None):
        # Eq. 2: Model Consolidation
        with torch.no_grad():
            # Eq. 4: "The regression target of the consolidated model is the concatenation of normalized logits of
            # the two specialist models."
            targets = torch.cat(targets_old[:t] + [targets_new[t]], dim=1)
            targets -= targets.mean(0)
        # Eq. 3: Double Distillation Loss
        return torch.nn.functional.mse_loss(torch.cat(outputs, dim=1), targets.detach(), reduction='mean')
