import torch
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data.dataloader import default_collate

from utils import cutmix_data
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the GDumb approach
    described in https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf
    Original code available at https://github.com/drimpossible/GDumb
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=0.0005, lr_factor=3, lr_patience=5, clipgrad=10.0,
                 momentum=0.9, wd=1e-6, multi_softmax=False, wu_nepochs=1, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, regularization='cutmix'):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.regularization = regularization
        self.init_model = None

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        assert (have_exemplars > 0), 'Error: GDumb needs exemplars.'

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--regularization', default='cutmix', required=False,
                            help='Use cutmix regularization (default=%(default)s)')
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        # 1. GDumb resets the network before learning a new task, relying only on the exemplars stored so far
        if t == 0:
            # Keep the randomly initialized model from time step 0
            self.init_model = deepcopy(self.model)
        else: 
            # Reinitialize the model (backbone) for very task from time step 1
            self.model.model = deepcopy(self.init_model.model)
            for layer in self.model.heads.children():
                layer.reset_parameters()

        # EXEMPLAR MANAGEMENT -- select training subset from current task and exemplar memory
        aux_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                 batch_size=trn_loader.batch_size,
                                                 shuffle=True,
                                                 num_workers=trn_loader.num_workers,
                                                 pin_memory=trn_loader.pin_memory)
        self.exemplars_dataset.collect_exemplars(self.model, aux_loader, val_loader.dataset.transform)

        # Set new set of exemplars as the only data to train on
        trn_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                 batch_size=trn_loader.batch_size,
                                                 shuffle=True,
                                                 num_workers=trn_loader.num_workers,
                                                 pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Get exemplars
            if len(self.exemplars_dataset) > 0:
                # 2. Balanced batches
                exemplar_indices = torch.randperm(len(self.exemplars_dataset))[:trn_loader.batch_size]
                images_exemplars, targets_exemplars = default_collate([self.exemplars_dataset[i]
                                                                       for i in exemplar_indices])
                images = torch.cat((images, images_exemplars), dim=0)
                targets = torch.cat((targets, targets_exemplars), dim=0) 
                
            # 3. Apply cutmix as regularization
            do_cutmix = self.regularization == 'cutmix' and np.random.rand(1) < 0.5  # cutmix_prob (Sec.4)
            if do_cutmix: 
                images, targets_a, targets_b, lamb = cutmix_data(x=images, y=targets, alpha=1.0)  # cutmix_alpha (Sec.4)
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = lamb * self.criterion(t, outputs, targets_a.to(self.device))
                loss += (1.0 - lamb) * self.criterion(t, outputs, targets_b.to(self.device))
            else:
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device))

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
