import torch
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the freezing baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, freeze_after=0, all_outputs=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.freeze_after = freeze_after
        self.all_out = all_outputs

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--freeze-after', default=0, type=int, required=False,
                            help='Freeze model except current head after the specified task (default=%(default)s)')
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        return torch.optim.SGD(self._train_parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _has_exemplars(self):
        """Returns True in case exemplars are being used"""
        return self.exemplars_dataset is not None and len(self.exemplars_dataset) > 0

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        if t >= self.freeze_after:
            self.model.freeze_backbone()

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if t > 0 and self._has_exemplars():
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self._model_train(t)
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._train_parameters(), self.clipgrad)
            self.optimizer.step()

    def _model_train(self, t):
        """Freezes the necessary weights"""
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        if self.freeze_after >= 0 and t <= self.freeze_after:  # non-frozen task - whole model to train
            self.model.train()
        else:
            self.model.model.eval()
            if self._has_exemplars():
                # with exemplars - use all heads
                for head in self.model.heads:
                    head.train()
            else:
                # no exemplars - use current head
                self.model.heads[-1].train()

    def _train_parameters(self):
        """Includes the necessary weights to the optimizer"""
        if len(self.model.heads) <= (self.freeze_after + 1):
            return self.model.parameters()
        else:
            if self._has_exemplars():
                return [p for head in self.model.heads for p in head.parameters()]
            else:
                return self.model.heads[-1].parameters()

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or self._has_exemplars():
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
