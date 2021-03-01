import torch
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Path Integral (aka Synaptic Intelligence) approach
    described in http://proceedings.mlr.press/v70/zenke17a.html
    Original code available at https://github.com/ganguli-lab/pathint
    """

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, lamb=0.1, damping=0.1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.lamb = lamb
        self.damping = damping

        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext = self.model.model
        # Page 3, following Eq. 3: "The w now have an intuitive interpretation as the parameter specific contribution to
        #  changes in the total loss."
        self.w = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters() if p.requires_grad}
        # Store current parameters as the initial parameters before first task starts
        self.older_params = {n: p.clone().detach().to(self.device) for n, p in feat_ext.named_parameters()
                             if p.requires_grad}
        # Store importance weights matrices
        self.importance = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                           if p.requires_grad}

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Eq. 4: lamb is the 'c' trade-off parameter from the surrogate loss -- 1e-3 < c < 0.1
        parser.add_argument('--lamb', default=0.1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Eq. 5: damping parameter is set to 0.1 in the MNIST case
        parser.add_argument('--damping', default=0.1, type=float, required=False,
                            help='Damping (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

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
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Eq. 5: accumulate Omega regularization strength (importance matrix)
        with torch.no_grad():
            curr_params = {n: p for n, p in self.model.model.named_parameters() if p.requires_grad}
            for n, p in self.importance.items():
                p += self.w[n] / ((curr_params[n] - self.older_params[n]) ** 2 + self.damping)
                self.w[n].zero_()

        # Store current parameters for the next task
        self.older_params = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # store current model without heads
            curr_feat_ext = {n: p.clone().detach() for n, p in self.model.model.named_parameters() if p.requires_grad}

            # Forward current model
            outputs = self.model(images.to(self.device))
            # theoretically this is the correct one for 2 tasks, however, for more tasks maybe is the current loss
            # check https://github.com/ganguli-lab/pathint/blob/master/pathint/optimizers.py line 123
            # cross-entropy loss on current task
            if len(self.exemplars_dataset) == 0:
                loss = torch.nn.functional.cross_entropy(outputs[t], targets.to(self.device) - self.model.task_offset[t])
            else:
                # with exemplars we check output from all heads (train data has all labels)
                loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets.to(self.device))
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # store gradients without regularization term
            unreg_grads = {n: p.grad.clone().detach() for n, p in self.model.model.named_parameters()
                            if p.grad is not None}
            # apply loss with path integral regularization
            loss = self.criterion(t, outputs, targets.to(self.device))

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            # Eq. 3: accumulate w, compute the path integral -- "In practice, we can approximate w online as the running
            #  sum of the product of the gradient with the parameter update".
            with torch.no_grad():
                for n, p in self.model.model.named_parameters():
                    if n in unreg_grads.keys():
                        # w[n] >=0, but minus for loss decrease
                        self.w[n] -= unreg_grads[n] * (p.detach() - curr_feat_ext[n])

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            loss_reg = 0
            # Eq. 4: quadratic surrogate loss
            for n, p in self.model.model.named_parameters():
                loss_reg += torch.sum(self.importance[n] * (p - self.older_params[n]).pow(2))
            loss += self.lamb * loss_reg
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
