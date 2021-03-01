import importlib
from copy import deepcopy
from argparse import ArgumentParser

import utils


class GridSearch:
    """Basic class for implementing hyperparameter grid search"""

    def __init__(self, appr_ft, seed, gs_config='gridsearch_config', acc_drop_thr=0.2, hparam_decay=0.5,
                 max_num_searches=7):
        self.seed = seed
        GridSearchConfig = getattr(importlib.import_module(name=gs_config), 'GridSearchConfig')
        self.appr_ft = appr_ft
        self.gs_config = GridSearchConfig()
        self.acc_drop_thr = acc_drop_thr
        self.hparam_decay = hparam_decay
        self.max_num_searches = max_num_searches
        self.lr_first = 1.0

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the GridSearch specific parameters"""
        parser = ArgumentParser()
        # Configuration file with a GridSearchConfig class with all necessary args
        parser.add_argument('--gridsearch-config', type=str, default='gridsearch_config', required=False,
                            help='Configuration file for GridSearch options (default=%(default)s)')
        # Accuracy threshold drop below which the search stops for that phase
        parser.add_argument('--gridsearch-acc-drop-thr', default=0.2, type=float, required=False,
                            help='GridSearch accuracy drop threshold (default=%(default)f)')
        # Value at which hyperparameters decay
        parser.add_argument('--gridsearch-hparam-decay', default=0.5, type=float, required=False,
                            help='GridSearch hyperparameter decay (default=%(default)f)')
        # Maximum number of searched before the search stops for that phase
        parser.add_argument('--gridsearch-max-num-searches', default=7, type=int, required=False,
                            help='GridSearch maximum number of hyperparameter search (default=%(default)f)')
        return parser.parse_known_args(args)

    def search_lr(self, model, t, trn_loader, val_loader):
        """Search for accuracy and best LR on finetuning"""
        best_ft_acc = 0.0
        best_ft_lr = 0.0

        # Get general parameters and fix the ones with only one value
        gen_params = self.gs_config.get_params('general')
        for k, v in gen_params.items():
            if not isinstance(v, list):
                setattr(self.appr_ft, k, v)
        if t > 0:
            # LR for search are 'lr_searches' largest LR below 'lr_first'
            list_lr = [lr for lr in gen_params['lr'] if lr < self.lr_first][:gen_params['lr_searches'][0]]
        else:
            # For first task, try larger LR range
            list_lr = gen_params['lr_first']

        # Iterate through the other variable parameters
        for curr_lr in list_lr:
            utils.seed_everything(seed=self.seed)
            self.appr_ft.model = deepcopy(model)
            self.appr_ft.lr = curr_lr
            self.appr_ft.train(t, trn_loader, val_loader)
            _, ft_acc_taw, _ = self.appr_ft.eval(t, val_loader)
            if ft_acc_taw > best_ft_acc:
                best_ft_acc = ft_acc_taw
                best_ft_lr = curr_lr
            print('Current best LR: ' + str(best_ft_lr))
        self.gs_config.current_lr = best_ft_lr
        print('Current best acc: {:5.1f}'.format(best_ft_acc * 100))
        # After first task, keep LR used
        if t == 0:
            self.lr_first = best_ft_lr

        return best_ft_acc, best_ft_lr

    def search_tradeoff(self, appr_name, appr, t, trn_loader, val_loader, best_ft_acc):
        """Search for less-forgetting tradeoff with minimum accuracy loss"""
        best_tradeoff = None
        tradeoff_name = None

        # Get general parameters and fix all the ones that have only one option
        appr_params = self.gs_config.get_params(appr_name)
        for k, v in appr_params.items():
            if isinstance(v, list):
                # get tradeoff name as the only one with multiple values
                tradeoff_name = k
            else:
                # Any other hyperparameters are fixed
                setattr(appr, k, v)

        # If there is no tradeoff, no need to gridsearch more
        if tradeoff_name is not None and t > 0:
            # get starting value for trade-off hyperparameter
            best_tradeoff = appr_params[tradeoff_name][0]
            # iterate through decreasing trade-off values -- limit to `max_num_searches` searches
            num_searches = 0
            while num_searches < self.max_num_searches:
                utils.seed_everything(seed=self.seed)
                # Make deepcopy of the appr without duplicating the logger
                appr_gs = type(appr)(deepcopy(appr.model), appr.device, exemplars_dataset=appr.exemplars_dataset)
                for attr, value in vars(appr).items():
                    if attr == 'logger':
                        setattr(appr_gs, attr, value)
                    else:
                        setattr(appr_gs, attr, deepcopy(value))

                # update tradeoff value
                setattr(appr_gs, tradeoff_name, best_tradeoff)
                # train this iteration
                appr_gs.train(t, trn_loader, val_loader)
                _, curr_acc, _ = appr_gs.eval(t, val_loader)
                print('Current acc: ' + str(curr_acc) + ' for ' + tradeoff_name + '=' + str(best_tradeoff))
                # Check if accuracy is within acceptable threshold drop
                if curr_acc < ((1 - self.acc_drop_thr) * best_ft_acc):
                    best_tradeoff = best_tradeoff * self.hparam_decay
                else:
                    break
                num_searches += 1
        else:
            print('There is no trade-off to gridsearch.')

        return best_tradeoff, tradeoff_name
