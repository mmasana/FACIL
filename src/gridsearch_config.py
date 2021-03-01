class GridSearchConfig():
    def __init__(self):
        self.params = {
            'general': {
                'lr_first': [5e-1, 1e-1, 5e-2],
                'lr': [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
                'lr_searches': [3],
                'lr_min': 1e-4,
                'lr_factor': 3,
                'lr_patience': 10,
                'clipping': 10000,
                'momentum': 0.9,
                'wd': 0.0002
            },
            'finetuning': {
            },
            'freezing': {
            },
            'joint': {
            },
            'lwf': {
                'lamb': [10],
                'T': 2
            },
            'icarl': {
                'lamb': [4]
            },
            'dmc': {
                'aux_dataset': 'imagenet_32_reduced',
                'aux_batch_size': 128
            },
            'il2m': {
            },
            'eeil': {
                'lamb': [10],
                'T': 2,
                'lr_finetuning_factor': 0.1,
                'nepochs_finetuning': 40,
                'noise_grad': False
            },
            'bic': {
                'T': 2,
                'val_percentage': 0.1,
                'bias_epochs': 200
            },
            'lucir': {
                'lamda_base': [10],
                'lamda_mr': 1.0,
                'dist': 0.5,
                'K': 2
            },
            'lwm': {
                'beta': [2],
                'gamma': 1.0
            },
            'ewc': {
                'lamb': [10000]
            },
            'mas': {
                'lamb': [400]
            },
            'path_integral': {
                'lamb': [10],
            },
            'r_walk': {
                'lamb': [20],
            },
        }
        self.current_lr = self.params['general']['lr'][0]
        self.current_tradeoff = 0

    def get_params(self, approach):
        return self.params[approach]
