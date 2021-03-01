import importlib
from argparse import ArgumentParser

from datasets.memory_dataset import MemoryDataset


class ExemplarsDataset(MemoryDataset):
    """Exemplar storage for approaches with an interface of Dataset"""

    def __init__(self, transform, class_indices,
                 num_exemplars=0, num_exemplars_per_class=0, exemplar_selection='random'):
        super().__init__({'x': [], 'y': []}, transform, class_indices=class_indices)
        self.max_num_exemplars_per_class = num_exemplars_per_class
        self.max_num_exemplars = num_exemplars
        assert (num_exemplars_per_class == 0) or (num_exemplars == 0), 'Cannot use both limits at once!'
        cls_name = "{}ExemplarsSelector".format(exemplar_selection.capitalize())
        selector_cls = getattr(importlib.import_module(name='datasets.exemplars_selection'), cls_name)
        self.exemplars_selector = selector_cls(self)

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser("Exemplars Management Parameters")
        _group = parser.add_mutually_exclusive_group()
        _group.add_argument('--num-exemplars', default=0, type=int, required=False,
                            help='Fixed memory, total number of exemplars (default=%(default)s)')
        _group.add_argument('--num-exemplars-per-class', default=0, type=int, required=False,
                            help='Growing memory, number of exemplars per class (default=%(default)s)')
        parser.add_argument('--exemplar-selection', default='random', type=str,
                            choices=['herding', 'random', 'entropy', 'distance'],
                            required=False, help='Exemplar selection strategy (default=%(default)s)')
        return parser.parse_known_args(args)

    def _is_active(self):
        return self.max_num_exemplars_per_class > 0 or self.max_num_exemplars > 0

    def collect_exemplars(self, model, trn_loader, selection_transform):
        if self._is_active():
            self.images, self.labels = self.exemplars_selector(model, trn_loader, selection_transform)
