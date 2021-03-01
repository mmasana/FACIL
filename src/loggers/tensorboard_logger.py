from torch.utils.tensorboard import SummaryWriter

from loggers.exp_logger import ExperimentLogger
import json
import numpy as np


class Logger(ExperimentLogger):
    """Characterizes a Tensorboard logger"""

    def __init__(self, log_path, exp_name, begin_time=None):
        super(Logger, self).__init__(log_path, exp_name, begin_time)
        self.tbwriter = SummaryWriter(self.exp_path)

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        self.tbwriter.add_scalar(tag="t{}/{}_{}".format(task, group, name),
                                 scalar_value=value,
                                 global_step=iter)
        self.tbwriter.file_writer.flush()

    def log_figure(self, name, iter, figure, curtime=None):
        self.tbwriter.add_figure(tag=name, figure=figure, global_step=iter)
        self.tbwriter.file_writer.flush()

    def log_args(self, args):
        self.tbwriter.add_text(
            'args',
            json.dumps(args.__dict__,
                       separators=(',\n', ' : '),
                       sort_keys=True))
        self.tbwriter.file_writer.flush()

    def log_result(self, array, name, step):
        if array.ndim == 1:
            # log as scalars
            self.tbwriter.add_scalar(f'results/{name}', array[step], step)

        elif array.ndim == 2:
            s = ""
            i = step
            # for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                s += '{:5.1f}% '.format(100 * array[i, j])
            if np.trace(array) == 0.0:
                if i > 0:
                    s += '\tAvg.:{:5.1f}% \n'.format(100 * array[i, :i].mean())
            else:
                s += '\tAvg.:{:5.1f}% \n'.format(100 * array[i, :i + 1].mean())
            self.tbwriter.add_text(f'results/{name}', s, step)

    def __del__(self):
        self.tbwriter.close()
