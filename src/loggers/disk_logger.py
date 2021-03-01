import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

from loggers.exp_logger import ExperimentLogger


class Logger(ExperimentLogger):
    """Characterizes a disk logger"""

    def __init__(self, log_path, exp_name, begin_time=None):
        super(Logger, self).__init__(log_path, exp_name, begin_time)

        self.begin_time_str = self.begin_time.strftime("%Y-%m-%d-%H-%M")

        # Duplicate standard outputs
        sys.stdout = FileOutputDuplicator(sys.stdout,
                                          os.path.join(self.exp_path, 'stdout-{}.txt'.format(self.begin_time_str)), 'w')
        sys.stderr = FileOutputDuplicator(sys.stderr,
                                          os.path.join(self.exp_path, 'stderr-{}.txt'.format(self.begin_time_str)), 'w')

        # Raw log file
        self.raw_log_file = open(os.path.join(self.exp_path, "raw_log-{}.txt".format(self.begin_time_str)), 'a')

    def log_scalar(self, task, iter, name, value, group=None, curtime=None):
        if curtime is None:
            curtime = datetime.now()

        # Raw dump
        entry = {"task": task, "iter": iter, "name": name, "value": value, "group": group,
                 "time": curtime.strftime("%Y-%m-%d-%H-%M")}
        self.raw_log_file.write(json.dumps(entry, sort_keys=True) + "\n")
        self.raw_log_file.flush()

    def log_args(self, args):
        with open(os.path.join(self.exp_path, 'args-{}.txt'.format(self.begin_time_str)), 'w') as f:
            json.dump(args.__dict__, f, separators=(',\n', ' : '), sort_keys=True)

    def log_result(self, array, name, step):
        if array.ndim <= 1:
            array = array[None]
        np.savetxt(os.path.join(self.exp_path, 'results', '{}-{}.txt'.format(name, self.begin_time_str)),
                   array, '%.6f', delimiter='\t')

    def log_figure(self, name, iter, figure, curtime=None):
        curtime = datetime.now()
        figure.savefig(os.path.join(self.exp_path, 'figures',
                                    '{}_{}-{}.png'.format(name, iter, curtime.strftime("%Y-%m-%d-%H-%M-%S"))))
        figure.savefig(os.path.join(self.exp_path, 'figures',
                                    '{}_{}-{}.pdf'.format(name, iter, curtime.strftime("%Y-%m-%d-%H-%M-%S"))))

    def save_model(self, state_dict, task):
        torch.save(state_dict, os.path.join(self.exp_path, "models", "task{}.ckpt".format(task)))

    def __del__(self):
        self.raw_log_file.close()


class FileOutputDuplicator(object):
    def __init__(self, duplicate, fname, mode):
        self.file = open(fname, mode)
        self.duplicate = duplicate

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.duplicate.write(data)

    def flush(self):
        self.file.flush()
