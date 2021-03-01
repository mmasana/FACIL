# Loggers

We include a disk logger, which logs into files and folders in the disk. We also provide a tensorboard logger which
provides a faster way of analysing a training process without need of further development. They can be specified with
`--log` followed by `disk`, `tensorboard` or both. Custom loggers can be defined by inheriting the `ExperimentLogger`
in [exp_logger.py](exp_logger.py).

When enabled, both loggers will output everything in the path `[RESULTS_PATH]/[DATASETS]_[APPROACH]_[EXP_NAME]` or
`[RESULTS_PATH]/[DATASETS]_[APPROACH]` if `--exp-name` is not set.

## Disk logger
The disk logger outputs the following file and folder structure:
- **figures/**: folder where generated figures are logged.
- **models/**: folder where model weight checkpoints are saved.
- **results/**: folder containing the results.
  - **acc_tag**: task-agnostic accuracy table.
  - **acc_taw**: task-aware accuracy table.
  - **avg_acc_tag**: task-agnostic average accuracies.
  - **avg_acc_taw**: task-agnostic average accuracies.
  - **forg_tag**: task-agnostic forgetting table.
  - **forg_taw**: task-aware forgetting table.
  - **wavg_acc_tag**: task-agnostic average accuracies weighted according to the number of classes of each task.
  - **wavg_acc_taw**: task-aware average accuracies weighted according to the number of classes of each task.
- **raw_log**: json file containing all the logged metrics easily read by many tools (e.g. `pandas`).
- stdout: a copy from the standard output of the terminal.
- stderr: a copy from the error output of the terminal.

## TensorBoard logger
The tensorboard logger outputs analogous metrics to the disk logger separated into different tabs according to the task
and different graphs according to the data splits.

Screenshot for a 10 task experiment, showing the last task plots:
<p align="center">
<img src="/docs/_static/tb2.png" alt="Tensorboard Screenshot" width="920"/>
</p>
