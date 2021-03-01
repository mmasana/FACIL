import pandas as pd
from pathlib import Path

from tests import run_main

FAST_LOCAL_TEST_ARGS = "--exp-name loggers_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 2 --lr-factor 10 --momentum 0.9 --lr-min 1e-7" \
                       " --num-workers 0 --approach finetuning"


def test_disk_and_tensorflow_logger():

    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --log disk tensorboard"
    result = run_main(args_line, 'results_test_loggers', clean_run=True)
    experiment_dir = Path(result[-1])

    # check disk logger
    assert experiment_dir.is_dir()
    raw_logs = list(experiment_dir.glob('raw_log-*.txt'))
    assert len(raw_logs) == 1
    df = pd.read_json(raw_logs[0], lines=True)
    assert sorted(df.iter.unique()) == [0, 1, 2]
    assert sorted(df.group.unique()) == ['test', 'train', 'valid']
    assert len(df.group.unique()) == 3

    # check tb logger
    tb_events_logs = list(experiment_dir.glob('events.out.tfevents*'))
    assert len(tb_events_logs) == 1
    assert experiment_dir.joinpath(tb_events_logs[0]).is_file()
