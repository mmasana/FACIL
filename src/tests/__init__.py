import os
import torch
import shutil

from main_incremental import main
import datasets.dataset_config as c


def run_main(args_line, result_dir='results_test', clean_run=False):
    assert "--results-path" not in args_line

    print('Staring dir:', os.getcwd())
    if os.getcwd().endswith('tests'):
        os.chdir('..')
    elif os.getcwd().endswith('IL_Survey'):
        os.chdir('src')
    elif os.getcwd().endswith('src'):
        print('CWD is OK.')
    print('Test CWD:', os.getcwd())
    test_results_path = os.getcwd() + f"/../{result_dir}"

    # for testing - use relative path to CWD
    c.dataset_config['mnist']['path'] = '../data'
    if os.path.exists(test_results_path) and clean_run:
        shutil.rmtree(test_results_path)
    os.makedirs(test_results_path, exist_ok=True)
    args_line += " --results-path {}".format(test_results_path)

    # if distributed test -- use all GPU
    worker_id = int(os.environ.get("PYTEST_XDIST_WORKER", "gw-1")[2:])
    if worker_id >= 0 and torch.cuda.is_available():
        gpu_idx = worker_id % torch.cuda.device_count()
        args_line += " --gpu {}".format(gpu_idx)

    print('ARGS:', args_line)
    return main(args_line.split(' '))


def run_main_and_assert(args_line,
                        taw_current_task_min=0.01,
                        tag_current_task_min=0.0,
                        result_dir='results_test'):
    acc_taw, acc_tag, forg_taw, forg_tag, exp_dir = run_main(args_line, result_dir)

    # acc matrices sanity check
    assert acc_tag.shape == acc_taw.shape
    assert acc_tag.shape == forg_tag.shape
    assert acc_tag.shape == forg_taw.shape

    # check current task performance
    assert all(acc_tag.diagonal() >= tag_current_task_min)
    assert all(acc_taw.diagonal() >= taw_current_task_min)
