from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 5 --seed 1 --batch-size 32" \
                       " --nepochs 2 --num-workers 0 --stop-at-task 3"


def test_finetuning_stop_at_task():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    run_main_and_assert(args_line)
