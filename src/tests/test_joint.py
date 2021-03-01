from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 2 --lr-factor 10 --momentum 0.9 --lr-min 1e-7" \
                       " --num-workers 0"


def test_joint():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach joint"
    run_main_and_assert(args_line)
