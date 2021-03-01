from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 2 --lr-factor 10 --momentum 0.9 --lr-min 1e-7" \
                       " --num-workers 0"


def test_finetuning_without_multisoftmax():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    run_main_and_assert(args_line)


def test_finetuning_with_multisoftmax():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    args_line += " --multi-softmax"
    run_main_and_assert(args_line)
