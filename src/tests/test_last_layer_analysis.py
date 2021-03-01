from tests import run_main

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 2 --lr-factor 10 --momentum 0.9 --lr-min 1e-7" \
                       " --num-workers 0" \
                       " --approach finetuning"


def test_last_layer_analysis():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --last-layer-analysis"
    run_main(args_line)

