from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 3" \
                       " --num-workers 0" \
                       " --approach il2m"


def test_il2m():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


def test_il2m_with_warmup():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --warmup-nepochs 5"
    args_line += " --warmup-lr-factor 0.5"
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)
