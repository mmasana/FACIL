from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 3" \
                       " --num-workers 0" \
                       " --gridsearch-tasks -1" \
                       " --approach lucir"


def test_lucir_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars-per-class 20"
    run_main_and_assert(args_line)


def test_lucir_exemplars_with_gridsearch():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars-per-class 20"
    args_line = args_line.replace('--gridsearch-tasks -1', '--gridsearch-tasks 3')
    run_main_and_assert(args_line)


def test_lucir_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars-per-class 20"
    run_main_and_assert(args_line)


def test_lucir_exemplars_remove_margin_ranking():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars-per-class 20"
    args_line += " --remove-margin-ranking"
    run_main_and_assert(args_line)


def test_lucir_exemplars_remove_adapt_lamda():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars-per-class 20"
    args_line += " --remove-adapt-lamda"
    run_main_and_assert(args_line)


def test_lucir_exemplars_warmup():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars-per-class 20"
    args_line += " --warmup-nepochs 5"
    args_line += " --warmup-lr-factor 0.5"
    run_main_and_assert(args_line)
