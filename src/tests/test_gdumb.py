import pytest

from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 2 --lr-factor 10 --momentum 0.9 --lr-min 1e-7" \
                       " --num-workers 0"



def test_gdumb():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach gdumb"
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


@pytest.mark.xfail
def test_gdumb_with_exemplars_per_class_and_herding():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach gdumb"
    args_line += " --num-exemplars-per-class 0"
    run_main_and_assert(args_line)


def test_gdumb_without_catmix():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach gdumb"
    args_line += " --regularization none"
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)
