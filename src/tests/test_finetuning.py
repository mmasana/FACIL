import torch
import pytest

from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 2 --lr-factor 10 --momentum 0.9 --lr-min 1e-7" \
                       " --num-workers 0"


def test_finetuning_without_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    run_main_and_assert(args_line)


def test_finetuning_with_exemplars():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    args_line += " --num-exemplars 200"
    run_main_and_assert(args_line)


@pytest.mark.xfail
def test_finetuning_with_exemplars_per_class_and_herding():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    args_line += " --num-exemplars-per-class 10"
    args_line += " --exemplar-selection herding"
    run_main_and_assert(args_line)


def test_finetuning_with_exemplars_per_class_and_entropy():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    args_line += " --num-exemplars-per-class 10"
    args_line += " --exemplar-selection entropy"
    run_main_and_assert(args_line)


def test_finetuning_with_exemplars_per_class_and_distance():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    args_line += " --num-exemplars-per-class 10"
    args_line += " --exemplar-selection distance"
    run_main_and_assert(args_line)


def test_wrong_args():
    with pytest.raises(SystemExit):  # error of providing both args
        args_line = FAST_LOCAL_TEST_ARGS
        args_line += " --approach finetuning"
        args_line += " --num-exemplars-per-class 10"
        args_line += " --num-exemplars 200"
        run_main_and_assert(args_line)


def test_finetuning_with_eval_on_train():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    args_line += " --num-exemplars-per-class 10"
    args_line += " --exemplar-selection distance"
    args_line += " --eval-on-train"
    run_main_and_assert(args_line)

def test_finetuning_with_no_cudnn_deterministic():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    args_line += " --num-exemplars-per-class 10"
    args_line += " --exemplar-selection distance"

    run_main_and_assert(args_line)
    assert torch.backends.cudnn.deterministic == True

    args_line += " --no-cudnn-deterministic"
    run_main_and_assert(args_line)
    assert torch.backends.cudnn.deterministic == False
