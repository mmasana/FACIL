from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets cifar100" \
                       " --network resnet32 --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 3" \
                       " --num-workers 0" \
                       " --approach dmc" \
                       " --aux-dataset cifar100"


def test_dmc():
    run_main_and_assert(FAST_LOCAL_TEST_ARGS)

