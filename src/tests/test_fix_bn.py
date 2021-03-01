from tests import run_main_and_assert

FAST_LOCAL_TEST_ARGS = "--exp-name local_test --datasets mnist" \
                       " --network LeNet --num-tasks 3 --seed 1 --batch-size 32" \
                       " --nepochs 2 --lr-factor 10 --momentum 0.9 --lr-min 1e-7" \
                       " --num-workers 0 --fix-bn"


def test_finetuning_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach finetuning"
    run_main_and_assert(args_line)


def test_joint_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach joint"
    run_main_and_assert(args_line)


def test_freezingt_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach freezing"
    run_main_and_assert(args_line)


def test_icarl_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --approach icarl"
    run_main_and_assert(args_line)


def test_eeil_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --num-exemplars 200"
    args_line += " --approach eeil"
    run_main_and_assert(args_line)


def test_mas_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach mas"
    run_main_and_assert(args_line)


def test_lwf_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach lwf"
    run_main_and_assert(args_line)


def test_lwm_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach lwm --gradcam-layer conv2 --log-gradcam-samples 16"
    run_main_and_assert(args_line)


def test_r_walk_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach r_walk"
    run_main_and_assert(args_line)


def test_path_integral_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach path_integral"
    run_main_and_assert(args_line)


def test_luci_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach lucir"
    run_main_and_assert(args_line)


def test_ewc_fix_bn():
    args_line = FAST_LOCAL_TEST_ARGS
    args_line += " --approach ewc"
    run_main_and_assert(args_line)
