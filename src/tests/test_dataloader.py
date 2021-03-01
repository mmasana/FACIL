import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from main_incremental import main


def test_dataloader_dataset_swap():
    # given
    data1 = TensorDataset(torch.arange(10))
    data2 = TensorDataset(torch.arange(10, 20))
    dl = DataLoader(data1, batch_size=2, shuffle=True, num_workers=1)
    # when
    batches1 = list(dl)

    try:
        dl.dataset += data2
    except ValueError:  
        # In new pytorch this raise an error
        # which is expected and OK behaviour
        return

    batches2 = list(dl)
    all_data = list(dl.dataset)

    # then
    assert len(all_data) == 20
    assert len(batches1) == 5
    assert len(batches2) == 5
    # ^ is troublesome!
    # Sampler is initialized in DataLoader __init__
    # and it holding reference to old DS.
    assert dl.sampler.data_source == data1
    # Thus, we will not see the new data.


def test_dataloader_multiple_datasets():
    args_line = "--exp-name local_test --approach finetuning --datasets mnist mnist mnist" \
                " --network LeNet --num-tasks 2 --batch-size 32" \
                " --results-path ../results/ --num-workers 0 --nepochs 2"
    print('ARGS:', args_line)
    main(args_line.split(' '))
