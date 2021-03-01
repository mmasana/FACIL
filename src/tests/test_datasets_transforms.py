import numpy as np
from torchvision.transforms import Lambda
from torch.utils.data.dataset import ConcatDataset

from datasets.memory_dataset import MemoryDataset
from datasets.exemplars_selection import override_dataset_transform


def pic(i):
    return np.array([[i]], dtype=np.int8)


def test_dataset_transform_override():
    # given
    data1 = MemoryDataset({
        'x': [pic(1), pic(2), pic(3)], 'y': ['a', 'b', 'c']
    }, transform=Lambda(lambda x: np.array(x)[0, 0] * 2))
    data2 = MemoryDataset({
        'x': [pic(4), pic(5), pic(6)], 'y': ['d', 'e', 'f']
    }, transform=Lambda(lambda x: np.array(x)[0, 0] * 3))
    data3 = MemoryDataset({
        'x': [pic(7), pic(8), pic(9)], 'y': ['g', 'h', 'i']
    }, transform=Lambda(lambda x: np.array(x)[0, 0] + 10))
    ds = ConcatDataset([data1, ConcatDataset([data2, data3])])

    # when
    x1, y1 = zip(*[ds[i] for i in range(len(ds))])
    with override_dataset_transform(ds, Lambda(lambda x: np.array(x)[0, 0])) as ds_overriden:
        x2, y2 = zip(*[ds_overriden[i] for i in range(len(ds_overriden))])
    x3, y3 = zip(*[ds[i] for i in range(len(ds))])

    # then
    assert np.array_equal(x1, [2, 4, 6, 12, 15, 18, 17, 18, 19])
    assert np.array_equal(x2, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert np.array_equal(x3, x1)  # after everything is back to normal
