# Datasets
We include predefined datasets MNIST, CIFAR-100, SVHN, VGGFace2, ImageNet, ImageNet-subset. The first three are
integrated directly from their respective torchvision classes. The other follow our proposed dataset implementation
(see below). We also include `imagenet_32_reduced` to be used with the DMC approach as an external dataset.

## Main usage
When running an experiment, datasets used can be defined in [main_incremental.py](../main_incremental.py) using
`--datasets`. A single dataset or a list of datasets can be provided, and will be learned in the given order. If
`--num_tasks` is larger than 1, each dataset will further be split into the given number of tasks. All tasks from the
first dataset will be learned before moving to the next dataset in the list. Other main arguments related to datasets
are:

* `--num-workers`: number of subprocesses to use for dataloader (default=4)
* `--pin-memory`: copy Tensors into CUDA pinned memory before returning them (default=False)
* `--batch-size`: number of samples per batch to load (default=64)
* `--nc-first-task`: number of classes of the first task (default=None)
* `--use-valid-only`: use validation split instead of test (default=False)
* `--stop-at-task`: stop training after specified task (0: no stop) (default=0)

Datasets are defined in [dataset_config.py](dataset_config.py). Each entry key is considered a dataset name. For a
proper configuration, the mandatory entry value is `path`, which contains the path to the dataset folder (or for
torchvision datasets, the path to where it will be downloaded). The rest of values are possible transformations to be
applied to the datasets (see [data_loader.py](data_loader.py)):

* `resize`: resize the input image to the given size on both train and eval
  [[source](https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.Resize)].
* `pad`: pad the given image on all sides with the given “pad” value on both train and eval
  [[source](https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.functional.pad)].
* `crop`: crop the given image to random size and aspect ratio for train
  [[source](https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.RandomResizedCrop)], and at the center
  on eval [[source](https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.CenterCrop)].
* `flip`: horizontally flip the given image randomly with a 50% probability on train only
  [[source](https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.RandomHorizontalFlip)].
* `normalize`: normalize a tensor image with mean and standard deviation
  [[source](https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.Normalize)].
* `class_order`: fixes the class order given the list of ordered labels. It also allows to limit the dataset to only the
  classes provided.
* `extend_channel`: times that the input channels have to be extended.

where the first ones are adapted from PyTorch transforms, and the last two are our own additions.

### Dataset types
MNIST, CIFAR-100 and SVHN are small enough to use [memory_dataset.py](memory_dataset.py), which loads all images in
memory. The other datasets are too large to fully load in memory and, therefore, use [base_dataset.py](base_dataset.py).
This dataset type loads the corresponding paths and labels in memory and the images make use the PyTorch DataLoader to
load the images in batches when needed. This can be modified in [data_loader.py](data_loader.py#L64) by modifying the
type of dataset used.

### Dataset path
Modify variable `_BASE_DATA_PATH` in [dataset_config.py](dataset_config.py) pointing to the root folder containing your
datasets. You can also modify the `path` field in the entries in [dataset_config.py](dataset_config.py) if a specific
dataset is found to a special location in your machine.

### Exemplars
For those approaches that can use exemplars, those can be used with either `--num_exemplars` or `--num_exemplars_per_class`. The first one is for a fixed memory where the number of exemplars is the same for all classes. The second one is a growing memory which specifies the number of exemplars per class that will be stored. 

Different exemplar sampling strategies are implemented in [exemplars_selection.py](exemplars_selection.py). Those can be selected by using `--exemplar_selection` followed by one of this strategies:

* `random`: produces a random list of samples.
* `herding`: sampling based on distance to the mean sample of each class. From [iCaRL](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf) algorithms 4 and 5.
* `entropy`: sampling based on entropy of each sample. From [RWalk](http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112).
* `distance`: sampling based on closeness to decision boundary of each sample. From [RWalk](http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112).


## Adding new datasets
To add a new dataset, follow this:

1. Create a new entry in [dataset_config.py](dataset_config.py), add the folder with the data in `path` and any other transformations or class ordering needed.
2. Depending if the dataset is in [torchvision](https://pytorch.org/docs/stable/torchvision/datasets.html) or custom:
   * If the new dataset is available in **torchvision**, you can add a new option like in lines 67, 79 or 91 from [data_loader.py](data_loader.py). If the dataset is too large to fit in memory, use `base_dataset`, else `memory_dataset`.
   * If the dataset is **custom**, the option from line 135 in [data_loader.py](data_loader.py) should be enough. In the same folder as the data add a `train.txt` and a `test.txt` files. Each containing one line per sample with the path and the corresponding class label:
      ```
      /data/train/sample1.jpg 0
      /data/train/sample2.jpg 1
      /data/train/sample3.jpg 2
      ...
      ```

No need to modify any other file. If the dataset is a subset or modification of an already added dataset, only use step 1.

### Available custom datasets
We provide `.txt` and `.zip` files for the following datasets for an easier integration with FACIL:

* (COMING SOON)

## Changing dataset transformations

There are some cases when it's necessary to get data from the existing dataset with a different transformation (i.e. self-supervision) or none at all (i.e. selecting exemplars). For this particular cases we prepared a simple solution to override transformations for a dataset within a selected context, e.g. taking raw pictures in `numpy` format: 

```python
   with override_dataset_transform(trn_loader.dataset, Lambda(lambda x: np.array(x))) as ds_for_raw:
      x, y = zip(*(ds_for_raw[idx] for idx in selected_indices))
```

This can be used in other cases too, like train/eval change of transformation when checking the training process, etc. Also, you can check out a simple unit test in [test_dataset_transforms.py](src/tests/test_datasets_transforms.py). 

## Notes
* As an example, we include two versions of CIFAR-100. The one with entry `cifar100` is the default one which by default
  shuffles the class order depending on the fixed seed. We also include `cifar100_icarl` which fixes the class order
  from iCaRL (given by seed 1993), and thus makes the comparison more fair with results from papers that use that class
  order (e.g. iCaRL, LUCIR, BiC).
* When using multiple machines, you can create different [dataset_config.py](dataset_config.py) files for each of them.
  Otherwise, remember you can create symbolic links that point to a specific folder with
  `ln -s TARGET_DIRECTORY LINK_NAME` in Linux/UNIX.
