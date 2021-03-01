# Networks
We include a core [network](network.py) class which handles the architecture used as well as the heads needed to do
image classification in an incremental learning setting.

## Main usage
When running an experiment, the network model used can be defined in [main_incremental.py](../main_incremental.py) using
`--network`. By default, the existing head of the architecture (usually with 1,000 outputs because of ImageNet) will be
removed since we create a head each time a task is learned. This default behaviour can be disabled by using
`--keep-existing-head`. If the architecture used has the option to use a pretrained model, it can be called with
`--pretrained`.

We define a [network](network.py) class which contains the architecture model class to be used (torchvision models or
custom), and also a `ModuleList()` of heads that grows incrementally as tasks are learned, called `model` and `heads`
respectively. When doing a forward pass, inputs are passed through the `model` specific forward pass, and the outputs of
it are then fed to the `heads`. This results in a list of outputs, corresponding to the different tasks learned so far
(multi-head base). However, it has to be noted that when using approaches for class-incremental learning, which has no
access to task-ID during test, the heads are treated as if they were concatenated, so the task-ID has no influence.

We use this system since it would be equivalent to having a head that grows at each task and it would concatenate the
heads after (or create a new head with the new number of outputs and copy the previous heads weights to their respective
positions). However, an advantage of this system is that it allows to update previous heads by adding them to the
optimizer (needed when using exemplars). This is also important when using some regularization such as weight decay,
which would affect previous task heads/outputs if the corresponding weights are included in the optimizer. Furthermore,
it makes it very easy to evaluate on both task-incremental and class-incremental scenarios.

### Torchvision models
* Alexnet: `alexnet`
* DenseNet: `densenet121, densenet169, densenet201, densenet161`
* Googlenet: `googlenet`
* Inception: `inception_v3`
* MobileNet: `mobilenet_v2`
* ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d`
* ShuffleNet: `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`
* Squeezenet: `squeezenet1_0`, `squeezenet1_1`
* VGG: `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19_bn`, `vgg19`
* WideResNet: `wide_resnet50_2`, `wide_resnet101_2`

### Custom models
We include versions of [LeNet](lenet.py), [ResNet-32](resnet32.py) and [VGGnet](vggnet.py), which use a smaller input
size than the torchvision models. LeNet together with MNIST is useful for quick tests and debugging.

## Adding new networks
To add a new custom model architecture, follow this:

1. Take as an example [vggnet.py](vggnet.py) and define a Class for the architecture. Initialize all necessary layers
   and modules and define a non-incremental last layer (e.g. `self.fc`). Then add `self.head_var = 'fc'` to point to the
   variable containing the non-incremental head (it is not important how many classes it has as output since we remove
   it when using it for incremental learning).
2. Define the forward pass of the architecture inside the Class and any other necessary functions.
3. Define a function outside of the Class to call the model. It needs to contain `num_out` and `pretrained` as inputs.
4. Include the import to [\_\_init\_\_.py](__init__.py) and add the architecture name to `allmodels`.

## Notes
* We provide an implementation of ResNet-32 (see [resnet32.py](resnet32.py)) which is commonly used by several works in
  the literature for learning CIFAR-100 in incremental learning scenarios. This network architecture is an adaptation of
  ResNet for smaller input size. The number of blocks can be modified in
  [this line](https://github.com/mmasana/IL_Survey/blob/9837386d9efddf48d22fc4d23e031248decce68d/src/networks/resnet32.py#L113)
  by changing `n=5` to `n=3` for ResNet-20, and `n=9` for ResNet-56.
