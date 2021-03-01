# Approaches
We include baselines (Finetuning, Freezing and Incremental Joint Training) and the approaches defined in 
_**Class-incremental learning: survey and performance evaluation**_ ([arxiv](https://arxiv.org/abs/2010.15277)).
The regularization-based approaches are EWC, MAS, PathInt, LwF, LwM and DMC. The rehearsal approaches are iCaRL, EEIL
and RWalk. The bias-correction approaches are IL2M, BiC and LUCIR.

## Main usage
When running an experiment, the approach used can be defined in [main_incremental.py](../main_incremental.py) using
`--approach`. Each approach is called by their respective `*.py` name. All approaches inherit from class
`Inc_Learning_Appr`, which has the following main arguments:

* `--nepochs`: number of epochs per training session (default=200)
* `--lr`: starting learning rate (default=0.1)
* `--lr-min`: minimum learning rate (default=1e-4)
* `--lr-factor`: learning rate decreasing factor (default=3)
* `--lr-patience`: maximum patience to wait before decreasing learning rate (default=5)
* `--clipping`: clip gradient norm (default=10000)
* `--momentum`: momentum factor (default=0.0)
* `--weight-decay`: weight decay (L2 penalty) (default=0.0)
* `--warmup-nepochs`: number of warm-up epochs (default=0)
* `--warmup-lr-factor`: warm-up learning rate factor (default=1.0)
* `--multi-softmax`: apply separate softmax for each task (default=False)
* `--fix-bn`: fix batch normalization after first task (default=False)
* `--eval-on-train`: show train loss and accuracy (default=False)

If the approach has some specific arguments, those are defined in the specific `extra_parser()` of each approach file
and are also listed below. All of this information is also available by using `--help`.

### Allowing rehearsal
For all approaches using exemplars, the corresponding arguments are:

* `--num-exemplars`: fixed memory, total number of exemplars (default=0)
* `--num-exemplars-per-class`: growing memory, number of exemplars per class (default=0)
* `--exemplar-selection`: exemplar selection strategy (default='random')

where `--num-exemplars` and `--num-exemplars-per-class` cannot be used at the same time. We extend LwF, EWC, MAS,
Path Integral to allow exemplar rehearsal.

## Adding new approaches
To add a new approach, follow this:

1. Create a new file similar to [finetuning.py](finetuning.py). The name used will be the one that can be called with
   `--approach`.
2. Implement the method as needed and overwrite necessary functions and methods from
   [incremental_learning.py](incremental_learning.py).
3. Add necessary arguments to the approach parser and make sure to not modify `calculate_metrics()` unless necessary to
   make sure that metrics are comparable.

## Baselines

### Finetuning
`--approach finetuning`

Learning approach which learns each task incrementally while not using any data or knowledge from previous tasks. By
default, weights corresponding to the outputs of previous classes are not updated. This can be changed by using
`--all-outputs`. This approach allows the use of exemplars.

### Freezing
`--approach freezing`

Learning approach which freezes the model after training the first task so only the heads are learned. The task after
which the model is frozen can be changed by using `--freeze-after num_task (int)`. As in Finetuning, by default the
corresponding to the current task outputs are updated, but can be changed by using `--all-outputs`.

### Incremental Joint Training
`--approach joint`

Learning approach which has access to all data from previous tasks and serves as an upperbound baseline. Joint training 
can be combined with Freezing by using `--freeze-after num_task (int)`. However, this option is disabled (default=-1).

## Approaches

### Learning without Forgetting
`--approach lwf`
[arxiv](https://arxiv.org/abs/1606.09282)
| [TPAMI 2017](https://ieeexplore.ieee.org/document/8107520)

* `--lamb`: forgetting-intransigence trade-off (default=1)
* `--T`: temperature scaling (default=2)

### iCaRL
`--approach icarl`
[arxiv](https://arxiv.org/abs/1611.07725)
| [CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf)
| [code](https://github.com/srebuffi/iCaRL)
* `--lamb`: forgetting-intransigence trade-off (default=1)

### Elastic Weight Consolidation
`--approach ewc`
[arxiv](http://arxiv.org/abs/1612.00796)
| [PNAS 2017](https://www.pnas.org/content/114/13/3521)

* `--lamb`: forgetting-intransigence trade-off (default=5000)
* `--alpha`: trade-off for how old and new fisher are fused (default=0.5)
* `--fi-sampling-type`: sampling type for Fisher information (default='max_pred')
* `--fi-num-samples`: number of samples for Fisher information (-1: all available) (default=-1)

### Path Integral (aka Synaptic Intelligence)
`--approach path_integral`
[arxiv](https://arxiv.org/abs/1703.04200)
| [ICML 2017](http://proceedings.mlr.press/v70/zenke17a.html)
| [code](https://github.com/ganguli-lab/pathint)

* `--lamb`: forgetting-intransigence trade-off (default=0.1)
* `--damping`: damping (default=0.1)

### Memory Aware Synapses
`--approach mas`
[arxiv](https://arxiv.org/abs/1711.09601)
| [ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf)
| [code](https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses)

* `--lamb`: forgetting-intransigence trade-off (default=1)
* `--alpha`: trade-off for how old and new fisher are fused (default=0.5)
* `--fi-num-samples`: number of samples for Fisher information (-1: all available) (default=-1)

### Riemannian Walk
`--approach r_walk`
[arxiv](https://arxiv.org/abs/1801.10112)
| [ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Arslan_Chaudhry__Riemannian_Walk_ECCV_2018_paper.pdf)
| [code](https://github.com/facebookresearch/agem)

* `--lamb`: forgetting-intransigence trade-off (default=1)
* `--alpha`: trade-off for how old and new fisher are fused (default=0.5)
* `--damping`: damping (default=0.1)
* `--fi-sampling-type`: sampling type for Fisher information (default='max_pred')
* `--fi-num-samples`: number of samples for Fisher information (-1: all available) (default=-1)

### End-to-End Incremental Learning
`--approach eeil`
[arxiv](https://arxiv.org/abs/1807.09536)
| [ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf)
| [code](https://github.com/fmcp/EndToEndIncrementalLearning)

* `--lamb`: forgetting-intransigence trade-off (default=1)
* `--T`: temperature scaling (default=2)
* `--lr-finetuning-factor`: finetuning learning rate factor (default=0.01)
* `--nepochs-finetuning`: number of epochs for balanced training (default=40)
* `--noise-grad`: add noise to gradients (default=False)

### Learning without Memorizing
`--approach lwm`
[arxiv](http://arxiv.org/abs/1811.08051)
| [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dhar_Learning_Without_Memorizing_CVPR_2019_paper.pdf)

* `--beta`: trade-off for distillation loss (default=1)
* `--gamma`: trade-off for attention loss (default=1)
* `--gradcam-layer`: which layer take for GradCAM calculations (default='layer3')
* `--log-gradcam-samples`: how many examples of GradCAM to log (default=0)

### Deep Model Consolidation
`--approach dmc`
[arxiv](https://arxiv.org/abs/1903.07864)
| [WACV 2020](http://openaccess.thecvf.com/content_WACV_2020/papers/Zhang_Class-incremental_Learning_via_Deep_Model_Consolidation_WACV_2020_paper.pdf)
| [code](https://github.com/juntingzh/incremental-learning-baselines)

* `--aux-dataset`: auxiliary dataset (default='imagenet_32_reduced')
* `--aux-batch-size`: batch size for auxiliary dataset (default=128)

### Bias Correction
`--approach bic`
[arxiv](https://arxiv.org/abs/1905.13260)
| [CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf)
| [code](https://github.com/wuyuebupt/LargeScaleIncrementalLearning)

* `--lamb`: forgetting-intransigence trade-off (-1: original moving trade-off) (default=-1)
* `--T`: temperature scaling (default=2)
* `--val-exemplar-percentage`: percentage of exemplars that will be used for validation (default=0.1)
* `--num-bias-epochs`: number of epochs for training bias (default=200)

### Learning a Unified Classifier Incrementally via Rebalancing
`--approach lucir`
[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf)
| [code](https://github.com/hshustc/CVPR19_Incremental_Learning)

* `--lamb`: trade-off for distillation loss (default=5)
* `--lamb-mr`: trade-off for the MR loss (default=1)
* `--dist`: margin threshold for the MR loss  (default=0.5)
* `--K`: Number of "new class embeddings chosen as hard negatives for MR loss (default=2)

### Dual Memory (IL2M)
`--approach il2m`
[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Belouadah_IL2M_Class_Incremental_Learning_With_Dual_Memory_ICCV_2019_paper.pdf)
| [code](https://github.com/EdenBelouadah/class-incremental-learning/tree/master/il2m)
