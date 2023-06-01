<div align="center">
<img src="./docs/_static/facil_logo.png" width="100px">

# Framework for Analysis of Class-Incremental Learning

---

<p align="center">
  <a href="#what-is-facil">What is FACIL</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="src/approach#approaches-1">Approaches</a> •
  <a href="src/datasets#datasets">Datasets</a> •
  <a href="src/networks#networks">Networks</a> •
  <a href="#license">License</a> •
  <a href="#cite">Cite</a>
</p>
</div>

---

## What is FACIL
FACIL started as code for the paper:  
_**Class-incremental learning: survey and performance evaluation**_  
*Marc Masana, Xialei Liu, Bartlomiej Twardowski, Mikel Menta, Andrew D. Bagdanov, Joost van de Weijer*  
([TPAMI](https://ieeexplore.ieee.org/abstract/document/9915459)) ([arxiv](https://arxiv.org/abs/2010.15277))

It allows to reproduce the results in the paper as well as provide a (hopefully!) helpful framework to develop new
methods for incremental learning and analyse existing ones. Our idea is to expand the available approaches
and tools with the help of the community. To help FACIL grow, don't forget to star this github repository and
share it to friends and coworkers!

## Key Features
We provide a framework based on class-incremental learning. However, task-incremental learning is also fully
supported. Experiments by default provide results on both task-aware and task-agnostic evaluation. Furthermore, if an
experiment runs with one task on one dataset, results would be equivalent to 'common' supervised learning.

| Setting | task-ID at train time | task-ID at test time | # of tasks |
| -----   | ------------------------- | ------------------------ | ------------ |
| [class-incremental learning](https://ieeexplore.ieee.org/abstract/document/9915459) | yes | no | ≥1 |
| [task-incremental learning](https://ieeexplore.ieee.org/abstract/document/9349197) | yes | yes | ≥1 |
| non-incremental supervised learning | yes | yes | 1 |

Current available approaches include:

<div align="center"><p align="center"><b>

  [Finetuning](./src/approach/finetuning.py) • [Freezing](./src/approach/freezing.py) • [Joint](./src/approach/joint.py)

  [LwF](./src/approach/lwf.py) • [iCaRL](./src/approach/icarl.py) • [EWC](./src/approach/ewc.py) • 
  [PathInt](./src/approach/path_integral.py) • [MAS](./src/approach/mas.py) • [RWalk](./src/approach/r_walk.py) • 
  [EEIL](./src/approach/eeil.py) • [LwM](./src/approach/lwm.py) • [DMC](./src/approach/dmc.py) • 
  [BiC](./src/approach/bic.py) • [LUCIR](./src/approach/lucir.py) • [IL2M](./src/approach/il2m.py) •
  [GDumb](./src/approach/gdumb.py)

</b></p>
</div>

## How To Use
Clone this github repository:
```
git clone https://github.com/mmasana/FACIL.git
cd FACIL
```

<details>
  <summary>Optionally, create an environment to run the code (click to expand).</summary>

  ### Using a requirements file
  The library requirements of the code are detailed in [requirements.txt](requirements.txt). You can install them
  using pip with:
  ```
  python3 -m pip install -r requirements.txt
  ```

  ### Using a conda environment
  Development environment based on Conda distribution. All dependencies are in `environment.yml` file.

  #### Create env
  To create a new environment check out the repository and type: 
  ```
  conda env create --file environment.yml --name FACIL
  ```
  *Notice:* set the appropriate version of your CUDA driver for `cudatoolkit` in `environment.yml`.

  #### Environment activation/deactivation
  ```
  conda activate FACIL
  conda deactivate
  ```

</details>

To run the basic code:
```
python3 -u src/main_incremental.py
```
More options are explained in the [`src`](./src), including GridSearch usage. Also, more specific options on approaches,
loggers, datasets and networks.

### Scripts
We provide scripts to reproduce the specific scenarios proposed in 
_**Class-incremental learning: survey and performance evaluation**_:

* CIFAR-100 (10 tasks) with ResNet-32 without exemplars
* CIFAR-100 (10 tasks) with ResNet-32 with fixed and growing memory
* _MORE COMING SOON..._

All scripts run 10 times to later calculate mean and standard deviation of the results.
Check out all available in the [scripts](scripts) folder.

## License
Please check the MIT license that is listed in this repository.

## Cite
If you want to cite the framework feel free to use this preprint citation while we await publication:
```bibtex
@article{masana2022class,
  title={Class-Incremental Learning: Survey and Performance Evaluation on Image Classification},
  author={Masana, Marc and Liu, Xialei and Twardowski, Bartłomiej and Menta, Mikel and Bagdanov, Andrew D. and van de Weijer, Joost},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  doi={10.1109/TPAMI.2022.3213473},
  year={2023},
  volume={45},
  number={5},
  pages={5513-5533}}
}
```

---

The basis of FACIL is made possible thanks to [Marc Masana](https://github.com/mmasana),
[Xialei Liu](https://github.com/xialeiliu), [Bartlomiej Twardowski](https://github.com/btwardow)
and [Mikel Menta](https://github.com/mkmenta). Code structure is inspired by [HAT](https://github.com/joansj/hat.). Feel free to contribute or propose new features by opening an issue!
