Dual Online Stein Variational Inference for Control and Dynamics
================================================================

Welcome! This is the code repository for the **DuSt-MPC** paper, published
at Robotics: Science and Systems (RSS) 2021.
This code is provided as an open-source package for Python 3 offering modules
and examples to quickly setup and run experiments with the framework
presented in our paper. This package is still under development.

**Abstract**:
Model predictive control (MPC) schemes have a proven track record for delivering
aggressive and robust performance in many challenging control tasks, coping with
nonlinear system dynamics, constraints, and observational noise. Despite their
success, these methods often rely on simple control distributions, which can
limit their performance in highly uncertain and complex environments. MPC
frameworks must be able to accommodate changing distributions over system
parameters, based on the most recent measurements. In this paper, we devise an
implicit variational inference algorithm able to estimate distributions over
model parameters and control inputs on-the-fly. The method incorporates Stein
Variational gradient descent to approximate the target distributions as a
collection of particles, and performs updates based on a Bayesian formulation.
This enables the approximation of complex multi-modal posterior distributions,
typically occurring in challenging and realistic robot navigation tasks. We
demonstrate our approach on both simulated and real-world experiments requiring
real-time execution in the face of dynamically changing environments.

Spotlight video
-----------------
[![DuSt-MPC Spotlight video](https://img.youtube.com/vi/aUVQ04pwlvE/0.jpg)](https://www.youtube.com/watch?v=aUVQ04pwlvE)

Installation
------------

DuSt is built for [Python](https://www.python.org/) (version 3.6 or later).

### From source

The package can be installed from source by running the following commands:
```shell
  $ git clone https://github.com/lubaroli/dust
  $ cd dust
  $ python setup.py install
```

### Using pip
Or directly from PyPi:
```shell
  $ pip install dust
```

### Using Conda
Additionally, we offer a requirements file in case you want to create a Conda
environment with the necessary dependencies:
```shell
  $ git clone https://github.com/lubaroli/dust
  $ cd dust
  $ conda create -n dust -f environment.yaml
  $ conda activate dust
  $ pip install dust
```

Running Experiments
-------------------

For convenience, scripts for running the simulation experiments found on the
paper are provided. These files are located on the `demo/` folder.

The configuration of each experiment is controlled by a separate `yaml` file,
located on the same folder. Although most variables are self explanatory, there
are a few worth highlighting:

```
    INIT_STATE (required) The initial states for all episodes. A tensor of
                          appropriate dimensions.
    EPISODES   (required) The number of episodes executed in for each test case.
    WARM-UP    (required) The number of warm-up steps executed before the first
                          action is taken.
    RENDER     (optional) Controls whether the experiments should be rendered
                          by gym.
    VERBOSE    (optional) Controls whether progress messages are printed to the
                          console.
```

Once an example file is executed, output files will be saved to the
`./data/local/<date-time>` folder.

Please refer to each module documentation for details on the other arguments.

Citing us:
----------

This code is licensed under the GPL-3.0 and you are welcome to use it and or
modify it accordingly. If you use it for future research, we ask you to cite
our papers:

[Dual Online Stein Variational Inference for Control and Dynamics](http://www.roboticsproceedings.org/rss17/p068.pdf)
```
@INPROCEEDINGS{Barcelos-RSS-21, 
    AUTHOR    = {Lucas Barcelos AND Alexander Lambert AND Rafael Oliveira AND Paulo Borges AND Byron Boots AND Fabio Ramos}, 
    TITLE     = {{Dual Online Stein Variational Inference for Control and Dynamics}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2021}, 
    ADDRESS   = {Virtual}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2021.XVII.068} 
} 
```

and [Stein Variational Model Predictive Control](https://corlconf.github.io/corl2020/paper_282/)

```
@INPROCEEDINGS{Lambert-CoRL-20, 
  AUTHOR    = {Alexander Lambert AND Adam Fishman AND Dieter Fox AND Byron Boots AND Fabio Ramos},
  TITLE     = {Stein Variational Model Predictive Control},
  BOOKTITLE = {Proceedings of Conference on Robot Learning},
  YEAR      = {2020},
  ADDRESS   = {Virtual},
  MONTH     = {November},
  URL       = {https://corlconf.github.io/corl2020/paper_282/},
```
