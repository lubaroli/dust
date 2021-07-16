- [DuSt-MPC's paper](http://www.roboticsproceedings.org/rss17/p068.pdf)
- [DuSt-MPC's spotlight video](https://www.youtube.com/watch?v=aUVQ04pwlvE)

# Table of Content
<!--ts-->
   <!-- * [DuSt-MPC's Introduction](#DuSt-MPCs-introduction)
   * [Installation](#installation)
   * [Quick Start](#quick-start)
      * [Event Extractor](#event-extractor)
         * [Steps](#steps)
      * [Fidelity Evaluator and Utility Evaluator](#fidelity-evaluator-and-utility-evaluator)
         * [Steps](#steps-1)
   * [Fully Reproduce and Reuse](#fully-reproduce-and-reuse)
      * [Event Extractor](#event-extractor-1)
         * [Fully Reproduce Event Extractor](#fully-reproduce-event-extractor)
         * [Reuse Event Extractor](#reuse-event-extractor)
            * [Steps](#steps-2)
      * [Fidelity Evaluator and Utility Evaluator](#fidelity-evaluator-and-utility-evaluator-1)
         * [Fully Reproduce Fidelity Evaluator and Utility Evaluator](#fully-reproduce-fidelity-evaluator-and-utility-evaluator)
            * [Steps](#steps-3)
         * [Reuse Fidelity Evaluator and Utility Evaluator](#reuse-fidelity-evaluator-and-utility-evaluator)
            * [Steps](#steps-4)
   * [DuSt-MPC's Final Datasets](#DuSt-MPCs-final-datasets)
      * [Header Description](#header-description)
      * [Results](#results)
   * [Guidance for New Techniques](#guidance-for-new-techniques)
      * [Prerequisite](#prerequisite)
      * [Steps](#steps-5) -->

<!-- Added by: yixue, at: Wed Sep  9 16:41:38 CDT 2020 -->

<!--te-->
<!-- run ./gh-md-toc --insert index.md to generate ToC automatically -->

DuSt-MPC: Introduction
======================

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