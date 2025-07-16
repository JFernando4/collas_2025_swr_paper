# Reinitializing Units vs Weights for Maintaining Plasticity in Neural Networks

## Contents

- [Overview](#overview)
- [Repository Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [License](./LICENSE)
- [Citation](./citation.bib)

## Overview

This repository contains the code to reproduce the experiments present in our paper titled [Reinitializing Weights vs Units for Maintaining
Plasticity in Neural Networks](404).
A talk about this work can be found [here](404).

### Abstract
Loss of plasticity is a phenomenon where a neural network loses its ability to learn when trained for an extended time on non-stationary data.
It is a crucial problem to overcome when designing systems that learn continually.
An effective technique for preventing loss of plasticity is reinitializing parts of the network.
In this paper, we compare two different reinitialization schemes: reinitializing units vs reinitializing weights.
We propose a new algorithm that we name \textit{selective weight reinitialization} for reinitializing the least useful weights in a network. 
We compare our algorithm to continual backpropagation and ReDo, two previously proposed algorithms that reinitialize units in the network.
Through our experiments in continual supervised learning problems, we identify two settings when reinitializing weights is more effective at maintaining plasticity than reinitializing units: (1) when the network has a small number of units and (2) when the network includes layer normalization.
Conversely, reinitializing weights and units are equally effective at maintaining plasticity when the network is of sufficient size and does not include layer normalization. 
We found that reinitializing weights maintains plasticity in a wider variety of settings than reinitializing units.

### Citation
Please cite our work if you find it useful:

```latex

```


## Repository Contents
- [src/swr_functions](./src/swr_functions): contains the selective weight reinitialization algorithm introduced in the paper.
- [src/networks](./src/networks): contains the network architectures and reinitialization layers for continual backpropagation and ReDO.
- [src/optimizers](./src/optimizers): contains a pytorch implementation of SGDW. 
- [src/utils](./src/utils): contains utility functions for training and evaluating networks.
- [experiments/permuted_mnist](./experiments/permuted_mnist): contains the code for the permuted MNIST experiment along with config files for running the experiments and analysing the results.
- [experiments/vit_incremental_cifar](./experiments/vit_incremental_cifar): contains the code for the incremental CIFAR-100 experiment using Vision Transformers.
- [experiments/experiment_analysis](./experiments/experiment_analysis): contains the code for analysing the results of the experiments.

The README files in each subdirectory contains further information on the contents of the subdirectory.

## System Requirements

This package only requires a standard computed with sufficient RAM (8GB+) to reproduce the experimental results.
However, a GPU is recommended for the incremental CIFAR-100 experiments.
Internet connection is required to download many of the datasets and packages.

The package has been tested on Ubuntu 20.04 and macOS 14.6.1 using python3.9.6. 
We expect this package to work on all machines that support all the packages listed in [`requirements.txt`](requirements.txt)


## Installation Guide

Download the repository and install the requirements
```sh
git clone **missing**
cd collas_2025_swr_paper
```

Create a virtual environment
```sh
python -m virtualenv ./venv --no-dowload --python=/usr/bin/python3.9.6
source ./venv/bin/activate
pip3 install --no-index --upgrade pip
```

Download the requirements
```sh
pip3 install -r requirements.txt
pip3 install -e .
```

Add this lines in your `~/.zshrc` or `~/.bashrc`
```sh
source ~/envs/lop/bin/activate
```

Installation on a normal laptop with good internet connection should only take a few minutes
