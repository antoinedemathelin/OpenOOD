# Maximum Weight Entropy OpenOOD Benchmark

This repository provides the experiments conducted for the Maximum Weight Entropy method (MaxWEnt) within the [OpenOOD Benchmark](https://arxiv.org/abs/2210.07242).

The OpenOOD Benchmark reproduces representative methods within the [`Generalized Out-of-Distribution Detection Framework`](https://arxiv.org/abs/2110.11334),
aiming to make a fair comparison across methods that initially developed for anomaly detection, novelty detection, open set recognition, and out-of-distribution detection.

## Get Started

To setup the environment, we use `conda` to manage our dependencies.

Our developers use `CUDA 10.1` to do experiments.

You can specify the appropriate `cudatoolkit` version to install on your machine in the `environment.yml` file, and then run the following to create the `conda` environment:
```bash
conda env create -f environment.yml
conda activate openood
```

Datasets and pretrained models are provided [here](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/Eso7IDKUKQ9AoY7hm9IU2gIBMWNnWGCYPwClpH0TASRLmg?e=kMrkVQ).
Please unzip the files if necessary.

The codebase accesses the datasets from `./data/` and pretrained models from `./networks/` by default.

## Run MaxWEnt Experiments

The scripts to run the MaxWEnt experiments can be found in the `./mwe_scripts/` folder.
