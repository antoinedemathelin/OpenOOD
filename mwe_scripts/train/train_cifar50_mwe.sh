#!/bin/bash

python main.py \
--config configs/datasets/osr_cifar50/cifar50_seed$1.yml \
configs/datasets/osr_cifar50/cifar50_seed$1_osr.yml \
configs/networks/mwe_resnet18_32x32.yml \
configs/pipelines/train/train_mwe.yml \
--network.checkpoint "./networks/osr/cifar50_seed$1.ckpt" \
--mark $1 \
--merge_option merge \
