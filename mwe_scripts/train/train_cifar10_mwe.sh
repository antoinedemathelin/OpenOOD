#!/bin/bash

python main.py \
--config configs/datasets/cifar10/cifar10.yml \
configs/datasets/cifar10/cifar10_ood.yml \
configs/networks/mwe_resnet18_32x32.yml \
configs/pipelines/train/train_mwe.yml \
--network.checkpoint "./networks/cifar10_res18_acc94.30.ckpt" \
--mark $1 \
--merge_option merge \