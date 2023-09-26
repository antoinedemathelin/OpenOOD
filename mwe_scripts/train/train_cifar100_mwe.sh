#!/bin/bash

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/mwe_resnet18_32x32.yml \
configs/pipelines/train/train_mwe.yml \
--network.checkpoint "./networks/cifar100_res18_acc78.20.ckpt" \
--mark $1 \
--merge_option merge \
