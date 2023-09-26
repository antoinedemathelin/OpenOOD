#!/bin/bash

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/mwe_resnet18_32x32.yml \
configs/pipelines/test/test_ood_mwe.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/mwe.yml \
--network.checkpoint "./results/cifar100_mwe_resnet18_32x32_mwe_$1/best.ckpt" \
--merge_option merge \
--mark $1 \