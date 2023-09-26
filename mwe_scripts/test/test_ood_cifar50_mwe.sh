#!/bin/bash

python main.py \
--config configs/datasets/osr_cifar50/cifar50_seed$1.yml \
configs/datasets/osr_cifar50/cifar50_seed$1_osr.yml \
configs/networks/mwe_resnet18_32x32.yml \
configs/pipelines/test/test_osr_mwe.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/mwe.yml \
--network.checkpoint "./results/cifar50_seed$1_mwe_resnet18_32x32_mwe_$1/best.ckpt" \
--merge_option merge \
--mark $1 \