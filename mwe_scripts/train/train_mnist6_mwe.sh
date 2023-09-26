#!/bin/bash

python main.py \
--config configs/datasets/osr_mnist6/mnist6_seed$1.yml \
configs/datasets/osr_mnist6/mnist6_seed$1_osr.yml \
configs/networks/mwe_lenet.yml \
configs/pipelines/train/train_mwe.yml \
--network.checkpoint "./networks/osr/mnist6_seed$1.ckpt" \
--mark $1 \
--merge_option merge \