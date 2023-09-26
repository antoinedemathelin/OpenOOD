#!/bin/bash

python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/mwe_lenet.yml \
configs/pipelines/train/train_mwe.yml \
--network.checkpoint "./networks/mnist_lenet_acc99.60.ckpt" \
--mark $1 \
--merge_option merge \