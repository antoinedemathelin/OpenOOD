#!/bin/bash

python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/mwe_lenet.yml \
configs/pipelines/test/test_ood_mwe.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/mwe.yml \
--network.checkpoint "./results/mnist_mwe_lenet_mwe_$1/best.ckpt" \
--merge_option merge \
--mark $1 \