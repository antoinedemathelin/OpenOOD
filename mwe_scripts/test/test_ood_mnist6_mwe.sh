#!/bin/bash

python main.py \
--config configs/datasets/osr_mnist6/mnist6_seed$1.yml \
configs/datasets/osr_mnist6/mnist6_seed$1_osr.yml \
configs/networks/mwe_lenet.yml \
configs/pipelines/test/test_osr_mwe.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/mwe.yml \
--network.checkpoint "./results/mnist6_seed$1_mwe_lenet_mwe_$1/best.ckpt" \
--merge_option merge \
--mark $1 \