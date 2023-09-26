#!/bin/bash

python main.py \
--config configs/datasets/osr_tin20/tin20_seed$1.yml \
configs/datasets/osr_tin20/tin20_seed$1_osr.yml \
configs/networks/mwe_resnet18_64x64.yml \
configs/pipelines/train/train_mwe.yml \
--network.checkpoint "./networks/osr/tin20_seed$1.ckpt" \
--mark $1 \
--merge_option merge \
