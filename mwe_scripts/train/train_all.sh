#!/bin/sh

names='mnist cifar10 cifar100 tin20 mnist6 cifar6 cifar50'
marks='1 2 3 4 5'

for mark in $marks; do
    for name in $names; do
        echo $mark $name
        sh "./scripts/train_${name}_mwe.sh" $mark
    done
done