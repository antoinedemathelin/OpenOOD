#!/bin/sh

names='mnist cifar10 cifar100 mnist6 cifar6 cifar50 tin20'
marks='1 2 3 4 5'

for mark in $marks; do
    for name in $names; do
        echo $mark $name
        sh "./scripts/test_ood_${name}_mwe.sh" $mark
    done
done