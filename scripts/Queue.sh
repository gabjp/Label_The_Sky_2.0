#!/bin/bash

for lrw in 0.0001 0.00001; do
    for lr in 0.00001 0.000001; do
        for dpout in 0.3 0.5; do
            for l2 in 0 0.0007; do
                python finetune.py $lrw $lr $l2 $dpout
            done
        done
    done
done


    