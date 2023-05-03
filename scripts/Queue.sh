#!/bin/bash

for lr in 0.0001 0.00001 0.00005; do
    for dpout in 0 0.3 0.5; do
        python mag_pretrain.py $lr 0 $dpout
    done
done




    