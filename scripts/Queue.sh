#!/bin/bash

for lr in 0.00001 0.00005; do
    for dpout in 0 0.3 0.5; do
        python from_scratch.py $lr 0 $dpout
    done
done

for lrw in 0.0001 0.00001; do
    for lr in 0.00001 0.000001; do
        for dpout in 0 0.3 0.5; do
            for l2 in 0 0.0007; do
                python finetune.py 0.0001 $lr $l2 $dpout
            done
        done
    done
done


    