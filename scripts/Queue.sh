#!/bin/bash


for lr in 0.00001 0.0001; do
    for dpout in 0.3 0.5; do
        for l2 in 0 0.0007; do
            python from_scratch.py $lr $l2 $dpout
        done
    done
done



    