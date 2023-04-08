#!/bin/bash

    for lr in 0.0001 0.00001; do
        for l2 in 0 0.0007; do
            for dpout in 0.2 0 0.5; do
                nohup python from_scratch.py $lr $l2 $dpout 
            done
        done
    done