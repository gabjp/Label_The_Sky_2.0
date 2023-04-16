#!/bin/bash
for f_lr in 0.00001 0.0001; do
            for l2 in 0 0.001 0.0007; do
                for dpout in  0 0.2 0.5; do
                    python finetune.py 0.0001 $f_lr $l2 $dpout 
                done
            done
        done
    