#!/bin/bash

for lr in 0.0001 0.0005 0.001; do
    python img_pretrain.py $lr 0
done
    