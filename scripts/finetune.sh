#!/bin/bash

python from_scratch.py 0.00001 0 0
python from_scratch.py 0.0001 0 0.5
python mag_pretrain.py 0.0001 0 0
python finetune.py 0.0001 0.00001 0.0007 0.5
    