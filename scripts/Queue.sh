#!/bin/bash


CUDA_VISIBLE_DEVICES=0 nohup python finetune.py  0.0001 0.00001 0.0007 0.3 300 0 1 > run1.out &
CUDA_VISIBLE_DEVICES=1 nohup python finetune.py  0.0001 0.00001 0.0007 0.3 300 0 2 > run2.out &
CUDA_VISIBLE_DEVICES=2 nohup python finetune.py  0.0001 0.00001 0.0007 0.3 300 0 3 > run3.out 

CUDA_VISIBLE_DEVICES=3 nohup python finetune.py  0.0001 0.00001 0.0007 0.3 300 1 1 > run1.out &
CUDA_VISIBLE_DEVICES=2 nohup python finetune.py  0.0001 0.00001 0.0007 0.3 300 1 2 > run2.out &
CUDA_VISIBLE_DEVICES=0 nohup python finetune.py  0.0001 0.00001 0.0007 0.3 300 1 3 > run3.out 





    