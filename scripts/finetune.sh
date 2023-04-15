#!/bin/bash

            for l2 in 0.001 0.0007; do
                for dpout in 0.2 0 0.5; do
                    python finetune.py 0.0001 0.00001 $l2 $dpout 
                done
            done
