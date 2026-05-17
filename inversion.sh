#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate attack

for i in {0..99}; do
    for j in {0..26}; do
        echo "Running: Image $i, Mu $j"
        python main.py --mu_index $j --image_index $i --dataset mnist >> "logs/ci_mu_${i}_image_${j}.txt"
    done
done