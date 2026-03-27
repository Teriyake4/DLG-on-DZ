#!/bin/bash

conda activate attack
python main.py --mu_index 0 --image_index 0 --dataset mnist > bisection_minist.txt
python main.py --mu_index 0 --image_index 0 --dataset mnist > bisection_cifar10.txt