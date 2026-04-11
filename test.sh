#!/bin/bash

conda activate attack
python main.py --mu_index 0 --image_index 0 --dataset mnist > bisection_minist_2.txt
python main.py --mu_index 0 --image_index 0 --dataset cifar10 > bisection_cifar10_2.txt