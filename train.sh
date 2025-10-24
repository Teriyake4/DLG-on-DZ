#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

nohup python experiments/sparse_gradient_training.py -o nohup.out