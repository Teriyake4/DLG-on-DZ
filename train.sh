#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python experiments/sparse_gradient_training.py \
  --network resnet20 \
  --dataset MNIST \
  --score layer_wise_random \
  --sparsity 0.9 \
  --sparsity-ckpt zo_grasp_0.9 \
  --gpus 3 \
  --lr 0.1 \
  --master-port 29500