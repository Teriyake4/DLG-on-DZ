import os
import numpy as np
import multiprocessing
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

from attack import main, init_process
from concurrent.futures import ProcessPoolExecutor
from functools import partial

class ModelArgs:
    def __init__(self, p, mu, alpha):
        self.sparsity_folder = "Layer_Sparsity"
        self.dataset = "mnist"
        self.network = "lenet"  # lenet, resnet20
        self.zero = True
        self.sparsity = p  # p
        self.sparsity_ckpt = f"zo_grasp_{self.sparsity}"
        self.lr = 0.1
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.nesterov = True
        self.epoch = 3
        self.warmup_epochs = 3
        self.scheduler = "cosine"
        self.gpus = [0]
        self.process_per_gpu = 2
        self.zoo_step_size = mu  # mu
        self.master_port = "29600"
        self.master_addr = "localhost"
        self.score = "layer_wise_random"
        self.mask_shuffle_interval = 5
        self.log = True
        self.sample_size = 10000
        self.alpha = alpha

class RunArgs:
    def __init__(self, resultPath):
        self.single = True
        self.saveFreq = 10
        self.printFreq = 3000
        self.resultPath = resultPath
        self.num_dummy = 1
        self.num_iterations = 300
        self.num_exp = 100

def run(p_value, mu_value, alpha_value):
    mArgs = ModelArgs(p_value, mu_value, alpha_value)

    dir = os.path.join('.', f'results/nudge_intermediate_{mArgs.dataset}/{mu_value}_{alpha_value}').replace('\\', '/')
    os.makedirs(dir, exist_ok=True)
    rArgs = RunArgs(dir)
    main(mArgs, rArgs)

if __name__ == '__main__':
    p = [1]
    # alpha = [0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
    # mu = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-15, 1e-17, 1e-19, 1e-21, 1e-23, 1e-30]
    # mu.extend(np.linspace(1e-15, 1e-17, 6)[1:-1].tolist())
    # mu.extend(np.linspace(1e-9, 1e-15, 14)[1:-1].tolist())

    mu = [1e-5,1e-7,1e-10]
    alpha = [0,0.5, 0.99]

    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = []
        for p_value in p:
            for mu_value in mu:
                for alpha_value in alpha:
                    # if os.path.isd
                    futures.append(executor.submit(run, p_value, mu_value, alpha_value))
        
        for future in futures:
            future.result()
                # for gpu
                # world_size = 1 + len(args.gpus) * args.process_per_gpu
                # init_process(0, world_size, args)
                # mp.spawn(init_process, args=(world_size, args), nprocs=world_size, join=True)