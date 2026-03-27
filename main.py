import argparse
import os

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

from attack import main, init_process
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Lock

class ModelArgs:
    def __init__(self, p, mu, dataset="mnist"):
        self.sparsity_folder = "Layer_Sparsity"
        self.dataset = dataset
        self.network = "lenet"  # lenet, resnet20, lenetAlt
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
        self.mask_shuffle_interval = 50
        self.log = True
        self.sample_size = 10000

class RunArgs:
    def __init__(self, mu_value, epsilon_squared, image_index, dataset):
        self.single = True
        self.printFreq = 20
        self.num_dummy = 1
        self.num_exp = 1
        self.image_index = image_index # when num_exp = 1 then set the specified image
        self.num_samples = 100 # Number of times to repeat attack with same alpha on same image
        self.num_attack_iterations = 600 # Number of iterations within attack
        self.num_alpha_search_evals = 50
        self.epsilon_squared = epsilon_squared
        self.exper_name = f'stoch_bisect_for_alpha_{dataset}_numattit={self.num_attack_iterations}_epssq={self.epsilon_squared}_numalphasearchevals={self.num_alpha_search_evals}'
        self.resultPath = os.path.join('.', f'results/1_{self.exper_name}/mu={mu_value}').replace('\\', '/')
        self.inversion_methods = ['iDLG']
        self.lock = Lock()
        os.makedirs(self.resultPath, exist_ok=True)


epsilon_squared = 0.1
num_alpha_search_iterations = 1000 # 100 if too long
ci_protection_num_init_samples = 10 # 5 if too long
ci_protection_delta = 0.1
ci_protection_tau = 0.1
ci_protection_tol = 0.01

def run(p_value, mu_value, dataset, epsilon=0.1, image_index=None):
    mArgs = ModelArgs(p_value, mu_value, dataset)
    rArgs = RunArgs(mu_value, epsilon, image_index, dataset)
    main(mArgs, rArgs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--mu_index", type=int, required=True)
    parser.add_argument("--image_index", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    print("mu index:", args.mu_index)
    print("image index:", args.image_index)
    print("dataset:", args.dataset)


    p = [1]
    # alpha = [0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
    mu = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-15, 1e-17, 1e-19, 1e-21, 1e-23, 1e-30]
    mu.extend(np.linspace(1e-15, 1e-17, 6)[1:-1].tolist())
    mu.extend(np.linspace(1e-9, 1e-15, 14)[1:-1].tolist())
    # epsilon = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    epsilon = [0.1, 0.01]
    alpha = [1]
    # mu = [1e-5]
    max_workers = 1
    if "SLURM_CPUS_PER_TASK" in os.environ:
        max_workers = os.environ["SLURM_CPUS_PER_TASK"]
    else:
        max_workers = os.cpu_count()
    print(f"Up to {max_workers} cores")
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))


    mu_value = mu[args.mu_index]
    image_index = args.image_index
    print(f"Job ID: {task_id} | mu = {mu_value}, image index = {image_index}, dataset = {args.dataset}")

    for epsilon_value in epsilon:
        print(f"Epsilon value = {epsilon_value}")
        run(float(p[0]), float(mu_value), args.dataset, float(epsilon_value), int(image_index))




    # with ProcessPoolExecutor(max_workers=1) as executor:
    #     futures = []
    #     for p_value in p:
    #         for mu_value in mu:
    #             for alpha_value in alpha:
    #                 # if os.path.isd
    #                 futures.append(executor.submit(run, p_value, mu_value))
        
        # for future in futures:
        #     future.result()
                # for gpu
                # world_size = 1 + len(args.gpus) * args.process_per_gpu
                # init_process(0, world_size, args)
                # mp.spawn(init_process, args=(world_size, args), nprocs=world_size, join=True)