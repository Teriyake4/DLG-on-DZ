import os

from attack import main, init_process
from concurrent.futures import ProcessPoolExecutor
from functools import partial

class ModelArgs:
    def __init__(self, p, mu, alpha):
        self.sparsity_folder = "Layer_Sparsity"
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
        self.printFreq = 10
        self.resultPath = resultPath
        self.num_dummy = 1
        self.num_iterations = 300
        self.num_exp = 1000

def run(p_value, mu_value, alpha_value):
    dir = os.path.join('.', f'results/nudge/{mu_value}_{alpha_value}').replace('\\', '/')
    os.makedirs(dir, exist_ok=True)
    mArgs = ModelArgs(p_value, mu_value, alpha_value)
    rArgs = RunArgs(dir)
    main(mArgs, rArgs)

if __name__ == '__main__':
    p = [0.1, 0.6, 0.9]  # 1e-10
    default_mu = 5e-3
    delta_mu_values = [10, 5, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    num_mus = len(delta_mu_values)
    mu = []
    for i in range(num_mus):
        mu_value = default_mu + delta_mu_values[i]
        mu.append(mu_value)
        mu.append(-mu_value)

    p = [1]
    mu = [1e-5, 1e-15, 1e-30]
    alpha = [0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for p_value in p:
            for mu_value in mu:
                for alpha_value in alpha:
                    futures.append(executor.submit(run, p_value, mu_value, alpha_value))
        
        for future in futures:
            future.result()
                # for gpu
                # world_size = 1 + len(args.gpus) * args.process_per_gpu
                # init_process(0, world_size, args)
                # mp.spawn(init_process, args=(world_size, args), nprocs=world_size, join=True)