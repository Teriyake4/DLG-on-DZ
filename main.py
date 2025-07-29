import os

from attack import main, init_process

class ModelArgs:
    def __init__(self, p, mu):
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

class RunArgs:
    def __init__(self, resultPath):
        self.single = True
        self.printFreq = 50
        self.resultPath = resultPath

if __name__ == '__main__':
    p_values = [0.1, 0.6, 0.9]  # 1e-10
    default_mu = 5e-3
    delta_mu_values = [10, 5, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    num_mus = len(delta_mu_values)
    mu_values = []
    for i in range(num_mus):
        mu_value = default_mu + delta_mu_values[i]
        mu_values.append(mu_value)
        mu_values.append(-mu_value)

    p_values = [0.9]
    mu_values = [5e-3]

    mArgs = ModelArgs(p_values, mu_values)
    rArgs = RunArgs(os.path.join('.', f'results/baseline/zo').replace('\\', '/'))

    # for gpu cge
    # for p_value in p_values:
    #     for mu_value in mu_values:
    #         args = Args(p_value, mu_value)
    #         # world_size = 1 + len(args.gpus) * args.process_per_gpu
    #         # init_process(0, world_size, args)
              # mp.spawn(init_process, args=(world_size, args), nprocs=world_size, join=True)

    for p_value in p_values:
        for mu_value in mu_values:
            mArgs = ModelArgs(p_value, mu_value)
            rArgs = RunArgs(os.path.join('.', f'results/baseline/zo').replace('\\', '/'))
            main(mArgs, rArgs)