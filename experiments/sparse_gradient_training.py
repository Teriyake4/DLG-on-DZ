import os
import time
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from torch.distributed import rpc
import torch.multiprocessing as mp

import sys

sys.path.append(".")
from cfg import results_path
from tools import *
from algorithm.prune import global_prune, check_sparsity, extract_mask, custom_prune, remove_prune
from algorithm.zoo import cge_weight_allocate_to_process, cge_calculation, network_synchronize
from data import prepare_dataset
from models.tools import time_consumption_per_layer
from models.distributed_model import DistributedCGEModel


class Args():
    def __init__(self, batch_size: int, lr: float, alpha: float, sparsity: float, sparsity_ckpt: str):
        self.dry_run = False
        self.seed = 123
        self.network = "lenet"
        self.dataset = "MNIST"
        self.batch_size = batch_size
        self.zoo_step_size = 1e-7
        self.epoch = 50
        self.lr = lr
        self.alpha = alpha
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.warmup_epochs = 3
        self.nesterov = True
        self.scheduler = "cosine"
        self.mask_shuffle_interval = 5
        self.score = "layer_wise_random"
        self.sparsity = sparsity
        self.sparsity_folder = "Layer_Sparsity"
        self.sparsity_ckpt = sparsity_ckpt
        self.gpus = [1, 2]
        self.process_per_gpu = 2
        self.master_addr = "localhost"
        self.master_port = "29500"
        self.log = True


def main(args):
    start_time = time.time()
    # Misc
    device = f"cuda:{args.gpus[-1]}"
    set_seed(args.seed)
    exp = os.path.basename(__file__.split('.')[0])
    # save_path = os.path.join(results_path, exp, gen_folder_name(args, ignore=['log', 'gpus', 'process_per_gpu', 'master_addr', 'master_port', 'momentum', 'weight_decay', 'sparsity_folder', 'sparsity_ckpt']))
    save_path = os.path.join(".", f"training/experiment3/training_batchSize_{args.batch_size}_lr_{args.lr}_alpha_{args.alpha}_p_{args.sparsity}/")
    print(save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Criterion for calculating vanilla gradients
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Data
    # loaders, class_num = prepare_dataset(args.dataset, args.batch_size)
    from data import alt_dataset
    loaders, class_num = alt_dataset(args.dataset, args.batch_size)

    # Network
    if args.network == "resnet20":
        from models.resnet_s import resnet20, param_name_to_module_id_rn20
        param_name_to_module_id = param_name_to_module_id_rn20
        network_init_func = resnet20
        network_kwargs = {
            'num_classes': class_num
        }
    elif args.network == "lenet":
        from models.lenet import lenet, param_name_to_module_id_lenet
        param_name_to_module_id = param_name_to_module_id_lenet
        network_init_func = lenet
        network_kwargs = {
            'num_classes': class_num
        }
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    network = network_init_func(**network_kwargs).to(device)

    # Load Lay-wise sparsity ckpt
    sparsity_ckpt = torch.load(os.path.join(args.sparsity_folder, args.network, args.sparsity_ckpt + '.pth'), map_location=device) if args.sparsity_ckpt is not None else None

    # Optimizer
    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=args.nesterov)
    global_length = (args.epoch - args.warmup_epochs) * len(loaders['train'])
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_length)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*global_length), int(0.75*global_length)], gamma=0.1)
    else:
        raise NotImplementedError(f'scheduler {args.scheduler} not implemented')

    # Makedir or Resume
    if args.log:
        # First time running code in a new session run the following command:
        # export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

        # If restarting uncomment the code below to remove dir
        import shutil
        shutil.rmtree(save_path)

        try:
            os.makedirs(save_path, exist_ok=False)
            best_acc = 0.
            epoch = 0
        except FileExistsError:
            state_dict = torch.load(os.path.join(save_path, "ckpt.pth"), map_location=device)
            for key, val in state_dict["state_dicts"].items():
                eval(f"{key}.load_state_dict(val)")
            current_mask = state_dict["current_mask"]
            best_acc = state_dict["best_acc"]
            epoch = state_dict["epoch"]
        logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    else:
        epoch = 0

    # Init subprocess networks
    remote_networks = {}
    os.makedirs('.cache', exist_ok=True)
    cache_file_path = f'.cache/pruned_model_{args.master_port}.pth'
    torch.save(network.state_dict(), cache_file_path)
    for gpu in args.gpus:
        for i in range(args.process_per_gpu):
            remote_networks[f"{gpu}-{i}"] = rpc.remote(f"{gpu}-{i}", DistributedCGEModel,
                                                       args=(f"cuda:{gpu}", partial(network_init_func, **network_kwargs),
                                                             F.cross_entropy, param_name_to_module_id, cache_file_path, False))

    # Subprocess resume
    if epoch > 0:
        state_dict_to_restore = network.state_dict()
        custom_prune(network, current_mask)
        cge_weight_allocate_to_process(remote_networks, network, args.gpus, args.process_per_gpu, param_name_to_module_id, time_consumption_per_layer(args.network))
        remove_prune(network)
        network.load_state_dict(state_dict_to_restore)

    while epoch < args.epoch:
        epoch += 1
        if (epoch-1) % args.mask_shuffle_interval == 0:
            # ReGenerate Mask
            state_dict_to_restore = network.state_dict()
            if 0. < args.sparsity < 1.:
                global_prune(network, args.sparsity, args.score, class_num, loaders['train'], zoo_sample_size=192, zoo_step_size=5e-3, layer_wise_sparsity=sparsity_ckpt)
            elif args.sparsity == 0:
                pass
            # else:
            #     raise ValueError('sparsity not valid')
            # assert abs(args.sparsity - (1 - check_sparsity(network, if_print=False) / 100)) < 0.01, check_sparsity(network, if_print=False)
            current_mask = extract_mask(network.state_dict())
            cge_weight_allocate_to_process(remote_networks, network, args.gpus, args.process_per_gpu, param_name_to_module_id, time_consumption_per_layer(args.network))
            remove_prune(network)
            network.load_state_dict(state_dict_to_restore)
        # Train
        network.train()
        acc = AverageMeter()
        loss = AverageMeter()
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                desc=f"Epo {epoch} Training", ncols=160)
        for i, (x, y) in enumerate(pbar):
            if epoch <= args.warmup_epochs:
                warmup_lr(optimizer, epoch-1, i+1, len(loaders['train']), args.warmup_epochs, args.lr)
            x_cuda, y_cuda = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                fx = network(x_cuda, return_interval = False)
                loss_batch = F.cross_entropy(fx, y_cuda).cpu()
            lr = optimizer.param_groups[0]['lr']
            cge_calculation(remote_networks, network, args.gpus, args.process_per_gpu, x, y, lr if args.zoo_step_size == -1 else args.zoo_step_size)

            zo_dy_dx = [p.grad for p in network.parameters()]

            # Compute original gradients
            out = network(x_cuda)
            model_y = criterion(out, y_cuda)
            vanilla_dy_dx = torch.autograd.grad(model_y, network.parameters())

            # Calculate nudge gradients
            zo_dy_dx_nudge = [zo_dy_dx[i] + (vanilla_dy_dx[i] - zo_dy_dx[i]) * args.alpha for i in range(len(zo_dy_dx))]
            zo_dy_dx_nudge = [grad.detach().clone() for grad in zo_dy_dx_nudge]

            for p, nudge in zip(network.parameters(), zo_dy_dx_nudge):
                p.grad.copy_(nudge)

            optimizer.step()
            network_synchronize(remote_networks, network, args.gpus, args.process_per_gpu)
            acc.update(torch.argmax(fx, 1).eq(y_cuda).float().mean().item(), y.size(0))
            loss.update(loss_batch.item(), y.size(0))
            if epoch > args.warmup_epochs:
                scheduler.step()
            pbar.set_postfix_str(f"Lr {lr:.2e} Acc {100*acc.avg:.2f}%")
        if args.log:
            logger.add_scalar("train/acc", acc.avg, epoch)
            logger.add_scalar("train/loss", loss.avg, epoch)

        # Test
        network.eval()
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=120)
        acc = AverageMeter()
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(x)
            acc.update(torch.argmax(fx, 1).eq(y).float().mean(), y.size(0))
            pbar.set_postfix_str(f"Acc {100*acc.avg:.2f}%")
        if args.log:
            logger.add_scalar("test/acc", acc.avg, epoch)

        # Save CKPT
        if args.log:
            state_dict = {
                "state_dicts":{
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                "current_mask": current_mask,
                "epoch": epoch,
                "best_acc": best_acc,
            }
            if acc.avg > best_acc:
                best_acc = acc.avg
                state_dict['best_acc'] = best_acc
                torch.save(state_dict, os.path.join(save_path, 'best.pth'))
            torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))

        # Cutoff
        if acc.avg == 0.9:
            print("Accuracy cutoff")
            return
        if epoch == 10:
            print("Epoch cutoff")
            return


def init_process(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    if rank == 0:
        rpc.init_rpc(
                f"master", rank=rank, world_size=world_size,
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=args.process_per_gpu*world_size+1, rpc_timeout=0.)
            )
        main(args)
    else:
        gpu = args.gpus[(rank-1)//args.process_per_gpu]
        i = (rank-1) % args.process_per_gpu
        rpc.init_rpc(
                f"{gpu}-{i}", rank=rank, world_size=world_size,
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=args.process_per_gpu*world_size+1, rpc_timeout=0.)
            )
    rpc.shutdown()


if __name__ == "__main__":
    # p = argparse.ArgumentParser()
    # p = process_cli(p)
    # args = p.parse_args()

    # args.gpus = args.gpus.split(',')

    # alpha = 0.9 - 1.0 increasing by intervals of 0.01
    # p = 0, alpha = 0, mu = 1e-7
    # try lr = 0.1, 0.01

    # 0.5-0.9 alpha

    for lr in [0.1, 0.01]:
        for batch_size in [128, 256, 512]:
            for alpha in np.arange(0.5, 1, 0.1).tolist():
                for p in [0]:
                    args = Args(batch_size, lr, alpha, p, f"zo_grasp_{p:.1f}")

                world_size = 1 + len(args.gpus) * args.process_per_gpu
                mp.spawn(init_process, args=(world_size, args), nprocs=world_size, join=True)
