import os
from functools import partial

import torch
from torch.distributed import rpc
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithm.prune import global_prune, check_sparsity, extract_mask, remove_prune
from algorithm.zoo import cge_weight_allocate_to_process, cge_calculation, network_synchronize
from models.distributed_model import DistributedCGEModel
from models.lenet import lenet, param_name_to_module_id_lenet
from models.tools import time_consumption_per_layer
from models.resnet_s import resnet20, param_name_to_module_id_rn20
from tools.meter import AverageMeter
from tools.training import warmup_lr

def dataset_loader(dataset, data_path):
    if dataset == 'MNIST':
        shape_img = (28, 28)
        num_classes = 10
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=True)
    elif dataset == 'cifar100':
        shape_img = (32, 32)
        num_classes = 100
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)
    elif dataset == 'cifar10':
        shape_img = (32, 32)
        num_classes = 10
        channel = 3
        hidden = 768
        dst = datasets.CIFAR10(data_path, download=True)
    else:
        exit('unknown dataset')

    return (shape_img, num_classes, channel, hidden, dst)

def init_model(device, mArgs, class_num, hidden, channel):
    if mArgs.network == "resnet20":
        # param_name_to_module_id = param_name_to_module_id_rn20
        network_init_func = resnet20
        network_kwargs = {
            'channel': channel,
            'num_classes': class_num
        }
    elif mArgs.network == "lenet":  # lenet
        network_init_func = lenet
        network_kwargs = {
            'channel': channel,
            'hidden': hidden,
            'num_classes': class_num
        }
    else:
        raise NotImplementedError
    net = network_init_func(**network_kwargs).to(device)
    net.apply(weights_init).to(device)
    # net = network_init_func(**network_kwargs)

    # Optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=mArgs.lr, weight_decay=mArgs.weight_decay,
    #                             momentum=mArgs.momentum, nesterov=mArgs.nesterov)

    net.train()
    return net

def net_prep(net, mArgs):
    class_num = 10
    hidden = 588
    channel = 1
    # prep
    if mArgs.network == "resnet20":
        network_init_func = resnet20
        param_name_to_module_id = param_name_to_module_id_rn20
    elif mArgs.network == "lenet":
        network_init_func = lenet
        param_name_to_module_id = param_name_to_module_id_lenet
    else:
        raise NotImplementedError
    network_kwargs = {
        'channel': channel,
        'hidden': hidden,
        'num_classes': class_num  # cifar10
    }
    useRemoteNet = False
    remote_networks = {}
    if useRemoteNet:
        os.makedirs('.cache', exist_ok=True)
        cache_file_path = f'.cache/pruned_model_{mArgs.master_port}.pth'
        torch.save(net.state_dict(), cache_file_path)
        for gpu in mArgs.gpus:  # gpus
            for i in range(mArgs.process_per_gpu):
                remote_networks[f"{gpu}-{i}"] = rpc.remote(f"{gpu}-{i}", DistributedCGEModel,
                                                           mArgs=(f"cuda:{gpu}",
                                                                 partial(network_init_func, **network_kwargs),
                                                                 F.cross_entropy, param_name_to_module_id,
                                                                 cache_file_path, False))

    # ReGenerate Mask
    sparsity_ckpt = torch.load(os.path.join(mArgs.sparsity_folder, mArgs.network, mArgs.sparsity_ckpt + '.pth'),
                               map_location=f"cuda:{mArgs.gpus[-1]}") if mArgs.sparsity_ckpt is not None else None
    state_dict_to_restore = net.state_dict()
    if 0. < mArgs.sparsity < 1.:
        global_prune(net, mArgs.sparsity, mArgs.score, class_num, None, zoo_sample_size=192,
                     zoo_step_size=5e-3, layer_wise_sparsity=sparsity_ckpt)
    elif mArgs.sparsity == 0:
        pass
    else:
        raise ValueError('sparsity not valid')
    assert abs(mArgs.sparsity - (1 - check_sparsity(net, if_print=False) / 100)) < 0.01, check_sparsity(net,
                                                                                                       if_print=False)
    current_mask = extract_mask(net.state_dict())

    # gets error can not find [layer_._.conv_.weight_mask] in mask_dict, skipping
    if useRemoteNet:
        cge_weight_allocate_to_process(remote_networks, net, mArgs.gpus, mArgs.process_per_gpu,
                                       param_name_to_module_id,
                                       time_consumption_per_layer(mArgs.network))
    remove_prune(net)
    net.load_state_dict(state_dict_to_restore)
    return net, remote_networks

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

def prep_data(gt_data, gt_label):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    gt_data = transform(gt_data)
    data = TensorDataset(gt_data, gt_label)
    data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    return {
        "train": data_loader,
        "test": data_loader,
    }, 10

def train(network, mArgs, remote_networks, gt_data, gt_label, save_path):

    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    # dataset = data.TensorDataset(
    #     gt_data.unsqueeze(0),  # Add batch dimension
    #     gt_label.unsqueeze(0)  # Add batch dimension
    # )
    # # loaders = DataLoader(dataset, batch_size=1, shuffle=False)
    param_name_to_module_id = param_name_to_module_id_rn20

    device = f"cuda:{mArgs.gpus[-1]}"
    loaders, class_num = prep_data(gt_data, gt_label)
    sparsity_ckpt = torch.load(os.path.join(mArgs.sparsity_folder, mArgs.network, mArgs.sparsity_ckpt + '.pth'),
                               map_location=device) if mArgs.sparsity_ckpt is not None else None
    optimizer = torch.optim.SGD(network.parameters(), lr=mArgs.lr, weight_decay=mArgs.weight_decay,
                                momentum=mArgs.momentum, nesterov=mArgs.nesterov)

    epoch = 0
    while epoch < mArgs.epoch:
        epoch += 1
        if (epoch - 1) % mArgs.mask_shuffle_interval == 0:
            # ReGenerate Mask
            state_dict_to_restore = network.state_dict()
            if 0. < mArgs.sparsity < 1.:
                global_prune(network, mArgs.sparsity, mArgs.score, class_num, loaders['train'], zoo_sample_size=192,
                             zoo_step_size=5e-3, layer_wise_sparsity=sparsity_ckpt)
            elif mArgs.sparsity == 0:
                pass
            else:
                raise ValueError('sparsity not valid')
            assert abs(mArgs.sparsity - (1 - check_sparsity(network, if_print=False) / 100)) < 0.01, check_sparsity(
                network, if_print=False)
            current_mask = extract_mask(network.state_dict())
            cge_weight_allocate_to_process(remote_networks, network, mArgs.gpus, mArgs.process_per_gpu,
                                           param_name_to_module_id, time_consumption_per_layer(mArgs.network))
            remove_prune(network)
            network.load_state_dict(state_dict_to_restore)
        # Train
        network.train()
        acc = AverageMeter()
        loss = AverageMeter()
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                    desc=f"Epo {epoch} Training", ncols=160)
        for i, (x, y) in enumerate(pbar):
            # need to move to cpu
            x = x.cpu()
            y = y.cpu()
            if epoch <= mArgs.warmup_epochs:
                warmup_lr(optimizer, epoch - 1, i + 1, len(loaders['train']), mArgs.warmup_epochs, mArgs.lr)
            x_cuda, y_cuda = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                fx = network(x_cuda, return_interval=False)
                loss_batch = F.cross_entropy(fx, y_cuda).cpu()
            lr = optimizer.param_groups[0]['lr']
            cge_calculation(remote_networks, network, mArgs.gpus, mArgs.process_per_gpu, x, y,
                            lr if mArgs.zoo_step_size == -1 else mArgs.zoo_step_size)
            optimizer.step()
            network_synchronize(remote_networks, network, mArgs.gpus, mArgs.process_per_gpu)
            acc.update(torch.argmax(fx, 1).eq(y_cuda).float().mean().item(), y.size(0))
            loss.update(loss_batch.item(), y.size(0))
            if epoch > mArgs.warmup_epochs:
                scheduler.step()
            pbar.set_postfix_str(f"Lr {lr:.2e} Acc {100 * acc.avg:.2f}%")
        if mArgs.log:
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
            pbar.set_postfix_str(f"Acc {100 * acc.avg:.2f}%")
        if mArgs.log:
            logger.add_scalar("test/acc", acc.avg, epoch)
    return network