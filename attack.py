
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch.nn as nn
from torchvision import transforms
import os
from torch.distributed import rpc

import sys
sys.path.append(".")

import closures
import optimizers as opt
from tools import *
from init_util import dataset_loader, init_model

# run command on first session in terminal when ssh
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# numpy 1.26.4
# pytorch 1.13.1

# mu: smoothness = cgs_step_size
# mu: cge_estimate
# step size:
# p: precentile start with higher value, will decrease privacy
# find the starting value of p
# high p


def main(mArgs, rArgs):
    dataset = 'cifar10'
    root_path = '.'
    print(os.path.join(root_path, '../data').replace('\\', '/'))
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    csvPath = os.path.join(rArgs.resultPath, "results.csv")
    with open(csvPath, 'w') as csv:
        csv.write("index,DLG loss,DLG MSE,iDLG loss,iDLG MSE\n")

    lr = 1
    use_cuda = torch.cuda.is_available()
    device = f'cuda:{mArgs.gpus[0]}' if use_cuda else 'cpu'
    device = "cpu"
    print(device)

    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', rArgs.resultPath)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(rArgs.resultPath):
        os.mkdir(rArgs.resultPath)

    ''' load data '''
    shape_img, num_classes, channel, hidden, dst = dataset_loader(dataset, data_path)

    idx_shuffle = np.random.default_rng(123).permutation(len(dst))

    net = init_model(device, mArgs, num_classes, hidden, channel)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    ''' train DLG and iDLG '''
    for idx_net in range(0, rArgs.num_exp):
        if not rArgs.single:
            idx_shuffle = np.random.default_rng(123).permutation(len(dst))

        print(f'Running {idx_net}|{rArgs.num_exp} experiment')
        for method in ['iDLG', "DLG"]:
            if rArgs.single:
                print(f'{method}, Trying to generate 1 image on [{idx_shuffle[idx_net]}]')
            else:
                print(f'{method}, Try to generate {rArgs.num_dummy} images')

            imidx_list = []
            for imidx in range(rArgs.num_dummy):
                if rArgs.single:
                    idx = idx_shuffle[idx_net]
                else:
                    idx = idx_shuffle[imidx]
                imidx_list.append(idx)
                tmp_datum = tt(dst[idx][0]).float().to(device)
                tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                tmp_label = tmp_label.view(1, )
                if imidx == 0:
                    gt_data = tmp_datum
                    gt_label = tmp_label
                else:
                    gt_data = torch.cat((gt_data, tmp_datum), dim=0)
                    gt_label = torch.cat((gt_label, tmp_label), dim=0)

            params_dict = {
                name: p for name, p in net.named_parameters() if p.requires_grad
            }
            mask_dict = {
                name: p for name, p in net.named_buffers() if 'mask' in name
            }
            dy_dx = opt.cge(opt.f, params_dict, mask_dict, mArgs.zoo_step_size, net, gt_data, gt_label, F.cross_entropy)
            # compute original gradient
            out = net(gt_data)
            y = criterion(out, gt_label)
            vanilla_dy_dx = torch.autograd.grad(y, net.parameters())
            

            zo_dy_dx = [grad.detach().clone() for grad in dy_dx]

            # generate dummy data and label
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            # dummy_label = gt_label.to(device)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)


                # label_pred = torch.argmin(torch.sum(zo_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            zo_dy_dx_nudge = [zo_dy_dx[i] + (vanilla_dy_dx[i] - zo_dy_dx[i]) * mArgs.alpha for i in range(len(zo_dy_dx))]
            zo_dy_dx_nudge = [grad.detach().clone() for grad in zo_dy_dx_nudge]
            losses, mses, = inv_attack(zo_dy_dx_nudge, optimizer, net, criterion, method, 
                                       gt_data, dummy_data, dummy_label, label_pred,
                                       rArgs.num_iterations, rArgs.printFreq, rArgs.num_dummy, 
                                       rArgs.resultPath, imidx_list, tp)
            
            if method == 'DLG':
                loss_DLG = losses
                label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                mse_DLG = mses
            elif method == 'iDLG':
                loss_iDLG = losses
                label_iDLG = label_pred.item()
                mse_iDLG = mses

        print('imidx_list:', imidx_list)
        if method == 'DLG':
            print('loss_DLG:', loss_DLG[-1], 'loss_iDLG:', loss_iDLG[-1])
            print('mse_DLG:', mse_DLG[-1], 'mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)
        if method == 'iDLG':
            print('loss_iDLG:', loss_iDLG[-1])
            print('mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_iDLG:', label_iDLG)

        print('----------------------\n\n')
        # index, loss, mse
        with open(csvPath, 'a') as csv:
            csv.write(f"{imidx_list[0]},{loss_DLG[-1]},{mse_DLG[-1]},{loss_iDLG[-1]},{mse_iDLG[-1]}\n")


def init_process(rank, world_size, mArgs, rArgs):
    os.environ['MASTER_ADDR'] = mArgs.master_addr
    os.environ['MASTER_PORT'] = mArgs.master_port

    if rank == 0:
        rpc.init_rpc(
            f"master", rank=rank, world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=mArgs.process_per_gpu * world_size + 1, rpc_timeout=0.)
        )
        main(mArgs, rArgs)
    else:
        gpu = mArgs.gpus[(rank - 1) // mArgs.process_per_gpu]
        i = (rank - 1) % mArgs.process_per_gpu
        rpc.init_rpc(
            f"{gpu}-{i}", rank=rank, world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=mArgs.process_per_gpu * world_size + 1, rpc_timeout=0.)
        )
    rpc.shutdown()


