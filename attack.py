
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traceback
import torch.nn as nn
from torchvision import transforms
import os
from torch.distributed import rpc

import sys
sys.path.append(".")

import closures
import optimizers as opt
from tools import *
import traceback
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

def log(msg, log_path, lock):
    with lock:
        with open(log_path, 'a') as f:
            f.write(f"{msg}\n")

def init_csv_write(csvPath, inversion_methods, lock):
    inversion_methods = set(inversion_methods)
    with lock:
        if inversion_methods == {'iDLG', 'DLG'}:
            with open(csvPath, 'w') as csv:
                csv.write("index,DLG loss,DLG MSE,DLG label correct,DLG alpha_star,DLG secs elapsed,iDLG loss,iDLG MSE,iDLG label correct,iDLG alpha_star,iDLG secs elapsed\n")
        elif inversion_methods == {'DLG'}:
            with open(csvPath, 'w') as csv:
                csv.write("index,DLG loss,DLG MSE,DLG label correct,DLG alpha_star,DLG secs elapsed\n")
        elif inversion_methods == {'iDLG'}:
            with open(csvPath, 'w') as csv:
                csv.write("index,iDLG loss,iDLG MSE,iDLG label correct,iDLG alpha_star,iDLG secs elapsed\n")
        else:
            raise ValueError('Entries in list of inversion methods are not in acceptable formats.  Please check and try again.')
        
def run_single_image(idx_net, idx_shuffle, rArgs, mArgs, log_path, csvPath, dst, net, criterion, num_classes, device):
    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])
    if not rArgs.single:
        idx_shuffle = np.random.default_rng(123).permutation(len(dst))
    
    print(f'Running {idx_net}|{rArgs.num_exp} experiment, with {rArgs.num_samples} samples per experiment')
    
    DLG_alpha_star = None
    iDLG_alpha_star = None
    for sample_num in range(0, rArgs.num_samples):
        for method in rArgs.inversion_methods:
            t0 = time.time()
            if rArgs.single:
                print(f'\n{method}, Trying to generate 1 image on [{idx_shuffle[idx_net]}]')
            else:
                print(f'\n{method}, Try to generate {rArgs.num_dummy} images')
            try:
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
                vanilla_dy_dx = [grad.detach().clone() for grad in vanilla_dy_dx]
                zo_dy_dx = [grad.detach().clone() for grad in dy_dx]
                verbose = True
                if method == 'DLG':
                    if DLG_alpha_star == None:
                        DLG_alpha_star = closures.optimize_alpha(vanilla_dy_dx, zo_dy_dx, net, criterion, method, gt_data, gt_label,
                                                            rArgs.num_attack_iterations, rArgs.num_dummy, imidx_list,
                                                            rArgs.num_alpha_search_evals, rArgs.epsilon_squared, verbose, 
                                                            device, num_classes, rArgs.printFreq)
                    alpha_star = DLG_alpha_star
                elif method == 'iDLG':
                    if iDLG_alpha_star == None:
                        iDLG_alpha_star = closures.optimize_alpha(vanilla_dy_dx, zo_dy_dx, net, criterion, method, gt_data, gt_label,
                                                            rArgs.num_attack_iterations, rArgs.num_dummy, imidx_list,
                                                            rArgs.num_alpha_search_evals, rArgs.epsilon_squared, verbose, 
                                                            device, num_classes, rArgs.printFreq)
                    alpha_star = iDLG_alpha_star
                
                zo_dy_dx_nudge = closures.nudge_estimate(zo_dy_dx, vanilla_dy_dx, alpha_star)
                print('Now performing gradient inversion on nudged CGE')
                mse, loss, x_inv, y_inv = closures.inv_attack(zo_dy_dx_nudge, net, criterion, method, gt_data, gt_label, rArgs.num_attack_iterations,
                                                                    rArgs.printFreq, rArgs.num_dummy, rArgs.resultPath, imidx_list, 
                                                                    tp, False, device, num_classes, alpha_star=alpha_star, sample_num=sample_num)
                t1 = time.time()
                if method == 'DLG':
                    loss_DLG = loss
                    label_DLG = torch.argmax(y_inv, dim=-1).detach().item()
                    label_acc_DLG = int(label_DLG == gt_label.item())
                    mse_DLG = mse
                    alpha_star_DLG = DLG_alpha_star
                    secs_elapsed_DLG = round(t1-t0, 2)
                elif method == 'iDLG':
                    loss_iDLG = loss
                    label_iDLG = y_inv.item()
                    label_acc_iDLG = int(label_iDLG == gt_label.item())
                    mse_iDLG = mse
                    alpha_star_iDLG = iDLG_alpha_star
                    secs_elapsed_iDLG = round(t1-t0, 2)
            except Exception as e:
                print('Error:', e)
                print(traceback.format_exc())
                log(f"Error encountered in {method}, at idx_net={idx_net}:{e}", log_path, rArgs.lock)
                log('Full traceback:', log_path, rArgs.lock)
                log(traceback.format_exc(), log_path, rArgs.lock)

        
        print('imidx_list:', imidx_list)
        if method == 'DLG':
            try:
                print('loss_DLG:', loss_DLG, 'loss_iDLG:', loss_iDLG)
                print('mse_DLG:', mse_DLG, 'mse_iDLG:', mse_iDLG)
                print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG, 'lab_iDLG:', label_iDLG)
            except:
                continue
        if method == 'iDLG':
            try:
                print('loss_iDLG:', loss_iDLG)
                print('mse_iDLG:', mse_iDLG)
                print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_iDLG:', label_iDLG)
            except:
                continue

        print('----------------------\n\n')

        loss_DLG, mse_DLG, label_acc_DLG, alpha_star_DLG, secs_elapsed_DLG, loss_iDLG, mse_iDLG, label_acc_iDLG, alpha_star_iDLG, secs_elapsed_iDLG = \
            locals().get('loss_DLG', None), locals().get('mse_DLG', None), locals().get('label_acc_DLG', None), locals().get('alpha_star_DLG', None), locals().get('secs_elapsed_DLG', None),\
            locals().get('loss_iDLG', None), locals().get('mse_iDLG', None), locals().get('label_acc_iDLG', None), locals().get('alpha_star_iDLG', None), locals().get('secs_elapsed_iDLG', None)
        
        write_to_csv(csvPath, method, imidx_list, loss_DLG, mse_DLG, label_acc_DLG, alpha_star_DLG, secs_elapsed_DLG,
                    loss_iDLG, mse_iDLG, label_acc_iDLG, alpha_star_iDLG, secs_elapsed_iDLG, rArgs.inversion_methods, rArgs.lock)

def main(mArgs, rArgs):
    dataset = mArgs.dataset
    root_path = '.'
    # print(os.path.join(root_path, '../data').replace('\\', '/'))
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    csvPath = os.path.join(rArgs.resultPath, "results.csv")
    init_csv_write(csvPath, rArgs.inversion_methods, rArgs.lock)
    log_path = os.path.join(rArgs.resultPath, 'log.txt')
    use_cuda = torch.cuda.is_available()
    device = f'cuda:{mArgs.gpus[0]}' if use_cuda else 'cpu'
    device = "cpu"
    print(device)

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(rArgs.resultPath):
        os.mkdir(rArgs.resultPath)

    ''' load data '''
    shape_img, num_classes, channel, hidden, dst = dataset_loader(dataset, data_path)

    idx_shuffle = list(range(len(dst))) # don't shuffle for now, to preserve experiment reproducibility and error tracking by index np.random.default_rng(123).permutation(len(dst))

    net = init_model(device, mArgs, num_classes, hidden, channel, shape_img)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    ''' train DLG and iDLG '''
    m = multiprocessing.Manager()
    rArgs.lock = m.Lock()
    max_workers = os.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx_net in range(0, rArgs.num_exp):
            futures.append(
                    executor.submit(run_single_image, idx_net, idx_shuffle, rArgs, mArgs, log_path, csvPath, dst, net, criterion, num_classes, device)
                )
            
        for future in futures:
            try:
                future.result() 
            except Exception as e:
                print(e)

        
        
def write_to_csv(csvPath, method, imidx_list, loss_DLG, mse_DLG, label_acc_DLG, alpha_star_DLG, secs_elapsed_DLG, loss_iDLG, mse_iDLG, label_acc_iDLG, alpha_star_iDLG, secs_elapsed_iDLG, inversion_methods, lock):
    with lock:
        if set(inversion_methods) == {'iDLG', 'DLG'}:
            with open(csvPath, 'a') as csv:
                csv.write(f"{imidx_list[0]},{loss_DLG},{mse_DLG},{label_acc_DLG},{alpha_star_DLG},{secs_elapsed_DLG},{loss_iDLG},{mse_iDLG},{label_acc_iDLG},{alpha_star_iDLG},{secs_elapsed_iDLG}\n")
        elif 'DLG' in inversion_methods:
            with open(csvPath, 'a') as csv:
                csv.write(f"{imidx_list[0]},{loss_DLG},{mse_DLG},{label_acc_DLG},{alpha_star_DLG},{secs_elapsed_DLG}\n")
        else:
            with open(csvPath, 'a') as csv:
                csv.write(f"{imidx_list[0]},{loss_iDLG},{mse_iDLG},{label_acc_iDLG},{alpha_star_iDLG},{secs_elapsed_iDLG}\n")

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