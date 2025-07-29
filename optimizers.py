from copy import deepcopy

import torch
from torch.distributed import rpc
from joblib import Parallel, delayed

@torch.no_grad()
def cge(func, params_dict, mask_dict, step_size, net, gt_data, gt_label, loss_func, base=None):
    if base is None:
        base = func(params_dict, net, gt_data, gt_label, loss_func)
    grads_dict = {}
    for key, param in params_dict.items():
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten()
        else:
            mask_flat = torch.ones_like(param).flatten()
        directional_derivative = torch.zeros_like(param)
        directional_derivative_flat = directional_derivative.flatten()
        for idx in mask_flat.nonzero().flatten():
            perturbed_params_dict = deepcopy(params_dict)
            p_flat = perturbed_params_dict[key].flatten()
            p_flat[idx] += step_size
            directional_derivative_flat[idx] = (func(perturbed_params_dict, net, gt_data, gt_label, loss_func) - base) / step_size
        grads_dict[key] = directional_derivative.to(param.device)
    return list(grads_dict.values())

@torch.no_grad()
def cge_multi(func, params_dict, mask_dict, step_size, net, gt_data, gt_label, loss_func, base=None):
    if base is None:
        base = func(params_dict, net, gt_data, gt_label, loss_func)
    from functools import partial
    compute = partial(grad_compute,
                      func=func,
                      params_dict=params_dict,
                      mask_dict=mask_dict,
                      step_size=step_size,
                      net=net,
                      gt_data=gt_data.detach().clone(),
                      gt_label=gt_label,
                      loss_func=loss_func,
                      base=base.detach().clone()
                      )
    # import torch.multiprocessing as mp
    # with mp.Pool() as pool:
    #     grads_dict = dict(pool.map(compute, params_dict.items()))
    grads = Parallel(n_jobs=-1)(delayed(compute)(item) for item in params_dict.items())
    grads_dict = dict(grads)

    for key, tensor in grads_dict.items():
        if isinstance(tensor, torch.Tensor):
            grads_dict[key] = tensor.requires_grad_(True)

    return list(grads_dict.values())

# cge_multi helper function
def grad_compute(item, func, params_dict, mask_dict, step_size, net, gt_data, gt_label, loss_func, base):
    key, param = item
    if 'orig' in key:
        mask_key = key.replace('orig', 'mask')
        mask_flat = mask_dict[mask_key].flatten()
    else:
        mask_flat = torch.ones_like(param).flatten()
    directional_derivative = torch.zeros_like(param)
    directional_derivative_flat = directional_derivative.flatten()
    # mask_flat = mask_flat.to("cpu")
    for idx in mask_flat.nonzero():  # .flatten()
        perturbed_params_dict = deepcopy(params_dict)
        # perturbed_params_dict = {
        #     k: v.clone() if k != key else v.clone().view(-1)
        #     for k, v in params_dict.items()
        # }
        p_clone = perturbed_params_dict[key].clone()
        p_flat = p_clone.flatten()
        p_flat[idx] = p_flat[idx] + step_size
        perturbed_params_dict[key] = p_flat.view(param.shape)
        directional_derivative_flat[idx] = (func(perturbed_params_dict, net, gt_data, gt_label,
                                                 loss_func) - base) / step_size
    return key, directional_derivative.to(param.device)

def cge_gpu(remote_networks, network, gpus, process_per_gpu, x, y, cge_step_size):
    # cge_step_size = args.zoo_step_size
    params_dict = {
        name: p for name, p in network.named_parameters() if p.requires_grad
    }
    device = next(network.parameters()).device
    x_rref, y_rref = rpc.RRef(x), rpc.RRef(y)
    grads_signal = []
    for gpu in gpus:
        for i in range(process_per_gpu):
            grads_signal.append(
                remote_networks[f"{gpu}-{i}"].rpc_async(timeout=0).calculate_grads(x_rref, y_rref, cge_step_size))
    grads = []
    for g in grads_signal:
        grads.append(g.wait())
    grads = torch.cat(grads, dim=0).to(device)

    # dlg_grads = {} # gradients for dlg attack
    grad_list = []
    for name_id, (name, param) in enumerate(params_dict.items()):
        param.grad = torch.zeros_like(param)
        grads_indices_and_values = grads[grads[:, 0] == name_id, 1:]
        param_grad_flat = param.grad.flatten()
        param_grad_flat[grads_indices_and_values[:, 0].long()] = grads_indices_and_values[:, 1]

        # dlg_grads[name] = param.grad.clone()
        grad_list.append(param.grad.clone())
    # return dlg_grads

    return grad_list

def rge(func, params_dict, sample_size, step_size, net, gt_data, gt_label, loss_func, base=None):
    if base == None:
        base = func(params_dict, net, gt_data, gt_label, loss_func)
    grads_dict = {}
    for _ in range(sample_size):
        perturbs_dict, perturbed_params_dict = {}, {}
        for key, param in params_dict.items():
            perturb = torch.randn_like(param)
            perturb /= (torch.norm(perturb) + 1e-8)
            perturb *= step_size
            perturbs_dict[key] = perturb
            perturbed_params_dict[key] = perturb + param
        directional_derivative = (func(perturbed_params_dict, net, gt_data, gt_label,
                                       loss_func) - base) / step_size
        if len(grads_dict.keys()) == len(params_dict.keys()):
            for key, perturb in perturbs_dict.items():
                grads_dict[key] += (perturb * directional_derivative / sample_size).to(param.device)
        else:
            for key, perturb in perturbs_dict.items():
                grads_dict[key] = (perturb * directional_derivative / sample_size).to(param.device)
    return list(grads_dict.values())

@torch.no_grad()
def f(params_dict, network, x, y, loss_func):
    state_dict_backup = network.state_dict()
    network.load_state_dict(params_dict, strict=False)
    loss = loss_func(network(x), y).detach().item()
    network.load_state_dict(state_dict_backup)
    return loss