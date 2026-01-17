import os
import sys
import argparse
import torch
sys.path.append(".")
from algorithm.prune import global_prune, layer_sparsity
from data import prepare_dataset, alt_dataset
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--network', type=str, choices=['resnet20', 'lenet', 'lenetAlt'])
    p.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'])
    p.add_argument('--method', type=str, choices=['zo_grasp', 'grasp'])
    p.add_argument('--sparsity', type=float, default=0.)
    args = p.parse_args()

    loaders, class_num = alt_dataset(args.dataset)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"device: {device}")

    if args.dataset == "mnist":
        hidden = 588
        channel = 1
        input_size = (28, 28)
    elif args.dataset == "cifar10":
        hidden = 768
        channel = 3
        input_size = (32, 32)
    # Network
    if args.network == "resnet20":
        from models.resnet_s import resnet20, param_name_to_module_id_rn20
        param_name_to_module_id = param_name_to_module_id_rn20
        network_init_func = resnet20
        network_kwargs = {
            "num_classes": class_num
        }
    elif args.network == "lenet":
        from models.lenet import lenet, param_name_to_module_id_lenet
        param_name_to_module_id = param_name_to_module_id_lenet
        network_init_func = lenet
        network_kwargs = {
            "num_classes": class_num,
            "hidden" : hidden,
            "channel": channel
        }
    elif args.network == "lenetAlt":
        from models.lenetAlt import LeNetAlt, param_name_to_module_id_lenet_alt
        param_name_to_module_id = param_name_to_module_id_lenet_alt
        network_init_func = LeNetAlt
        network_kwargs = {
            "num_classes": class_num,
            "channels" : channel,
            "input_size": input_size
        }
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    network = network_init_func(**network_kwargs).to(device)
    loss_function = torch.nn.CrossEntropyLoss()

    global_prune(network, args.sparsity, args.method, 10, loaders['train'], zoo_sample_size=192, zoo_step_size=5e-3)

    os.makedirs(f"Layer_Sparsity/{args.network}", exist_ok=True)
    torch.save(layer_sparsity(network), f"Layer_Sparsity/{args.network}/{args.method}_{args.sparsity}.pth")