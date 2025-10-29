import torch

class BaseClosure:
    def __init__(self, optimizer, net, criterion, method, dummy_data, dummy_label, label_pred, original_dy_dx):
        self.optimizer = optimizer
        self.net = net
        self.criterion = criterion
        self.method = method
        self.dummy_data = dummy_data
        self.dummy_label = dummy_label
        self.label_pred = label_pred
        self.original_dy_dx = original_dy_dx

    def __call__(self):
        self.optimizer.zero_grad()
        pred = self.net(self.dummy_data)
        if self.method == 'DLG':
            dummy_loss = - torch.mean(
                torch.sum(torch.softmax(self.dummy_label.float(), -1) * torch.log(torch.softmax(pred, -1)),
                          dim=-1))
        elif self.method == 'iDLG':
            dummy_loss = self.criterion(pred, self.label_pred)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, self.net.parameters(), create_graph=True)
        # dummy_dy_dx = cge(f, params_dict, mask_dict, mArgs.zoo_step_size, net, dummy_data, dummy_label, F.cross_entropy)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, self.original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        return grad_diff




def inv_attack(ghat, optimizer, net, criterion, method, gt_data, dummy_data, dummy_label, label_pred, 
               num_iterations, printfreq, num_dummy, result_path, imidx_list, tp, doplot, tol=0.000001):
    if method == 'DLG':
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
    elif method == 'iDLG':
        optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
        # predict the ground-truth label
        label_pred = gt_label
    history = []
    history_iters = []
    losses = []
    mses = []
    train_iters = []
    for iteration in range(num_iterations):
        closure = closures.BaseClosure(optimizer, net, criterion, method, dummy_data, dummy_label, label_pred, ghat)

        loss = optimizer.step(closure)
        current_loss = loss.item()
        train_iters.append(iteration)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())
        if doplot:
            if iteration % printfreq == 0:
                current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                print(current_time, iteration, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                history_iters.append(iteration)

                for imidx in range(num_dummy):
                    plt.figure(figsize=(12, 8))
                    plt.subplot(3, 10, 1)
                    plt.imshow(tp(gt_data[imidx].cpu()))
                    for i in range(min(len(history), 29)):
                        plt.subplot(3, 10, i + 2)
                        plt.imshow(history[i][imidx])
                        plt.title('iter=%d' % (history_iters[i]))
                        plt.axis('off')
                    if method == 'DLG':
                        plt.savefig('%s/DLG_on_%s_%05d.png' % (result_path, imidx_list, imidx_list[imidx]))
                        plt.close()
                    elif method == 'iDLG':
                        plt.savefig('%s/iDLG_on_%s_%05d.png' % (result_path, imidx_list, imidx_list[imidx]))
                        plt.close()

                if current_loss < tol:  # converge
                            break
    return mses


def search_for_alpha(g_alpha, optimizer, net, criterion, method, gt_data, 
                     dummy_data, dummy_label, label_pred, num_attack_iterations, num_dummy, result_path, imidx_list, tp,
                     num_alpha_seaarch_iterations, epsilon_squared):
     
def search_for_alpha_wrapper(vanilla_dy_dx, zo_dy_dx, optimizer, net, criterion, method, gt_data, 
                     dummy_data, dummy_label, label_pred, num_attack_iterations, num_dummy, result_path, imidx_list, tp,
                     num_alpha_seaarch_iterations, epsilon_squared):
     # alpha = 0
     if inv_attack(vanilla_dy_dx, optimizer, net, criterion, method, gt_data, dummy_data.detach().clone(), dummy_label, label_pred, num_attack_iterations, 
                   None, num_dummy, None, None, tp, False)[-1] >= epsilon_squared:
          return vanilla_dy_dx
     
     
     
