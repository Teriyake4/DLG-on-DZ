import torch
import time
from matplotlib import pyplot as plt
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




def inv_attack(ghat, net, criterion, method, gt_data, gt_label,
               num_iterations, printfreq, num_dummy, result_path, imidx_list, tp, doplot, device, num_classes, tol=0.000001, lr=1):
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    if method == 'DLG':
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
        label_pred = None
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
        closure = BaseClosure(optimizer, net, criterion, method, dummy_data, dummy_label, label_pred, ghat)
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
    return mses[-1], losses[-1], dummy_data, dummy_label

def nudge_estimate(estimate, true, alpha):
    nudge = [true[i] + (estimate[i] - true[i]) * alpha for i in range(len(true))]
    nudge = [grad.detach().clone() for grad in nudge]
    return nudge

def initialize_bounds(lo, hi, eval_fn, is_pos, num_evals, epsilon_squared, mse_z_ghat_0, verbose):
    num_evals_init = num_evals
    init_range = hi - lo
    if is_pos:
        mse, z, ghat = eval_fn(hi)
        while mse < epsilon_squared and num_evals > 0:
            hi += init_range
            mse, z, ghat = eval_fn(hi)
            num_evals -= 1
        bound_dict = {'lo':mse_z_ghat_0, 'hi':(mse, z, ghat)}
    else:
        mse, z, ghat = eval_fn(lo)
        while mse > epsilon_squared and num_evals > 0: # remember epsilon_squared will be negative in this case
            lo -= init_range
            mse, z, ghat = eval_fn(lo)
            num_evals -= 1
        bound_dict = {'lo':(mse, z, ghat), 'hi':mse_z_ghat_0}
    if num_evals == 0:
        raise ValueError('Exceeded number of allowable evaluations during initialization of search bounds.')
    elif verbose:
        print(f'Initial bounds: ({lo}, {hi})')
        print(f'{num_evals}/{num_evals_init} evaluations remaining after initialization.')
    return lo, hi, bound_dict, num_evals

def bisection_search(lo, hi , eval_fn, num_evals, epsilon_squared, mse_z_ghat_0, verbose, tol=0):
    is_pos = hi > 0
    lo, hi, bound_dict, num_evals = initialize_bounds(lo, hi, eval_fn, is_pos, num_evals, epsilon_squared, mse_z_ghat_0, verbose)
    while lo < hi - tol and num_evals > 0:
        mid = (lo + hi) / 2
        mse, z, ghat = eval_fn(mid)
        if verbose:
            print(f'alpha:{mid}, mse:{mse}, lo:{lo}, hi:{hi}')
        if mse < epsilon_squared:
            lo = mid
            bound_dict['lo'] = (mse, z, ghat)
        elif mse > epsilon_squared:
            hi = mid
            bound_dict['hi'] = (mse, z, ghat)
        else:
            return (mid, mse, z, ghat)
        num_evals -= 1
    return (hi,) + bound_dict['hi'] if is_pos else (lo,) + bound_dict['lo']

            
def decide_between_neg_and_pos_alpha(alpha_pos, alpha_neg, mse_pos, mse_neg):
    if abs(alpha_neg) < alpha_pos:
        return alpha_neg
    elif abs(alpha_neg) > alpha_pos:
        return alpha_pos
    else:
        return alpha_pos if mse_pos >= -mse_neg else alpha_neg

def optimize_alpha(vanilla_dy_dx, zo_dy_dx, net, criterion, method, gt_data, gt_label,
                   num_attack_iterations, num_dummy, imidx_list,
                   num_alpha_search_evals, epsilon_squared, verbose, device, num_classes):
     inv_attack_closure = lambda ghat: inv_attack(ghat, net, criterion, method, gt_data, gt_label,
                                       num_attack_iterations, None, num_dummy, None, 
                                       imidx_list, None, False, device, num_classes)
     def get_bisection_search_eval_fn(sign):
        if sign == 'pos':
            def bisection_search_eval_fn(alpha):
                ghat = nudge_estimate(zo_dy_dx, vanilla_dy_dx, alpha)
                return inv_attack_closure(ghat) + (ghat,)
        elif sign == 'neg':
            def bisection_search_eval_fn(alpha):
                ghat = nudge_estimate(zo_dy_dx, vanilla_dy_dx, alpha)
                mse, z, ghat = inv_attack_closure(ghat)
                return  (-mse, z, ghat)
        else:
            raise ValueError('sign must be either `pos` or `neg')
        return bisection_search_eval_fn

            
     # alpha = 0
     mse_0, z_0 = inv_attack_closure(vanilla_dy_dx)
     if mse_0 >= epsilon_squared:
          # if the vanilla gradient (alpha=0) is already larger than the error tol, then we are satisfying the constraint and can't reduce alpha any further. return 
          return 0, vanilla_dy_dx
     alpha_star_pos, mse_star_pos, ghat_alpha_star_pos = bisection_search(0, 1, get_bisection_search_eval_fn('pos'), num_alpha_search_evals, epsilon_squared, 
                                                                          (mse_0, z_0, vanilla_dy_dx), verbose=verbose)
     assert mse_star_pos >= epsilon_squared, ValueError('The mse violates the constraint after optimizing alpha_pos, which shouldnt happen.  Check the code for bugs.')
     alpha_star_neg, mse_star_neg, ghat_alpha_star_neg = bisection_search(-1, 0, get_bisection_search_eval_fn('neg'), num_alpha_search_evals, -epsilon_squared,
                                                                          (mse_0, z_0, vanilla_dy_dx), verbose=verbose)
     assert -mse_star_neg >= epsilon_squared, ValueError('The mse violates the constraint after optimizing alpha_neg, which shouldnt happen.  Check the code for bugs.')
     alpha_star = decide_between_neg_and_pos_alpha(alpha_star_pos, alpha_star_neg, mse_star_pos, mse_star_neg)
     return alpha_star
     
     
     
     
     
     
