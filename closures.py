import torch
import time
import math
from matplotlib import pyplot as plt
class BaseClosure:
    def __init__(self, optimizer, net, criterion, method, dummy_data, dummy_label, original_dy_dx):
        self.optimizer = optimizer
        self.net = net
        self.criterion = criterion
        self.method = method
        self.dummy_data = dummy_data
        self.dummy_label = dummy_label
        self.original_dy_dx = original_dy_dx

    def __call__(self):
        self.optimizer.zero_grad()
        pred = self.net(self.dummy_data)
        if self.method == 'DLG':
            dummy_loss = - torch.mean(
                torch.sum(torch.softmax(self.dummy_label.float(), -1) * torch.log(torch.softmax(pred, -1)),
                          dim=-1))
        elif self.method == 'iDLG':
            dummy_loss = self.criterion(pred, self.dummy_label) # in this case the dummy label is the analytically reverse engineered label
        dummy_dy_dx = torch.autograd.grad(dummy_loss, self.net.parameters(), create_graph=True)
        # dummy_dy_dx = cge(f, params_dict, mask_dict, mArgs.zoo_step_size, net, dummy_data, dummy_label, F.cross_entropy)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, self.original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        return grad_diff




def inv_attack(ghat, net, criterion, method, gt_data, gt_label,
               num_iterations, printfreq, num_dummy, result_path, imidx_list, tp, doplot, device, num_classes, 
               tol=0.000001, lr=1, verbose=True, alpha_star=None):
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    if method == 'DLG':
        dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
    elif method == 'iDLG':
        optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
        dummy_label = torch.argmin(torch.sum(ghat[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False) # = gt_label
    history = []
    history_iters = []
    losses = []
    mses = []
    train_iters = []
    for iteration in range(num_iterations):
        closure = BaseClosure(optimizer, net, criterion, method, dummy_data, dummy_label, ghat)
        loss = optimizer.step(closure)
        current_loss = loss.item()
        train_iters.append(iteration)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())
        if verbose:
            if iteration % printfreq == 0:
                current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                print(current_time, iteration, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
                if doplot:
                    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                    history_iters.append(iteration)
        if current_loss < tol:  # converge
                    if verbose:
                        print('Attack stopped early due to loss dropping below tol.')
                    break
    if doplot:
        do_the_plot(num_dummy, num_classes, gt_data, gt_label, dummy_label, history, history_iters, mses, alpha_star, result_path, imidx_list, method, tp)
    return float(mses[-1]), float(losses[-1]), dummy_data, dummy_label

def do_the_plot(num_dummy, num_classes, gt_data, gt_label, dummy_label, history, history_iters, mses, alpha_star, result_path, imidx_list, method, tp):
    for imidx in range(num_dummy):
        num_history = len(history)
        total_plots = 1 + num_history  # Ground truth + history
        
        # Always use 3 rows, 10 columns (max 30 plots)
        nrows = 3
        ncols = 10
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(nrows, ncols, 1)
        ax.imshow(tp(gt_data[imidx].cpu()))
        ax.set_title('Ground Truth', fontsize=8)
        ax.axis('off')
        
        # Plot history (up to 29 images to fit in remaining slots)
        upp = min(num_history, 29)
        for i in range(upp):
            ax = plt.subplot(nrows, ncols, i + 2)
            ax.imshow(history[-upp+i][imidx])
            ax.set_title(f'iter={history_iters[-upp+i]}', fontsize=8)
            ax.axis('off')
        
        # Remove unused subplots
        for i in range(total_plots, nrows * ncols):
            ax = plt.subplot(nrows, ncols, i + 1)
            ax.remove()
        
        # If we have fewer than 2 rows worth of data, remove entire empty rows
        if total_plots <= ncols:  # Only need 1 row
            for i in range(ncols, nrows * ncols):
                ax = plt.subplot(nrows, ncols, i + 1)
                ax.remove()
        elif total_plots <= 2 * ncols:  # Only need 2 rows
            for i in range(2 * ncols, nrows * ncols):
                ax = plt.subplot(nrows, ncols, i + 1)
                ax.remove()
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.suptitle(f'Final MSE: {round(mses[-1], 5)}\n' + rf'$\alpha^\star={alpha_star}$')
    # Replace line 77 in closures.py with this:

    # Add probability table to the figure (embedded, not separate file)
    if method == 'DLG':
        probs = torch.softmax(dummy_label[0].detach(), dim=-1).cpu().numpy()
    else:  # iDLG
        probs = torch.zeros(num_classes)
        probs[dummy_label.item()] = 1.0
        probs = probs.cpu().numpy()
    
    gt_class = gt_label[0].item()
    pred_class = probs.argmax()
    
    # Add a new subplot for the probability table at the top
    ax_label = plt.gcf().add_axes([0.02, 0.915, 0.12, 0.04])
    ax_label.axis('off')
    ax_label.text(0.5, 0.5, 'Final\nreverse\nengineered\nprobabilities', 
                    ha='center', va='center', fontsize=8, 
                    multialignment='center')
    
    # Position the table
    ax_table = plt.gcf().add_axes([0.15, 0.915, 0.7, 0.04])
    ax_table.axis('off')
    
    # Create table
    col_labels = [str(i) for i in range(num_classes)]
    try:
        cell_text = [[f'{probs[i]:.3f}' for i in range(num_classes)]]
    except:
        import pdb
        pdb.set_trace()
    
    table = ax_table.table(cellText=cell_text, colLabels=col_labels,
                            cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color cells
    for i in range(num_classes):
        cell = table[(1, i)]  # Data row
        if i == pred_class and i == gt_class:
            # Correct prediction - lime green
            cell.set_facecolor('#00FF00')
            cell.set_text_props(color='black', weight='bold')
        elif i == pred_class:
            # Wrong prediction - red with white text
            cell.set_facecolor('#FF0000')
            cell.set_text_props(color='white', weight='bold')
        elif i == gt_class:
            # Ground truth (missed) - lime green
            cell.set_facecolor('#00FF00')
            cell.set_text_props(color='black', weight='bold')
        
        # Style header
        header_cell = table[(0, i)]
        header_cell.set_facecolor('#E0E0E0')
        header_cell.set_text_props(weight='bold', fontsize=8)
    
    # Create suptitle with more space between MSE and alpha
    alpha_display = locals().get('alpha_star', 'N/A')
    plt.suptitle(f'Final MSE: {round(mses[-1], 5)}\n\n\n\n' + 
                rf'$\alpha^\star={alpha_display}$', 
                y=0.98, fontsize=12)

    if method == 'DLG':
        plt.savefig('%s/DLG_on_%s_%05d.png' % (result_path, imidx_list, imidx_list[imidx]))
        plt.close()
    elif method == 'iDLG':
        plt.savefig('%s/iDLG_on_%s_%05d.png' % (result_path, imidx_list, imidx_list[imidx]))
        plt.close()


def nudge_estimate(estimate, true, alpha):
    nudge = [true[i] + (estimate[i] - true[i]) * alpha for i in range(len(true))]
    nudge = [grad.detach().clone() for grad in nudge]
    return nudge

def initialize_bounds(lo, hi, eval_fn, is_pos, num_evals, epsilon_squared, mse_z_ghat_0, verbose):
    num_evals_init = num_evals
    init_range = hi - lo
    if is_pos:
        mse, x, y, ghat = eval_fn(hi)
        while mse < epsilon_squared and num_evals > 0:
            lo = hi
            mse_z_ghat_0 = (mse, x, y, ghat)
            hi += init_range
            mse, x, y, ghat = eval_fn(hi)
            num_evals -= 1
        bound_dict = {'lo':mse_z_ghat_0, 'hi':(mse, x, y, ghat)}
    else:
        mse, x, y, ghat = eval_fn(lo)
        while mse > epsilon_squared and num_evals > 0: # remember epsilon_squared and mse will be negative in this case
            hi = lo
            mse_z_ghat_0 = (mse, x, y, ghat)
            lo -= init_range
            mse, x, y, ghat = eval_fn(lo)
            num_evals -= 1
        bound_dict = {'lo':(mse, x, y, ghat), 'hi':mse_z_ghat_0}
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
        mse, x, y, ghat = eval_fn(mid)
        num_evals -= 1
        if verbose:
            print(f'Evals remaning: {num_evals}, alpha:{mid}, mse:{mse}, lo:{lo}, hi:{hi}')
        if mse < epsilon_squared:
            lo = mid
            bound_dict['lo'] = (mse, x, y, ghat)
        elif mse > epsilon_squared:
            hi = mid
            bound_dict['hi'] = (mse, x, y, ghat)
        else:
            return (mid, mse, x, y, ghat)
        
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
                   num_alpha_search_evals, epsilon_squared, verbose, 
                   device, num_classes, printfreq):
    inv_attack_closure = lambda ghat: inv_attack(ghat, net, criterion, method, gt_data, gt_label,
                                       num_attack_iterations, printfreq, num_dummy, None, 
                                       imidx_list, None, False, device, num_classes, verbose=False)
    def get_bisection_search_eval_fn(sign):
        if sign == 'pos':
            def bisection_search_eval_fn(alpha):
                ghat = nudge_estimate(zo_dy_dx, vanilla_dy_dx, alpha)
                mse, _, x, y = inv_attack_closure(ghat)
                return (mse, x, y, ghat)
        elif sign == 'neg':
            def bisection_search_eval_fn(alpha):
                ghat = nudge_estimate(zo_dy_dx, vanilla_dy_dx, alpha)
                mse, _, x, y = inv_attack_closure(ghat)
                return  (-mse, x, y, ghat)
        else:
            raise ValueError('sign must be either `pos` or `neg')
        return bisection_search_eval_fn

            
    # alpha = 0
    mse_0, _, x_0, y_0 = inv_attack_closure(vanilla_dy_dx)
    if mse_0 >= epsilon_squared:
        # if the vanilla gradient (alpha=0) is already larger than the error tol, then we are satisfying the constraint and can't reduce alpha any further. return 
        if verbose:
            print(f'FO gradient already exceeds epsilon_squared={epsilon_squared}, no improvement possible.')
        return 0.0
    num_pos_alpha_search_evals = math.ceil(num_alpha_search_evals / 2)
    if verbose:
        print(f'\nSearching for best alpha starting from FO-gradient and in the positive direction (towards ZO)...')
    alpha_star_pos, mse_star_pos, x_inv_pos, y_inv_pos, ghat_alpha_star_pos = bisection_search(0, 1, get_bisection_search_eval_fn('pos'), num_pos_alpha_search_evals, epsilon_squared, 
                                                                          (mse_0, x_0, y_0, vanilla_dy_dx), verbose=verbose)
    assert mse_star_pos >= epsilon_squared, ValueError('The mse violates the constraint after optimizing alpha_pos, which shouldnt happen.  Check the code for bugs.')
    if verbose:
        print(f'Found alpha={alpha_star_pos} searching in positive direction.')
    print()
    num_neg_alpha_search_evals = num_alpha_search_evals - num_pos_alpha_search_evals
    if verbose:
         print(f'Searching for best alpha starting from FO-gradient and in the negative direction (away from ZO)...')
    alpha_star_neg, mse_star_neg, x_inv_neg, y_inv_neg, ghat_alpha_star_neg = bisection_search(-1, 0, get_bisection_search_eval_fn('neg'), num_neg_alpha_search_evals, -epsilon_squared,
                                                                          (mse_0, x_0, y_0, vanilla_dy_dx), verbose=verbose)
    assert -mse_star_neg >= epsilon_squared, ValueError('The mse violates the constraint after optimizing alpha_neg, which shouldnt happen.  Check the code for bugs.')
    if verbose:
        print(f'Found alpha={alpha_star_neg} searching in negative direction.')
    alpha_star = decide_between_neg_and_pos_alpha(alpha_star_pos, alpha_star_neg, mse_star_pos, mse_star_neg)
    if verbose:
        print(f'\nFound alpha={alpha_star} after {num_alpha_search_evals} inversion attack evaluations.\n')
    return alpha_star
     
     
     
     
     
     
