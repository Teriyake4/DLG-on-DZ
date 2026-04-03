import os

import scipy
stats = scipy.stats
import torch
import time
import json
import math
import numpy as np
from scipy.stats import beta
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


# Getting Alpha

epsilon_squared = 0.1
num_alpha_search_iterations = 1000 # 100 if too long
ci_protection_num_init_samples = 10 # 5 if too long
ci_protection_delta = 0.1
ci_protection_tau = 0.1
ci_protection_tol = 0.01

def inv_attack(ghat, net, criterion, method, gt_data, gt_label,
               num_iterations, printfreq, num_dummy, result_path, imidx_list, tp, doplot, device, num_classes, 
               tol=0.000001, lr=1, verbose=True, alpha_star=None, sample_num=0):
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
        do_the_plot(num_dummy, num_classes, gt_data, gt_label, dummy_label, history, history_iters, mses, alpha_star, result_path, imidx_list, method, tp, sample_num)
    return float(mses[-1]), float(losses[-1]), dummy_data, dummy_label

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
    start = time.time()
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
    mse_0, _, x_0, y_0 = inv_attack_closure(vanilla_dy_dx)
    if mse_0 >= epsilon_squared:
        # if the vanilla gradient (alpha=0) is already larger than the error tol, then we are satisfying the constraint and can't reduce alpha any further. return 
        if verbose:
            print(f'FO gradient already exceeds epsilon_squared={epsilon_squared}, no improvement possible.')
        return 0.0
    num_pos_alpha_search_evals = math.ceil(num_alpha_search_evals / 2)
    if verbose:
        print(f'\nSearching for best alpha starting from FO-gradient and in the positive direction (towards ZO)...')
    # alpha_star_pos, mse_star_pos, x_inv_pos, y_inv_pos, ghat_alpha_star_pos = bisection_search(0, 1, get_bisection_search_eval_fn('pos'), num_pos_alpha_search_evals, epsilon_squared, 
    #                                                                       (mse_0, x_0, y_0, vanilla_dy_dx), verbose=verbose)
    alpha_star_pos, search_history, status_history, eval_cnt_history, out_fold = custom_bisection_search(method, 0, 1, get_bisection_search_eval_fn('pos'), num_alpha_search_iterations, epsilon_squared, 
                                                                               (mse_0, x_0, y_0, vanilla_dy_dx), True, ci_protection_num_init_samples, ci_protection_tau, ci_protection_delta, ci_protection_tol)
    print("Elpased time for alpha search: ", time.time() - start)
    return alpha_star_pos
    # assert mse_star_pos >= epsilon_squared, ValueError('The mse violates the constraint after optimizing alpha_pos, which shouldnt happen.  Check the code for bugs.')
    if verbose:
        print(f'Found alpha={alpha_star_pos} searching in positive direction.')
    print()
    num_neg_alpha_search_evals = num_alpha_search_evals - num_pos_alpha_search_evals
    if verbose:
         print(f'Searching for best alpha starting from FO-gradient and in the negative direction (away from ZO)...')
    # alpha_star_neg, mse_star_neg, x_inv_neg, y_inv_neg, ghat_alpha_star_neg = bisection_search(-1, 0, get_bisection_search_eval_fn('neg'), num_neg_alpha_search_evals, -epsilon_squared,
    #                                                                       (mse_0, x_0, y_0, vanilla_dy_dx), verbose=verbose)
    assert -mse_star_neg >= epsilon_squared, ValueError('The mse violates the constraint after optimizing alpha_neg, which shouldnt happen.  Check the code for bugs.')
    if verbose:
        print(f'Found alpha={alpha_star_neg} searching in negative direction.')
    alpha_star = decide_between_neg_and_pos_alpha(alpha_star_pos, alpha_star_neg, mse_star_pos, mse_star_neg)
    if verbose:
        print(f'\nFound alpha={alpha_star} after {num_alpha_search_evals} inversion attack evaluations.\n')
    return alpha_star


# Bisection Search for Alpha

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


# Stochastic Bisection Search for Alpha

# Notes
# Redundant variables: mse_z_ghat_0, do_plot, get_ci_protection_output

def initialize_bounds_b(lo, hi, eval_fn, is_pos, num_evals, epsilon_squared, mse_z_ghat_0, verbose):
    num_evals_init = num_evals
    init_range = hi - lo
    if is_pos:
        mse, z, ghat = eval_fn(hi)
        num_evals -= 1
        while mse < epsilon_squared and num_evals > 0:
            hi += init_range
            mse, z, ghat = eval_fn(hi)
            num_evals -= 1
        bound_dict = {'lo':mse_z_ghat_0, 'hi':(mse, z, ghat)}
    else:
        mse, z, ghat = eval_fn(lo)
        num_evals -= 1
        while mse > epsilon_squared and num_evals > 0: # remember epsilon_squared will be negative in this case
            lo -= init_range
            mse, z, ghat = eval_fn(lo)
            num_evals -= 1
        bound_dict = {'lo':(mse, z, ghat), 'hi':mse_z_ghat_0}
    if num_evals <= 0:
        raise ValueError('Exceeded number of allowable evaluations during initialization of search bounds.')
    elif verbose:
        print(f'Initial bounds: ({lo}, {hi})')
        print(f'{num_evals}/{num_evals_init} evaluations remaining after initialization.')
    return lo, hi, bound_dict, num_evals

def get_v_statistic(mses, epsilon_squared):
    return int(sum(mses < epsilon_squared))

def get_ci_bounds(mses, epsilon_squared, delta):
    n = len(mses)
    if n == 0:
        raise ValueError("mses must be non-empty")
    v = get_v_statistic(np.array(mses), epsilon_squared)
    if v == 0:
        theta_l = 0
        theta_u = stats.beta.ppf(1 - delta/2, 1, n) 
    elif v == n:
        theta_l = stats.beta.ppf(delta/2, n, 1)       # Beta(n, 1)
        theta_u = 1.0
    else:
        theta_l = stats.beta.ppf(delta/2, v, n-v+1)
        theta_u = stats.beta.ppf(1-delta/2, v+1, n-v)
    return theta_l, theta_u

def get_ci_protection_output(alpha, epsilon_squared, eval_fn, num_evals, num_init_samples, tau=0.1, delta=0.1, tol=0.0000001, doplot=False):
    if num_evals <= 0:
        return 'inconclusive', num_evals
    num_init_samples = min(num_evals, num_init_samples)
    mses = [eval_fn(alpha)[0] for _ in range(num_init_samples)]
    num_evals -= num_init_samples
    theta_l, theta_u = get_ci_bounds(mses, epsilon_squared, delta)
    if tau > theta_u: return 'safe', num_evals
    if tau < theta_l: return 'unsafe', num_evals
    while True:
        if theta_u - theta_l < tol: return 'boundary', num_evals
        if num_evals <= 0: return 'inconclusive', num_evals
        new_mse = eval_fn(alpha)[0]
        num_evals -= 1
        mses.append(new_mse)
        theta_l, theta_u = get_ci_bounds(mses, epsilon_squared, delta)
        if tau > theta_u:
            return 'safe', num_evals
        elif tau < theta_l:
            return 'unsafe', num_evals
    
def custom_bisection_search(method, lo, hi, eval_fn, num_evals, epsilon_squared, mse_z_ghat_0, verbose, ci_protection_num_init_samples, 
    ci_protection_tau, ci_protection_delta, ci_protection_tol, tol=0, log_root='ci_logs'):
    exper_str = f'{method}_epssq={epsilon_squared}_numevals={num_alpha_search_iterations}_ciinit={ci_protection_num_init_samples}_cidelta={ci_protection_delta}_citau={ci_protection_tau}_citol={ci_protection_tol}'
    run_id = time.strftime("%Y%m%d_%H%M%S") + '_' + exper_str
    run_root = os.path.join(log_root + "/test12", run_id)
    print(run_root)
    os.makedirs(run_root, exist_ok=True)
    is_pos = hi > 0
    lo, hi, _, num_evals = initialize_bounds_b(lo, hi, eval_fn, is_pos, num_evals, epsilon_squared, mse_z_ghat_0, verbose)
    attempt = 0
    alpha_list = []
    protection_status_list = []
    evals_per_alpha = []
    while lo < hi - tol and num_evals > 0:
        if verbose:
            print(f'Evals remaining: {num_evals}, lo:{lo}, hi:{hi}')
        mid = (lo + hi) / 2
        alpha_dir = os.path.join(run_root, f"attempt_{attempt:03d}_alpha_{mid:+.6f}")
        alpha_list.append(mid)
        is_protected, num_evals, evals_used_for_ci = get_ci_protection_output_logged(mid, epsilon_squared, eval_fn, num_evals, 
            ci_protection_num_init_samples, ci_protection_tau, ci_protection_delta, ci_protection_tol, log_dir=alpha_dir)
        protection_status_list.append(is_protected)
        evals_per_alpha.append(evals_used_for_ci)
        attempt += 1
        if verbose:
            print(f'Protection status for alpha={mid}: {is_protected}\n')
        if is_protected == 'unsafe':
            lo = mid
        elif is_protected == 'safe':
            hi = mid
        elif is_protected == 'boundary':
            return mid, alpha_list, protection_status_list, evals_per_alpha, run_root
        else: # is_protected == 'inconclusive'
            assert num_evals <= 0
            return (hi if is_pos else lo), alpha_list, protection_status_list, evals_per_alpha, run_root
    return (hi if is_pos else lo), alpha_list, protection_status_list, evals_per_alpha, run_root



# Plotting

def cp_ci(v, n, delta):
    # two-sided Clopper–Pearson with edge cases
    if n == 0:
        raise ValueError("n must be >= 1")
    if v == 0:
        theta_l = 0.0
        theta_u = beta.ppf(1 - delta/2, 1, n)
    elif v == n:
        theta_l = beta.ppf(delta/2, n, 1)
        theta_u = 1.0
    else:
        theta_l = beta.ppf(delta/2, v, n - v + 1)
        theta_u = beta.ppf(1 - delta/2, v + 1, n - v)
    return float(theta_l), float(theta_u)

def plot_ci_step(out_path, alpha, tau, delta, eps2, noise_var, log_rows, step_idx):
    """
    log_rows: list of dicts with keys:
      n, v, theta_l, theta_u, mse (last), status (optional)
    """
    ns = [r["n"] for r in log_rows]
    L  = [r["theta_l"] for r in log_rows]
    U  = [r["theta_u"] for r in log_rows]
    v  = [r["v"] for r in log_rows]

    plt.figure()
    plt.plot(ns, L, marker="o", label=f"theta_L")
    plt.plot(ns, U, marker="o", label=f"theta_U")
    plt.axhline(tau, linestyle="--", label=f"tau ={tau} (target violation prob)")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("n (samples at this alpha)")
    plt.ylabel(r"CI bounds for $\theta = P(\mathrm{MSE} < \varepsilon^2)$")
    plt.title(f"alpha={alpha:.6g}  delta={delta}  last v/n={v[-1]}/{ns[-1]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"ci_step_{step_idx:03d}.png"))
    plt.close()

def get_ci_protection_output_logged(
    alpha, epsilon_squared, eval_fn, num_evals, num_init_samples,
    tau=0.1, delta=0.1, tol=1e-7,
    log_dir=None
):
    """
    Returns (status, num_evals_remaining).
    status in {"safe","unsafe","boundary","inconclusive"}.
    Also saves plots per sampling step if log_dir is provided.
    """
    if num_evals <= 0:
        return "inconclusive", num_evals, 0

    os.makedirs(log_dir, exist_ok=True) if log_dir else None

    # initial samples
    k0 = min(num_evals, num_init_samples)
    mses = [eval_fn(alpha)[0] for _ in range(k0)]
    num_evals -= k0

    log_rows = []
    v = int(np.sum(np.array(mses) < epsilon_squared))
    n = len(mses)
    theta_l, theta_u = cp_ci(v, n, delta)

    log_rows.append({"n": n, "v": v, "theta_l": theta_l, "theta_u": theta_u, "mse": float(mses[-1])})

    step_idx = k0
    # if log_dir:
    #     plot_ci_step(log_dir, alpha, tau, delta, epsilon_squared, None, log_rows, step_idx)
    #     with open(os.path.join(log_dir, "trace.jsonl"), "a") as f:
    #         f.write(json.dumps(log_rows[-1]) + "\n")

    # early decisive
    if tau > theta_u:
        plotit(log_dir, alpha, tau, delta, epsilon_squared, None, log_rows, step_idx)
        return "safe", num_evals, step_idx
    if tau < theta_l:
        plotit(log_dir, alpha, tau, delta, epsilon_squared, None, log_rows, step_idx)
        return "unsafe", num_evals, step_idx

    # sequential tightening
    while True:
        if (theta_u - theta_l) < tol:
            plotit(log_dir, alpha, tau, delta, epsilon_squared, None, log_rows, step_idx)
            return "boundary", num_evals, step_idx
        if num_evals <= 0:
            plotit(log_dir, alpha, tau, delta, epsilon_squared, None, log_rows, step_idx)
            return "inconclusive", num_evals, step_idx

        mses.append(eval_fn(alpha)[0])
        num_evals -= 1

        v = int(np.sum(np.array(mses) < epsilon_squared))
        n = len(mses)
        theta_l, theta_u = cp_ci(v, n, delta)

        step_idx += 1
        log_rows.append({"n": n, "v": v, "theta_l": theta_l, "theta_u": theta_u, "mse": float(mses[-1])})

        # if log_dir:
        #     plot_ci_step(log_dir, alpha, tau, delta, epsilon_squared, None, log_rows, step_idx)
        #     with open(os.path.join(log_dir, "trace.jsonl"), "a") as f:
        #         f.write(json.dumps(log_rows[-1]) + "\n")

        if tau > theta_u:
            plotit(log_dir, alpha, tau, delta, epsilon_squared, None, log_rows, step_idx)
            return "safe", num_evals, step_idx
        if tau < theta_l:
            plotit(log_dir, alpha, tau, delta, epsilon_squared, None, log_rows, step_idx)
            return "unsafe", num_evals, step_idx
        
def plotit(log_dir, alpha, tau, delta, epsilon_squared, noise_var, log_rows, step_idx):
    plot_ci_step(log_dir, alpha, tau, delta, epsilon_squared, None, log_rows, step_idx)
    with open(os.path.join(log_dir, "trace.jsonl"), "a") as f:
        f.write(json.dumps(log_rows[-1]) + "\n")

def do_the_plot(num_dummy, num_classes, gt_data, gt_label, dummy_label, history, history_iters, mses, alpha_star, result_path, imidx_list, method, tp, sample_num):
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
        plt.savefig('%s/DLG_on_%s_%05d_%03d.png' % (result_path, imidx_list, imidx_list[imidx], sample_num))
        plt.close()
    elif method == 'iDLG':
        plt.savefig('%s/iDLG_on_%s_%05d_%03d.png' % (result_path, imidx_list, imidx_list[imidx], sample_num))
        plt.close()

