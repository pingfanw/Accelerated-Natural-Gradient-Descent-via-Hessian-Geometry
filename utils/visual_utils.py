import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from pathlib import Path

import math, torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

def plot_spectrum_comparision(original_imgs, recon_matrix,
                       title='', label='', exp_type='',iter_num=''):
    """
    original_imgs : Tensor [n, C, H, W] 或 [n, H, W]
    recon_matrix  : Tensor [C·H·W, n]  or [n, C, H, W]
    """
    if isinstance(original_imgs, list):
        original_imgs = torch.stack(original_imgs)
    n = original_imgs.shape[0]
    m = int(torch.sqrt(torch.tensor(n)).item())
    assert m * m == n, "n_samples = m**2"
    if isinstance(original_imgs, list):
        original_imgs = torch.stack(original_imgs)
    n_samples = original_imgs.shape[0]
    m = int(math.isqrt(n_samples))
    assert m * m == n_samples, (
        f"n_samples={n_samples} make sure n_samples = m*m.")

    def tile_square(imgs):
        """
        imgs : [n, C, H, W]  or [n, H, W] (grey)
        out  : [H·m, W·m] (grey)  or  [H·m, W·m, C] (color)
        """
        if imgs.dim() == 3:                          # [n, H, W]
            H, W = imgs.shape[-2:]
            grid = imgs.view(m, m, H, W).permute(0, 2, 1, 3)
            return grid.reshape(m*H, m*W)            # [H·m, W·m]
        elif imgs.dim() == 4:                        # [n, C, H, W]
            C, H, W = imgs.shape[1:]
            if C == 1:
                imgs = imgs.squeeze(1)               # [n, H, W]
                return tile_square(imgs)          
            grid = imgs.view(m, m, C, H, W).permute(0, 3, 1, 4, 2)
            return grid.reshape(m*H, m*W, C)         # [H·m, W·m, C]
        else:
            raise ValueError("Unsupported img tensor shape.")
        
    def single_spectrum(img):
        # img: [C, H, W] or [H, W]
        if img.dim() == 3:
            spec = torch.fft.fft2(img, norm='ortho')
            mag  = torch.log1p(spec.abs()).mean(0)   # [H,W]
        else:
            spec = torch.fft.fft2(img, norm='ortho')
            mag  = torch.log1p(spec.abs())           # [H,W]
        return torch.fft.fftshift(mag, dim=(-2, -1))
    if recon_matrix.dim() == 2:
        C = 1 if original_imgs.dim() == 3 else 3
        H, W = original_imgs.shape[-2:]
        recon_imgs = recon_matrix.T.view(n, C, H, W)
    else:
        recon_imgs = recon_matrix
    recon_imgs = recon_imgs.to(original_imgs.device)
    orig_specs  = torch.stack([single_spectrum(img) for img in original_imgs])
    recon_specs = torch.stack([single_spectrum(img) for img in recon_imgs])
    orig_big  = tile_square(orig_specs)          # [mH, mW]
    recon_big = tile_square(recon_specs)
    Path('images').mkdir(exist_ok=True)
    for tag, big in zip(['original', 'recover'], [orig_big, recon_big]):
        plt.imsave(f'images/{tag}_spectrum_{label.lower()}_iter{iter_num}.png',
                   big.numpy(), cmap='binary_r')

def plot_mean_spectrum_comparison(original_imgs, recon_matrix,
                             title='', label='', exp_type='',iter_num=''):
    """
    original_imgs : Tensor [n, C, H, W] or [n, H, W]   (0-1)
    recon_matrix  : Tensor [C·H·W, n]  or [n, C, H, W]
    """
    if isinstance(original_imgs, list):
        original_imgs = torch.stack(original_imgs)
    n = original_imgs.shape[0]

    def avg_spectrum(imgs):
        if imgs.dim() == 3:                
            imgs = imgs.unsqueeze(1)              # [n, 1, H, W]
        _, _, H, W = imgs.shape
        spec = torch.fft.fft2(imgs, norm='ortho')
        mag  = torch.log1p(spec.abs())            # [n, C, H, W]
        mag  = torch.fft.fftshift(mag, dim=(-2, -1))
        return mag.mean(dim=(0,1)).cpu()          # [H, W]
    if recon_matrix.dim() == 2:                   # (D, n)
        C = 1 if original_imgs.dim() == 3 else 3
        H, W = original_imgs.shape[-2:]
        recon_imgs = recon_matrix.T.view(n, C, H, W)
    else:
        recon_imgs = recon_matrix
    recon_imgs = recon_imgs.to(original_imgs.device)

    orig_spec  = avg_spectrum(original_imgs)
    recon_spec = avg_spectrum(recon_imgs)
    Path('images').mkdir(exist_ok=True)
    for tag, spec in zip(['original', 'recover'], [orig_spec, recon_spec]):
        plt.imsave(f'images/{tag}_mean_spectrum_{label.lower()}_iter{iter_num}.png',
                   spec.numpy(), cmap='binary_r')

def plot_comparison(original_imgs, recon_matrix, title='', label='', exp_type=''):
    """
    original_imgs : Tensor  [n_samples, 28, 28]
    recon_matrix  : Tensor  [784, n_samples]
    """
    if isinstance(original_imgs, list):
        original_imgs = torch.stack(original_imgs)
    n_samples = original_imgs.shape[0]
    m = int(math.isqrt(n_samples))
    assert m * m == n_samples, (
        f"n_samples={n_samples} make sure n_samples = m*m.")

    def tile_square(imgs):
        """
        imgs : [n, C, H, W]  or [n, H, W] (grey)
        out  : [H·m, W·m] (grey)  or  [H·m, W·m, C] (color)
        """
        if imgs.dim() == 3:                          # [n, H, W]
            H, W = imgs.shape[-2:]
            grid = imgs.view(m, m, H, W).permute(0, 2, 1, 3)
            return grid.reshape(m*H, m*W)            # [H·m, W·m]
        elif imgs.dim() == 4:                        # [n, C, H, W]
            C, H, W = imgs.shape[1:]
            if C == 1:
                imgs = imgs.squeeze(1)               # [n, H, W]
                return tile_square(imgs)          
            grid = imgs.view(m, m, C, H, W).permute(0, 3, 1, 4, 2)
            return grid.reshape(m*H, m*W, C)         # [H·m, W·m, C]
        else:
            raise ValueError("Unsupported img tensor shape.")
        
    orig_big   = tile_square(original_imgs.cpu())
    if recon_matrix.dim() == 2:           # (D, n)
        C, H, W = (1, 28, 28) if orig_big.ndim == 2 else (3, 32, 32)
        recon_imgs = recon_matrix.T.view(-1, C, H, W).cpu()
    else:
        recon_imgs = recon_matrix.cpu()
    recon_big  = tile_square(recon_imgs)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=300)
    for ax, big in zip(axes, [orig_big, recon_big]):
        if big.ndim == 2:
           ax.imshow(big, cmap='gray', interpolation='nearest')
        else:
           ax.imshow(big, interpolation='nearest')
        ax.axis('off')
    fig.subplots_adjust(wspace=0.1, hspace=0, left=0, right=1,
                        top=1, bottom=0)
    # plt.show()       
    plt.close(fig)
    Path('images').mkdir(exist_ok=True)

    def to_uint8(arr):
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255 + 0.5).astype(np.uint8)

    orig_np  = to_uint8(orig_big.numpy())
    recon_np = to_uint8(recon_big.numpy())
    kwargs = dict(cmap='gray') if orig_np.ndim == 2 else {}
 
    plt.imsave(f'images/original_{exp_type}_{label.lower()}.png',
               orig_np,  **kwargs)
    plt.imsave(f'images/recover_{exp_type}_{label.lower()}.png',
               recon_np, **kwargs)


# -------- Ackley -------- #
a, b, c = 20.0, 0.2, 2 * np.pi
d = 2

def _tri(z):
    return (2.0 / np.pi) * np.arcsin(np.sin(z))

def multtri(x):
    """x shape (...,2)"""
    x = np.asarray(x)
    return _tri(x[..., 0]) * _tri(x[..., 1])

def multtri_grad(x):
     """
     x : ndarray [..., 2]
     """
     x = np.asarray(x)
     gx = (2.0/np.pi) * np.sign(np.cos(x[..., 0])) * _tri(x[..., 1])
     gy = (2.0/np.pi) * np.sign(np.cos(x[..., 1])) * _tri(x[..., 0])
     return np.stack([gx, gy], axis=-1)

def ackley(x):
    x = np.asarray(x)
    sum_sq = np.sum(x ** 2, axis=-1)
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(np.sum(np.cos(c * x), axis=-1) / d)
    return term1 + term2 + a + np.e

def ackley_grad(x):
    x = np.asarray(x)
    sum_sq = np.sum(x ** 2, axis=-1, keepdims=True)
    r = np.sqrt(sum_sq / d)
    r_safe = np.where(r == 0, 1e-12, r)
    term1 = (a * b / d) * np.exp(-b * r) * x / r_safe
    q = np.sum(np.cos(c * x), axis=-1, keepdims=True) / d
    term2 = (c / d) * np.exp(q) * np.sin(c * x)
    return term1 + term2

def exp_angd_numpy(x0, beta_t, beta_t_type: str, eta=0.1,
                   a_t_ratio=1.0, n_iter=150,
                   grad_fun=ackley_grad):
    x, v = x0.copy(), x0.copy()
    path = [x.copy()]
    for t in range(n_iter):
        beta = t / (t + 3.0) if beta_t_type == "nestrov" else beta_t
        y = v + beta * (x - v)
        grad = grad_fun(y)
        nat_grad = y * grad
        x = y * np.exp(-eta * nat_grad)
        v = v * np.exp(-(a_t_ratio * eta) * nat_grad)
        path.append(x.copy())
    return np.array(path)

# ---------------- Zeroth-Order Accelerated Hessian Barrier (ZO-AHessB) ------------ #
def fd_grad_est(fun, x, h):
    """Forward-difference zeroth-order gradient estimator U(h,x)."""
    x = np.asarray(x, dtype=float)
    d = x.size
    grads = np.empty_like(x)
    for i in range(d):
        ei = np.zeros_like(x); ei[i] = 1.0
        grads[i] = (fun(x + h*ei) - fun(x)) / h
    return grads

def zoahessb_numpy(x0,               
                   beta_t,         
                   beta_t_type:str,
                   eta=1e-5,       
                   a_t_ratio=3.0,  
                   h0=1e-3,        
                   rho=0.9,        
                   n_iter=150,
                   fun=ackley):
    """
    Zeroth-Order Accelerated Hessian Barrier (positive-orthant metric) for Ackley.
    Returns the trajectory array of shape (n_iter+1, d).
    """
    x  = x0.astype(float).copy()
    v  = x.copy()         # v_0 = x_0
    h  = h0
    path = [x.copy()]
    for t in range(n_iter):
        # ----- momentum coefficient β_t -----
        beta = t/(t+3.0) if beta_t_type.lower()=="nestrov" else beta_t
        # ----- interpolation point -----
        y = v + beta*(x - v)
        # Zeroth-order gradient estimate U(h,y)
        U = fd_grad_est(fun, y, h)
        # H^{-1}(y)  = diag(y_i^2)  (positive-orthant Hessian metric)
        direction = (y**2) * U          # element-wise
        # ----- updates -----
        x = y - eta*direction
        v = v - (a_t_ratio*eta)*direction
        # ----- record & schedule -----
        path.append(x.copy())
        h *= rho        # h_{t+1} = ρ h_t
    return np.array(path)

def plot_trajectories(optim,save_path,objective=''):
    # ---------------- Run paths ---------------- #
    optim_name = optim.lower()
    if optim_name not in ["angd", "zoahessb"]:
        raise ValueError("Unsupported optimization method. Use 'ANGD' or 'ZOAHessB'.")
    np.random.seed(0)
    if objective.lower() == "ackley":
        x0 = np.array([-1.5, 1.0])
        target_fun, target_grad = ackley, ackley_grad
        fun_label = "Ackley Function"
        if optim_name == "angd":
            print(f"Using ANGD on {fun_label}, x0 =", x0)
            optim_label = ["ANGD","ANGD,β=0.6","NGD"]
            path_nest   = exp_angd_numpy(x0, 0.0, "nestrov",
                                         0.1, 0.9, 200, grad_fun=target_grad)
            path_nomom  = exp_angd_numpy(x0, 0.0, "constant",
                                         0.1, 0.9, 200, grad_fun=target_grad)
            path_const06= exp_angd_numpy(x0, 0.6, "constant",
                                         0.1, 0.9, 200, grad_fun=target_grad)
        elif optim_name == "zoahessb":
            print(f"Using ZO-AHessB on {fun_label}, x0 =", x0)
            optim_label = ["ZO-AHessB", "ZO-AHessB,β=0.6", "ZO-HessB"]
            path_nest   = zoahessb_numpy(x0, 0.0, "nestrov", eta=1e-1, a_t_ratio=3.0,
                                         h0=1e-1, rho=0.9, n_iter=200, fun=target_fun)
            path_const06= zoahessb_numpy(x0, 0.6, "constant", eta=1e-1, a_t_ratio=3.0,
                                         h0=1e-1, rho=0.9, n_iter=200, fun=target_fun)
            # Non-accelerated baseline: set a_t_ratio = 0 ➔ v_t
            path_nomom  = zoahessb_numpy(x0, 0.0, "constant", eta=1e-1, a_t_ratio=0.0,
                                         h0=1e-1, rho=0.9, n_iter=200, fun=target_fun)
        # ---------------- Main contour ---------------- #
        xmin, xmax, ymin, ymax = -2.5, 0.5, -0.5, 1.5
        xs = np.linspace(xmin, xmax, 400)
        ys = np.linspace(ymin, ymax, 400)
        X, Y = np.meshgrid(xs, ys)
        Z = target_fun(np.stack([X, Y], axis=-1))
        levels_main = np.linspace(Z.min(), Z.max(), 30)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.contour(X, Y, Z, levels=levels_main, linewidths=0.5,cmap='viridis')
        ax.contourf(X, Y, Z, levels=levels_main, alpha=0.3,cmap='viridis')
        # Trajectories
        ax.plot(path_nest[:, 0], path_nest[:, 1], marker='o', markersize=2.0, linewidth=0.8, label=optim_label[0])
        ax.plot(path_const06[:, 0], path_const06[:, 1], marker='^', markersize=2.0, linewidth=0.8, label=optim_label[1])
        ax.plot(path_nomom[:, 0], path_nomom[:, 1], marker='s', markersize=2.0, linewidth=0.8, label=optim_label[2])
        # Markers
        ax.scatter([x0[0]], [x0[1]], edgecolors='black', color='white', label='Start', marker='*', linewidths=1.0)
        ax.scatter(path_nest[-1, 0], path_nest[-1, 1], marker='o', edgecolors='black', color='white', label='End('+optim_label[0]+')', linewidths=1.0)
        ax.scatter(path_const06[-1, 0], path_const06[-1, 1], marker='^', edgecolors='black', color='white', label='End('+optim_label[1]+')', linewidths=1.0)
        ax.scatter(path_nomom[-1, 0], path_nomom[-1, 1], marker='s', edgecolors='black', color='white', label='End('+optim_label[2]+')', linewidths=1.0)
        ax.set_title(f"{optim_label[0]} Trajectories on {fun_label}")
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        if optim_name == "angd":
            ax.legend(
              fontsize=10, title_fontsize=10,
              markerscale=1.4, handlelength=1.9,
              handletextpad=0.9, borderpad=0.9,
              labelspacing=0.5,
              loc='upper left')
        elif optim_name == "zoahessb":
            ax.legend(
              fontsize=10, title_fontsize=10,
              markerscale=1.4, handlelength=1.9,
              handletextpad=0.9, borderpad=0.9,
              labelspacing=0.5,
              loc='upper right')
        # -------- Inset (dense) -------- #
        zoom_xlim = (-0.05, 0.02)
        zoom_ylim = (-0.02, 0.05)
    
        axins = inset_axes(ax, width="45%", height="45%", loc='lower left', borderpad=1.5)
        xin = np.linspace(*zoom_xlim, 300)
        yin = np.linspace(*zoom_ylim, 300)
        Xin, Yin = np.meshgrid(xin, yin)
        Zin = target_fun(np.stack([Xin, Yin], axis=-1))
        levels_inset = np.linspace(Zin.min(), Zin.max(), 15)
        cs_in = axins.contour(
            Xin, Yin, Zin,
            levels=levels_inset,
            linewidths=0.4,
            cmap='viridis'
        )
        axins.contourf(
            Xin, Yin, Zin,
            levels=levels_inset,
            alpha=0.3,         
            cmap='viridis'
        )
        axins.clabel(
            cs_in,              
            levels=levels_inset,
            inline=True,
            fmt='%.2f',         
            fontsize=4          
        )
        # Trajectories inset
        axins.plot(path_nest[:, 0], path_nest[:, 1], marker='o', markersize=1.5, linewidth=0.8)
        axins.plot(path_const06[:, 0], path_const06[:, 1], marker='^', markersize=1.5, linewidth=0.8)
        axins.plot(path_nomom[:, 0], path_nomom[:, 1], marker='s', markersize=1.5, linewidth=0.8)
        axins.scatter(path_nest[-1, 0], path_nest[-1, 1], marker='o', color='white', edgecolors='black', linewidths=0.8)
        axins.scatter(path_const06[-1, 0], path_const06[-1, 1], marker='^', color='white', edgecolors='black', linewidths=0.8)
        axins.scatter(path_nomom[-1, 0], path_nomom[-1, 1], marker='s', color='white', edgecolors='black', linewidths=0.8)
        axins.set_xlim(*zoom_xlim)
        axins.set_ylim(*zoom_ylim)
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor('grey')
            spine.set_linestyle('--')
            spine.set_linewidth(0.8)
        # Dashed rectangle
        rect = plt.Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1]-zoom_xlim[0],
                             zoom_ylim[1]-zoom_ylim[0], linewidth=0.8, edgecolor='gray',
                             facecolor='none', linestyle='--')
        ax.add_patch(rect)
        # Light grey dashed connectors
        mark_inset(ax, axins, loc1=1, loc2=4, fc="none",
                   ec='gray', linestyle='--', linewidth=0.8)
        plt.savefig(f"{save_path}/{optim_name}_{objective.lower()}_trajectories.png",
                    dpi=300, bbox_inches='tight')
        plt.show()
    elif objective.lower() == "multtri":
        x0 = np.array([1.8, 1.0])        
        target_fun, target_grad = multtri, multtri_grad
        fun_label = "Multiplied-Triangular Function"
        if optim_name == "angd":
            print(f"Using ANGD on {fun_label}, x0 =", x0)
            optim_label = ["ANGD","ANGD,β=0.6","NGD"]
            path_nest   = exp_angd_numpy(x0, 0.0, "nestrov",
                                         1e-2, 0.5, 200, grad_fun=target_grad)
            path_nomom  = exp_angd_numpy(x0, 0.0, "constant",
                                         1e-2, 0.5, 200, grad_fun=target_grad)
            path_const06= exp_angd_numpy(x0, 0.6, "constant",
                                         1e-2, 0.5, 200, grad_fun=target_grad)
        elif optim_name == "zoahessb":
            print(f"Using ZO-AHessB on {fun_label}, x0 =", x0)
            optim_label = ["ZO-AHessB", "ZO-AHessB,β=0.6", "ZO-HessB"]
            path_nest   = zoahessb_numpy(x0, 0.0, "nestrov", eta=1e-2, a_t_ratio=3.0,
                                         h0=1e-2, rho=0.6, n_iter=200, fun=target_fun)
            path_const06= zoahessb_numpy(x0, 0.6, "constant", eta=1e-2, a_t_ratio=3.0,
                                         h0=1e-2, rho=0.6, n_iter=200, fun=target_fun)
            # Non-accelerated baseline: set a_t_ratio = 0 ➔ v_t
            path_nomom  = zoahessb_numpy(x0, 0.0, "constant", eta=5e-1, a_t_ratio=0.0,
                                         h0=5e-1, rho=0.9, n_iter=200, fun=target_fun)
        # ---------------- Main contour ---------------- #
        xmin, xmax, ymin, ymax = -1.0, 6.5, -1.0, 6.5
        xs = np.linspace(xmin, xmax, 400)
        ys = np.linspace(ymin, ymax, 400)
        X, Y = np.meshgrid(xs, ys)
        Z = target_fun(np.stack([X, Y], axis=-1))
        Z = gaussian_filter(Z, sigma=10.0)
        levels_main = np.linspace(Z.min(), Z.max(), 30)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.contour(X, Y, Z, levels=levels_main, linewidths=0.5,cmap='viridis')
        ax.contourf(X, Y, Z, levels=levels_main, alpha=0.3,cmap='viridis')
        # Trajectories
        ax.plot(path_nest[:, 0], path_nest[:, 1], marker='o', markersize=2.0, linewidth=0.8, label=optim_label[0])
        ax.plot(path_const06[:, 0], path_const06[:, 1], marker='^', markersize=2.0, linewidth=0.8, label=optim_label[1])
        ax.plot(path_nomom[:, 0], path_nomom[:, 1], marker='s', markersize=2.0, linewidth=0.8, label=optim_label[2])
        # Markers
        ax.scatter([x0[0]], [x0[1]], edgecolors='black', color='white', label='Start', marker='*', linewidths=1.0)
        ax.scatter(path_nest[-1, 0], path_nest[-1, 1], marker='o', edgecolors='black', color='white', label='End('+optim_label[0]+')', linewidths=1.0)
        ax.scatter(path_const06[-1, 0], path_const06[-1, 1], marker='^', edgecolors='black', color='white', label='End('+optim_label[1]+')', linewidths=1.0)
        ax.scatter(path_nomom[-1, 0], path_nomom[-1, 1], marker='s', edgecolors='black', color='white', label='End('+optim_label[2]+')', linewidths=1.0)
        ax.set_title(f"{optim_label[0]} Trajectories on {fun_label}")
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        if optim_name == "angd":
            ax.legend(
              fontsize=10, title_fontsize=10,
              markerscale=1.4, handlelength=1.9,
              handletextpad=0.9, borderpad=0.9,
              labelspacing=0.5,
              loc='upper left')
            zoom_xlim = (4.5, 5)
            zoom_ylim = (0.8, 1.6)
        elif optim_name == "zoahessb":
            ax.legend(
              fontsize=10, title_fontsize=10,
              markerscale=1.4, handlelength=1.9,
              handletextpad=0.9, borderpad=0.9,
              labelspacing=0.5,
              loc='upper right')
            zoom_xlim = (4, 5)
            zoom_ylim = (0.9, 1.9)
        # -------- Inset (dense) -------- #

        axins = inset_axes(ax, width="30%", height="30%", loc='lower left', borderpad=1.5)
        xin = np.linspace(*zoom_xlim, 300)
        yin = np.linspace(*zoom_ylim, 300)
        Xin, Yin = np.meshgrid(xin, yin)
        Zin = target_fun(np.stack([Xin, Yin], axis=-1))
        Zin = gaussian_filter(Zin, sigma=20.0)
        levels_inset = np.linspace(-1.0, 1.0, 15)
        cs_in = axins.contour(
            Xin, Yin, Zin,
            levels=levels_inset,
            linewidths=0.4,
            cmap='viridis'
        )
        axins.contourf(
            Xin, Yin, Zin,
            levels=levels_inset,
            alpha=0.3,         
            cmap='viridis'
        )
        axins.clabel(
            cs_in,              
            levels=levels_inset,
            inline=True,
            fmt='%.2f',         
            fontsize=4          
        )
        # Trajectories inset
        axins.plot(path_nest[:, 0], path_nest[:, 1], marker='o', markersize=1.5, linewidth=0.8)
        axins.plot(path_const06[:, 0], path_const06[:, 1], marker='^', markersize=1.5, linewidth=0.8)
        axins.plot(path_nomom[:, 0], path_nomom[:, 1], marker='s', markersize=1.5, linewidth=0.8)
        axins.scatter(path_nest[-1, 0], path_nest[-1, 1], marker='o', color='white', edgecolors='black', linewidths=0.8)
        axins.scatter(path_const06[-1, 0], path_const06[-1, 1], marker='^', color='white', edgecolors='black', linewidths=0.8)
        axins.scatter(path_nomom[-1, 0], path_nomom[-1, 1], marker='s', color='white', edgecolors='black', linewidths=0.8)
        axins.set_xlim(*zoom_xlim)
        axins.set_ylim(*zoom_ylim)
        axins.set_xticks([])
        axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor('grey')
            spine.set_linestyle('--')
            spine.set_linewidth(0.8)
        # Dashed rectangle
        rect = plt.Rectangle((zoom_xlim[0], zoom_ylim[0]), zoom_xlim[1]-zoom_xlim[0],
                             zoom_ylim[1]-zoom_ylim[0], linewidth=0.8, edgecolor='gray',
                             facecolor='none', linestyle='--')
        ax.add_patch(rect)
        # Light grey dashed connectors
        mark_inset(ax, axins, loc1=1, loc2=4, fc="none",
                   ec='gray', linestyle='--', linewidth=0.8)
        plt.savefig(f"{save_path}/{optim_name}_{objective.lower()}_trajectories.png",
                    dpi=300, bbox_inches='tight')
        plt.show()
    else:
        raise ValueError("objective must be 'ackley' or 'multtri'")

def plot_loss_curve(log_path: str, optimizer_name, save_path: str = None, dim=None, legend_title=None):
    """
    Reads CSV log files and plots loss curves for two optimizers with a zoomed-in inset.

    Parameters:
        log_path (str): Directory containing the log file.
        optimizer_name (tuple[str, str]): Tuple of optimizer names.
        save_path (str): Path to save the generated plot. If None, plot will be shown instead.
        dim (str): Dimension identifier to locate the correct CSV files.
    """
    save_path ='./images/loss_curve_dim'+dim+'.png'

    # Construct file paths
    lee_name = optimizer_name[0].lower()
    angd_name = optimizer_name[1].lower()
    ngd_name = optimizer_name[2].lower()
    zoahessb_name = optimizer_name[3].lower()
    lee_csv_file = os.path.join(log_path, 'error', lee_name, f'{lee_name}_loss_dim{dim}.csv')
    angd_csv_file = os.path.join(log_path, 'error', angd_name, f'{angd_name}_loss_dim{dim}.csv')
    ngd_csv_file = os.path.join(log_path, 'error', ngd_name, f'{ngd_name}_loss_dim{dim}.csv')
    zoahessb_csv_file = os.path.join(log_path, 'error', zoahessb_name, f'{zoahessb_name}_loss_dim{dim}.csv')
    # Check existence
    if not os.path.exists(lee_csv_file):
        raise FileNotFoundError(f"CSV file not found at: {lee_csv_file}")
    if not os.path.exists(angd_csv_file):
        raise FileNotFoundError(f"CSV file not found at: {angd_csv_file}")
    if not os.path.exists(ngd_csv_file):
        raise FileNotFoundError(f"CSV file not found at: {ngd_csv_file}")
    if not os.path.exists(zoahessb_csv_file):
        raise FileNotFoundError(f"CSV file not found at: {zoahessb_csv_file}")
    # Load data
    df_lee = pd.read_csv(lee_csv_file)
    df_angd = pd.read_csv(angd_csv_file)
    df_ngd = pd.read_csv(ngd_csv_file)
    df_zoahessb = pd.read_csv(zoahessb_csv_file)
    # Plot settings
    sns.set(style="whitegrid", palette="deep", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6),dpi=300)
    # Plot full loss curves
    sns.lineplot(x='Iteration', y='Loss', data=df_lee, ax=ax, label=optimizer_name[0], linewidth=1.0,color="#6CA6CD",markers="--")
    sns.lineplot(x='Iteration', y='Loss', data=df_angd, ax=ax, label=optimizer_name[1], linewidth=1.0,color="#FA6E6E",markers="--")
    sns.lineplot(x='Iteration', y='Loss', data=df_ngd, ax=ax, label=optimizer_name[2], linewidth=1.0,color="#39D264",markers="--")
    sns.lineplot(x='Iteration', y='Loss', data=df_zoahessb, ax=ax, label=optimizer_name[3], linewidth=1.0,color="#A84ED1",markers="--")
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Loss (Frobenius Norm)", fontsize=14)
    ax.legend(title=legend_title, fontsize=12, title_fontsize=13, loc='upper right')
    inset_ax = inset_axes(ax, width="45%", height="40%", loc='center', borderpad=3)
    # Define zoom range
    max_iter = min(df_lee['Iteration'].max(), df_angd['Iteration'].max())
    zoom_start = max_iter - 1000
    # Subset data
    df_lee_zoom = df_lee[df_lee['Iteration'] >= zoom_start]
    df_angd_zoom = df_angd[df_angd['Iteration'] >= zoom_start]
    df_ngd_zoom = df_ngd[df_ngd['Iteration'] >= zoom_start]
    df_zoahessb_zoom = df_zoahessb[df_zoahessb['Iteration'] >= zoom_start]
    # Plot zoomed-in curves
    sns.lineplot(x='Iteration', y='Loss', data=df_lee_zoom, ax=inset_ax, label=None, linewidth=1.2,color="#6CA6CD",markers="--")
    sns.lineplot(x='Iteration', y='Loss', data=df_angd_zoom, ax=inset_ax, label=None, linewidth=1.2,color="#FA6E6E",markers="--")
    sns.lineplot(x='Iteration', y='Loss', data=df_ngd_zoom, ax=inset_ax, label=None, linewidth=1.2,color="#39D264")
    sns.lineplot(x='Iteration', y='Loss', data=df_zoahessb_zoom, ax=inset_ax, label=None, linewidth=1.2,color="#A84ED1",markers="--")
    inset_ax.set_title("Last 1000 Iterations", fontsize=10)
    inset_ax.tick_params(axis='both', labelsize=8)
    inset_ax.set_xlabel("")
    inset_ax.set_ylabel("")
    for spine in inset_ax.spines.values():
        spine.set_edgecolor('darkgrey')
        spine.set_linestyle('--')
        spine.set_linewidth(1.0)
    # Optional: draw lines linking inset to main plot
    mark_inset(ax, inset_ax, loc1=3, loc2=4, fc="none", ec='darkgrey', linestyle="--")
    # Save & show
    # plt.tight_layout()
    plt.savefig(save_path, dpi=300,bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()