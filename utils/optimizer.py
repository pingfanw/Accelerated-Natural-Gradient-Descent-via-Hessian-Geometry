import torch
import torch.optim as optim

class ZOAHessB(optim.Optimizer):
    r"""
    Zeroth-Order Accelerated Hessian Barrier (Algorithm 3, ACML 2025).
    Parameters

    ----------
    params : list [W, H]          factor matrices (torch.nn.Parameter)

    V : torch.Tensor              target matrix

    a_const : float               auxiliary step a_{t+1}

    beta_type : str               'nesterov' | 'constant'

    beta_const : float            β when beta_type='constant'

    h_init : float                initial smoothing parameter h₀

    rho : float                   h ← ρ·h, 0<ρ<1

    alpha_init : float            initial step size α₀

    armijo : bool                 use Armijo line search 

    armijo_max_iters              Armijo

    finite_difference : bool      Accurate Diff

    eps : float                   positivity guard

    device : torch.device
    """
    def __init__(self, params, V, *,
                 a_const=3e-5,
                 beta_type='nesterov', beta_const=0.9,
                 h_init=1e-2, rho=0.9,
                 alpha_init=1e-2,
                 armijo=False, armijo_beta=0.5,
                 armijo_c=1e-4, armijo_max_iters=5,
                 finite_difference=False,
                 device=None, eps=1e-12):
        if len(params) != 2:
            raise ValueError("ZOAHessB expects exactly two parameters [W, H].")
        defaults = dict(a_const=a_const, beta_type=beta_type,
                        beta_const=beta_const, h_init=h_init, rho=rho,
                        alpha_init=alpha_init, armijo=armijo,
                        armijo_beta=armijo_beta, armijo_c=armijo_c,
                        armijo_max_iters=armijo_max_iters,
                        finite_difference=finite_difference,
                        eps=eps)
        super().__init__(params, defaults)
        self.V = V.to(device)
        self.device = device
        # algorithm states
        W, H = self.param_groups[0]['params']
        state = self.state
        state['t'] = 0
        state['V_W'] = W.data.clone().detach()
        state['V_H'] = H.data.clone().detach()
        state['h']  = h_init
    def _loss(self, W, H):
        diff = W @ H - self.V
        return 0.5 * (diff * diff).sum()
    # coordinate-wise finite difference (optional, very slow)
    def _finite_diff_U(self, X, loss_fn, h):
        U = torch.zeros_like(X)
        with torch.no_grad():
            base = loss_fn()
            it = torch.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                X[idx] += h
                f_plus = loss_fn()          # f(x+he_i)
                U[idx] = (f_plus - base) / h
                X[idx] -= h                 # restore
                it.iternext()
        return U
    # analytical gradient equals zeroth-order limit
    def _analytic_U(self, Y_W, Y_H):
        diff = Y_W @ Y_H - self.V
        U_W = diff @ Y_H.t()
        U_H = Y_W.t() @ diff
        return U_W, U_H
    @torch.no_grad()
    def step(self, closure=None):
        params      = self.param_groups[0]['params']
        W, H        = params
        p           = self.param_groups[0]
        s           = self.state
        t           = s['t']
        beta_type   = p['beta_type']
        beta_const  = p['beta_const']
        a_const     = p['a_const']
        alpha       = p['alpha_init']
        h           = s['h']
        rho         = p['rho']
        eps         = p['eps']
        armijo      = p['armijo']
        # β_t
        beta_t = t / (t + 3.0) if beta_type == 'nesterov' else beta_const
        # y_t
        V_W, V_H = s['V_W'], s['V_H']
        Y_W = V_W + beta_t * (W.data - V_W)
        Y_H = V_H + beta_t * (H.data - V_H)
        # zeroth-order gradient U(h,y)
        if p['finite_difference']:
            # extremely expensive, use only on toy sizes
            def f(): return self._loss(Y_W, Y_H)
            U_W = self._finite_diff_U(Y_W, f, h)
            U_H = self._finite_diff_U(Y_H, f, h)
        else:                           # analytical, still zeroth-order limit
            U_W, U_H = self._analytic_U(Y_W, Y_H)
        # Hessian-inverse scaling: diag(x_i^2)
        D_W = (Y_W * Y_W) * U_W        # H^{-1}(Y_W)·U_W
        D_H = (Y_H * Y_H) * U_H
        # Armijo back-tracking on α if requested
        if armijo:
            f0    = self._loss(Y_W, Y_H)
            normD = (D_W.pow(2).sum() + D_H.pow(2).sum())
            for _ in range(p['armijo_max_iters']):
                W_new = (Y_W - alpha * D_W).clamp_min(eps)
                H_new = (Y_H - alpha * D_H).clamp_min(eps)
                f_new = self._loss(W_new, H_new)
                if f_new <= f0 - p['armijo_c'] * alpha * normD:
                    break
                alpha *= p['armijo_beta']
        # x_{t+1}
        W.data = (Y_W - alpha * D_W).clamp_min(eps)
        H.data = (Y_H - alpha * D_H).clamp_min(eps)
        # v_{t+1}
        s['V_W'] = (V_W - a_const * D_W).clamp_min(eps)
        s['V_H'] = (V_H - a_const * D_H).clamp_min(eps)
        # h_{t+1}
        s['h']  = rho * h
        s['t']  = t + 1
        return None

class ANGD(optim.Optimizer):
    def __init__(self, params, V, eta=1e-5, a_const=3e-5, beta_type='nesterov', beta_const=0.9,
                 grad_clip=None, eps=1e-12):
        if len(params) != 2:
            raise ValueError("ANGD_NMF requires exactly two parameters: [W, H]")
        defaults = {
            'eta': eta,
            'a_const': a_const,
            'beta_type': beta_type,
            'beta_const': beta_const,
            'grad_clip': grad_clip,
            'eps': eps,
        }
        super(ANGD, self).__init__(params, defaults)
        self.V = V
        self.state['t'] = 0
        W, H = self.param_groups[0]['params']
        self.state['V_W'] = W.data.clone().detach()
        self.state['V_H'] = H.data.clone().detach()
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one step of the ANGD_NMF update.
        W <- Y^W * exp(-η * G^W)
        H <- Y^H * exp(-η * G^H)
        V^W <- V^W * exp(-a_const * G^W)
        """
        params = self.param_groups[0]['params']
        W, H = params
        V = self.V
        state = self.state
        t = state['t']
        eta = self.param_groups[0]['eta']
        a_const = self.param_groups[0]['a_const']
        beta_type = self.param_groups[0]['beta_type']
        beta_const = self.param_groups[0]['beta_const']
        grad_clip = self.param_groups[0]['grad_clip']
        if beta_type == 'nesterov':
            beta_t = t / (t + 3.0)
        else:
            beta_t = beta_const
        V_W = state['V_W']
        V_H = state['V_H']
        Y_W = V_W + beta_t * (W.data - V_W)
        Y_H = V_H + beta_t * (H.data - V_H)
        WH = Y_W @ Y_H  # (m, n)
        diff = WH - V  # (m, n)
        G_W = diff @ Y_H.t()
        # G^H = Y_W.T @ diff  -> shape (r, n)
        G_H = Y_W.t() @ diff
        if grad_clip is not None:
            G_W = torch.clamp(G_W, -grad_clip, grad_clip)
            G_H = torch.clamp(G_H, -grad_clip, grad_clip)
        # exp_W = exp(-eta * G_W)，exp_VW = exp(-a_const * G_W)
        exp_W = torch.exp(-eta * G_W)
        exp_VW = torch.exp(-a_const * G_W)
        exp_H = torch.exp(-eta * G_H)
        exp_VH = torch.exp(-a_const * G_H)
        W.data = Y_W * exp_W
        H.data = Y_H * exp_H
        state['V_W'] = V_W * exp_VW
        state['V_H'] = V_H * exp_VH
        state['t'] = t + 1
        return None

class LeeSeung(optim.Optimizer):
    def __init__(self, params, V, eps=0.1):
        # params: list of torch.nn.Parameter [W, H]
        defaults = {'eps': eps}
        super(LeeSeung, self).__init__(params, defaults)
        self.V = V  # Data matrix V to factorize
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one step of multiplicative update for NMF as per Lee & Seung.
        W <- W * (V H^T) / (W H H^T)
        H <- H * (W^T V) / (W^T W H)
        """
        # Extract parameters W and H from the single parameter group
        params = self.param_groups[0]['params']
        if len(params) != 2:
            raise ValueError("LeeSeungNMF requires exactly two parameters: [W, H]")
        W, H = params
        eps = self.defaults['eps']
        # Multiplicative updates with stability epsilon
        W *= (self.V @ H.T) / ((W @ H) @ H.T+eps)
        H *= (W.T @ self.V) / ((W.T @ W) @ H+eps)
        return None