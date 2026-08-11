import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import proto_scales.ssm_model.scales_ssm as scales_ssm
import os
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import math

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def sinh_arcsinh_flow_nll_conditional(y, mu, log_sigma, eps_skew, log_delta, eps=1e-6):
    """
    Conditional elementwise Sinh–Arcsinh flow emission.

    y:         [B, Dy] observed (real-valued, can be negative)
    mu:        [B, Dy] base normal mean
    log_sigma: [B, Dy] base normal log-scale
    eps_skew:  [B, Dy] skew parameter ε_t
    log_delta: [B, Dy] tail parameter (unconstrained); delta = softplus(log_delta)+eps

    returns:   [B] negative log-likelihood summed over Dy
    """
    sigma = torch.exp(torch.clamp(log_sigma, -8.0, 6.0)) + eps
    delta = F.softplus(log_delta) + eps

    # Inverse transform: x = sinh(delta * asinh(y) - eps_skew)
    a = torch.asinh(y)
    t = delta * a - eps_skew
    x = torch.sinh(t)

    # log|dx/dy| for inverse transform:
    # dx/dy = cosh(t) * delta / sqrt(1 + y^2)
    log_abs_det = torch.log(torch.cosh(t) + eps) + torch.log(delta) - 0.5 * torch.log1p(y * y)

    # base log-prob Normal(mu, sigma) evaluated at x
    r = (x - mu) / sigma
    logp_base = -0.5 * (r * r) - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)

    # change of variables: log p(y) = log p_base(x) + log|dx/dy|
    logp = logp_base + log_abs_det
    return (-logp).sum(dim=-1)


class UnifiedWindowDataset(Dataset):
    """
    Works for:
      - one long series: y [T, Dy], pr [T, Dy], u [T, Du]
      - many series:     y [N, T, Dy],pr [N, T, Dy],  u [N, T, Du]

    Each dataset item corresponds to a window defined by (series_idx, start).
    It returns:
      y_ctx: [Tc, Dy]
      u_ctx: [Tc, Du]
      u_fut: [H,  Du]
      y_fut: [H,  Dy]
    """
    def __init__(self, y, pr, u, context_len=40, horizon=12, stride=1, start_mode="all"):
        """
        stride: step between consecutive window starts (reduces overlap if >1)
        start_mode:
          - "all": generate all valid starts for each series...like sliding window
          - "zero": only start at 0 for each series (behaves like your WindowDataset)
        """
        y = np.asarray(y, dtype=np.float32)
        pr = np.asarray(pr, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)

        # Normalize shapes to [N, T, D]
        if y.ndim == 2:
            y = y[None, ...]  # [1, T, Dy]
        if pr.ndim ==2:
            pr = pr[None,...] # [1,T, Dy]
        if u.ndim == 2:
            u = u[None, ...]  # [1, T, Du]

        assert y.shape[0] == u.shape[0] and y.shape[1] == u.shape[1]
        assert pr.shape[0]==u.shape[0] and pr.shape[1] == u.shape[1]
        assert pr.shape[2] ==y.shape[2]
        self.y = y
        self.u = u
        self.pr = pr
        self.N, self.T, self.Dy = y.shape
        self.Du = u.shape[2]
        self.Tc = int(context_len)
        self.H = int(horizon)
        self.stride = int(stride)

        if self.Tc + self.H > self.T:
            raise ValueError("context_len + horizon must be <= T")

        # Build an index map: each item maps to (series_idx, start)
        self.index = []
        max_start = self.T - (self.Tc + self.H)
        for s in range(self.N):
            if start_mode == "zero":
                self.index.append((s, 0))
            elif start_mode == "all":
                for start in range(0, max_start + 1, self.stride):
                    self.index.append((s, start))
            else:
                raise ValueError("start_mode must be 'all' or 'zero'")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s, start = self.index[idx]
        Tc, H = self.Tc, self.H

        y = self.y[s]  # [T, Dy]
        u = self.u[s]  # [T, Du]
        pr = self.pr[s]


        y_ctx = y[start : start + Tc]
        u_ctx = u[start : start + Tc]
        pr_ctx = pr[start : start + Tc]
        u_fut = u[start + Tc : start + Tc + H]
        y_fut = y[start + Tc : start + Tc + H]
        pr_fut = pr[start + Tc : start + Tc + H]

        return y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut

def sinh_arcsinh_forward(x, eps_skew, log_delta, eps=1e-6):
    delta = F.softplus(log_delta) + eps
    return torch.sinh((torch.asinh(x) + eps_skew) / delta)

class DeepSSMPatternConditioned(nn.Module):
    """
    q(z_t | y_{1:t}, u_{1:t}) via GRU on [y,u]
    p(z_t | z_{t-1}, u_t) via MLP([z_{t-1},u_t]) -> (mu, logvar)
    p(y_t | z_t) via MLP(z_t) -> (mu_y) with learned global sigma_y
    """
    def __init__(self, y_dim, u_dim, z_dim=16, rnn_hidden=62, u_rnn_hidden=64, mlp_hidden=128,
                 emission_uses_u=False, use_linear_model=True, reservoir_dim=2,
                 init_alpha=0.01, init_omega=1.0, alpha_max=0.02, cov_rank=5):
        super().__init__()
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.emission_uses_u = emission_uses_u
        self.use_linear_model = use_linear_model
        self.u_rnn_hidden = u_rnn_hidden
        self.reservoir_dim = reservoir_dim
        self.alpha_max = alpha_max
        self.cov_rank = cov_rank

        self.gru = nn.GRU(input_size=y_dim + u_dim, hidden_size=rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        self.trans = scales_ssm.MLP(z_dim + u_dim, 2 * z_dim, hidden=mlp_hidden)

        if emission_uses_u:
            self.u_gru = nn.GRU(input_size=u_dim, hidden_size=u_rnn_hidden, batch_first=True)
            # sigmoid(log_alpha) = alpha; initialise so alpha ≈ init_alpha
            log_alpha_init = math.log(init_alpha / (1.0 - init_alpha))
            self.log_alpha = nn.Parameter(torch.full((y_dim, reservoir_dim), log_alpha_init))
            # omega_lin projects uh_t [B, u_rnn_hidden] -> target [B, y_dim * reservoir_dim]
            self.omega_lin = nn.Linear(u_rnn_hidden, y_dim * reservoir_dim, bias=False)

        emit_in = z_dim + (u_rnn_hidden + y_dim * reservoir_dim if emission_uses_u else 0)
        # emit outputs: y_dim (mean) + y_dim (log cov_diag) + y_dim*cov_rank (cov_factor)
        self.emit = scales_ssm.MLP(emit_in, y_dim * (2 + cov_rank), hidden=mlp_hidden)

        self.emit_pr = scales_ssm.MLP(emit_in,4*y_dim,hidden = 230)

        #pattern scaling like head for emission
        if(self.use_linear_model):
            self.ctrl_lin = nn.Linear(u_dim, y_dim,bias=True)

        self.eps = 1e-6
        

    def sample(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def _reservoir_step(self, s, uh_t):
        """
        s:    [B, y_dim, reservoir_dim]
        uh_t: [B, u_rnn_hidden]  — hidden state of u_gru at time t
        returns s_next: [B, y_dim, reservoir_dim]
        s_{t+1,r} = s_{t,r} + alpha_r * (omega_lin(uh_t)_r - s_{t,r})
        """
        alpha = torch.sigmoid(self.log_alpha) * self.alpha_max   # [y_dim, reservoir_dim], in (0, alpha_max)
        target = self.omega_lin(uh_t).reshape(uh_t.shape[0], self.y_dim, self.reservoir_dim)  # [B, y_dim, reservoir_dim]
        return s + alpha[None] * (target - s)

    def _parse_emit(self, emit_out, B):
        """Split emit output into (mean, cov_factor [B,D,r], cov_diag [B,D])."""
        D, r = self.y_dim, self.cov_rank
        res        = emit_out[:, :D]
        log_diag   = emit_out[:, D:2*D]
        factor_raw = emit_out[:, 2*D:]                      # [B, D*r]
        cov_diag   = F.softplus(log_diag) + self.eps        # [B, D], positive
        cov_factor = factor_raw.reshape(B, D, r)            # [B, D, r]
        return res, cov_factor, cov_diag

    def forward_elbo(self, y, pr,u, kl_free_bits=0.5):
        """
        y,u: [B, T, Dy/Du]
        Returns loss, stats.
        """
        B, T, _ = y.shape

        rnn_in = torch.cat([y, u], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h)
        mu_q, logvar_q = torch.chunk(q_params, 2, dim=-1)

        # stabilize logvar range...numbers not tuned
        logvar_q = torch.clamp(logvar_q, -12.0, 6.0)

        # prior for z0
        mu_p0 = torch.zeros(B, self.z_dim, device=y.device)
        logvar_p0 = torch.zeros(B, self.z_dim, device=y.device)

        nll = 0.0
        nll_pr = 0.0
        kl = 0.0

        if self.emission_uses_u:
            uh, _ = self.u_gru(u)  # [B, T, u_rnn_hidden]
            # initialise s at analytical equilibrium given first u step
            s = self.omega_lin(uh[:, 0]).reshape(B, self.y_dim, self.reservoir_dim).detach()

        z_prev = None
        for t in range(T):
            z_t = self.sample(mu_q[:, t], logvar_q[:, t])
            if(self.use_linear_model):
                ctrl = self.ctrl_lin(u[:, t])              # [B, y_dim]

            if self.emission_uses_u:
                e_in = torch.cat([z_t, uh[:, t], s.reshape(B, -1)], dim=-1)
            else:
                e_in = z_t

            emit_out = self.emit(e_in)
            res, cov_factor, cov_diag = self._parse_emit(emit_out, B)
            if self.use_linear_model:
                y_hat = ctrl + res
            else:
                y_hat = res

            dist_y = torch.distributions.LowRankMultivariateNormal(
                loc=y_hat, cov_factor=cov_factor, cov_diag=cov_diag)
            nll_t = -dist_y.log_prob(y[:, t])  # [B]
            nll = nll + nll_t
            
            # Predict pr emission + flow params
            out = self.emit_pr(e_in)  # [B, 4*Dy]
            mu_t, log_sigma_t, eps_skew_t, log_delta_t = torch.chunk(out, 4, dim=-1)

            # NLL under conditional flow emission
            nll_pr_t = sinh_arcsinh_flow_nll_conditional(
                pr[:, t], mu_t, log_sigma_t, eps_skew_t, log_delta_t, eps=self.eps
            )
            nll_pr = nll_pr + nll_pr_t

            

            if t == 0:
                kl_t = scales_ssm.diag_gaussian_kl(mu_q[:, 0], logvar_q[:, 0], mu_p0, logvar_p0)
            else:
                trans_in = torch.cat([z_prev, u[:, t]], dim=-1)
                mu_p, logvar_p = torch.chunk(self.trans(trans_in), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                kl_t = scales_ssm.diag_gaussian_kl(mu_q[:, t], logvar_q[:, t], mu_p, logvar_p)

            # free-bits: don't over-penalize small KL; helps avoid posterior collapse: from Claude
            # Applied per-sample
            kl = kl + torch.clamp(kl_t, min=kl_free_bits)

            if self.emission_uses_u:
                s = self._reservoir_step(s, uh[:, t])
            z_prev = z_t

        # per-batch means
        nll = nll.mean()
        nll_pr = nll_pr.mean()
        kl = kl.mean()
        return nll, kl, nll_pr

    @torch.no_grad()
    def forecast(self, y_ctx, u_ctx, u_fut, steps, n_samples=50):
        B, Tc, _ = y_ctx.shape

        rnn_in = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT = torch.clamp(logvar_qT, -12.0, 6.0)

        if self.emission_uses_u:
            uh_ctx, h_u = self.u_gru(u_ctx)  # uh_ctx: [B, Tc, u_rnn_hidden], h_u: [1, B, u_rnn_hidden]
            s_ctx = torch.zeros(B, self.y_dim, self.reservoir_dim, device=u_ctx.device)
            for k in range(u_ctx.shape[1]):
                s_ctx = self._reservoir_step(s_ctx, uh_ctx[:, k])

        ysamps = []
        ysamps_pr = []
        for _ in range(n_samples):
            z = self.sample(mu_qT, logvar_qT)
            h_u_s = h_u.clone() if self.emission_uses_u else None
            s = s_ctx.clone() if self.emission_uses_u else None

            preds = []
            preds_pr = []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(self.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = self.sample(mu_p, logvar_p)

                if(self.use_linear_model):
                    ctrl = self.ctrl_lin(u_t)

                if self.emission_uses_u:
                    uh_t, h_u_s = self.u_gru(u_t.unsqueeze(1), h_u_s)
                    uh_t = uh_t.squeeze(1)  # [B, u_rnn_hidden]
                    e_in = torch.cat([z, uh_t, s.reshape(B, -1)], dim=-1)
                    s = self._reservoir_step(s, uh_t)
                else:
                    e_in = z

                emit_out = self.emit(e_in)
                res, cov_factor, cov_diag = self._parse_emit(emit_out, B)
                if self.use_linear_model:
                    y_hat = ctrl + res
                else:
                    y_hat = res

                out = self.emit_pr(e_in)
                mu_t, log_sigma_t, eps_skew_t, log_delta_t = torch.chunk(out, 4, dim=-1)
                sigma_pr = torch.exp(log_sigma_t) + self.eps
                x_samp = mu_t + sigma_pr * torch.randn_like(mu_t)

                dist_y = torch.distributions.LowRankMultivariateNormal(
                    loc=y_hat, cov_factor=cov_factor, cov_diag=cov_diag)
                y_s = dist_y.sample()
                y_s_pr = sinh_arcsinh_forward(x_samp, eps_skew_t, log_delta_t, eps=self.eps)
                preds.append(y_s)
                preds_pr.append(y_s_pr)

            ysamps.append(torch.stack(preds, dim=1))  # [B,steps,Dy]
            ysamps_pr.append(torch.stack(preds_pr, dim=1)) # [B,steps,Dy]
        

        samp = torch.stack(ysamps, dim=0)  # [S,B,steps,Dy]
        samp_pr = torch.stack(ysamps_pr, dim=0)  # [S,B,steps,Dy]
        return samp.mean(0), samp.quantile(0.10, 0), samp.quantile(0.90, 0),samp_pr.mean(0),samp_pr.quantile(0.10, 0), samp_pr.quantile(0.90, 0)

    @torch.no_grad()
    def forecast_deterministic(self, y_ctx, u_ctx, u_fut, steps, n_samples=50):
        B, Tc, _ = y_ctx.shape

        rnn_in = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT = torch.clamp(logvar_qT, -12.0, 6.0)

        if self.emission_uses_u:
            uh_ctx, h_u = self.u_gru(u_ctx)  # uh_ctx: [B, Tc, u_rnn_hidden], h_u: [1, B, u_rnn_hidden]
            s_ctx = torch.zeros(B, self.y_dim, self.reservoir_dim, device=u_ctx.device)
            for k in range(u_ctx.shape[1]):
                s_ctx = self._reservoir_step(s_ctx, uh_ctx[:, k])

        ysamps = []
        ysamps_pr = []
        for _ in range(n_samples):
            z = self.sample(mu_qT, logvar_qT)
            h_u_s = h_u.clone() if self.emission_uses_u else None
            s = s_ctx.clone() if self.emission_uses_u else None

            preds = []
            preds_pr = []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(self.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = self.sample(mu_p, logvar_p)
                if(self.use_linear_model):
                    ctrl = self.ctrl_lin(u_t)
                if self.emission_uses_u:
                    uh_t, h_u_s = self.u_gru(u_t.unsqueeze(1), h_u_s)
                    uh_t = uh_t.squeeze(1)  # [B, u_rnn_hidden]
                    e_in = torch.cat([z, uh_t, s.reshape(B, -1)], dim=-1)
                    s = self._reservoir_step(s, uh_t)
                else:
                    e_in = z

                emit_out = self.emit(e_in)
                res = emit_out[:, :self.y_dim]  # only mean needed
                if self.use_linear_model:
                    y_hat = ctrl + res
                else:
                    y_hat = res

                out = self.emit_pr(e_in)
                mu_t, log_sigma_t, eps_skew_t, log_delta_t = torch.chunk(out, 4, dim=-1)

                y_s = y_hat
                y_s_pr = sinh_arcsinh_forward(mu_t, eps_skew_t, log_delta_t, eps=self.eps)
                preds.append(y_s)
                preds_pr.append(y_s_pr)

            ysamps.append(torch.stack(preds, dim=1))  # [B,steps,Dy]
            ysamps_pr.append(torch.stack(preds_pr, dim=1)) # [B,steps,Dy]

        samp = torch.stack(ysamps, dim=0)  # [S,B,steps,Dy]
        samp_pr = torch.stack(ysamps_pr, dim=0)  # [S,B,steps,Dy]
        return samp.mean(0), samp.quantile(0.10, 0), samp.quantile(0.90, 0),samp_pr.mean(0),samp_pr.quantile(0.10, 0), samp_pr.quantile(0.90, 0)


def batch_acf(x, max_lag, eps=1e-6):
    """
    Differentiable normalized ACF via FFT for lags 1..max_lag.
    x: [B, T, D]
    returns: [B, max_lag, D]  values in approximately [-1, 1]
    """
    B, T, D = x.shape
    x = x - x.mean(dim=1, keepdim=True)
    xp = F.pad(x, (0, 0, 0, T))                              # zero-pad to 2T, [B, 2T, D]
    Xf = torch.fft.rfft(xp, dim=1)                           # [B, T+1, D]
    acf = torch.fft.irfft(Xf * Xf.conj(), dim=1, n=2 * T)   # [B, 2T, D]
    acf0 = acf[:, 0:1, :].clamp(min=eps)                     # lag-0 = sum x^2
    return acf[:, 1:max_lag + 1, :] / acf0                   # [B, max_lag, D]


def fit_control_mahalanobis(u_train_norm, eps=1e-6):
    """
    u_train_norm: [N, T, Du] standardized using train-only scaler
    Returns mu [Du], inv_cov [Du, Du]
    """
    U = u_train_norm.reshape(-1, u_train_norm.shape[-1])  # [N*T, Du]
    mu = U.mean(axis=0)
    cov = np.cov(U.T) + eps * np.eye(U.shape[1])
    inv_cov = np.linalg.inv(cov)
    return mu.astype(np.float32), inv_cov.astype(np.float32)

def mahalanobis_score(u_fut_norm, mu, inv_cov):
    """
    u_fut_norm: [B, H, Du] standardized
    returns score: [B, H] (per-step)
    """
    diff = u_fut_norm - mu[None, None, :]
    # score[b,h] = diff^T inv_cov diff
    return np.einsum("bhd,dd,bhd->bh", diff, inv_cov, diff)

def pick_threshold_from_val(val_loader, mu, inv_cov, percentile=99.0):
    all_scores = []
    for _, _,_, u_fut,_, _ in val_loader:
        s = mahalanobis_score(u_fut.numpy(), mu, inv_cov)  # [B,H]
        all_scores.append(s.reshape(-1))
    all_scores = np.concatenate(all_scores)
    return float(np.percentile(all_scores, percentile))

import numpy as np
import torch

def fit_ridge_D(u_train, y_train, alpha=1e-2, fit_intercept=True):
    """
    Fit ridge regression: y ≈ u @ W + b

    u_train: np.ndarray [N, T, Du]  (preferably normalized)
    y_train: np.ndarray [N, T, Dy]  (preferably normalized)
    alpha: ridge strength (λ)
    Returns:
      W: [Du, Dy]
      b: [Dy] (zeros if fit_intercept=False)
    """
    U = u_train.reshape(-1, u_train.shape[-1])  # [N*T, Du]
    Y = y_train.reshape(-1, y_train.shape[-1])  # [N*T, Dy]

    if fit_intercept:
        # augment with ones column for bias
        ones = np.ones((U.shape[0], 1))#, dtype=U.dtype)
        X = np.concatenate([U, ones], axis=1)  # [M, Du+1]
    else:
        X = U  # [M, Du]

    # Ridge closed form: (X^T X + α I)^{-1} X^T Y
    XtX = X.T @ X
    I = np.eye(XtX.shape[0])#, dtype=X.dtype)
    if fit_intercept:
        # usually do NOT regularize bias
        I[-1, -1] = 0.0

    Wb = np.linalg.solve(XtX + alpha * I, X.T @ Y)  # [Du(+1), Dy]

    if fit_intercept:
        W = Wb[:-1, :]          # [Du, Dy]
        b = Wb[-1, :]           # [Dy]
    else:
        W = Wb
        b = np.zeros((Y.shape[1],), dtype=Y.dtype)

    return W.astype(np.float32), b.astype(np.float32)


def load_into_ctrl_lin(model, W, b, freeze=True):
    """
    model.ctrl_lin is nn.Linear(u_dim, y_dim, bias=True)
    W: [Du, Dy] in numpy
    b: [Dy] in numpy
    """
    assert hasattr(model, "ctrl_lin"), "Model must have ctrl_lin = nn.Linear(u_dim, y_dim)"
    Du, Dy = W.shape
    assert model.ctrl_lin.in_features == Du
    assert model.ctrl_lin.out_features == Dy

    with torch.no_grad():
        # PyTorch Linear expects weight shape [out_features, in_features] = [Dy, Du]
        model.ctrl_lin.weight.copy_(torch.from_numpy(W.T))
        model.ctrl_lin.bias.copy_(torch.from_numpy(b))

    if freeze:
        for p in model.ctrl_lin.parameters():
            p.requires_grad = False




# -------------------------
# Train / eval loop with early stopping
# -------------------------
def run_train(
    y_np, pr_np, u_np,
    context_len=40, horizon=12,
    batch_size=64,
    epochs=50,
    lr=2e-3,
    z_dim=16,
    rnn_hidden=62,
    use_linear_model = True,
    resevoir_dim = 2,
    alpha_max = 0.02,
    cov_rank = 5,
    run_dir = None,
    weights_file = None,
    acf_max_lag = 120,
    acf_weight = 5000.0,
):

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend)
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")

    # split
    N = y_np.shape[0]
    idx = np.random.permutation(N)
    n_train = int(0.8 * N)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    y_tr, pr_tr,u_tr = y_np[tr_idx], pr_np[tr_idx],u_np[tr_idx]
    y_va, pr_va,u_va = y_np[va_idx],pr_np[va_idx], u_np[va_idx]

    print("training start")

    # normalize (fit on train only)
    y_scaler = scales_ssm.StandardScaler().fit(y_tr)
    u_scaler = scales_ssm.StandardScaler().fit(u_tr)
    pr_scaler = scales_ssm.StandardScaler().fit(pr_tr)
    print("Created standard scaler")
    if run_dir is not None:
        y_scaler.save(os.path.join(run_dir, "y_scaler.out"))
        u_scaler.save(os.path.join(run_dir, "u_scaler.out"))
        pr_scaler.save(os.path.join(run_dir, "pr_scaler.out"))
    y_trn = y_scaler.transform(y_tr)
    pr_trn = pr_scaler.transform(pr_tr)
    y_van = y_scaler.transform(y_va)
    pr_van = pr_scaler.transform(pr_va)
    u_trn = u_scaler.transform(u_tr)
    u_van = u_scaler.transform(u_va)

    print("data standardised.")

    # Fit ridge on TRAIN only
    if(use_linear_model):
        W, b = fit_ridge_D(u_trn, y_trn, alpha=1e-2, fit_intercept=True)
        print("ridge regression completed")

    #mu_control, inv_cov_control = fit_control_mahalanobis(u_trn)

    train_ds = UnifiedWindowDataset(y_trn, pr_trn, u_trn, context_len=context_len, horizon=horizon)
    val_ds   = UnifiedWindowDataset(y_van, pr_van, u_van, context_len=context_len, horizon=horizon)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("dataloaders prepared")

    #tau_ood = pick_threshold_from_val(val_dl,mu_control,inv_cov_control)

    Dy = y_np.shape[-1]
    Du = u_np.shape[-1]
    raw_model = DeepSSMPatternConditioned(y_dim=Dy, u_dim=Du, z_dim=z_dim,rnn_hidden=rnn_hidden,
        use_linear_model=use_linear_model,emission_uses_u=True,reservoir_dim=resevoir_dim,
        alpha_max=alpha_max, cov_rank=cov_rank).to(device)
    if weights_file is not None:
        raw_model.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"Loaded weights from {weights_file}")
    # Copy into raw_model.ctrl_lin and freeze (recommended for fallback)
    if(use_linear_model):
        load_into_ctrl_lin(raw_model, W, b, freeze=True)
    ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if use_cuda else {}
    model = DDP(raw_model, **ddp_kwargs)

    #opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)

    best_val = float("inf")
    best_state = None
    patience, patience_left = 15, 15

    # KL annealing schedule: ramp from 0 -> 1 over first ~30% of training
    total_steps = epochs * len(train_dl)
    global_step = 0

    unfreeze_epoch = int(0.7 * epochs) + 1
    ctrl_lin_unfrozen = False

    for epoch in range(1, epochs + 1):
        # if(use_linear_model):
        #     if epoch == unfreeze_epoch and not ctrl_lin_unfrozen:
        #         for p in raw_model.ctrl_lin.parameters():
        #             p.requires_grad = True
        #         opt.add_param_group({"params": list(raw_model.ctrl_lin.parameters()), "lr": 2e-4})
        #         ctrl_lin_unfrozen = True
        #         print(f"Unfreezing ctrl_lin at epoch {epoch}")

        model.train()

        tr_loss = []

        for y_ctx, pr_ctx,u_ctx, u_fut, pr_fut,y_fut in train_dl:


            y_ctx = torch.tensor(y_ctx, device=device)
            pr_ctx = torch.tensor(pr_ctx, device=device)
            u_ctx = torch.tensor(u_ctx, device=device)
            u_fut = torch.tensor(u_fut, device=device)
            y_fut = torch.tensor(y_fut, device=device)
            pr_fut = torch.tensor(pr_fut, device=device)

            # We train on full (context+horizon) to teach dynamics across the boundary:
            y_full = torch.cat([y_ctx, y_fut], dim=1)
            pr_full = torch.cat([pr_ctx, pr_fut], dim=1)
            u_full = torch.cat([u_ctx, u_fut], dim=1)


            B, T, _ = y_full.shape

            nll, kl, nll_pr = raw_model.forward_elbo(y_full, pr_full, u_full, kl_free_bits=0.2)
            mean, _, _ ,mean_pr,_,_= raw_model.forecast_deterministic(y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
            roll_out_mse = F.huber_loss(mean, y_fut, delta=1.0)
            roll_out_mse_pr = F.huber_loss(mean_pr, pr_fut, delta=1.0)
            if(use_linear_model):
                lin_mean = raw_model.ctrl_lin(u_full.reshape(-1, Du)).reshape(B, T, Dy)
                lin_mse = ((lin_mean-y_full)**2).mean()

            # yearly average loss: penalise trend errors over complete 12-month blocks
            n_complete_years = horizon // 12
            if n_complete_years > 0:
                H_yr = n_complete_years * 12
                mean_yr = mean[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
                y_fut_yr = y_fut[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
                roll_out_mse_yearly = ((mean_yr - y_fut_yr) ** 2).mean()
            else:
                roll_out_mse_yearly = torch.tensor(0.0, device=device)

            # anneal KL weight and rollout weights
            global_step += 1
            frac = min(1.0, global_step / int(0.3 * total_steps))
            kl_w = 5
            alpha = 10000 * frac
            omega = 500   * frac
            gamma = 80000  # yearly trend loss weight — no warmup

            # ACF loss: match normalized autocorrelation of rollout vs ground truth
            # tas: compute on ctrl_lin residual to focus on internal variability
            # pr:  batch_acf removes temporal mean internally
            if acf_max_lag > 0 and horizon >= 2 * acf_max_lag:
                with torch.no_grad():
                    ctrl_fut = raw_model.ctrl_lin(u_fut.reshape(-1, Du)).reshape(B, horizon, Dy)
                acf_pred_tas = batch_acf(mean - ctrl_fut, acf_max_lag)
                acf_true_tas = batch_acf(y_fut - ctrl_fut, acf_max_lag)
                acf_pred_pr  = batch_acf(mean_pr, acf_max_lag)
                acf_true_pr  = batch_acf(pr_fut,  acf_max_lag)
                acf_loss = (
                    ((acf_pred_tas - acf_true_tas) ** 2).mean()
                    + ((acf_pred_pr - acf_true_pr) ** 2).mean()
                )
            else:
                acf_loss = torch.tensor(0.0, device=device)

            loss = nll + nll_pr + kl_w * kl + alpha*roll_out_mse + omega*roll_out_mse_pr + gamma*roll_out_mse_yearly + acf_weight * frac * acf_loss
            #loss = nll + nll_pr + kl_w * kl + alpha*roll_out_mse + gamma*roll_out_mse_yearly


            if(global_step%100==0):
                if(use_linear_model):
                    print("loss: ",nll.item(),kl_w,kl.item(),roll_out_mse.item(),lin_mse.item(),nll_pr.item(),roll_out_mse_pr.item(),roll_out_mse_yearly.item(),acf_loss.item())
                else:
                    print("loss: ",nll.item(),kl_w,kl.item(),roll_out_mse.item(),nll_pr.item(),roll_out_mse_pr.item(),roll_out_mse_yearly.item(),acf_loss.item())

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()


            tr_loss.append(loss.item())

        # validation: one-step objective + forecast MSE on horizon
        model.eval()

        va_loss = []
        va_mse = []

        with torch.no_grad():
            for y_ctx, pr_ctx, u_ctx, u_fut, pr_fut,y_fut in val_dl:

                y_ctx = torch.tensor(y_ctx, device=device)
                pr_ctx = torch.tensor(pr_ctx, device=device)
                u_ctx = torch.tensor(u_ctx, device=device)
                u_fut = torch.tensor(u_fut, device=device)
                y_fut = torch.tensor(y_fut, device=device)
                pr_fut = torch.tensor(pr_fut, device=device)

                y_full = torch.cat([y_ctx, y_fut], dim=1)
                u_full = torch.cat([u_ctx, u_fut], dim=1)
                pr_full = torch.cat([pr_ctx, pr_fut], dim=1)

                nll, kl, nll_pr = raw_model.forward_elbo(y_full, pr_full, u_full, kl_free_bits=0.2)
                mean, _, _,mean_pr,_,_ = raw_model.forecast(y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
                mse = ((mean - y_fut) ** 2).mean().item()
                mse_pr = ((mean_pr - pr_fut) ** 2).mean().item()
                loss = nll + nll_pr + kl_w * kl + alpha*mse + omega*mse_pr
                va_loss.append(loss.item())
                va_mse.append(mse)

        tr = float(np.mean(tr_loss))
        va = float(np.mean(va_loss))
        mse = float(np.mean(va_mse))
        print(f"epoch {epoch:03d} | train {tr:.4f} | val_elbo {va:.4f} | val_mse {mse:.4f}")
        if run_dir is not None and epoch % 10 == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, f"model_epoch{epoch:04d}.pt"))

        # early stopping on val_elbo
        if va < best_val - 1e-4:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            # if patience_left <= 0:
            #     print("Early stopping.")
            #     break

    if best_state is not None:
        raw_model.load_state_dict(best_state)

    return raw_model, y_scaler, u_scaler, pr_scaler#,mu_control,inv_cov_control,tau_ood
