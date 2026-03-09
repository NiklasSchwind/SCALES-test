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
    def __init__(self, y_dim, u_dim, z_dim=16, rnn_hidden=62, mlp_hidden=128, emission_uses_u=False):
        super().__init__()
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.emission_uses_u = emission_uses_u

        self.gru = nn.GRU(input_size=y_dim + u_dim, hidden_size=rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        self.trans = scales_ssm.MLP(z_dim + u_dim, 2 * z_dim, hidden=mlp_hidden)

        emit_in = z_dim + (u_dim if emission_uses_u else 0)
        self.emit = scales_ssm.MLP(emit_in, 2 * y_dim, hidden=mlp_hidden)

        self.emit_pr = scales_ssm.MLP(emit_in,4*y_dim,hidden = 230)

        #pattern scaling like head for emission
        self.ctrl_lin = nn.Linear(u_dim, y_dim,bias=True)

        self.eps = 1e-6
        

    def sample(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

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

        z_prev = None
        for t in range(T):
            z_t = self.sample(mu_q[:, t], logvar_q[:, t])
            ctrl = self.ctrl_lin(u[:, t])              # [B, y_dim]
                    
            if self.emission_uses_u:
                emit_out = self.emit(torch.cat([z_t, u[:, t]], dim=-1))  # [B, 2*y_dim]
                e_in = torch.cat([z_t, u[:, t]], dim=-1)
            else:
                emit_out = self.emit(z_t)  # [B, 2*y_dim]
                e_in = z_t

            res, log_sigma_y_t = torch.chunk(emit_out, 2, dim=-1)
            log_sigma_y_t = torch.clamp(log_sigma_y_t, -8.0, 4.0)
            sigma_y_t = torch.exp(log_sigma_y_t) + self.eps
            y_hat = ctrl + res

            # Gaussian NLL with per-step sigma
            r = (y[:, t] - y_hat) / sigma_y_t
            nll_t = (0.5 * r ** 2 + log_sigma_y_t + 0.5 * math.log(2.0 * math.pi)).sum(-1)
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

        ysamps = []
        ysamps_pr = []
        for _ in range(n_samples):
            z = self.sample(mu_qT, logvar_qT)

            preds = []
            preds_pr = []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(self.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = self.sample(mu_p, logvar_p)

                ctrl = self.ctrl_lin(u_t)

                if self.emission_uses_u:
                    emit_out = self.emit(torch.cat([z, u_t], dim=-1))
                    e_in = torch.cat([z, u_t], dim=-1)
                else:
                    emit_out = self.emit(z)
                    e_in = z

                res, log_sigma_y_t = torch.chunk(emit_out, 2, dim=-1)
                log_sigma_y_t = torch.clamp(log_sigma_y_t, -8.0, 4.0)
                sigma_y_t = torch.exp(log_sigma_y_t) + self.eps
                y_hat = ctrl + res

                out = self.emit_pr(e_in)
                mu_t, log_sigma_t, eps_skew_t, log_delta_t = torch.chunk(out, 4, dim=-1)
                sigma_pr = torch.exp(log_sigma_t) + self.eps
                x_samp = mu_t + sigma_pr * torch.randn_like(mu_t)

                y_s = y_hat + sigma_y_t * torch.randn_like(y_hat)
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

        ysamps = []
        ysamps_pr = []
        for _ in range(n_samples):
            z = self.sample(mu_qT, logvar_qT)

            preds = []
            preds_pr = []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(self.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = self.sample(mu_p, logvar_p)

                ctrl = self.ctrl_lin(u_t)
                if self.emission_uses_u:
                    emit_out = self.emit(torch.cat([z, u_t], dim=-1))
                    e_in = torch.cat([z, u_t], dim=-1)
                else:
                    emit_out = self.emit(z)
                    e_in = z

                res, _ = torch.chunk(emit_out, 2, dim=-1)
                y_hat = ctrl + res

                out = self.emit_pr(e_in)
                mu_t, log_sigma_t, eps_skew_t, log_delta_t = torch.chunk(out, 4, dim=-1)
                sigma_pr = torch.exp(log_sigma_t) + self.eps
                x_samp = mu_t + sigma_pr * torch.randn_like(mu_t)

                y_s = y_hat
                y_s_pr = sinh_arcsinh_forward(x_samp, eps_skew_t, log_delta_t, eps=self.eps)
                preds.append(y_s)
                preds_pr.append(y_s_pr)

            ysamps.append(torch.stack(preds, dim=1))  # [B,steps,Dy]
            ysamps_pr.append(torch.stack(preds_pr, dim=1)) # [B,steps,Dy]

        samp = torch.stack(ysamps, dim=0)  # [S,B,steps,Dy]
        samp_pr = torch.stack(ysamps_pr, dim=0)  # [S,B,steps,Dy]
        return samp.mean(0), samp.quantile(0.10, 0), samp.quantile(0.90, 0),samp_pr.mean(0),samp_pr.quantile(0.10, 0), samp_pr.quantile(0.90, 0)


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
    device="cpu",
):
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
    y_trn = y_scaler.transform(y_tr)
    pr_trn = pr_scaler.transform(pr_tr)
    y_van = y_scaler.transform(y_va)
    pr_van = pr_scaler.transform(pr_va)
    u_trn = u_scaler.transform(u_tr)
    u_van = u_scaler.transform(u_va)

    print("data standardised.")

    # Fit ridge on TRAIN only
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
    model = DeepSSMPatternConditioned(y_dim=Dy, u_dim=Du, z_dim=z_dim).to(device)
    # Copy into model.ctrl_lin and freeze (recommended for fallback)
    load_into_ctrl_lin(model, W, b, freeze=True)
    #opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)

    best_val = float("inf")
    best_state = None
    patience, patience_left = 15, 15
  
    # KL annealing schedule: ramp from 0 -> 1 over first ~30% of training
    total_steps = epochs * len(train_dl)
    global_step = 0

    for epoch in range(1, epochs + 1):
       
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
       
            nll, kl, nll_pr = model.forward_elbo(y_full, pr_full, u_full, kl_free_bits=0.2)
            mean, _, _ ,mean_pr,_,_= model.forecast_deterministic(y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
            roll_out_mse =((mean - y_fut) ** 2).mean()
            roll_out_mse_pr = ((mean_pr - pr_fut) ** 2).mean()
            lin_mean = model.ctrl_lin(u_full.reshape(-1, Du)).reshape(B, T, Dy)
            lin_mse = ((lin_mean-y_full)**2).mean()
            
           
            # anneal KL weight
            global_step += 1
            frac = min(1.0, global_step / int(0.3 * total_steps))
            kl_w = frac  # 0->1
            kl_w = 5
            alpha = 10000
            omega = 100
         
         
            loss = nll + nll_pr + kl_w * kl + alpha*roll_out_mse + omega*roll_out_mse_pr
            

            if(global_step%100==0):
                print("loss: ",nll.item(),kl_w,kl.item(),roll_out_mse.item(),lin_mse.item(),nll_pr.item(),roll_out_mse_pr.item())
            
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

                nll, kl, nll_pr = model.forward_elbo(y_full, pr_full, u_full, kl_free_bits=0.2)
                mean, _, _,mean_pr,_,_ = model.forecast(y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
                mse = ((mean - y_fut) ** 2).mean().item()
                mse_pr = ((mean_pr - pr_fut) ** 2).mean().item()
                loss = nll + nll_pr + kl_w * kl + alpha*mse + omega*mse_pr
                va_loss.append(loss.item())
                va_mse.append(mse)

        tr = float(np.mean(tr_loss))
        va = float(np.mean(va_loss))
        mse = float(np.mean(va_mse))
        print(f"epoch {epoch:03d} | train {tr:.4f} | val_elbo {va:.4f} | val_mse {mse:.4f}")
        os.makedirs("outputs_ssm_scales", exist_ok=True)
        torch.save(model.state_dict(),"outputs_ssm_scales/model_out")

        # early stopping on val_elbo
        if va < best_val - 1e-4:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, y_scaler, u_scaler, pr_scaler#,mu_control,inv_cov_control,tau_ood
