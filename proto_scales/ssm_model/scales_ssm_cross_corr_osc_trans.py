"""
scales_ssm_cross_corr_osc_trans
================================

Variant of `scales_ssm_tas_pr_hyst_correlated` that addresses the three
structural improvements identified in the design review.

Improvements vs. the original module
------------------------------------

1. Joint tas–pr emission via a Gaussian copula
   ---------------------------------------------
   The two separate emission heads (`emit` for tas as low-rank MVN, `emit_pr`
   for pr as elementwise sinh–arcsinh flow) are merged into a single joint
   head. The latent Gaussian for tas and the *base* Gaussian of the sinh–
   arcsinh flow for pr are drawn jointly from a low-rank multivariate
   Gaussian of dimension 2·y_dim; the pr half is then warped elementwise
   by the sinh–arcsinh forward map. This gives:
     - normal marginals for tas (unchanged from original),
     - sinh–arcsinh marginals for pr (unchanged from original),
     - low-rank cross-correlation across regions AND across tas/pr (new).
   The log-likelihood decomposes as:
     log p(y_tas, y_pr) = log N([y_tas; x_pr]; μ, LLᵀ+D) + log|dx_pr/dy_pr|
   where x_pr is the inverse of the sinh–arcsinh flow applied to y_pr.

2. Oscillatory latent transition
   -------------------------------
   The latent transition p(z_t | z_{t-1}, u_t) is augmented with a linear
   block-diagonal oscillatory core:
     mu_p = R(ω)·exp(-damping)·z_{t-1} + B·u_t + MLP_correction(z_{t-1}, u_t)
     logvar_p = MLP_var(z_{t-1}, u_t)
   The oscillator pairs adjacent z-dimensions into 2D rotations with learned
   per-pair frequency ω and damping. Frequencies are initialised across the
   ENSO band (period 12–240 months). This gives complex eigenvalues by
   construction, so the model has an inductive bias for damped oscillatory
   modes in the latent state — the mechanism that was missing to represent
   ENSO-like internal variability.

3. Rebalanced training loss
   -------------------------
   The rollout-MSE weights (`alpha`, `omega`, `gamma`) that overwhelmed the
   ELBO in the original are reduced by ~5×, so the joint NLL and KL carry
   meaningful gradient signal. This is a `run_train` internal change; the
   function signature is unchanged.

Public API
----------
All class and function signatures are identical to
`scales_ssm_tas_pr_hyst_correlated`:
  - `DeepSSMPatternConditioned(...)` with the same constructor kwargs
    (plus three new optional oscillator kwargs at the end),
  - `forward_elbo(y, pr, u, kl_free_bits)` returns `(nll, kl, nll_pr)` where
    `nll` is the joint NLL and `nll_pr` is a zero placeholder (pr is folded
    into the joint NLL),
  - `forecast`, `forecast_deterministic` return the same six tensors,
  - `run_train(...)` has the same signature (incl. `acf_max_lag`,
    `acf_weight`).

Note: model weights are NOT compatible with the original module (the emit
head has a different output size and there is no separate `emit_pr`). Train
from scratch or write a new bootstrap script.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import proto_scales.ssm_model.scales_ssm as scales_ssm


# ─────────────────────────────────────────────────────────────────────────────
# Sinh–arcsinh flow (kept for legacy callers; joint model uses inline versions)
# ─────────────────────────────────────────────────────────────────────────────
def sinh_arcsinh_flow_nll_conditional(y, mu, log_sigma, eps_skew, log_delta, eps=1e-6):
    sigma = torch.exp(torch.clamp(log_sigma, -8.0, 6.0)) + eps
    delta = F.softplus(log_delta) + eps
    a = torch.asinh(y)
    t = delta * a - eps_skew
    x = torch.sinh(t)
    log_abs_det = torch.log(torch.cosh(t) + eps) + torch.log(delta) - 0.5 * torch.log1p(y * y)
    r = (x - mu) / sigma
    logp_base = -0.5 * (r * r) - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)
    logp = logp_base + log_abs_det
    return (-logp).sum(dim=-1)


def sinh_arcsinh_forward(x, eps_skew, log_delta, eps=1e-6):
    delta = F.softplus(log_delta) + eps
    return torch.sinh((torch.asinh(x) + eps_skew) / delta)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (identical to original)
# ─────────────────────────────────────────────────────────────────────────────
class UnifiedWindowDataset(Dataset):
    def __init__(self, y, pr, u, context_len=40, horizon=12, stride=1, start_mode="all"):
        y = np.asarray(y, dtype=np.float32)
        pr = np.asarray(pr, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)
        if y.ndim == 2:
            y = y[None, ...]
        if pr.ndim == 2:
            pr = pr[None, ...]
        if u.ndim == 2:
            u = u[None, ...]
        assert y.shape[0] == u.shape[0] and y.shape[1] == u.shape[1]
        assert pr.shape[0] == u.shape[0] and pr.shape[1] == u.shape[1]
        assert pr.shape[2] == y.shape[2]
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
        y = self.y[s]
        u = self.u[s]
        pr = self.pr[s]
        y_ctx = y[start:start + Tc]
        u_ctx = u[start:start + Tc]
        pr_ctx = pr[start:start + Tc]
        u_fut = u[start + Tc:start + Tc + H]
        y_fut = y[start + Tc:start + Tc + H]
        pr_fut = pr[start + Tc:start + Tc + H]
        return y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut


# ─────────────────────────────────────────────────────────────────────────────
# Deep SSM with joint emission and oscillatory transition
# ─────────────────────────────────────────────────────────────────────────────
class DeepSSMPatternConditioned(nn.Module):
    """
    Joint-emission variant with an oscillatory linear transition.

    Emission (joint, low-rank Gaussian copula):
      [x_tas; x_pr] ~ N(μ, LLᵀ + D)          — 2·y_dim latent Gaussian
      y_tas = x_tas + ctrl_lin(u)            — identity marginals for tas
      y_pr  = sinh((asinh(x_pr) + ε) / δ)    — SAS marginals for pr

    Transition (oscillatory + MLP correction):
      z_next = R(ω)·exp(-damping)·z + B·u + MLP_μ(z, u)
      logvar = MLP_σ(z, u)
    """

    def __init__(self, y_dim, u_dim, z_dim=16, rnn_hidden=62, u_rnn_hidden=64, mlp_hidden=128,
                 emission_uses_u=False, use_linear_model=True, reservoir_dim=2,
                 init_alpha=0.01, init_omega=1.0, alpha_max=0.02, cov_rank=5,
                 osc_damping_max=0.1, osc_freq_range=(2 * math.pi / 240.0, 2 * math.pi / 24.0),
                 osc_init_damp_ratio=0.05):
        super().__init__()
        if z_dim % 2 != 0:
            raise ValueError("z_dim must be even (paired into 2D oscillators)")
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.emission_uses_u = emission_uses_u
        self.use_linear_model = use_linear_model
        self.u_rnn_hidden = u_rnn_hidden
        self.reservoir_dim = reservoir_dim
        self.alpha_max = alpha_max
        self.cov_rank = cov_rank
        self.n_osc = z_dim // 2
        self.osc_damping_max = osc_damping_max

        # Inference GRU on [y, u]
        self.gru = nn.GRU(input_size=y_dim + u_dim, hidden_size=rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        # Oscillatory transition parameters
        # log_damping raw such that sigmoid(log_damping) * osc_damping_max = init damp
        init_raw = math.log(osc_init_damp_ratio / (1.0 - osc_init_damp_ratio))
        self.log_damping = nn.Parameter(torch.full((self.n_osc,), init_raw))
        # Spread ω across the requested range (log-spaced)
        omega_lo, omega_hi = osc_freq_range
        self.omega = nn.Parameter(torch.linspace(omega_lo, omega_hi, self.n_osc))
        # Learnable per-oscillator amplitude gate; sigmoid(-2.2) ≈ 0.1 so oscillators
        # start small and must earn their amplitude via the ACF loss.
        self.log_amp = nn.Parameter(torch.full((self.n_osc,), -2.2))
        # Linear input coupling (B·u_t)
        self.B_osc = nn.Linear(u_dim, z_dim, bias=False)
        # MLP correction on top of linear oscillator
        self.trans_corr = scales_ssm.MLP(z_dim + u_dim, 2 * z_dim, hidden=mlp_hidden)

        # Reservoir (unchanged from original)
        if emission_uses_u:
            self.u_gru = nn.GRU(input_size=u_dim, hidden_size=u_rnn_hidden, batch_first=True)
            log_alpha_init = math.log(init_alpha / (1.0 - init_alpha))
            self.log_alpha = nn.Parameter(torch.full((y_dim, reservoir_dim), log_alpha_init))
            self.omega_lin = nn.Linear(u_rnn_hidden, y_dim * reservoir_dim, bias=False)

        # Joint emission head:
        # output layout per timestep [B, out_dim]:
        #   [ μ_joint (2·D) | log_diag (2·D) | cov_factor (2·D·r) | ε_skew (D) | log δ (D) ]
        # total out_dim = 2·D·(3 + r)
        emit_in = z_dim + (u_rnn_hidden + y_dim * reservoir_dim if emission_uses_u else 0)
        emit_out = 2 * y_dim * (3 + cov_rank)
        self.emit = scales_ssm.MLP(emit_in, emit_out, hidden=mlp_hidden)

        # Pattern-scaling backbone (unchanged; applies only to tas mean)
        if self.use_linear_model:
            self.ctrl_lin = nn.Linear(u_dim, y_dim, bias=True)

        self.eps = 1e-6

    # -----------------------
    # Helpers
    # -----------------------
    def sample(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def _reservoir_step(self, s, uh_t):
        alpha = torch.sigmoid(self.log_alpha) * self.alpha_max
        target = self.omega_lin(uh_t).reshape(uh_t.shape[0], self.y_dim, self.reservoir_dim)
        return s + alpha[None] * (target - s)

    def _osc_transition(self, z_prev, u_t):
        """Oscillatory prior mean + MLP correction; returns (mu_p, logvar_p)."""
        B = z_prev.shape[0]
        # Block-diagonal 2D rotations with damping
        z_pairs = z_prev.reshape(B, self.n_osc, 2)
        damp = torch.sigmoid(self.log_damping) * self.osc_damping_max        # [n_osc]
        decay = torch.exp(-damp)                                             # [n_osc]
        amp = torch.sigmoid(self.log_amp)                                    # [n_osc]
        cos_w = torch.cos(self.omega)                                        # [n_osc]
        sin_w = torch.sin(self.omega)                                        # [n_osc]
        z0 = amp * decay * (cos_w * z_pairs[..., 0] - sin_w * z_pairs[..., 1])   # [B, n_osc]
        z1 = amp * decay * (sin_w * z_pairs[..., 0] + cos_w * z_pairs[..., 1])   # [B, n_osc]
        z_osc = torch.stack([z0, z1], dim=-1).reshape(B, self.z_dim)
        z_input = self.B_osc(u_t)                                            # [B, z_dim]
        corr = self.trans_corr(torch.cat([z_prev, u_t], dim=-1))             # [B, 2·z_dim]
        mu_corr, logvar_p = torch.chunk(corr, 2, dim=-1)
        mu_p = z_osc + z_input + mu_corr
        return mu_p, logvar_p

    def _parse_emit(self, emit_out, B):
        """
        Returns:
          mu_joint    [B, 2·D]  joint latent mean for [x_tas; x_pr]
          cov_factor  [B, 2·D, r]
          cov_diag    [B, 2·D]
          eps_skew    [B, D]    SAS skew  (pr)
          log_delta   [B, D]    SAS tail  (pr)
        """
        D, r = self.y_dim, self.cov_rank
        i = 0
        mu_joint = emit_out[:, i:i + 2 * D];               i += 2 * D
        log_diag = emit_out[:, i:i + 2 * D];               i += 2 * D
        factor   = emit_out[:, i:i + 2 * D * r];           i += 2 * D * r
        eps_skew = emit_out[:, i:i + D];                   i += D
        log_delta = emit_out[:, i:i + D];                  i += D
        # Clamp log_diag before softplus to avoid degenerate cov_diag (Cholesky
        # in LowRankMultivariateNormal fails when I + Wᵀ D⁻¹ W becomes
        # ill-conditioned — happens early in training when D underflows to 0).
        log_diag = torch.clamp(log_diag, -8.0, 8.0)
        cov_diag = F.softplus(log_diag) + 1e-4
        cov_factor = factor.reshape(B, 2 * D, r)
        return mu_joint, cov_factor, cov_diag, eps_skew, log_delta

    def _emit_step(self, e_in, u_t, B):
        """Run one emission step; returns everything needed for NLL and sampling."""
        emit_out = self.emit(e_in)
        mu_joint, cov_factor, cov_diag, eps_skew, log_delta = self._parse_emit(emit_out, B)
        mu_tas = mu_joint[:, :self.y_dim]
        mu_pr  = mu_joint[:, self.y_dim:]
        if self.use_linear_model:
            mu_tas = mu_tas + self.ctrl_lin(u_t)
        mu_x = torch.cat([mu_tas, mu_pr], dim=-1)
        mu_x = torch.clamp(mu_x, -50.0, 50.0)
        return mu_x, cov_factor, cov_diag, eps_skew, log_delta

    # -----------------------
    # ELBO
    # -----------------------
    def forward_elbo(self, y, pr, u, kl_free_bits=0.5):
        """
        Joint NLL over (y_tas, y_pr) with Gaussian copula and SAS flow on pr.
        Returns (nll_joint, kl, nll_pr_placeholder) — the third value is a
        zero tensor, kept for API compatibility with the original module.
        """
        B, T, _ = y.shape

        rnn_in = torch.cat([y, u], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h)
        mu_q, logvar_q = torch.chunk(q_params, 2, dim=-1)
        logvar_q = torch.clamp(logvar_q, -12.0, 6.0)

        mu_p0 = torch.zeros(B, self.z_dim, device=y.device)
        logvar_p0 = torch.zeros(B, self.z_dim, device=y.device)

        nll = 0.0
        kl = 0.0

        if self.emission_uses_u:
            uh, _ = self.u_gru(u)
            s = self.omega_lin(uh[:, 0]).reshape(B, self.y_dim, self.reservoir_dim).detach()

        z_prev = None
        for t in range(T):
            z_t = self.sample(mu_q[:, t], logvar_q[:, t])

            if self.emission_uses_u:
                e_in = torch.cat([z_t, uh[:, t], s.reshape(B, -1)], dim=-1)
            else:
                e_in = z_t

            mu_x, cov_factor, cov_diag, eps_skew, log_delta = self._emit_step(e_in, u[:, t], B)

            # Invert SAS on observed pr to obtain latent x_pr and its Jacobian
            delta = F.softplus(log_delta) + self.eps            # [B, D]
            y_pr_t = pr[:, t]                                   # [B, D]
            a = torch.asinh(y_pr_t)
            t_sas = delta * a - eps_skew
            x_pr = torch.sinh(t_sas)                            # [B, D]
            log_abs_det = (
                torch.log(torch.cosh(t_sas) + self.eps)
                + torch.log(delta)
                - 0.5 * torch.log1p(y_pr_t * y_pr_t)
            )                                                   # [B, D]

            x_joint = torch.cat([y[:, t], x_pr], dim=-1)        # [B, 2·D]
            dist_joint = torch.distributions.LowRankMultivariateNormal(
                loc=mu_x, cov_factor=cov_factor, cov_diag=cov_diag)
            nll_t = -dist_joint.log_prob(x_joint) - log_abs_det.sum(dim=-1)  # [B]
            nll = nll + nll_t

            if t == 0:
                kl_t = scales_ssm.diag_gaussian_kl(mu_q[:, 0], logvar_q[:, 0], mu_p0, logvar_p0)
            else:
                mu_p, logvar_p = self._osc_transition(z_prev, u[:, t])
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                kl_t = scales_ssm.diag_gaussian_kl(mu_q[:, t], logvar_q[:, t], mu_p, logvar_p)
            kl = kl + torch.clamp(kl_t, min=kl_free_bits)

            if self.emission_uses_u:
                s = self._reservoir_step(s, uh[:, t])
            z_prev = z_t

        nll = nll.mean()
        kl = kl.mean()
        # Placeholder for API compatibility with original (which returned nll_pr separately)
        nll_pr = torch.zeros((), device=y.device)
        return nll, kl, nll_pr

    # -----------------------
    # Stochastic forecast
    # -----------------------
    @torch.no_grad()
    def forecast(self, y_ctx, u_ctx, u_fut, steps, n_samples=50):
        B, Tc, _ = y_ctx.shape

        rnn_in = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT = torch.clamp(logvar_qT, -12.0, 6.0)

        if self.emission_uses_u:
            uh_ctx, h_u = self.u_gru(u_ctx)
            s_ctx = torch.zeros(B, self.y_dim, self.reservoir_dim, device=u_ctx.device)
            for k in range(u_ctx.shape[1]):
                s_ctx = self._reservoir_step(s_ctx, uh_ctx[:, k])

        ysamps_tas = []
        ysamps_pr = []
        for _ in range(n_samples):
            z = self.sample(mu_qT, logvar_qT)
            h_u_s = h_u.clone() if self.emission_uses_u else None
            s = s_ctx.clone() if self.emission_uses_u else None

            preds_tas = []
            preds_pr = []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = self._osc_transition(z, u_t)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = self.sample(mu_p, logvar_p)
                z = torch.clamp(z, -10.0, 10.0)

                if self.emission_uses_u:
                    uh_t, h_u_s = self.u_gru(u_t.unsqueeze(1), h_u_s)
                    uh_t = uh_t.squeeze(1)
                    e_in = torch.cat([z, uh_t, s.reshape(B, -1)], dim=-1)
                    s = self._reservoir_step(s, uh_t)
                else:
                    e_in = z

                mu_x, cov_factor, cov_diag, eps_skew, log_delta = self._emit_step(e_in, u_t, B)
                dist_joint = torch.distributions.LowRankMultivariateNormal(
                    loc=mu_x, cov_factor=cov_factor, cov_diag=cov_diag)
                x_samp = dist_joint.sample()
                y_tas_s = x_samp[:, :self.y_dim]
                x_pr_s  = x_samp[:, self.y_dim:]
                y_pr_s = sinh_arcsinh_forward(x_pr_s, eps_skew, log_delta, eps=self.eps)
                preds_tas.append(y_tas_s)
                preds_pr.append(y_pr_s)

            ysamps_tas.append(torch.stack(preds_tas, dim=1))
            ysamps_pr.append(torch.stack(preds_pr, dim=1))

        samp = torch.stack(ysamps_tas, dim=0)
        samp_pr = torch.stack(ysamps_pr, dim=0)
        return (samp.mean(0), samp.quantile(0.10, 0), samp.quantile(0.90, 0),
                samp_pr.mean(0), samp_pr.quantile(0.10, 0), samp_pr.quantile(0.90, 0))

    # -----------------------
    # Deterministic forecast (mean-only)
    # -----------------------
    @torch.no_grad()
    def forecast_deterministic(self, y_ctx, u_ctx, u_fut, steps, n_samples=50):
        B, Tc, _ = y_ctx.shape

        rnn_in = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT = torch.clamp(logvar_qT, -12.0, 6.0)

        if self.emission_uses_u:
            uh_ctx, h_u = self.u_gru(u_ctx)
            s_ctx = torch.zeros(B, self.y_dim, self.reservoir_dim, device=u_ctx.device)
            for k in range(u_ctx.shape[1]):
                s_ctx = self._reservoir_step(s_ctx, uh_ctx[:, k])

        ysamps_tas = []
        ysamps_pr = []
        for _ in range(n_samples):
            z = self.sample(mu_qT, logvar_qT)
            h_u_s = h_u.clone() if self.emission_uses_u else None
            s = s_ctx.clone() if self.emission_uses_u else None

            preds_tas = []
            preds_pr = []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = self._osc_transition(z, u_t)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = self.sample(mu_p, logvar_p)
                z = torch.clamp(z, -10.0, 10.0)

                if self.emission_uses_u:
                    uh_t, h_u_s = self.u_gru(u_t.unsqueeze(1), h_u_s)
                    uh_t = uh_t.squeeze(1)
                    e_in = torch.cat([z, uh_t, s.reshape(B, -1)], dim=-1)
                    s = self._reservoir_step(s, uh_t)
                else:
                    e_in = z

                mu_x, _cov_factor, _cov_diag, eps_skew, log_delta = self._emit_step(e_in, u_t, B)
                # Deterministic: use the mean of x directly
                y_tas_s = mu_x[:, :self.y_dim]
                x_pr_mean = mu_x[:, self.y_dim:]
                y_pr_s = sinh_arcsinh_forward(x_pr_mean, eps_skew, log_delta, eps=self.eps)
                preds_tas.append(y_tas_s)
                preds_pr.append(y_pr_s)

            ysamps_tas.append(torch.stack(preds_tas, dim=1))
            ysamps_pr.append(torch.stack(preds_pr, dim=1))

        samp = torch.stack(ysamps_tas, dim=0)
        samp_pr = torch.stack(ysamps_pr, dim=0)
        return (samp.mean(0), samp.quantile(0.10, 0), samp.quantile(0.90, 0),
                samp_pr.mean(0), samp_pr.quantile(0.10, 0), samp_pr.quantile(0.90, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Utilities (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def fit_control_mahalanobis(u_train_norm, eps=1e-6):
    U = u_train_norm.reshape(-1, u_train_norm.shape[-1])
    mu = U.mean(axis=0)
    cov = np.cov(U.T) + eps * np.eye(U.shape[1])
    inv_cov = np.linalg.inv(cov)
    return mu.astype(np.float32), inv_cov.astype(np.float32)


def mahalanobis_score(u_fut_norm, mu, inv_cov):
    diff = u_fut_norm - mu[None, None, :]
    return np.einsum("bhd,dd,bhd->bh", diff, inv_cov, diff)


def pick_threshold_from_val(val_loader, mu, inv_cov, percentile=99.0):
    all_scores = []
    for _, _, _, u_fut, _, _ in val_loader:
        s = mahalanobis_score(u_fut.numpy(), mu, inv_cov)
        all_scores.append(s.reshape(-1))
    all_scores = np.concatenate(all_scores)
    return float(np.percentile(all_scores, percentile))


def batch_acf(x, max_lag, eps=1e-6):
    """Differentiable normalized ACF via FFT for lags 1..max_lag. x: [B, T, D]."""
    B, T, D = x.shape
    x = x - x.mean(dim=1, keepdim=True)
    xp = F.pad(x, (0, 0, 0, T))
    Xf = torch.fft.rfft(xp, dim=1)
    acf = torch.fft.irfft(Xf * Xf.conj(), dim=1, n=2 * T)
    acf0 = acf[:, 0:1, :].clamp(min=eps)
    return acf[:, 1:max_lag + 1, :] / acf0


def fit_ridge_D(u_train, y_train, alpha=1e-2, fit_intercept=True):
    U = u_train.reshape(-1, u_train.shape[-1])
    Y = y_train.reshape(-1, y_train.shape[-1])
    if fit_intercept:
        ones = np.ones((U.shape[0], 1))
        X = np.concatenate([U, ones], axis=1)
    else:
        X = U
    XtX = X.T @ X
    I = np.eye(XtX.shape[0])
    if fit_intercept:
        I[-1, -1] = 0.0
    Wb = np.linalg.solve(XtX + alpha * I, X.T @ Y)
    if fit_intercept:
        W = Wb[:-1, :]
        b = Wb[-1, :]
    else:
        W = Wb
        b = np.zeros((Y.shape[1],), dtype=Y.dtype)
    return W.astype(np.float32), b.astype(np.float32)


def load_into_ctrl_lin(model, W, b, freeze=True):
    assert hasattr(model, "ctrl_lin")
    Du, Dy = W.shape
    assert model.ctrl_lin.in_features == Du
    assert model.ctrl_lin.out_features == Dy
    with torch.no_grad():
        model.ctrl_lin.weight.copy_(torch.from_numpy(W.T))
        model.ctrl_lin.bias.copy_(torch.from_numpy(b))
    if freeze:
        for p in model.ctrl_lin.parameters():
            p.requires_grad = False


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def run_train(
    y_np, pr_np, u_np,
    context_len=40, horizon=12,
    batch_size=64,
    epochs=50,
    lr=2e-3,
    z_dim=16,
    rnn_hidden=62,
    use_linear_model=True,
    resevoir_dim=2,
    alpha_max=0.02,
    cov_rank=5,
    run_dir=None,
    weights_file=None,
    acf_max_lag=120,
    acf_weight=5000.0,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend)
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")

    N = y_np.shape[0]
    idx = np.random.permutation(N)
    n_train = int(0.8 * N)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    y_tr, pr_tr, u_tr = y_np[tr_idx], pr_np[tr_idx], u_np[tr_idx]
    y_va, pr_va, u_va = y_np[va_idx], pr_np[va_idx], u_np[va_idx]

    print("training start")

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

    if use_linear_model:
        W, b = fit_ridge_D(u_trn, y_trn, alpha=1e-2, fit_intercept=True)
        print("ridge regression completed")

    train_ds = UnifiedWindowDataset(y_trn, pr_trn, u_trn, context_len=context_len, horizon=horizon)
    val_ds   = UnifiedWindowDataset(y_van, pr_van, u_van, context_len=context_len, horizon=horizon)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print("dataloaders prepared")

    Dy = y_np.shape[-1]
    Du = u_np.shape[-1]
    raw_model = DeepSSMPatternConditioned(
        y_dim=Dy, u_dim=Du, z_dim=z_dim, rnn_hidden=rnn_hidden,
        use_linear_model=use_linear_model, emission_uses_u=True,
        reservoir_dim=resevoir_dim, alpha_max=alpha_max, cov_rank=cov_rank,
    ).to(device)
    if weights_file is not None:
        ckpt = torch.load(weights_file, map_location=device)
        missing, unexpected = raw_model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"Warm-start: initialising missing keys {missing}")
        if unexpected:
            print(f"Warm-start: ignoring unexpected keys {unexpected}")
        # Re-initialise oscillator frequencies and amplitude gates so they start
        # in the correct range rather than carrying over values tuned for the
        # old (wider) frequency range.
        omega_lo = 2 * math.pi / 240.0
        omega_hi = 2 * math.pi / 24.0
        with torch.no_grad():
            raw_model.omega.copy_(torch.linspace(omega_lo, omega_hi, raw_model.n_osc))
            raw_model.log_amp.fill_(-2.2)
        print(f"Loaded weights from {weights_file} (oscillator params reset to new range)")
    if use_linear_model:
        load_into_ctrl_lin(raw_model, W, b, freeze=True)

    ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if use_cuda else {}
    model = DDP(raw_model, **ddp_kwargs)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)

    best_val = float("inf")
    best_state = None
    patience, patience_left = 15, 15

    total_steps = epochs * len(train_dl)
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = []

        for y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut in train_dl:
            y_ctx = torch.tensor(y_ctx, device=device)
            pr_ctx = torch.tensor(pr_ctx, device=device)
            u_ctx = torch.tensor(u_ctx, device=device)
            u_fut = torch.tensor(u_fut, device=device)
            y_fut = torch.tensor(y_fut, device=device)
            pr_fut = torch.tensor(pr_fut, device=device)

            y_full = torch.cat([y_ctx, y_fut], dim=1)
            pr_full = torch.cat([pr_ctx, pr_fut], dim=1)
            u_full = torch.cat([u_ctx, u_fut], dim=1)

            B, T, _ = y_full.shape

            nll, kl, nll_pr = raw_model.forward_elbo(y_full, pr_full, u_full, kl_free_bits=0.2)
            mean, _, _, mean_pr, _, _ = raw_model.forecast_deterministic(
                y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
            roll_out_mse = F.huber_loss(mean, y_fut, delta=1.0)
            roll_out_mse_pr = F.huber_loss(mean_pr, pr_fut, delta=1.0)
            if use_linear_model:
                lin_mean = raw_model.ctrl_lin(u_full.reshape(-1, Du)).reshape(B, T, Dy)
                lin_mse = ((lin_mean - y_full) ** 2).mean()

            n_complete_years = horizon // 12
            if n_complete_years > 0:
                H_yr = n_complete_years * 12
                mean_yr = mean[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
                y_fut_yr = y_fut[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
                roll_out_mse_yearly = ((mean_yr - y_fut_yr) ** 2).mean()
            else:
                roll_out_mse_yearly = torch.tensor(0.0, device=device)

            # Loss rebalance (improvement 3): reduce rollout/yearly weights so the
            # joint ELBO carries meaningful gradient signal. Original values were
            # alpha=10000, omega=500, gamma=80000.
            global_step += 1
            frac = min(1.0, global_step / int(0.3 * total_steps))
            kl_w = 5
            alpha = 2000 * frac
            omega = 100 * frac
            gamma = 20000

            # Clip max_lag so the FFT-based ACF is valid: need sequence > 2*lag.
            # With a typical horizon of 12 months, effective_lag=6 is enough to
            # penalise a 12-month oscillation via its half-period anticorrelation
            # (ACF at lag 6 = −1 for a pure annual cycle, 0 for white noise).
            effective_lag = min(acf_max_lag, (horizon - 1) // 2)
            if acf_max_lag > 0 and effective_lag >= 2:
                with torch.no_grad():
                    ctrl_fut = raw_model.ctrl_lin(u_fut.reshape(-1, Du)).reshape(B, horizon, Dy)
                acf_pred_tas = batch_acf(mean - ctrl_fut, effective_lag)
                acf_true_tas = batch_acf(y_fut - ctrl_fut, effective_lag)
                acf_pred_pr  = batch_acf(mean_pr, effective_lag)
                acf_true_pr  = batch_acf(pr_fut,  effective_lag)
                acf_loss = (
                    ((acf_pred_tas - acf_true_tas) ** 2).mean()
                    + ((acf_pred_pr - acf_true_pr) ** 2).mean()
                )
            else:
                acf_loss = torch.tensor(0.0, device=device)

            loss = (nll + nll_pr + kl_w * kl
                    + alpha * roll_out_mse + omega * roll_out_mse_pr
                    + gamma * roll_out_mse_yearly + acf_weight * frac * acf_loss)

            if global_step % 100 == 0:
                if use_linear_model:
                    print("loss: ", nll.item(), kl_w, kl.item(), roll_out_mse.item(),
                          lin_mse.item(), nll_pr.item(), roll_out_mse_pr.item(),
                          roll_out_mse_yearly.item(), acf_loss.item())
                else:
                    print("loss: ", nll.item(), kl_w, kl.item(), roll_out_mse.item(),
                          nll_pr.item(), roll_out_mse_pr.item(),
                          roll_out_mse_yearly.item(), acf_loss.item())

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss.append(loss.item())

        model.eval()
        va_loss = []
        va_mse = []
        with torch.no_grad():
            for y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut in val_dl:
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
                mean, _, _, mean_pr, _, _ = raw_model.forecast(
                    y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
                mse = ((mean - y_fut) ** 2).mean().item()
                mse_pr = ((mean_pr - pr_fut) ** 2).mean().item()
                loss = nll + nll_pr + kl_w * kl + alpha * mse + omega * mse_pr
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

        if va < best_val - 1e-4:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

    if best_state is not None:
        raw_model.load_state_dict(best_state)

    return raw_model, y_scaler, u_scaler, pr_scaler
