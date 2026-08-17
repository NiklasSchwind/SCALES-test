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

2. Contractive linear latent transition (replaced the explicit oscillator)
   ------------------------------------------------------------------------
     mu_p = ρ·A·z_{t-1} + B·u_t + MLP_correction(z_{t-1}, u_t)
     logvar_p = MLP_var(z_{t-1}, u_t)
   with A spectral-normalised and ρ = sigmoid(ρ_raw) ∈ (0,1), so every
   eigenvalue has modulus < ρ < 1 and the latent is stable by construction.
   A is initialised near the identity, giving genuine persistence from the
   first step.

   This replaced a bank of explicit 2D rotations, for two reasons.

   *The amplitude gate destroyed the latent's memory.* Oscillator amplitude
   was gated by sigmoid(log_amp) initialised at -2.2 ≈ 0.0998, so the linear
   recurrence had |λ| ≈ 0.099 **per month** — a memory half-life of 0.3
   months, retaining 9e-13 of its amplitude after a year. z could therefore
   carry neither temporal correlation (bad ACF) nor slow forced information
   (bad hysteresis / delayed warming). The gate existed so oscillators would
   "earn" amplitude from the ACF loss, which was inert. An ENSO-like mode
   needs |λ| ≈ 0.9835; the default ρ = 0.98 gives ~4 years.

   *The frequency band excluded the seasonal cycle.* Periods were initialised
   over 24–240 months, so the 12-month cycle — the dominant feature of the
   observed ACF — was a factor of two below anything representable. A general
   matrix has no band limit.

   Oscillations are not lost by removing the explicit rotations: a linear
   operator has complex eigenvalues generically. Reading them off a fitted
   transition is Principal Oscillation Pattern analysis; see `modes()`, which
   returns learned periods and e-folding times in months.

3. Differentiable rollout for the auxiliary losses
   ------------------------------------------------
   The rollout-MSE and ACF terms were previously computed from
   `forecast_deterministic`, which is decorated `@torch.no_grad()`. They were
   therefore constants and contributed exactly zero gradient — the effective
   training objective was `nll + kl_w * kl` alone. `rollout_samples` is the
   trainable counterpart (reparameterised `rsample`, per-sample trajectories),
   and `run_train` now evaluates the auxiliary terms on a shorter
   `rollout_steps` window that fits in memory with the graph retained.

   The ACF loss is also now evaluated per realisation rather than on the
   ensemble mean, since the ACF of an ensemble mean measures the forced signal,
   not the internal variability the term is meant to constrain.

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
#
# Numerical safety
# ----------------
# sinh() overflows float32 at an argument of ~89, and the SAS forward map is
# sinh((asinh(x) + eps_skew) / delta) with delta = softplus(log_delta). Nothing
# in the network bounds log_delta, so if it drifts negative during training
# delta collapses toward 0, the argument explodes and the map returns inf. That
# inf then reaches the ACF loss as NaN, the NaN reaches the gradients, and the
# optimiser writes NaN into every emission weight — after which the *next*
# forward pass fails with "invalid values" in the LowRankMultivariateNormal loc,
# several steps downstream of the actual cause.
#
# SINH_ARG_CLAMP is the hard safety net (sinh(20) ~ 2.4e8, comfortably finite
# with headroom for the cosh() in the Jacobian). The parameter clamps below keep
# normal operation well away from it; they are deliberately wide so that a
# trained checkpoint's behaviour is unchanged.
# ─────────────────────────────────────────────────────────────────────────────
SINH_ARG_CLAMP = 20.0
LOG_DELTA_CLAMP = (-2.0, 4.0)
EPS_SKEW_CLAMP = (-5.0, 5.0)


def sinh_arcsinh_flow_nll_conditional(y, mu, log_sigma, eps_skew, log_delta, eps=1e-6):
    sigma = torch.exp(torch.clamp(log_sigma, -8.0, 6.0)) + eps
    log_delta = torch.clamp(log_delta, *LOG_DELTA_CLAMP)
    eps_skew = torch.clamp(eps_skew, *EPS_SKEW_CLAMP)
    delta = F.softplus(log_delta) + eps
    a = torch.asinh(y)
    t = torch.clamp(delta * a - eps_skew, -SINH_ARG_CLAMP, SINH_ARG_CLAMP)
    x = torch.sinh(t)
    log_abs_det = torch.log(torch.cosh(t) + eps) + torch.log(delta) - 0.5 * torch.log1p(y * y)
    r = (x - mu) / sigma
    logp_base = -0.5 * (r * r) - torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)
    logp = logp_base + log_abs_det
    return (-logp).sum(dim=-1)


def sinh_arcsinh_forward(x, eps_skew, log_delta, eps=1e-6):
    log_delta = torch.clamp(log_delta, *LOG_DELTA_CLAMP)
    eps_skew = torch.clamp(eps_skew, *EPS_SKEW_CLAMP)
    delta = F.softplus(log_delta) + eps
    arg = torch.clamp((torch.asinh(x) + eps_skew) / delta,
                      -SINH_ARG_CLAMP, SINH_ARG_CLAMP)
    return torch.sinh(arg)


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
                 init_spectral_radius=0.98):
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

        # Inference GRU on [y, u]
        self.gru = nn.GRU(input_size=y_dim + u_dim, hidden_size=rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        # ── Linear transition core ───────────────────────────────────────────
        # Replaces the explicit bank of 2D rotations. Oscillations do not have to
        # be hard-coded: a linear operator has complex eigenvalues generically,
        # and reading them off a *fitted* transition matrix is exactly Principal
        # Oscillation Pattern analysis. The rotation bank was the special case of
        # a normal matrix with non-interacting pairs — strictly less expressive —
        # and it forced a frequency band to be chosen up front. That band was
        # 24–240 months, which excluded the annual cycle entirely and so made the
        # dominant feature of the observed ACF structurally unreachable.
        #
        # It also removes the amplitude gate, which was the real damage: with
        # sigmoid(-2.2) ≈ 0.0998 the latent recurrence had |λ| ≈ 0.099 per MONTH,
        # a memory half-life of 0.3 months. z therefore carried almost no
        # temporal correlation (bad ACF) and could not hold slow forced
        # information (bad hysteresis / delayed warming). The gate was meant to
        # let oscillators "earn" amplitude from the ACF loss, which was inert.
        #
        # A is spectral-normalised (‖A‖₂ = 1) and scaled by ρ = sigmoid(ρ_raw),
        # so every eigenvalue has modulus < ρ < 1 and the latent process is
        # stable by construction — the one useful property damping provided.
        #
        # A is initialised near the IDENTITY, not randomly. ρ bounds the spectral
        # radius but does not set it: a random matrix normalised to unit spectral
        # *norm* has eigenvalues filling a disk of roughly half that radius, so
        # random init would give ~1-year memory and reintroduce the very problem
        # this change fixes. Persistence is also the right prior for a slow
        # latent; the small random part breaks symmetry and lets eigenvalues
        # become complex if the data asks for it.
        _A = nn.Linear(z_dim, z_dim, bias=False)
        with torch.no_grad():
            _A.weight.copy_(torch.eye(z_dim)
                            + 0.05 * torch.randn(z_dim, z_dim) / math.sqrt(z_dim))
        self.A = nn.utils.parametrizations.spectral_norm(_A)
        r = float(init_spectral_radius)
        if not (0.0 < r < 1.0):
            raise ValueError("init_spectral_radius must be in (0, 1)")
        self.rho_raw = nn.Parameter(torch.tensor(math.log(r / (1.0 - r))))
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

    def _transition(self, z_prev, u_t):
        """Contractive linear core + forcing + MLP correction -> (mu_p, logvar_p)."""
        rho = torch.sigmoid(self.rho_raw)
        z_lin = rho * self.A(z_prev)                                         # [B, z_dim]
        z_input = self.B_osc(u_t)                                            # [B, z_dim]
        corr = self.trans_corr(torch.cat([z_prev, u_t], dim=-1))             # [B, 2·z_dim]
        mu_corr, logvar_p = torch.chunk(corr, 2, dim=-1)
        mu_p = z_lin + z_input + mu_corr
        return mu_p, logvar_p

    # Kept so existing callers (and notebooks) keep working after the rename.
    _osc_transition = _transition

    @torch.no_grad()
    def modes(self, months_per_year=12, tol=1e-3):
        """
        Principal Oscillation Patterns of the learned transition.

        Returns (periods_months, efolding_months, moduli) sorted by persistence.
        An eigenvalue λ = r·e^{iθ} is a mode of period 2π/θ months damped with
        e-folding −1/ln(r). Real eigenvalues report an infinite period. Nothing
        here is imposed — it is read back out of the fit, which is the point of
        dropping the explicit oscillator.
        """
        W = self.A.weight * torch.sigmoid(self.rho_raw)
        lam = torch.linalg.eigvals(W.to(torch.float32))
        r = lam.abs().clamp(1e-8, 1 - 1e-8)
        theta = lam.angle().abs()
        period = torch.where(theta > tol, 2.0 * math.pi / theta.clamp(min=tol),
                             torch.full_like(theta, float("inf")))
        efold = -1.0 / torch.log(r)
        order = torch.argsort(efold, descending=True)
        return period[order].cpu(), efold[order].cpu(), r[order].cpu()

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
        # Bound the SAS parameters at source. Unbounded, log_delta can drift
        # negative until delta ~ 0 and the flow overflows to inf; see the note
        # at the top of this module.
        eps_skew = torch.clamp(eps_skew, *EPS_SKEW_CLAMP)
        log_delta = torch.clamp(log_delta, *LOG_DELTA_CLAMP)
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
            # Mirror hazard to the forward map: a large delta drives t_sas past
            # the sinh overflow point, giving inf in x_pr and NaN in log_prob.
            t_sas = torch.clamp(delta * a - eps_skew,
                                -SINH_ARG_CLAMP, SINH_ARG_CLAMP)
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

    # -----------------------
    # Differentiable rollout (training-time auxiliary losses)
    # -----------------------
    def rollout_samples(self, y_ctx, u_ctx, u_fut, steps, n_samples=1):
        """
        Prior rollout that KEEPS the autograd graph.

        `forecast` and `forecast_deterministic` are both decorated
        `@torch.no_grad()` because they are inference entry points. Using them
        inside a training loss silently produces a constant: the rollout-MSE
        and ACF terms then contribute exactly zero gradient. This method is the
        trainable counterpart.

        It returns the individual sample trajectories rather than their mean,
        because the two auxiliary losses need different things:
          - rollout MSE wants the ensemble mean (an estimator of the forced
            response),
          - the ACF loss must be evaluated per realisation — the ACF of an
            ensemble *mean* is not the ACF of the process, it is the ACF of the
            forced signal, so averaging first defeats the purpose of the term.

        Emission draws use `rsample` (reparameterised); `sample` would detach.

        Returns (tas, pr), each [n_samples, B, steps, y_dim].
        """
        B = y_ctx.shape[0]

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

        samps_tas = []
        samps_pr = []
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
                x_samp = dist_joint.rsample()
                preds_tas.append(x_samp[:, :self.y_dim])
                preds_pr.append(sinh_arcsinh_forward(
                    x_samp[:, self.y_dim:], eps_skew, log_delta, eps=self.eps))

            samps_tas.append(torch.stack(preds_tas, dim=1))
            samps_pr.append(torch.stack(preds_pr, dim=1))

        return torch.stack(samps_tas, dim=0), torch.stack(samps_pr, dim=0)


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
    rollout_steps=120,
    rollout_samples=1,
    kl_w=5.0,
    rollout_mse_weight=0.0,
    rollout_mse_pr_weight=0.0,
    rollout_mse_yearly_weight=0.0,
    init_spectral_radius=0.98,
):
    """
    On the rollout-MSE weights (previously hardcoded at 2000 / 100 / 20000):

    `mean = tas_s.mean(0)` is the mean over `rollout_samples` draws, so at the
    default of 1 it *is* a single reparameterised sample. Minimising
    huber(sample, truth) decomposes as

        E[(mu + sigma*eps - y)^2] = (mu - E[y])^2 + sigma^2 + Var(y)

    and that sigma^2 term penalises the model's own predictive variance. In a
    direct test this collapsed sigma from 1.0 to 0.016 while mu converged
    correctly. Internal variability is unpredictable by construction, so asking
    a single realisation to match the truth pointwise can only be satisfied by
    becoming deterministic.

    They therefore default to 0, which reproduces the ELBO-dominated behaviour
    these weights had while the rollout was still non-differentiable. If you do
    want them, set `rollout_samples >= 4` first so `mean` is an actual ensemble
    mean; a warning is printed otherwise.

    They also only ever constrain `rollout_steps` months (120 by default = 10
    years), so they cannot teach a multi-century response such as Southern Ocean
    delayed warming. That signal lives in the ELBO over the full context+horizon
    window.
    """
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

    if (max(rollout_mse_weight, rollout_mse_pr_weight, rollout_mse_yearly_weight) > 0
            and rollout_samples < 4):
        print(f"[warn] rollout-MSE weights are non-zero with rollout_samples="
              f"{rollout_samples}: `mean` is then (near) a single reparameterised "
              f"draw and the term penalises the model's own predictive variance. "
              f"Use rollout_samples >= 4 or set the weights to 0.")

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
        init_spectral_radius=init_spectral_radius,
    ).to(device)
    if weights_file is not None:
        ckpt = torch.load(weights_file, map_location=device)
        missing, unexpected = raw_model.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"Warm-start: initialising missing keys {missing}")
        if unexpected:
            print(f"Warm-start: ignoring unexpected keys {unexpected}")
        print(f"Loaded weights from {weights_file}")
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

            # Differentiable prior rollout. The previous code called
            # forecast_deterministic (@torch.no_grad), so every term below
            # contributed zero gradient — the effective objective was nll+kl_w*kl.
            # A full `horizon`-step differentiable rollout does not fit in memory,
            # so the auxiliary losses are evaluated on the first `rollout_steps`.
            R = min(int(rollout_steps), horizon)
            tas_s, pr_s = raw_model.rollout_samples(
                y_ctx, u_ctx, u_fut[:, :R], steps=R, n_samples=rollout_samples)
            y_fut_R = y_fut[:, :R]
            pr_fut_R = pr_fut[:, :R]

            mean = tas_s.mean(0)
            mean_pr = pr_s.mean(0)
            roll_out_mse = F.huber_loss(mean, y_fut_R, delta=1.0)
            roll_out_mse_pr = F.huber_loss(mean_pr, pr_fut_R, delta=1.0)
            if use_linear_model:
                lin_mean = raw_model.ctrl_lin(u_full.reshape(-1, Du)).reshape(B, T, Dy)
                lin_mse = ((lin_mean - y_full) ** 2).mean()

            n_complete_years = R // 12
            if n_complete_years > 0:
                H_yr = n_complete_years * 12
                mean_yr = mean[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
                y_fut_yr = y_fut_R[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
                roll_out_mse_yearly = ((mean_yr - y_fut_yr) ** 2).mean()
            else:
                roll_out_mse_yearly = torch.tensor(0.0, device=device)

            # Rollout-MSE weights now default to 0 — see the note on
            # `rollout_mse_weight` in the run_train signature. They are a
            # variance-collapse term at rollout_samples=1, which is what made
            # the ACF and the delayed-warming response worse once the rollout
            # became differentiable.
            global_step += 1
            frac = min(1.0, global_step / int(0.3 * total_steps))
            alpha = rollout_mse_weight * frac
            omega = rollout_mse_pr_weight * frac
            gamma = rollout_mse_yearly_weight

            # Clip max_lag so the FFT-based ACF is valid: need sequence > 2*lag.
            # The rollout length R therefore caps which periods can be penalised:
            # only oscillations with a half-period below effective_lag are visible.
            effective_lag = min(acf_max_lag, (R - 1) // 2)
            if acf_max_lag > 0 and effective_lag >= 2:
                if use_linear_model:
                    with torch.no_grad():
                        ctrl_fut = raw_model.ctrl_lin(
                            u_fut[:, :R].reshape(-1, Du)).reshape(B, R, Dy)
                else:
                    ctrl_fut = torch.zeros_like(y_fut_R)
                acf_true_tas = batch_acf(y_fut_R - ctrl_fut, effective_lag)
                acf_true_pr = batch_acf(pr_fut_R, effective_lag)
                # Per realisation, not on the ensemble mean: averaging samples
                # first would remove exactly the internal variability whose
                # autocorrelation this term is meant to constrain.
                acf_loss = torch.zeros((), device=device)
                for si in range(tas_s.shape[0]):
                    acf_loss = acf_loss + (
                        ((batch_acf(tas_s[si] - ctrl_fut, effective_lag) - acf_true_tas) ** 2).mean()
                        + ((batch_acf(pr_s[si], effective_lag) - acf_true_pr) ** 2).mean()
                    )
                acf_loss = acf_loss / tas_s.shape[0]
            else:
                acf_loss = torch.tensor(0.0, device=device)

            loss = (nll + nll_pr + kl_w * kl
                    + alpha * roll_out_mse + omega * roll_out_mse_pr
                    + gamma * roll_out_mse_yearly + acf_weight * frac * acf_loss)

            if global_step % 100 == 0:
                # Weighted contributions, so the balance between the ELBO and the
                # auxiliary terms is directly readable. These terms only started
                # carrying gradient once the rollout became differentiable, so
                # their weights have never actually been tuned against the ELBO.
                print(
                    f"step {global_step} | nll {nll.item():.1f} "
                    f"| kl {kl_w * kl.item():.1f} "
                    f"| roll {alpha * roll_out_mse.item():.1f} "
                    f"| roll_pr {omega * roll_out_mse_pr.item():.1f} "
                    f"| yearly {gamma * roll_out_mse_yearly.item():.1f} "
                    f"| acf {acf_weight * frac * acf_loss.item():.1f} "
                    f"| (raw acf {acf_loss.item():.4f})"
                    + (f" | lin_mse {lin_mse.item():.4f}" if use_linear_model else "")
                )

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
