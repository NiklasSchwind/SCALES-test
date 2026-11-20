"""
annual_elbo_dit
================

Annual outpainting DiT conditioned on a probabilistic state-space latent z,
trained by an ELBO on regional tas only. Same public interface as
`annual_jepa_kl_dit` (`build_model`, `load_elbo_dit_annual`, the latent
process's `posterior` / `roll_forward` / `encoder_inputs`), so forecast code
can swap one import.

Why this exists
----------------
In the JEPA-KL model z is defined by self-prediction (KL to a target-encoder
prior, a rollout MSE in z-space, a covariance penalty, per-dim
standardization). Those terms reward any predictable feature, not the forced
response, and the anti-collapse machinery fills every z-dim with unit
variance. Restart evals showed 15 noise dims and one GRU-age clock, with the
forced response reaching the DiT through u only. This module goes back to the
structure of the original monthly SSM, where reconstruction defines z and
prevents collapse by itself:

    q(z_t | tas_<=t, u_<=t)  = N(mu_q, diag)              GRU, monthly input
    p(z_t | z_t-1, u_t)      = N(d * z_t-1 + B u_t, Q)    (+ gate / MLP)
    p(tas_t | z_t, u_t)      = N(P u_t + c + res(z_t), R) annual regional tas

`P`, `c` are a pattern-scaling regression of annual regional tas on annual
GSAT, fitted once on the training split and frozen (buffers). The residual
reads z only, so z has to carry everything instantaneous pattern scaling
misses: delayed (ocean) warming and the state the DiT lacks, since it already
gets u directly.

Losses (`AnnualELBOOutpaintingDiT.ssm_aux_loss`)
-------------------------------------------------
    nll + beta * kl          ELBO over the super-window, z sampled; the prior
                             at t comes from the sampled z at t-1, at t = 0
                             it is N(0, I). Both terms are summed over their
                             own dims and divided by the same (years x
                             regions), so beta means what it meant in the
                             original monthly SSM (default 5): beta = 1 is the
                             true ELBO.
    rollout_weight * roll    MSE of the emission mean along the mean rollout
                             from the context posterior's z_last, over the
                             horizon (MSE, not the one-step variance, which
                             would underweight century-scale drift).

KL makes encoding unpredictable interannual noise expensive, so unused dims
collapse onto the prior (harmless: their mean rollout is ~0). Watch
`kl_active_dims` instead of using free bits (default 0).

pr head and residual DiT (optional, `pr_emission`, `dit_residual`)
--------------------------------------------------------------------
    p(pr_t | z_t, u_t)      = N(P_pr u_t + c_pr + res_pr(z_t), R_pr)

trained outside the ELBO on detached z (see `ForcedAnnualELBO.pr_loss`).
With `dit_residual` the DiT generates field - (tas, pr) baseline and the
baseline is added back at sampling (see `AnnualELBOOutpaintingDiT`).

DiT gradients
--------------
`detach_latents=True` (default): the diffusion loss does not reach the latent
process; z is defined by the ELBO and rollout terms alone, which compute their
own encodings. The DiT then cannot reshape z into a clock or a copy of u.

No target encoder, no EMA of the encoder, no z standardization, no z-stat fit.
"""

import ast
import glob
import math
import os
import re
import warnings

import numpy as np
import torch
import torch.nn as nn

from proto_scales.ssm_dit_model.annual_ssm import (
    MONTHS_PER_YEAR, broadcast_to_monthly, diag_gaussian_kl_per_dim, to_annual)
from proto_scales.ssm_dit_model.annual_ssm_dit import AnnualSSMOutpaintingDiT

MONTHS = 12


def gru_year_reads(gru, x, monthly, window_years=0):
    """
    Run `gru` over x [B, T, F] and return one state per year [B, Y, h]
    (December if `monthly`, else every step). With `window_years` > 0, each
    year with a full lookback is a separate GRU run over just its last
    `window_years` years, so the read does not depend on how long the GRU has
    been running; earlier years keep the prefix read. Same as
    `ForcedAnnualJEPAKL._run_gru`.
    """
    T = x.shape[1]
    step = MONTHS if monthly else 1
    idx = torch.arange(step - 1, T + step - 1, step, device=x.device).clamp(max=T - 1)
    L = int(window_years) * step
    if L <= 0 or T <= L:
        h, _ = gru(x)
        return h[:, idx] if monthly else h
    full = idx >= L - 1
    h_pre, _ = gru(x[:, :L])
    ends = idx[full]
    win = x.unfold(1, L, 1)[:, ends - L + 1]          # [B, nW, F, L]
    Bsz, nW, F, _ = win.shape
    hw, _ = gru(win.permute(0, 1, 3, 2).reshape(Bsz * nW, L, F))
    hw = hw[:, -1].reshape(Bsz, nW, -1)
    return torch.cat([h_pre[:, idx[~full]], hw], dim=1)


def fit_pattern_scaling(tas_monthly, u_monthly, ridge=1e-3):
    """
    Ridge regression of annual regional tas on annual u, with an unpenalised
    intercept. Inputs are normalised monthly arrays [N, T, Dy] / [N, T, Du]
    (training split only); trailing partial years are dropped.
    Returns (W [Dy, Du], b [Dy]) as float32 tensors.
    """
    tas = np.asarray(tas_monthly, dtype=np.float64)
    u = np.asarray(u_monthly, dtype=np.float64)
    Y = tas.shape[1] // MONTHS
    tas_a = tas[:, :Y * MONTHS].reshape(tas.shape[0], Y, MONTHS, -1).mean(2).reshape(-1, tas.shape[-1])
    u_a = u[:, :Y * MONTHS].reshape(u.shape[0], Y, MONTHS, -1).mean(2).reshape(-1, u.shape[-1])
    X = np.concatenate([u_a, np.ones((u_a.shape[0], 1))], axis=1)
    reg = ridge * X.shape[0] * np.eye(X.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(X.T @ X + reg, X.T @ tas_a)        # [Du+1, Dy]
    W = torch.tensor(coef[:-1].T, dtype=torch.float32)
    b = torch.tensor(coef[-1], dtype=torch.float32)
    resid = tas_a - X @ coef
    r2 = 1.0 - resid.var(0) / tas_a.var(0).clip(1e-12)
    return W, b, r2


# ─────────────────────────────────────────────────────────────────────────────
# Latent process
# ─────────────────────────────────────────────────────────────────────────────
class ForcedAnnualELBO(nn.Module):
    """
    Annual state-space latent process, ELBO-trained on regional tas.

    Parameters
    ----------
    y_dim             : number of regions (tas channels; the field the
                        wrapper passes has 2*y_dim channels, tas first).
    u_dim             : forcing dimension (1 for GSAT).
    z_dim, rnn_hidden : latent size and GRU width.
    trans_hidden      : optional MLP correction on the transition mean.
    decay_efold_range : e-folding times in YEARS; spread at init and, in
                        linear mode, a hard clamp; in gated mode, the smooth
                        bounds of the gate.
    gated_decay       : tau_t = tau(z_t-1, u_t), as in `ForcedAnnualJEPAKL`.
    monthly_encoder   : GRU over monthly input, z read each December.
    enc_window_years  : fixed lookback per year (0 = whole window from h0=0);
                        stored as a buffer, so checkpoints carry it.
    encoder_reads_pr  : encoder also reads pr (default: tas + u only, as in
                        the original SSM).
    emission_hidden   : 0 -> linear residual res(z); > 0 -> one hidden layer.
    pr_emission       : add an annual pr head, PS_pr(u) + res_pr(z), with its
                        own frozen pattern scaling. It is not part of the
                        ELBO: it is trained by `pr_loss` (Gaussian NLL on the
                        posterior z + MSE along the rollout), on detached z by
                        default, so pr's noise does not redefine z. It gives
                        the DiT a pr baseline to generate residuals around.
    pr_emission_hidden: width of res_pr's hidden layer (0 -> linear). The
                        last layer is zero-initialised, so the head starts as
                        pure pattern scaling.
    """

    def __init__(
        self,
        y_dim,
        u_dim,
        z_dim=16,
        rnn_hidden=64,
        trans_hidden=0,
        decay_efold_range=(1.0, 50.0),
        gated_decay=False,
        gate_hidden=32,
        monthly_encoder=True,
        enc_window_years=0,
        encoder_reads_pr=False,
        emission_hidden=0,
        pr_emission=False,
        pr_emission_hidden=64,
    ):
        super().__init__()
        lo, hi = decay_efold_range
        if not (0 < lo <= hi):
            raise ValueError("decay_efold_range must satisfy 0 < low <= high (years)")
        self.y_dim = y_dim
        self.obs_dim = y_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.gated_decay = bool(gated_decay)
        self.monthly_encoder = bool(monthly_encoder)
        self.encoder_reads_pr = bool(encoder_reads_pr)

        enc_in = (2 * y_dim if self.encoder_reads_pr else y_dim) + u_dim
        self.gru = nn.GRU(enc_in, rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        if self.gated_decay:
            self.register_buffer("_log_tau_lo", torch.tensor(math.log(lo)))
            self.register_buffer("_log_tau_hi", torch.tensor(math.log(hi)))
            frac = torch.linspace(0.05, 0.95, z_dim)
            self.logit_tau = nn.Parameter(torch.log(frac / (1.0 - frac)))
            self.decay_gate = nn.Sequential(
                nn.Linear(z_dim + u_dim, gate_hidden), nn.SiLU(),
                nn.Linear(gate_hidden, z_dim))
            nn.init.zeros_(self.decay_gate[-1].weight)
            nn.init.zeros_(self.decay_gate[-1].bias)
        else:
            tau = torch.logspace(math.log10(lo), math.log10(hi), z_dim)
            d0 = torch.exp(-1.0 / tau).clamp(1e-4, 1 - 1e-4)
            self.logit_decay = nn.Parameter(torch.log(d0 / (1.0 - d0)))
            self.register_buffer("_decay_min", torch.tensor(math.exp(-1.0 / lo)))
            self.register_buffer("_decay_max", torch.tensor(math.exp(-1.0 / hi)))

        self.B = nn.Linear(u_dim, z_dim, bias=False)
        self.trans_mlp = (
            nn.Sequential(nn.Linear(z_dim + u_dim, trans_hidden), nn.SiLU(),
                          nn.Linear(trans_hidden, z_dim))
            if trans_hidden else None
        )
        if self.trans_mlp is not None:
            nn.init.zeros_(self.trans_mlp[-1].weight)
            nn.init.zeros_(self.trans_mlp[-1].bias)
        self.log_process_var = nn.Parameter(torch.zeros(z_dim))

        # Emission: frozen pattern scaling from u + learned residual from z.
        self.register_buffer("ps_weight", torch.zeros(y_dim, u_dim))
        self.register_buffer("ps_bias", torch.zeros(y_dim))
        self.register_buffer("ps_fitted", torch.tensor(False))
        self.emit_res = (
            nn.Sequential(nn.Linear(z_dim, emission_hidden), nn.SiLU(),
                          nn.Linear(emission_hidden, y_dim))
            if emission_hidden else nn.Linear(z_dim, y_dim)
        )
        self.log_obs_var = nn.Parameter(torch.zeros(y_dim))

        # Optional pr head (not in the ELBO; see `pr_loss`).
        self.pr_emission = bool(pr_emission)
        if self.pr_emission:
            self.register_buffer("ps_pr_weight", torch.zeros(y_dim, u_dim))
            self.register_buffer("ps_pr_bias", torch.zeros(y_dim))
            self.register_buffer("ps_pr_fitted", torch.tensor(False))
            self.emit_res_pr = (
                nn.Sequential(nn.Linear(z_dim, pr_emission_hidden), nn.SiLU(),
                              nn.Linear(pr_emission_hidden, y_dim))
                if pr_emission_hidden else nn.Linear(z_dim, y_dim)
            )
            last = self.emit_res_pr[-1] if pr_emission_hidden else self.emit_res_pr
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.log_obs_var_pr = nn.Parameter(torch.zeros(y_dim))

        self.register_buffer("enc_window_years", torch.tensor(int(enc_window_years)))

    # -----------------------
    # pattern scaling
    # -----------------------
    @torch.no_grad()
    def set_pattern_scaling(self, W, b):
        self.ps_weight.copy_(W)
        self.ps_bias.copy_(b)
        self.ps_fitted.fill_(True)

    def pattern_scaling(self, u):
        return u @ self.ps_weight.T + self.ps_bias

    def emit_mean(self, z, u):
        """Annual tas mean, [..., y_dim], from z and annual u."""
        return self.pattern_scaling(u) + self.emit_res(z)

    @torch.no_grad()
    def set_pr_pattern_scaling(self, W, b):
        if not self.pr_emission:
            raise ValueError("set_pr_pattern_scaling needs pr_emission=True")
        self.ps_pr_weight.copy_(W)
        self.ps_pr_bias.copy_(b)
        self.ps_pr_fitted.fill_(True)

    def emit_mean_pr(self, z, u):
        """Annual pr mean, [..., y_dim], from z and annual u (pr head only)."""
        return u @ self.ps_pr_weight.T + self.ps_pr_bias + self.emit_res_pr(z)

    def emit_mean_field(self, z, u):
        """[..., 2*y_dim]: (tas, pr) annual means, the DiT's residual baseline."""
        return torch.cat([self.emit_mean(z, u), self.emit_mean_pr(z, u)], dim=-1)

    # -----------------------
    # transition
    # -----------------------
    def _log_tau(self, z_prev=None, u_t=None):
        a = self.logit_tau
        if z_prev is not None:
            a = a + self.decay_gate(torch.cat([z_prev, u_t], dim=-1))
        return self._log_tau_lo + (self._log_tau_hi - self._log_tau_lo) * torch.sigmoid(a)

    def decay(self, z_prev=None, u_t=None):
        if self.gated_decay:
            return torch.exp(-torch.exp(-self._log_tau(z_prev, u_t)))
        return torch.sigmoid(self.logit_decay).clamp(self._decay_min, self._decay_max)

    def efolding_years(self):
        with torch.no_grad():
            d = self.decay().clamp(1e-6, 1 - 1e-9)
            return (-1.0 / torch.log(d)).cpu()

    @torch.no_grad()
    def gated_efolding_years(self, z0, u_fut):
        if not self.gated_decay:
            raise ValueError("gated_efolding_years needs gated_decay=True")
        z = z0
        taus = []
        for k in range(u_fut.shape[1]):
            taus.append(torch.exp(self._log_tau(z, u_fut[:, k])))
            z = self.transition_mean(z, u_fut[:, k])
        return torch.stack(taus, dim=1).cpu()

    def transition_mean(self, z_prev, u_t):
        d = self.decay(z_prev, u_t) if self.gated_decay else self.decay()
        mu = d * z_prev + self.B(u_t)
        if self.trans_mlp is not None:
            mu = mu + self.trans_mlp(torch.cat([z_prev, u_t], dim=-1))
        return mu

    def transition(self, z_prev, u_t):
        mu = self.transition_mean(z_prev, u_t)
        logvar = torch.clamp(self.log_process_var, -12.0, 6.0)
        return mu, logvar.unsqueeze(0).expand(z_prev.shape[0], -1)

    @staticmethod
    def sample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    # -----------------------
    # encoder
    # -----------------------
    def encoder_inputs(self, y_monthly, u_monthly):
        """
        From the monthly field [B, T, 2*y_dim] (tas then pr, as the wrapper
        passes it) and forcing: tas (+ pr if `encoder_reads_pr`) and u,
        monthly with `monthly_encoder`, annual means otherwise.
        """
        y = y_monthly if self.encoder_reads_pr else y_monthly[..., :self.y_dim]
        if self.monthly_encoder:
            return y, u_monthly
        return to_annual(y), to_annual(u_monthly)

    def _encode_dist(self, ye, ue):
        h = gru_year_reads(self.gru, torch.cat([ye, ue], dim=-1),
                           self.monthly_encoder, int(self.enc_window_years))
        mu, logvar = torch.chunk(self.q_head(h), 2, dim=-1)
        return mu, torch.clamp(logvar, -12.0, 6.0)

    def posterior(self, ye, ue):
        """Posterior mean over the window, annual. Returns ([B,Y,z], [B,z])."""
        mu, _ = self._encode_dist(ye, ue)
        return mu, mu[:, -1]

    def roll_forward(self, z0, u_fut, stochastic=False):
        """Iterate the transition over annual `u_fut`. Returns [B, Y_fut, z]."""
        z = z0
        out = []
        for k in range(u_fut.shape[1]):
            if stochastic:
                z = self.sample(*self.transition(z, u_fut[:, k]))
            else:
                z = self.transition_mean(z, u_fut[:, k])
            out.append(z)
        return torch.stack(out, dim=1)

    # -----------------------
    # objectives
    # -----------------------
    def elbo(self, ye, ue, tas_ann, u_ann, kl_free_bits=0.0, return_z=False):
        """
        Returns (nll, kl, kl_per_dim). nll and kl are summed over regions /
        z-dims and years, averaged over the batch, and both divided by
        (years x regions), so their ratio is the summed ELBO's. kl_per_dim
        [z] is the mean KL per year per dim in nats (diagnostic).
        `kl_free_bits` clamps each step's total KL from below, as in the
        original monthly SSM (default 0). With `return_z`, the posterior
        sample z_s [B, T, z] is appended (for the pr head, no second encode).
        """
        mu_q, logvar_q = self._encode_dist(ye, ue)
        B, T, _ = mu_q.shape
        log_obs_var = torch.clamp(self.log_obs_var, -12.0, 6.0)
        inv_var = torch.exp(-log_obs_var)

        z_s = self.sample(mu_q, logvar_q)                         # [B, T, z]
        resid = tas_ann - self.emit_mean(z_s, u_ann)
        nll = 0.5 * (resid ** 2 * inv_var + log_obs_var
                     + math.log(2.0 * math.pi)).sum(dim=(1, 2))   # [B]

        mu_p = torch.cat([torch.zeros_like(mu_q[:, :1]),
                          self.transition_mean(z_s[:, :-1], u_ann[:, 1:])], dim=1)
        lv = torch.clamp(self.log_process_var, -12.0, 6.0).view(1, 1, -1)
        logvar_p = torch.cat([torch.zeros_like(logvar_q[:, :1]),
                              lv.expand(B, T - 1, -1)], dim=1)
        kl_dim = diag_gaussian_kl_per_dim(mu_q, logvar_q, mu_p, logvar_p)  # [B, T, z]
        kl_step = kl_dim.sum(dim=-1)
        if kl_free_bits > 0:
            kl_step = torch.clamp(kl_step, min=kl_free_bits)
        kl = kl_step.sum(dim=1)

        norm = T * self.y_dim
        out = (nll.mean() / norm, kl.mean() / norm, kl_dim.detach().mean(dim=(0, 1)))
        return out + (z_s,) if return_z else out

    def rollout_loss(self, ye_ctx, ue_ctx, tas_fut_ann, u_fut_ann, return_z=False,
                     obs_var_scaled=False):
        """
        Error of the emission mean along the mean rollout from z_last.

        Default: plain MSE. With `obs_var_scaled`, 0.5 * r^2 / sigma^2 with the
        ELBO's learned per-region observation variance (gradient stopped):
        the rollout then pulls on the mean exactly as hard as the NLL does
        (NLL gradient r / sigma^2 vs MSE gradient 2r, so at weight 1 a plain
        MSE is 1 / (2 sigma^2) times weaker; with annual means sigma^2 is
        small), and each region is weighted by its own noise level, so
        low-noise regions such as SOO are not drowned out by noisy land.
        """
        _, z_last = self.posterior(ye_ctx, ue_ctx)
        z_traj = self.roll_forward(z_last, u_fut_ann)
        r2 = (self.emit_mean(z_traj, u_fut_ann) - tas_fut_ann) ** 2
        if obs_var_scaled:
            inv_var = torch.exp(-torch.clamp(self.log_obs_var, -12.0, 6.0)).detach()
            r2 = 0.5 * r2 * inv_var
        loss = r2.mean()
        return (loss, z_traj) if return_z else loss

    def pr_loss(self, z_post, u_ann, pr_ann, z_roll, u_fut_ann, pr_fut_ann, detach_z=True,
                obs_var_scaled=False):
        """
        pr head objective: Gaussian NLL of annual pr on the posterior z
        (per year per region, same normalisation as the tas NLL) plus the MSE
        of the pr mean along the forcing-only rollout, the z the DiT baseline
        uses in the future half. Returns (nll, roll_mse).

        Gaussian, not sinh-arcsinh: the head only supplies the mean the DiT
        generates residuals around; the shape of the pr distribution
        (skew, tails) is the DiT's job, and annual means are close to
        Gaussian anyway.
        """
        if detach_z:
            z_post, z_roll = z_post.detach(), z_roll.detach()
        lv = torch.clamp(self.log_obs_var_pr, -12.0, 6.0)
        r = pr_ann - self.emit_mean_pr(z_post, u_ann)
        nll = 0.5 * (r ** 2 * torch.exp(-lv) + lv + math.log(2.0 * math.pi)).mean()
        r2 = (self.emit_mean_pr(z_roll, u_fut_ann) - pr_fut_ann) ** 2
        if obs_var_scaled:
            r2 = 0.5 * r2 * torch.exp(-lv).detach()
        return nll, r2.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AnnualELBOOutpaintingDiT(AnnualSSMOutpaintingDiT):
    """
    `AnnualSSMOutpaintingDiT` with a `ForcedAnnualELBO` latent process.
    Overrides `latents` (detach), `ssm_aux_loss`, `loss` and `sample_blocks`;
    the DiT, conditioning and flow-matching loss are inherited.

    `kl_weight` is beta; the training loop scales it by `kl_beta_scale`
    (warm-up, set via `set_kl_beta_scale`).

    Residual DiT (`dit_residual=True`, needs the pr head)
    ------------------------------------------------------
    The DiT generates r = field - b(z, u), where b is the latent process's
    annual (tas, pr) mean `emit_mean_field`, evaluated along the same z the
    DiT is conditioned on (posterior over the context, mean rollout over the
    future) and repeated over each year's months. Its context is residual
    too. The level of the forecast then comes from the SSM, which is trained
    with a rollout loss, and a free-running DiT has no level to drift on:
    without this the DiT took the level from its own generated context and
    drifted region by region (eval_elbo_dit_bias: teacher-forced fine,
    free-running +0.2 K in SOO, -0.5 K in GIC). The baseline carries no DiT
    gradient. `sample_blocks` adds it back, so callers get the full field.

    Encoder context (`enc_context_len`)
    -----------------------------------
    The latent process reads `enc_context_len` months of history (training
    windows supply that much context), the DiT still attends over the last
    `context_len`. z_last, and with it every rollout, then starts from a
    read long enough to place the slow (ocean) dims; 10 yr from h0 = 0 is
    not. `loss` crops z, field and u to the DiT frame after encoding;
    `sample_blocks` encodes the last `enc_context_len` months it is given.

    pr head
    -------
    With `ssm.pr_emission`, `pr_weight * (pr_nll + pr_rollout)` is added to
    the aux loss. On detached z by default (`pr_detach_z`), so it trains the
    head only.
    """

    def __init__(self, ssm, y_dim, u_dim, nll_weight=1.0, kl_weight=5.0,
                 kl_free_bits=0.0, rollout_weight=1.0, detach_latents=True,
                 dit_residual=False, pr_weight=1.0, pr_detach_z=True,
                 enc_context_len=None, rollout_obs_var=False,
                 **kwargs):
        super().__init__(
            ssm=ssm, y_dim=y_dim, u_dim=u_dim,
            ssm_elbo_weight=0.0, ssm_kl_weight=0.0, ssm_kl_free_bits=0.0,
            ssm_rollout_weight=0.0, **kwargs,
        )
        self.nll_weight = float(nll_weight)
        self.kl_weight = float(kl_weight)
        self.kl_free_bits = float(kl_free_bits)
        self.rollout_weight = float(rollout_weight)
        self.detach_latents = bool(detach_latents)
        self.kl_beta_scale = 1.0
        self.dit_residual = bool(dit_residual)
        self.pr_detach_z = bool(pr_detach_z)
        self.rollout_obs_var = bool(rollout_obs_var)
        enc_context_len = self.context_len if enc_context_len is None else int(enc_context_len)
        if enc_context_len % MONTHS_PER_YEAR or enc_context_len < self.context_len:
            raise ValueError(
                f"enc_context_len must be whole years and >= context_len "
                f"({self.context_len}); got {enc_context_len}")
        self.enc_context_len = enc_context_len
        self.pr_weight = float(pr_weight) if ssm.pr_emission else 0.0
        if self.dit_residual and not ssm.pr_emission:
            raise ValueError("dit_residual needs a pr baseline: build the latent "
                             "process with pr_emission=True")
        frozen = self.ssm_encoder.frozen
        if frozen and self.pr_weight > 0:
            # A frozen latent process can still get a new pr head on top.
            for m in (ssm.emit_res_pr,):
                for prm in m.parameters():
                    prm.requires_grad = True
            ssm.log_obs_var_pr.requires_grad = True
            if not self.pr_detach_z:
                raise ValueError("pr_detach_z=False makes no sense with a frozen latent process")
        self.use_ssm_aux = (self.pr_weight > 0) or (not frozen and (
            self.nll_weight > 0 or self.rollout_weight > 0))

    def set_kl_beta_scale(self, s):
        self.kl_beta_scale = float(s)

    def _dit_offset(self, n_hist):
        """Months of history before the DiT window starts (whole years)."""
        off = int(n_hist) - self.context_len
        if off < 0 or off % MONTHS_PER_YEAR:
            raise ValueError(
                f"history of {n_hist} months: need >= context_len "
                f"({self.context_len}) and a whole number of years beyond it")
        return off

    def latents(self, tas_ctx, pr_ctx, u_ctx, u_fut):
        z = self.ssm_encoder(tas_ctx, pr_ctx, u_ctx, u_fut, stochastic=self.stochastic_z)
        return z.detach() if self.detach_latents else z

    # ------------------------------------------------------------------
    # residual baseline
    # ------------------------------------------------------------------
    @torch.no_grad()
    def field_baseline(self, z_all, u_all):
        """
        Monthly baseline [B, T, 2*y_dim] for z_all [B, T, z] (annual latents
        repeated over months, as `latents` returns them) and monthly u_all.
        Years are 12-month chunks from the window start, the same chunks z
        is built on. Zero if not `dit_residual`.
        """
        B, T, _ = z_all.shape
        if not self.dit_residual:
            return z_all.new_zeros(B, T, self.in_channels)
        u_ann_m = broadcast_to_monthly(to_annual(u_all), T)
        return self.ssm_encoder.ssm.emit_mean_field(z_all, u_ann_m)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def loss(self, tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut,
             block_idx=(0,), start_month=0, t=None, return_parts=False):
        """`AnnualSSMOutpaintingDiT.loss`, on the residual field if `dit_residual`."""
        n_fut = u_fut.shape[1]
        z_all = self.latents(tas_ctx, pr_ctx, u_ctx, u_fut)
        field_all = torch.cat([torch.cat([tas_ctx, tas_fut], dim=1),
                               torch.cat([pr_ctx, pr_fut], dim=1)], dim=-1)
        u_all = torch.cat([u_ctx, u_fut], dim=1)
        # The encoder read the whole (enc_context_len) context; the DiT frame
        # starts context_len months before the future. Whole years, so the
        # annual chunks of the baseline stay aligned with z's.
        off = self._dit_offset(tas_ctx.shape[1])
        if off:
            z_all, field_all, u_all = z_all[:, off:], field_all[:, off:], u_all[:, off:]
            start_month = int(start_month) + off
        if self.dit_residual:
            field_all = field_all - self.field_baseline(z_all.detach(), u_all)

        if isinstance(block_idx, torch.Tensor):
            block_idx = block_idx.tolist()
        elif isinstance(block_idx, int):
            block_idx = [block_idx]

        flow_terms, per_block = [], {}
        for bi in block_idx:
            w0, bs, nb = self.block_bounds(bi, n_fut)
            win = slice(w0, bs + nb)
            d = self.flow_matching_loss(
                z_win=z_all[:, win], u_win=u_all[:, win], field_win=field_all[:, win],
                n_block=nb, start_month=(int(start_month) + w0) % MONTHS_PER_YEAR, t=t)
            flow_terms.append(d)
            per_block[f"flow_blk{int(bi)}"] = d.detach()
        flow_loss = torch.stack(flow_terms).mean()

        parts = {"flow_matching": flow_loss.detach()}
        if len(per_block) > 1:
            parts.update(per_block)
        total = flow_loss
        if self.use_ssm_aux:
            aux, aux_parts = self.ssm_aux_loss(
                tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut)
            total = total + aux
            parts.update(aux_parts)
            parts["ssm_aux_weighted"] = aux.detach()
        return (total, parts) if return_parts else total

    def ssm_aux_loss(self, tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut):
        ssm = self.ssm_encoder.ssm
        frozen = self.ssm_encoder.frozen
        ye_ctx, ue_ctx = ssm.encoder_inputs(torch.cat([tas_ctx, pr_ctx], dim=-1), u_ctx)
        ye_fut, ue_fut = ssm.encoder_inputs(torch.cat([tas_fut, pr_fut], dim=-1), u_fut)
        u_ctx_ann, u_fut_ann = to_annual(u_ctx), to_annual(u_fut)
        u_all_ann = torch.cat([u_ctx_ann, u_fut_ann], dim=1)
        tas_fut_ann = to_annual(tas_fut)
        ye_all = torch.cat([ye_ctx, ye_fut], dim=1)
        ue_all = torch.cat([ue_ctx, ue_fut], dim=1)

        parts = {}
        total = torch.zeros((), device=tas_ctx.device)
        z_post = z_roll = None
        if self.nll_weight > 0 and not frozen:
            nll, kl, kl_dim, z_post = ssm.elbo(
                ye_all, ue_all,
                torch.cat([to_annual(tas_ctx), tas_fut_ann], dim=1), u_all_ann,
                kl_free_bits=self.kl_free_bits, return_z=True)
            beta = self.kl_weight * self.kl_beta_scale
            total = total + self.nll_weight * nll + beta * kl
            parts["ssm_nll"] = nll.detach()
            parts["ssm_kl"] = kl.detach()
            parts["kl_active_dims"] = (kl_dim > 0.01).sum().float()
        if self.rollout_weight > 0 and not frozen:
            roll, z_roll = ssm.rollout_loss(ye_ctx, ue_ctx, tas_fut_ann, u_fut_ann,
                                            return_z=True,
                                            obs_var_scaled=self.rollout_obs_var)
            total = total + self.rollout_weight * roll
            parts["ssm_rollout"] = roll.detach()

        if self.pr_weight > 0:
            # Reuse the encodings above; recompute only what was skipped.
            if z_post is None:
                with torch.set_grad_enabled(not self.pr_detach_z):
                    z_post = ssm.posterior(ye_all, ue_all)[0]
            if z_roll is None:
                with torch.set_grad_enabled(not self.pr_detach_z):
                    z_roll = ssm.roll_forward(ssm.posterior(ye_ctx, ue_ctx)[1], u_fut_ann)
            pr_nll, pr_roll = ssm.pr_loss(
                z_post, u_all_ann, to_annual(torch.cat([pr_ctx, pr_fut], dim=1)),
                z_roll, u_fut_ann, to_annual(pr_fut), detach_z=self.pr_detach_z,
                obs_var_scaled=self.rollout_obs_var)
            total = total + self.pr_weight * (pr_nll + pr_roll)
            parts["pr_nll"] = pr_nll.detach()
            parts["pr_rollout"] = pr_roll.detach()
        return total, parts

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_blocks(self, tas_ctx, pr_ctx, u_ctx, u_fut, start_month=0,
                      n_steps=None, generator=None, z_all=None,
                      tas_fut_true=None, pr_fut_true=None, return_baseline=False):
        """
        `AnnualSSMOutpaintingDiT.sample_blocks`, plus:
          - residual mode: the DiT works on field - baseline, and the
            baseline is added back to the output;
          - the history may be longer than `context_len` (whole years beyond
            it): the latent process reads its last `enc_context_len` months,
            the DiT the last `context_len`;
          - `z_all` [B, context_len + n_fut, z] monthly (DiT frame): condition
            on this latent trajectory instead (e.g. a full-history read);
          - `tas_fut_true`/`pr_fut_true`: teacher forcing, every block's
            context is the true field instead of the generated one
            (diagnostic only);
          - `return_baseline`: also return the baseline's (tas, pr) over the
            future (zeros if not residual).
        Returns (tas, pr[, tas_base, pr_base]), each [B, len(u_fut), y_dim].
        """
        H_total = u_fut.shape[1]
        n_ctx = self.context_len
        off = self._dit_offset(tas_ctx.shape[1])
        if z_all is None:
            if tas_ctx.shape[1] < self.enc_context_len:
                warnings.warn(
                    f"sample_blocks got {tas_ctx.shape[1]} months of history; the latent "
                    f"process was trained on {self.enc_context_len}. Pass more history "
                    f"(only the last {self.context_len} months reach the DiT).")
            n_enc = min(tas_ctx.shape[1], self.enc_context_len)
            n_enc -= n_enc % MONTHS_PER_YEAR
            z_all = self.latents(tas_ctx[:, -n_enc:], pr_ctx[:, -n_enc:],
                                 u_ctx[:, -n_enc:], u_fut)[:, n_enc - n_ctx:]
        tas_ctx, pr_ctx, u_ctx = tas_ctx[:, off:], pr_ctx[:, off:], u_ctx[:, off:]
        u_all = torch.cat([u_ctx, u_fut], dim=1)
        base = self.field_baseline(z_all, u_all)
        hist = torch.cat([tas_ctx, pr_ctx], dim=-1) - base[:, :n_ctx]
        tf = tas_fut_true is not None
        if tf:
            true_fut = torch.cat([tas_fut_true, pr_fut_true], dim=-1) - base[:, n_ctx:]
            hist_true = torch.cat([hist, true_fut], dim=1)

        out, done = [], 0
        while done < H_total:
            w0, bs, nb = self.block_bounds(done // self.block_len, H_total)
            win = slice(w0, bs + nb)
            field_ctx = (hist_true if tf else hist)[:, w0:bs]
            cond = self.window_conditioning(
                z_win=z_all[:, win], u_win=u_all[:, win], field_ctx=field_ctx,
                n_block=nb, start_month=(int(start_month) + w0) % MONTHS_PER_YEAR)
            blk = self._sample_window(cond, field_ctx, nb, n_steps, generator)
            hist = torch.cat([hist, blk], dim=1)
            out.append(blk)
            done += nb

        fut_base = base[:, n_ctx:n_ctx + H_total]
        fut = torch.cat(out, dim=1) + fut_base
        res = (fut[..., :self.y_dim], fut[..., self.y_dim:])
        if return_baseline:
            res = res + (fut_base[..., :self.y_dim], fut_base[..., self.y_dim:])
        return res


_SSM_KEYS = ("z_dim", "rnn_hidden", "trans_hidden", "decay_efold_range",
             "gated_decay", "gate_hidden", "monthly_encoder", "enc_window_years",
             "encoder_reads_pr", "emission_hidden", "pr_emission",
             "pr_emission_hidden")


def load_matching(module, state_dict, tag="warm start"):
    """
    Non-strict load that also skips tensors whose shape differs (e.g. a
    linear emission into an MLP one), which `load_state_dict(strict=False)`
    would raise on. Returns (loaded, missing, unexpected, mismatched) key lists.
    """
    own = module.state_dict()
    keep = {k: v for k, v in state_dict.items() if k in own and own[k].shape == v.shape}
    mismatched = [k for k in state_dict if k in own and own[k].shape != state_dict[k].shape]
    unexpected = [k for k in state_dict if k not in own]
    missing = [k for k in own if k not in keep]
    module.load_state_dict(keep, strict=False)
    print(f"[{tag}] loaded {len(keep)}/{len(own)} | missing {len(missing)} | "
          f"shape-mismatched {len(mismatched)} | unexpected {len(unexpected)}")
    if missing:
        print(f"[{tag}]   missing: {missing}")
    if mismatched:
        print(f"[{tag}]   shape-mismatched (left at init): {mismatched}")
    return list(keep), missing, unexpected, mismatched


def ssm_state_from_file(path, key="ema"):
    """
    Latent-process weights from any of: a bare `ForcedAnnualELBO` state dict,
    a full-model `model_out`, or an epoch checkpoint {"model", "ema"} (`key`
    picks which). Full-model keys are stripped of `ssm_encoder.ssm.`.
    """
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and ("model" in sd or "ema" in sd):
        sd = sd[key] if key in sd else sd["model"]
    if any(k.startswith(_PFX) for k in sd):
        sd = {k[len(_PFX):]: v for k, v in sd.items() if k.startswith(_PFX)}
    return sd


def build_model(y_dim, u_dim, device="cpu", ssm_weights=None, **kwargs):
    """
    Construct `AnnualELBOOutpaintingDiT`. Latent-process kwargs (`_SSM_KEYS`)
    go to `ForcedAnnualELBO`, the rest to the DiT. `ssm_weights` optionally
    warm-starts the latent process (non-strict, shape mismatches skipped; a
    full-model checkpoint works too).
    """
    ssm_kw = {k: kwargs.pop(k) for k in _SSM_KEYS if k in kwargs}
    if "decay_efold_range" in ssm_kw:
        ssm_kw["decay_efold_range"] = tuple(ssm_kw["decay_efold_range"])
    ssm = ForcedAnnualELBO(y_dim=y_dim, u_dim=u_dim, **ssm_kw)
    if ssm_weights is not None:
        load_matching(ssm, ssm_state_from_file(ssm_weights), tag="build_model ssm")
    model = AnnualELBOOutpaintingDiT(ssm=ssm, y_dim=y_dim, u_dim=u_dim, **kwargs)
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
_CKPT_EPOCH_RE = re.compile(r"annual_elbo_dit_epoch(\d+)\.pt$")
_PFX = "ssm_encoder.ssm."

# config.txt (CLI arg names) -> build_model kwargs
_CFG_MAP = {
    "z_dim": "z_dim", "rnn_hidden": "rnn_hidden", "trans_hidden": "trans_hidden",
    "decay_efold": "decay_efold_range", "gated_decay": "gated_decay",
    "gate_hidden": "gate_hidden", "monthly_encoder": "monthly_encoder",
    "context_len": "context_len", "block_len": "block_len",
    "cond_dim": "cond_dim", "hidden": "hidden", "depth": "depth", "heads": "heads",
    "n_flow_steps": "n_flow_steps", "timescales": "timescales_years",
    "field_memory_rank": "field_memory_rank",
    "kl_weight": "kl_weight", "kl_free_bits": "kl_free_bits",
    "rollout_weight": "rollout_weight",
    "dit_residual": "dit_residual", "pr_weight": "pr_weight",
    "pr_detach_z": "pr_detach_z",
    "enc_context_len": "enc_context_len", "rollout_obs_var": "rollout_obs_var",
}


def read_run_config(run_dir):
    """Parse `config.txt` (key=value lines) into a dict; '' if missing."""
    path = os.path.join(run_dir, "config.txt")
    cfg = {}
    if not os.path.exists(path):
        return cfg
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.rstrip("\n").split("=", 1)
            try:
                cfg[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                cfg[k] = v
    return cfg


def model_kwargs_from_config(cfg):
    kw = {dst: cfg[src] for src, dst in _CFG_MAP.items() if src in cfg}
    if "fixed_timescales" in cfg:
        kw["learnable_timescales"] = not cfg["fixed_timescales"]
    if "timescales_years" in kw:
        kw["timescales_years"] = tuple(kw["timescales_years"])
    return kw


def ssm_kwargs_from_state_dict(state_dict, y_dim, u_dim):
    """Emission width, encoder inputs and lookback, read off the weights."""
    kw = {}
    if _PFX + "emit_res.0.weight" in state_dict:
        kw["emission_hidden"] = int(state_dict[_PFX + "emit_res.0.weight"].shape[0])
    else:
        kw["emission_hidden"] = 0
    enc_in = int(state_dict[_PFX + "gru.weight_ih_l0"].shape[1])
    kw["encoder_reads_pr"] = enc_in == 2 * y_dim + u_dim
    kw["gated_decay"] = (_PFX + "logit_tau") in state_dict
    if _PFX + "enc_window_years" in state_dict:
        kw["enc_window_years"] = int(state_dict[_PFX + "enc_window_years"])
    if _PFX + "trans_mlp.0.weight" in state_dict:
        kw["trans_hidden"] = int(state_dict[_PFX + "trans_mlp.0.weight"].shape[0])
    if _PFX + "decay_gate.0.weight" in state_dict:
        kw["gate_hidden"] = int(state_dict[_PFX + "decay_gate.0.weight"].shape[0])
    kw["pr_emission"] = (_PFX + "ps_pr_weight") in state_dict
    if kw["pr_emission"]:
        kw["pr_emission_hidden"] = (
            int(state_dict[_PFX + "emit_res_pr.0.weight"].shape[0])
            if _PFX + "emit_res_pr.0.weight" in state_dict else 0)
    kw["z_dim"] = int(state_dict[_PFX + "log_process_var"].shape[0])
    kw["rnn_hidden"] = int(state_dict[_PFX + "gru.weight_hh_l0"].shape[1])
    return kw


def load_elbo_dit_annual(run_dir, y_dim, u_dim=1, device="gpu", ema=True,
                         checkpoint=None, prefer_zstats=True, **model_kwargs):
    """
    Load a trained `AnnualELBOOutpaintingDiT` plus its tas/pr/u scalers from a
    `train_annual_elbo_dit.py` run directory. Same signature and checkpoint
    selection as `annual_jepa_kl_dit.load_jepa_dit_annual`:

    `checkpoint=None`/"latest": highest epoch in `run_dir/checkpoints/`, else
    `model_out`; "model_out" or an epoch number explicitly. `ema` picks the
    weights in an epoch checkpoint (`model_out` is already the EMA shadow).
    `prefer_zstats` is accepted for signature compatibility and ignored (this
    model has no z stats).

    Architecture comes from, in increasing priority: `run_dir/config.txt`,
    the weights themselves (z_dim, widths, emission, gate, encoder inputs,
    lookback), and explicit `model_kwargs`.

    Returns (model, tas_scaler, pr_scaler, u_scaler), model in eval() mode.
    """
    from proto_scales.ssm_model.scales_ssm import StandardScaler

    if device in ("gpu", "cuda"):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("[load_elbo_dit_annual] no GPU visible, falling back to CPU")
            device = "cpu"

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    epochs = {}
    for f in glob.glob(os.path.join(ckpt_dir, "annual_elbo_dit_epoch*.pt")):
        m = _CKPT_EPOCH_RE.search(f)
        if m:
            epochs[int(m.group(1))] = f
    model_out = os.path.join(run_dir, "model_out")

    if checkpoint is None or checkpoint == "latest":
        if checkpoint == "latest" and not epochs:
            raise FileNotFoundError(f"no epoch checkpoints in {ckpt_dir}")
        checkpoint = max(epochs) if epochs else "model_out"
    if checkpoint == "model_out":
        if not ema:
            raise ValueError(f"{model_out} is already the EMA shadow; pick an epoch "
                             f"checkpoint for raw weights")
        if not os.path.exists(model_out):
            raise FileNotFoundError(f"{model_out} doesn't exist and there are no "
                                    f"epoch checkpoints in {ckpt_dir}")
        path = model_out
    else:
        if int(checkpoint) not in epochs:
            raise FileNotFoundError(
                f"no epoch {checkpoint} in {ckpt_dir}; available: {sorted(epochs)}")
        path = epochs[int(checkpoint)]

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
        which = "ema" if ema else "model"
        state_dict = ckpt[which]
    else:
        state_dict, which = ckpt, "ema"
    print(f"[load_elbo_dit_annual] loaded {which} weights from {path}")

    kw = model_kwargs_from_config(read_run_config(run_dir))
    kw.update(model_kwargs)
    # Architecture read off the weights wins over config.txt and explicit
    # kwargs: a contradicting value could never load (strict load below).
    for k, v in ssm_kwargs_from_state_dict(state_dict, y_dim, u_dim).items():
        if k in model_kwargs and model_kwargs[k] != v:
            print(f"[load_elbo_dit_annual] ignoring {k}={model_kwargs[k]!r}: "
                  f"checkpoint has {k}={v!r}")
        kw[k] = v
    model = build_model(y_dim, u_dim, device=device, **kw)
    model.load_state_dict(state_dict)
    model.eval()

    tas_scaler = StandardScaler.from_file(os.path.join(run_dir, "y_scaler.out"))
    pr_scaler = StandardScaler.from_file(os.path.join(run_dir, "pr_scaler.out"))
    u_scaler = StandardScaler.from_file(os.path.join(run_dir, "u_scaler.out"))
    return model, tas_scaler, pr_scaler, u_scaler
