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
from proto_scales.ssm_dit_model.memory_kernel import MultiTimescaleMemory

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
    reservoir_k       : > 0 adds the forcing-only pathway of the original
                        monthly SSM, annual here: a GRU over annual u
                        (`u_rnn_hidden`) and `reservoir_k` slow reservoirs per
                        region, s <- s + alpha (target(h_u) - s), with
                        timescales in `reservoir_tau_range` years (log-spaced
                        init, learned within the bounds). Both emissions
                        (tas, pr) read [z, h_u, s]. The state starts at the
                        window start (h_u = 0, s at the equilibrium of the
                        first year's target) and is a deterministic function
                        of the forcing, so the forced slow response does not
                        have to be read off noisy tas by the encoder.
    z_forced          : keep B u_t in the transition (the filter-bank z).
                        With reservoirs, off is the intended setting: z is
                        then the unforced state (damped modes, mean rollout
                        decays to 0) and cannot duplicate the forced response.
                        `trans_hidden` > 0 adds an MLP(z, u) on top, the
                        original SSM's state-dependent transition.
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
        reservoir_k=0,
        u_rnn_hidden=32,
        reservoir_tau_range=(2.0, 1000.0),
        z_forced=True,
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

        self.z_forced = bool(z_forced)
        self.B = nn.Linear(u_dim, z_dim, bias=False) if self.z_forced else None
        self.trans_mlp = (
            nn.Sequential(nn.Linear(z_dim + u_dim, trans_hidden), nn.SiLU(),
                          nn.Linear(trans_hidden, z_dim))
            if trans_hidden else None
        )
        if self.trans_mlp is not None:
            nn.init.zeros_(self.trans_mlp[-1].weight)
            nn.init.zeros_(self.trans_mlp[-1].bias)
        self.log_process_var = nn.Parameter(torch.zeros(z_dim))

        # Forcing-only pathway (u-GRU + per-region reservoirs).
        self.reservoir_k = int(reservoir_k)
        feat_dim = 0
        if self.reservoir_k > 0:
            t_lo, t_hi = reservoir_tau_range
            if not (0 < t_lo <= t_hi):
                raise ValueError("reservoir_tau_range must satisfy 0 < low <= high (years)")
            self.u_gru = nn.GRU(u_dim, u_rnn_hidden, batch_first=True)
            self.register_buffer("_res_log_tau_lo", torch.tensor(math.log(t_lo)))
            self.register_buffer("_res_log_tau_hi", torch.tensor(math.log(t_hi)))
            frac = torch.linspace(0.05, 0.95, self.reservoir_k)
            self.res_logit_tau = nn.Parameter(
                torch.log(frac / (1.0 - frac)).repeat(y_dim, 1))       # [R, K]
            self.res_target = nn.Linear(u_rnn_hidden, y_dim * self.reservoir_k)
            self.n_spin = 200
            feat_dim = u_rnn_hidden + y_dim * self.reservoir_k
        self.feat_dim = feat_dim
        emit_in = z_dim + feat_dim

        # Emission: frozen pattern scaling from u + learned residual from z.
        self.register_buffer("ps_weight", torch.zeros(y_dim, u_dim))
        self.register_buffer("ps_bias", torch.zeros(y_dim))
        self.register_buffer("ps_fitted", torch.tensor(False))
        self.emit_res = (
            nn.Sequential(nn.Linear(emit_in, emission_hidden), nn.SiLU(),
                          nn.Linear(emission_hidden, y_dim))
            if emission_hidden else nn.Linear(emit_in, y_dim)
        )
        self.log_obs_var = nn.Parameter(torch.zeros(y_dim))

        # Optional pr head (not in the ELBO; see `pr_loss`).
        self.pr_emission = bool(pr_emission)
        if self.pr_emission:
            self.register_buffer("ps_pr_weight", torch.zeros(y_dim, u_dim))
            self.register_buffer("ps_pr_bias", torch.zeros(y_dim))
            self.register_buffer("ps_pr_fitted", torch.tensor(False))
            self.emit_res_pr = (
                nn.Sequential(nn.Linear(emit_in, pr_emission_hidden), nn.SiLU(),
                              nn.Linear(pr_emission_hidden, y_dim))
                if pr_emission_hidden else nn.Linear(emit_in, y_dim)
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

    # -----------------------
    # forcing-only pathway
    # -----------------------
    def reservoir_efolding_years(self):
        """[R, K] reservoir e-folding times in years."""
        with torch.no_grad():
            return torch.exp(self._res_log_tau()).cpu()

    def _res_log_tau(self):
        return self._res_log_tau_lo + (self._res_log_tau_hi - self._res_log_tau_lo) \
            * torch.sigmoid(self.res_logit_tau)

    def forcing_features(self, u_ann):
        """
        [B, Y, feat_dim] = [h_u, s] for annual u [B, Y, u_dim], run from the
        start of `u_ann`. None without reservoirs. Year t's features include
        year t's forcing. Slice the result to match whatever z covers.

        Start state: equilibrium with the first year's forcing, for both
        parts. The u-GRU is spun up on that value held for `n_spin` (200) years
        and continues from there; the reservoirs start at the target of the
        spun-up state. Constant forcing therefore gives constant features.
        Starting the GRU from 0 instead would put its warm-up transient into
        the targets, and the slowest reservoirs would carry it for centuries.
        Same assumption as the original SSM (equilibrium at the window
        start), applied consistently in training and at inference.
        """
        if self.reservoir_k == 0:
            return None
        B, Y, _ = u_ann.shape
        R, K = self.y_dim, self.reservoir_k
        _, h_eq = self.u_gru(u_ann[:, :1].expand(-1, self.n_spin, -1))
        h, _ = self.u_gru(u_ann, h_eq)                              # [B, Y, H]
        target = self.res_target(h).view(B, Y, R, K)
        alpha = 1.0 - torch.exp(-torch.exp(-self._res_log_tau()))   # [R, K]
        s = self.res_target(h_eq[0]).view(B, R, K)
        out = []
        for t in range(Y):
            s = s + alpha * (target[:, t] - s)
            out.append(s)
        s_all = torch.stack(out, dim=1).reshape(B, Y, R * K)
        return torch.cat([h, s_all], dim=-1)

    def _emit_in(self, z, f):
        if self.reservoir_k == 0:
            return z
        if f is None:
            raise ValueError("this latent process has reservoirs: pass the forcing "
                             "features (`forcing_features`) aligned with z")
        return torch.cat([z, f], dim=-1)

    def emit_mean(self, z, u, f=None):
        """Annual tas mean, [..., y_dim], from z, annual u and forcing features f."""
        return self.pattern_scaling(u) + self.emit_res(self._emit_in(z, f))

    @torch.no_grad()
    def set_pr_pattern_scaling(self, W, b):
        if not self.pr_emission:
            raise ValueError("set_pr_pattern_scaling needs pr_emission=True")
        self.ps_pr_weight.copy_(W)
        self.ps_pr_bias.copy_(b)
        self.ps_pr_fitted.fill_(True)

    def emit_mean_pr(self, z, u, f=None):
        """Annual pr mean, [..., y_dim], from z, annual u, features f (pr head only)."""
        return u @ self.ps_pr_weight.T + self.ps_pr_bias + self.emit_res_pr(self._emit_in(z, f))

    def emit_mean_field(self, z, u, f=None):
        """
        [..., 2*y_dim] (tas, pr) annual means with the pr head, [..., y_dim]
        (tas) without; the DiT's baseline.
        """
        tas = self.emit_mean(z, u, f)
        if not self.pr_emission:
            return tas
        return torch.cat([tas, self.emit_mean_pr(z, u, f)], dim=-1)

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
        mu = d * z_prev
        if self.B is not None:
            mu = mu + self.B(u_t)
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
        sample z_s [B, T, z] and the forcing features (or None) are appended
        (for the pr head, no second encode).
        """
        mu_q, logvar_q = self._encode_dist(ye, ue)
        B, T, _ = mu_q.shape
        log_obs_var = torch.clamp(self.log_obs_var, -12.0, 6.0)
        inv_var = torch.exp(-log_obs_var)

        z_s = self.sample(mu_q, logvar_q)                         # [B, T, z]
        f = self.forcing_features(u_ann)
        resid = tas_ann - self.emit_mean(z_s, u_ann, f)
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
        return out + (z_s, f) if return_z else out

    def rollout_loss(self, ye_ctx, ue_ctx, tas_fut_ann, u_fut_ann, return_z=False,
                     obs_var_scaled=False, u_ctx_ann=None):
        """
        Error of the emission mean along the mean rollout from z_last.

        Default: plain MSE. With `obs_var_scaled`, 0.5 * r^2 / sigma^2 with the
        ELBO's learned per-region observation variance (gradient stopped):
        the rollout then pulls on the mean exactly as hard as the NLL does
        (NLL gradient r / sigma^2 vs MSE gradient 2r, so at weight 1 a plain
        MSE is 1 / (2 sigma^2) times weaker; with annual means sigma^2 is
        small), and each region is weighted by its own noise level, so
        low-noise regions such as SOO are not drowned out by noisy land.

        With reservoirs, `u_ctx_ann` (annual forcing over the context) is
        needed: the forcing state is run over context + future, as at
        inference. `return_z` appends (z_traj, features over the future).
        """
        _, z_last = self.posterior(ye_ctx, ue_ctx)
        z_traj = self.roll_forward(z_last, u_fut_ann)
        f_fut = None
        if self.reservoir_k > 0:
            if u_ctx_ann is None:
                raise ValueError("rollout_loss with reservoirs needs u_ctx_ann")
            f_fut = self.forcing_features(
                torch.cat([u_ctx_ann, u_fut_ann], dim=1))[:, u_ctx_ann.shape[1]:]
        r2 = (self.emit_mean(z_traj, u_fut_ann, f_fut) - tas_fut_ann) ** 2
        self.last_rollout_mse = r2.detach().mean()   # plain MSE, for logging
        if obs_var_scaled:
            inv_var = torch.exp(-torch.clamp(self.log_obs_var, -12.0, 6.0)).detach()
            r2 = 0.5 * r2 * inv_var
        loss = r2.mean()
        return (loss, z_traj, f_fut) if return_z else loss

    def pr_loss(self, z_post, u_ann, pr_ann, z_roll, u_fut_ann, pr_fut_ann, detach_z=True,
                obs_var_scaled=False, f_post=None, f_roll=None):
        """
        pr head objective: Gaussian NLL of annual pr on the posterior z
        (per year per region, same normalisation as the tas NLL) plus the MSE
        of the pr mean along the forcing-only rollout, the z the DiT baseline
        uses in the future half. Returns (nll, roll_mse).

        Gaussian, not sinh-arcsinh: the head only supplies the mean the DiT
        generates residuals around; the shape of the pr distribution
        (skew, tails) is the DiT's job, and annual means are close to
        Gaussian anyway. `f_post` / `f_roll`: forcing features aligned with
        z_post / z_roll (reservoirs only). With `detach_z` they are detached
        too: the pr head reads them but does not train the forcing pathway.
        """
        if detach_z:
            z_post, z_roll = z_post.detach(), z_roll.detach()
            f_post = f_post.detach() if f_post is not None else None
            f_roll = f_roll.detach() if f_roll is not None else None
        lv = torch.clamp(self.log_obs_var_pr, -12.0, 6.0)
        r = pr_ann - self.emit_mean_pr(z_post, u_ann, f_post)
        nll = 0.5 * (r ** 2 * torch.exp(-lv) + lv + math.log(2.0 * math.pi)).mean()
        r2 = (self.emit_mean_pr(z_roll, u_fut_ann, f_roll) - pr_fut_ann) ** 2
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
    the DiT and flow-matching loss are inherited.

    Forced state and conditioning (`cond_baseline=True`)
    ----------------------------------------------------
    Every path (training, `sample_blocks`, `ssm_mean`) goes through
    `_prepare`: encode the history, roll z forward, run the forcing-only
    pathway (u-GRU + reservoirs, if any) over the same history + future, and
    evaluate the emission ("baseline", the SSM's (tas, pr) mean) along it.
    With `cond_baseline` the DiT is conditioned on z, a low-rank projection
    of that baseline (all forced information, in the DiT's own variables),
    the instantaneous u, the month and the field memory; the window-local
    GSAT memory kernels (`mem_u`) are gone. Otherwise the inherited
    `ConditioningBuilder` (z, mem_u, ...) is used.

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
    `sample_blocks` encodes all the history it is given (`enc_months`).

    SSM-only training (`ssm_only=True`)
    -----------------------------------
    `loss` is the latent-process objective alone (ELBO, rollout, pr head);
    no DiT pass, DiT parameters frozen. With `detach_latents` the DiT never
    shaped z anyway, so the latent process can be trained and checked on its
    own (cheap), then reused via `ssm_weights` + `freeze_ssm`.

    Context-offset augmentation (`ctx_offset_std` > 0, training mode only)
    ----------------------------------------------------------------------
    Each DiT window's context gets a constant offset per sample and channel,
    N(0, s^2) with s ~ U(0, ctx_offset_std) per sample (normalised units);
    the generated half and its target are untouched. Teacher forcing alone
    teaches "continue the level of the context", which is right for a true
    context and is what makes a free-running DiT carry and amplify its own
    offsets from block to block. With the offsets the level of the context
    is no longer a reliable predictor, so the DiT learns to return to what
    z, u (and the baseline) say. Cost: it also discounts real decadal
    persistence, by about var_persist / (var_persist + s^2). Pick
    ctx_offset_std near the drift to suppress, not larger.

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
                 ssm_only=False, ctx_offset_std=0.0,
                 cond_baseline=False, baseline_rank=32,
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
        self.base_dim = 2 * y_dim if ssm.pr_emission else y_dim
        self.cond_baseline = bool(cond_baseline)
        if self.cond_baseline:
            self.cond_builder = BaselineConditioning(
                y_dim=y_dim, u_dim=u_dim, z_dim=ssm.z_dim, base_dim=self.base_dim,
                cond_dim=kwargs.get("cond_dim", 256),
                timescales_years=kwargs.get("timescales_years", (1.0, 5.0, 20.0)),
                learnable_timescales=kwargs.get("learnable_timescales", True),
                field_memory_rank=kwargs.get("field_memory_rank", 32),
                baseline_rank=baseline_rank)
        # A frozen latent process is frozen entirely, pr head included: the
        # DiT's baseline must not move under it.
        frozen = self.ssm_encoder.frozen
        self.use_ssm_aux = (not frozen) and (
            self.nll_weight > 0 or self.rollout_weight > 0 or self.pr_weight > 0)
        self.ctx_offset_std = float(ctx_offset_std)
        self.ssm_only = bool(ssm_only)
        if self.ssm_only:
            if not self.use_ssm_aux:
                raise ValueError("ssm_only: no latent-process objective is active "
                                 "(frozen latent process or all weights 0)")
            # The DiT is not on the path to the loss: freeze it, so DDP does
            # not raise on parameters without gradient and the optimiser
            # leaves it alone. Checkpoints keep the full-model layout.
            for mod in (self.dit, self.cond_builder):
                for prm in mod.parameters():
                    prm.requires_grad = False

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
    # forced state: z, forcing features, emission (the baseline)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _emission_monthly(self, z_all, u_all):
        """
        The SSM's mean (tas[, pr]) [B, T, base_dim], monthly (annual values
        repeated), along z_all [B, T, z] (monthly, as `latents` returns it).
        The forcing pathway runs over u_all from its first month, so z_all /
        u_all must start where the history starts. Years are 12-month chunks
        from that start, the chunks z is built on.
        """
        B, T, _ = z_all.shape
        u_ann = to_annual(u_all)
        z_ann = z_all[:, ::MONTHS_PER_YEAR][:, :u_ann.shape[1]]
        ssm = self.ssm_encoder.ssm
        base = ssm.emit_mean_field(z_ann, u_ann, ssm.forcing_features(u_ann))
        return broadcast_to_monthly(base, T)

    def _prepare(self, tas_hist, pr_hist, u_hist, u_fut, n_enc):
        """
        Encode the last `n_enc` months of history, roll z over u_fut, run the
        forcing pathway over the same span and evaluate the baseline; return
        (z, baseline, u) cropped to the DiT frame (last `context_len` months
        of history + future), each [B, context_len + n_fut, .].
        """
        n_ctx = self.context_len
        th, ph, uh = tas_hist[:, -n_enc:], pr_hist[:, -n_enc:], u_hist[:, -n_enc:]
        z_full = self.latents(th, ph, uh, u_fut)
        u_full = torch.cat([uh, u_fut], dim=1)
        base_full = self._emission_monthly(z_full.detach(), u_full)
        c = n_enc - n_ctx
        return z_full[:, c:], base_full[:, c:], u_full[:, c:]

    def _cond_z(self, z, base):
        """What goes into the conditioning's z slot: [z, baseline] with `cond_baseline`."""
        return torch.cat([z, base], dim=-1) if self.cond_baseline else z

    def _enc_len(self, n_hist, enc_months):
        if enc_months is None:
            want = self.enc_context_len
        elif enc_months == "all":
            want = n_hist
        else:
            want = int(enc_months)
        if want < self.context_len:
            raise ValueError(f"enc_months={enc_months}: the encoder must read at "
                             f"least context_len ({self.context_len}) months")
        if n_hist < want or (enc_months == "all" and n_hist < self.enc_context_len):
            warnings.warn(
                f"got {n_hist} months of history; the latent process was trained on "
                f"reads of up to {self.enc_context_len}. Pass more history (only the "
                f"last {self.context_len} months reach the DiT).")
        n_enc = min(n_hist, want)
        return n_enc - (n_enc - self.context_len) % MONTHS_PER_YEAR

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def loss(self, tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut,
             block_idx=(0,), start_month=0, t=None, return_parts=False):
        """
        `AnnualSSMOutpaintingDiT.loss`, on the residual field if
        `dit_residual`; the latent-process objective alone if `ssm_only`.
        """
        if self.ssm_only:
            aux, parts = self.ssm_aux_loss(tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut)
            parts["ssm_aux_weighted"] = aux.detach()
            return (aux, parts) if return_parts else aux
        n_fut = u_fut.shape[1]
        off = self._dit_offset(tas_ctx.shape[1])
        z_all, base, u_all = self._prepare(tas_ctx, pr_ctx, u_ctx, u_fut, tas_ctx.shape[1])
        field_all = torch.cat([torch.cat([tas_ctx, tas_fut], dim=1),
                               torch.cat([pr_ctx, pr_fut], dim=1)], dim=-1)[:, off:]
        start_month = int(start_month) + off
        if self.dit_residual:
            field_all = field_all - base
        zc = self._cond_z(z_all, base)

        if isinstance(block_idx, torch.Tensor):
            block_idx = block_idx.tolist()
        elif isinstance(block_idx, int):
            block_idx = [block_idx]

        flow_terms, per_block = [], {}
        for bi in block_idx:
            w0, bs, nb = self.block_bounds(bi, n_fut)
            win = slice(w0, bs + nb)
            field_win = field_all[:, win]
            if self.training and self.ctx_offset_std > 0:
                # Offset the context only. `flow_matching_loss` takes both the
                # conditioning and the clean context tokens from the first
                # context_len steps and scores only the generated ones.
                B, _, C = field_win.shape
                s_b = torch.rand(B, 1, 1, device=field_win.device) * self.ctx_offset_std
                shift = s_b * torch.randn(B, 1, C, device=field_win.device)
                field_win = torch.cat([field_win[:, :self.context_len] + shift,
                                       field_win[:, self.context_len:]], dim=1)
            d = self.flow_matching_loss(
                z_win=zc[:, win], u_win=u_all[:, win], field_win=field_win,
                n_block=nb, start_month=(start_month + w0) % MONTHS_PER_YEAR, t=t)
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
        z_post = z_roll = f_post = f_roll = None
        if self.nll_weight > 0 and not frozen:
            nll, kl, kl_dim, z_post, f_post = ssm.elbo(
                ye_all, ue_all,
                torch.cat([to_annual(tas_ctx), tas_fut_ann], dim=1), u_all_ann,
                kl_free_bits=self.kl_free_bits, return_z=True)
            beta = self.kl_weight * self.kl_beta_scale
            total = total + self.nll_weight * nll + beta * kl
            parts["ssm_nll"] = nll.detach()
            parts["ssm_kl"] = kl.detach()
            parts["kl_active_dims"] = (kl_dim > 0.01).sum().float()
        if self.rollout_weight > 0 and not frozen:
            roll, z_roll, f_roll = ssm.rollout_loss(
                ye_ctx, ue_ctx, tas_fut_ann, u_fut_ann, return_z=True,
                obs_var_scaled=self.rollout_obs_var, u_ctx_ann=u_ctx_ann)
            total = total + self.rollout_weight * roll
            parts["ssm_rollout"] = roll.detach()
            if self.rollout_obs_var:
                # `ssm_rollout` is 0.5 MSE / sigma^2 and rises as the ELBO's
                # sigma^2 shrinks; the plain MSE is the comparable number.
                parts["ssm_rollout_mse"] = ssm.last_rollout_mse
                parts["obs_var_med"] = torch.exp(ssm.log_obs_var.detach()).median()

        if self.pr_weight > 0:
            # Reuse the encodings above; recompute only what was skipped.
            with torch.set_grad_enabled(not self.pr_detach_z):
                if z_post is None:
                    z_post = ssm.posterior(ye_all, ue_all)[0]
                    f_post = ssm.forcing_features(u_all_ann)
                if z_roll is None:
                    z_roll = ssm.roll_forward(ssm.posterior(ye_ctx, ue_ctx)[1], u_fut_ann)
                    f_all = ssm.forcing_features(u_all_ann)
                    f_roll = None if f_all is None else f_all[:, u_ctx_ann.shape[1]:]
            pr_nll, pr_roll = ssm.pr_loss(
                z_post, u_all_ann, to_annual(torch.cat([pr_ctx, pr_fut], dim=1)),
                z_roll, u_fut_ann, to_annual(pr_fut), detach_z=self.pr_detach_z,
                obs_var_scaled=self.rollout_obs_var, f_post=f_post, f_roll=f_roll)
            total = total + self.pr_weight * (pr_nll + pr_roll)
            parts["pr_nll"] = pr_nll.detach()
            parts["pr_rollout"] = pr_roll.detach()
        return total, parts

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ssm_mean(self, tas_hist, pr_hist, u_hist, u_fut, enc_months="all"):
        """
        The latent process's own forecast: annual (tas, pr) means over u_fut,
        each [B, Y_fut, y_dim] (pr None without the pr head). History and
        `enc_months` as in `sample_blocks`. Deterministic (mean rollout).
        """
        self._dit_offset(tas_hist.shape[1])
        n_enc = self._enc_len(tas_hist.shape[1], enc_months)
        _, base, _ = self._prepare(tas_hist, pr_hist, u_hist, u_fut, n_enc)
        fut = to_annual(base[:, self.context_len:])
        return fut[..., :self.y_dim], (fut[..., self.y_dim:] if self.base_dim > self.y_dim else None)

    @torch.no_grad()
    def sample_blocks(self, tas_ctx, pr_ctx, u_ctx, u_fut, start_month=0,
                      n_steps=None, generator=None,
                      tas_fut_true=None, pr_fut_true=None, return_baseline=False,
                      enc_months="all"):
        """
        `AnnualSSMOutpaintingDiT.sample_blocks`, plus:
          - residual mode: the DiT works on field - baseline, and the
            baseline is added back to the output;
          - the history may be longer than `context_len` (whole years beyond
            it). The DiT reads the last `context_len` months; the latent
            process (encoder and forcing pathway) the last `enc_months`:
              "all"  -> everything passed (default; like the original SSM,
                        the GRUs read whatever history they are given)
              None   -> the last `enc_context_len` months (the training length)
              int    -> that many months (whole years, capped at the history)
          - `tas_fut_true`/`pr_fut_true`: teacher forcing, every block's
            context is the true field instead of the generated one
            (diagnostic only);
          - `return_baseline`: also return the SSM's (tas, pr) mean over the
            future (pr None without the pr head).
        Returns (tas, pr[, tas_base, pr_base]), each [B, len(u_fut), y_dim].
        """
        H_total = u_fut.shape[1]
        n_ctx = self.context_len
        off = self._dit_offset(tas_ctx.shape[1])
        n_enc = self._enc_len(tas_ctx.shape[1], enc_months)
        z_all, base, u_all = self._prepare(tas_ctx, pr_ctx, u_ctx, u_fut, n_enc)
        zc = self._cond_z(z_all, base)
        resid = base if self.dit_residual else torch.zeros_like(
            base[..., :1]).expand(-1, -1, self.in_channels)
        hist = torch.cat([tas_ctx[:, off:], pr_ctx[:, off:]], dim=-1) - resid[:, :n_ctx]
        tf = tas_fut_true is not None
        if tf:
            true_fut = torch.cat([tas_fut_true, pr_fut_true], dim=-1) - resid[:, n_ctx:]
            hist_true = torch.cat([hist, true_fut], dim=1)

        out, done = [], 0
        while done < H_total:
            w0, bs, nb = self.block_bounds(done // self.block_len, H_total)
            win = slice(w0, bs + nb)
            field_ctx = (hist_true if tf else hist)[:, w0:bs]
            cond = self.window_conditioning(
                z_win=zc[:, win], u_win=u_all[:, win], field_ctx=field_ctx,
                n_block=nb, start_month=(int(start_month) + off + w0) % MONTHS_PER_YEAR)
            blk = self._sample_window(cond, field_ctx, nb, n_steps, generator)
            hist = torch.cat([hist, blk], dim=1)
            out.append(blk)
            done += nb

        fut = torch.cat(out, dim=1) + resid[:, n_ctx:n_ctx + H_total]
        res = (fut[..., :self.y_dim], fut[..., self.y_dim:])
        if return_baseline:
            fb = base[:, n_ctx:n_ctx + H_total]
            res = res + (fb[..., :self.y_dim],
                         fb[..., self.y_dim:] if self.base_dim > self.y_dim else None)
        return res


class BaselineConditioning(nn.Module):
    """
    Conditioning for `cond_baseline`: like `ConditioningBuilder` without the
    window-local GSAT memory, plus a low-rank projection of the SSM baseline.
    The baseline arrives in the z slot, concatenated after z (see `_cond_z`),
    so the inherited DiT code passes it through unchanged.

        c_t = MLP([z_t, P_b b_t, P_f mem_y, month_t, u_t])
    """

    def __init__(self, y_dim, u_dim, z_dim, base_dim, cond_dim=256,
                 timescales_years=(1.0, 5.0, 20.0), learnable_timescales=True,
                 field_memory_rank=32, baseline_rank=32, month_embed_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.base_dim = base_dim
        self.mem_y = MultiTimescaleMemory(timescales_years, learnable=learnable_timescales)
        self.field_mem_proj = nn.Linear(2 * y_dim * self.mem_y.n_tau, field_memory_rank)
        self.base_proj = nn.Linear(base_dim, baseline_rank)
        self.month_embed = nn.Embedding(MONTHS_PER_YEAR, month_embed_dim)
        in_dim = z_dim + baseline_rank + field_memory_rank + month_embed_dim + u_dim
        self.proj = nn.Sequential(nn.Linear(in_dim, cond_dim), nn.SiLU(),
                                  nn.Linear(cond_dim, cond_dim))

    def forward(self, z, u_full, field_ctx, n_ctx, n_total, start_month=0):
        B = z.shape[0]
        zz, base = z[..., :self.z_dim], z[..., self.z_dim:]
        mem_y_ctx = self.mem_y(field_ctx)
        last = mem_y_ctx[:, -1:] if n_ctx > 0 else mem_y_ctx.new_zeros(
            B, 1, mem_y_ctx.shape[-1])
        mem_y = torch.cat([mem_y_ctx, last.expand(B, n_total - n_ctx, -1)], dim=1)
        months = (torch.arange(n_total, device=z.device) + int(start_month)) % MONTHS_PER_YEAR
        m_emb = self.month_embed(months).unsqueeze(0).expand(B, -1, -1)
        c = torch.cat([zz, self.base_proj(base), self.field_mem_proj(mem_y), m_emb, u_full], dim=-1)
        return self.proj(c)


_SSM_KEYS = ("z_dim", "rnn_hidden", "trans_hidden", "decay_efold_range",
             "gated_decay", "gate_hidden", "monthly_encoder", "enc_window_years",
             "encoder_reads_pr", "emission_hidden", "pr_emission",
             "pr_emission_hidden", "reservoir_k", "u_rnn_hidden",
             "reservoir_tau_range", "z_forced")


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
    for k in ("decay_efold_range", "reservoir_tau_range"):
        if k in ssm_kw:
            ssm_kw[k] = tuple(ssm_kw[k])
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
    "reservoir_tau": "reservoir_tau_range", "baseline_rank": "baseline_rank",
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
    kw["z_forced"] = (_PFX + "B.weight") in state_dict
    if _PFX + "u_gru.weight_ih_l0" in state_dict:
        kw["u_rnn_hidden"] = int(state_dict[_PFX + "u_gru.weight_hh_l0"].shape[1])
        kw["reservoir_k"] = int(state_dict[_PFX + "res_logit_tau"].shape[1])
    else:
        kw["reservoir_k"] = 0
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
    # DiT conditioning type, also read off the weights.
    kw["cond_baseline"] = "cond_builder.base_proj.weight" in state_dict
    if kw["cond_baseline"]:
        kw["baseline_rank"] = int(state_dict["cond_builder.base_proj.weight"].shape[0])
    model = build_model(y_dim, u_dim, device=device, **kw)
    model.load_state_dict(state_dict)
    model.eval()

    tas_scaler = StandardScaler.from_file(os.path.join(run_dir, "y_scaler.out"))
    pr_scaler = StandardScaler.from_file(os.path.join(run_dir, "pr_scaler.out"))
    u_scaler = StandardScaler.from_file(os.path.join(run_dir, "u_scaler.out"))
    return model, tas_scaler, pr_scaler, u_scaler
