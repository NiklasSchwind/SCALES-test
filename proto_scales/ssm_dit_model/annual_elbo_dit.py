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

import numpy as np
import torch
import torch.nn as nn

from proto_scales.ssm_dit_model.annual_ssm import diag_gaussian_kl_per_dim, to_annual
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
    def elbo(self, ye, ue, tas_ann, u_ann, kl_free_bits=0.0):
        """
        Returns (nll, kl, kl_per_dim). nll and kl are summed over regions /
        z-dims and years, averaged over the batch, and both divided by
        (years x regions), so their ratio is the summed ELBO's. kl_per_dim
        [z] is the mean KL per year per dim in nats (diagnostic).
        `kl_free_bits` clamps each step's total KL from below, as in the
        original monthly SSM (default 0).
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
        return nll.mean() / norm, kl.mean() / norm, kl_dim.detach().mean(dim=(0, 1))

    def rollout_loss(self, ye_ctx, ue_ctx, tas_fut_ann, u_fut_ann):
        """MSE of the emission mean along the mean rollout from z_last."""
        _, z_last = self.posterior(ye_ctx, ue_ctx)
        z_traj = self.roll_forward(z_last, u_fut_ann)
        return ((self.emit_mean(z_traj, u_fut_ann) - tas_fut_ann) ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AnnualELBOOutpaintingDiT(AnnualSSMOutpaintingDiT):
    """
    `AnnualSSMOutpaintingDiT` with a `ForcedAnnualELBO` latent process.
    Overrides `latents` (detach) and `ssm_aux_loss`; DiT, conditioning,
    flow-matching loss and block-wise sampling are inherited.

    `kl_weight` is beta; the training loop scales it by `kl_beta_scale`
    (warm-up, set via `set_kl_beta_scale`).
    """

    def __init__(self, ssm, y_dim, u_dim, nll_weight=1.0, kl_weight=5.0,
                 kl_free_bits=0.0, rollout_weight=1.0, detach_latents=True,
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
        self.use_ssm_aux = (not self.ssm_encoder.frozen) and (
            self.nll_weight > 0 or self.rollout_weight > 0)

    def set_kl_beta_scale(self, s):
        self.kl_beta_scale = float(s)

    def latents(self, tas_ctx, pr_ctx, u_ctx, u_fut):
        z = self.ssm_encoder(tas_ctx, pr_ctx, u_ctx, u_fut, stochastic=self.stochastic_z)
        return z.detach() if self.detach_latents else z

    def ssm_aux_loss(self, tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut):
        ssm = self.ssm_encoder.ssm
        ye_ctx, ue_ctx = ssm.encoder_inputs(torch.cat([tas_ctx, pr_ctx], dim=-1), u_ctx)
        ye_fut, ue_fut = ssm.encoder_inputs(torch.cat([tas_fut, pr_fut], dim=-1), u_fut)
        u_ctx_ann, u_fut_ann = to_annual(u_ctx), to_annual(u_fut)
        tas_fut_ann = to_annual(tas_fut)

        parts = {}
        total = torch.zeros((), device=tas_ctx.device)
        if self.nll_weight > 0:
            nll, kl, kl_dim = ssm.elbo(
                torch.cat([ye_ctx, ye_fut], dim=1), torch.cat([ue_ctx, ue_fut], dim=1),
                torch.cat([to_annual(tas_ctx), tas_fut_ann], dim=1),
                torch.cat([u_ctx_ann, u_fut_ann], dim=1),
                kl_free_bits=self.kl_free_bits)
            beta = self.kl_weight * self.kl_beta_scale
            total = total + self.nll_weight * nll + beta * kl
            parts["ssm_nll"] = nll.detach()
            parts["ssm_kl"] = kl.detach()
            parts["kl_active_dims"] = (kl_dim > 0.01).sum().float()
        if self.rollout_weight > 0:
            roll = ssm.rollout_loss(ye_ctx, ue_ctx, tas_fut_ann, u_fut_ann)
            total = total + self.rollout_weight * roll
            parts["ssm_rollout"] = roll.detach()
        return total, parts


_SSM_KEYS = ("z_dim", "rnn_hidden", "trans_hidden", "decay_efold_range",
             "gated_decay", "gate_hidden", "monthly_encoder", "enc_window_years",
             "encoder_reads_pr", "emission_hidden")


def build_model(y_dim, u_dim, device="cpu", ssm_weights=None, **kwargs):
    """
    Construct `AnnualELBOOutpaintingDiT`. Latent-process kwargs (`_SSM_KEYS`)
    go to `ForcedAnnualELBO`, the rest to the DiT. `ssm_weights` optionally
    warm-starts the latent process (non-strict).
    """
    ssm_kw = {k: kwargs.pop(k) for k in _SSM_KEYS if k in kwargs}
    if "decay_efold_range" in ssm_kw:
        ssm_kw["decay_efold_range"] = tuple(ssm_kw["decay_efold_range"])
    ssm = ForcedAnnualELBO(y_dim=y_dim, u_dim=u_dim, **ssm_kw)
    if ssm_weights is not None:
        ckpt = torch.load(ssm_weights, map_location="cpu")
        missing, unexpected = ssm.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"[build_model] ELBO missing keys: {missing}")
        if unexpected:
            print(f"[build_model] ELBO unexpected keys ignored: {len(unexpected)}")
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
    kw.update(ssm_kwargs_from_state_dict(state_dict, y_dim, u_dim))
    kw.update(model_kwargs)
    model = build_model(y_dim, u_dim, device=device, **kw)
    model.load_state_dict(state_dict)
    model.eval()

    tas_scaler = StandardScaler.from_file(os.path.join(run_dir, "y_scaler.out"))
    pr_scaler = StandardScaler.from_file(os.path.join(run_dir, "pr_scaler.out"))
    u_scaler = StandardScaler.from_file(os.path.join(run_dir, "u_scaler.out"))
    return model, tas_scaler, pr_scaler, u_scaler
