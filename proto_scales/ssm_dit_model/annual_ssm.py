"""
annual_ssm
==========

A deliberately small state-space model on **annual means**, trained with the
ELBO and nothing else.

Why this replaces `scales_ssm_cross_corr_osc_trans` in the DiT stack
--------------------------------------------------------------------
The original SSM was the whole emulator, so it had to produce calibrated
monthly tas/pr distributions by itself. That forced the SAS flow (non-Gaussian
pr), the low-rank MVN (spatial covariance), the ACF loss (temporal
correlation), the rollout-MSE terms (forced-response accuracy), the slow
reservoir (GSAT memory) and the ridge `ctrl_lin` (pattern scaling) — around 30
kwargs and five hand-tuned loss weights.

In the DiT stack every one of those jobs belongs to something else: marginals
and spatial covariance to the DiT, within-window temporal structure to the DiT,
long-range forced memory to the annual memory kernels, seasonality to the month
embedding. The only job left for the SSM is:

    a low-dimensional latent carrying slow internal variability,
    conditioned causally on GSAT.

Everything here follows from that.

Design choices and their justifications
---------------------------------------
*Annual resolution.* The seasonal cycle is averaged out before the model sees
anything, so `z` never has to represent it — the exact problem that forced the
reservoir. The ELBO loop also shrinks from ~1800 steps to ~150, which is what
makes end-to-end training with the DiT affordable.

*Diagonal Gaussian emission.* Not a compromise: the DiT owns the spatial
covariance, so the SSM has no reason to model it. No Cholesky, no `cov_rank`,
and none of the conditioning failures that came with them.

*No sinh-arcsinh flow.* Annual-mean precipitation is far closer to Gaussian
than monthly precipitation (central limit over 12 months), so the flow that
motivated it — and that overflowed to inf in training — is unnecessary.

*Deliberately weak emission.* Linear by default (`emit_hidden=0`). This is the
condition that makes ELBO-only work. The original needed an explicit ACF loss
not because the ELBO is inadequate but because a powerful emission head plus a
dominant rollout-MSE term let the model explain the data without `z`, and `z`
went slack. With a bottlenecked emission the KL term — which compares the
posterior against the oscillatory transition prior — is exactly what identifies
the latent dynamics. Widen the emission and posterior collapse comes back.

*Linear-Gaussian transition by default.* `trans_hidden=0` gives a pure damped
oscillator bank plus linear forcing input. Stability is structural: the rotation
is scaled by exp(-damping) with damping > 0, so every eigenvalue has modulus < 1
by construction and the latent process cannot diverge.

*Emission sees u.* `emit(cat([z, u]))` means the linear u -> y path is pattern
scaling, learned as part of the ELBO. `z` is then free to carry the residual
internal variability rather than the forced response. This is why no separate
ridge fit is needed.

Normalisation convention
------------------------
This model always consumes **annual means of monthly-standardised fields**, in
both standalone training and inside the DiT. Fitting separate annual scalers
would make the two paths disagree; keeping one convention makes them identical
by construction. Annual means of standardised monthly data simply have std < 1,
which the model handles fine.

Loss scale
----------
`forward_elbo` returns NLL and KL already normalised per (timestep x dimension),
so both are O(1) regardless of sequence length, number of regions or z_dim.
Weights therefore mean the same thing across configurations — including after
adding new indicators.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MONTHS_PER_YEAR = 12


# ─────────────────────────────────────────────────────────────────────────────
# Monthly -> annual
# ─────────────────────────────────────────────────────────────────────────────
def to_annual(x, months_per_year=MONTHS_PER_YEAR):
    """
    [B, T, D] monthly -> [B, ceil(T/12), D] annual means.

    A trailing partial year is averaged over the months actually present rather
    than dropped, so that horizons which are not a whole number of years (the
    last block of `sample_blocks`) still produce a latent for every month.

    Note this differs from `memory_kernel.annual_means`, which *drops* the
    partial year — there it is a feature (a partial mean would alias the
    seasonal cycle into a memory feature), here it would leave months without a
    latent.
    """
    B, T, D = x.shape
    m = int(months_per_year)
    if T == 0:
        return x.new_zeros(B, 0, D)
    Y = (T + m - 1) // m
    pad = Y * m - T
    if pad:
        xp = F.pad(x, (0, 0, 0, pad))
        counts = torch.full((Y,), float(m), device=x.device, dtype=x.dtype)
        counts[-1] = float(m - pad)
        return xp.reshape(B, Y, m, D).sum(dim=2) / counts.view(1, Y, 1)
    return x.reshape(B, Y, m, D).mean(dim=2)


def broadcast_to_monthly(z_annual, n_months, months_per_year=MONTHS_PER_YEAR):
    """
    [B, Y, z] -> [B, n_months, z] by repeating each year's latent across its
    months. Repetition rather than interpolation keeps the mapping strictly
    causal and free of any question about leaking the next year's state.
    """
    B, Y, Z = z_annual.shape
    idx = torch.arange(n_months, device=z_annual.device) // months_per_year
    return z_annual[:, idx.clamp(max=Y - 1)]


def diag_gaussian_kl_per_dim(mu_q, logvar_q, mu_p, logvar_p):
    """KL(q||p) for diagonal Gaussians, kept per-dimension for free bits."""
    return 0.5 * (
        logvar_p - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        - 1.0
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AnnualSSM(nn.Module):
    """
    Parameters
    ----------
    obs_dim  : channels of the annual observation. For joint tas+pr this is
               2 * n_regions.
    u_dim    : forcing dimension (1 for GSAT).
    z_dim    : latent size; must be even (paired into 2D rotations).
    rnn_hidden : inference GRU width.
    emit_hidden : 0 for a linear emission (default, and recommended — see the
               module docstring on posterior collapse).
    trans_hidden : 0 for a purely linear-Gaussian transition (default).
    osc_period_range : oscillator periods in YEARS. The default (2, 20) spans
               ENSO through multi-decadal.
    """

    def __init__(
        self,
        obs_dim,
        u_dim,
        z_dim=16,
        rnn_hidden=64,
        emit_hidden=0,
        trans_hidden=0,
        osc_period_range=(2.0, 20.0),
        osc_damping_max=1.0,
        osc_init_damp_ratio=0.05,
    ):
        super().__init__()
        if z_dim % 2 != 0:
            raise ValueError("z_dim must be even (paired into 2D oscillators)")
        lo_p, hi_p = osc_period_range
        if not (0 < lo_p < hi_p):
            raise ValueError("osc_period_range must satisfy 0 < low < high (years)")
        if lo_p < 2.0:
            raise ValueError(
                f"osc_period_range low end is {lo_p} yr, below the Nyquist limit "
                f"of 2 yr for annual sampling")

        self.obs_dim = obs_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.n_osc = z_dim // 2
        self.osc_damping_max = float(osc_damping_max)

        # Inference network: q(z_t | y_{<=t}, u_{<=t})
        self.gru = nn.GRU(obs_dim + u_dim, rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        # Oscillatory transition. No amplitude gate: the old model had one so
        # oscillators had to "earn" amplitude from the ACF loss, which no longer
        # exists. Damping alone controls amplitude decay.
        init_raw = math.log(osc_init_damp_ratio / (1.0 - osc_init_damp_ratio))
        self.log_damping = nn.Parameter(torch.full((self.n_osc,), init_raw))
        omega_hi = 2.0 * math.pi / lo_p          # rad / year
        omega_lo = 2.0 * math.pi / hi_p
        self.omega = nn.Parameter(torch.linspace(omega_lo, omega_hi, self.n_osc))
        self.B = nn.Linear(u_dim, z_dim, bias=False)
        self.trans_mlp = (
            nn.Sequential(nn.Linear(z_dim + u_dim, trans_hidden), nn.SiLU(),
                          nn.Linear(trans_hidden, z_dim))
            if trans_hidden else None
        )
        # State-independent process noise: the classic linear-Gaussian SSM.
        self.log_process_var = nn.Parameter(torch.zeros(z_dim))

        # Emission: mean only. Observation noise is a learned per-dimension
        # constant rather than an output of the network — an input-dependent
        # variance is a standard way for a model to game the Gaussian NLL.
        emit_in = z_dim + u_dim
        self.emit = (
            nn.Linear(emit_in, obs_dim) if not emit_hidden
            else nn.Sequential(nn.Linear(emit_in, emit_hidden), nn.SiLU(),
                               nn.Linear(emit_hidden, obs_dim))
        )
        self.log_obs_var = nn.Parameter(torch.zeros(obs_dim))

    # -----------------------
    # pieces
    # -----------------------
    def sample(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def transition(self, z_prev, u_t):
        """
        Damped rotation + linear forcing (+ optional MLP correction).

        Every eigenvalue has modulus exp(-damping) < 1, so the latent process is
        stable by construction — no clamping of z is needed anywhere.
        """
        B = z_prev.shape[0]
        pairs = z_prev.reshape(B, self.n_osc, 2)
        damp = torch.sigmoid(self.log_damping) * self.osc_damping_max
        decay = torch.exp(-damp)
        cos_w, sin_w = torch.cos(self.omega), torch.sin(self.omega)
        z0 = decay * (cos_w * pairs[..., 0] - sin_w * pairs[..., 1])
        z1 = decay * (sin_w * pairs[..., 0] + cos_w * pairs[..., 1])
        mu_p = torch.stack([z0, z1], dim=-1).reshape(B, self.z_dim) + self.B(u_t)
        if self.trans_mlp is not None:
            mu_p = mu_p + self.trans_mlp(torch.cat([z_prev, u_t], dim=-1))
        logvar_p = torch.clamp(self.log_process_var, -12.0, 6.0)
        return mu_p, logvar_p.unsqueeze(0).expand(B, -1)

    def emit_mean(self, z_t, u_t):
        return self.emit(torch.cat([z_t, u_t], dim=-1))

    def oscillator_periods(self):
        """Learned oscillator periods in years — the main thing to inspect."""
        with torch.no_grad():
            return (2.0 * math.pi / self.omega.abs().clamp(min=1e-6)).cpu()

    def oscillator_efolding(self):
        """Damping e-folding times in years."""
        with torch.no_grad():
            damp = torch.sigmoid(self.log_damping) * self.osc_damping_max
            return (1.0 / damp.clamp(min=1e-6)).cpu()

    # -----------------------
    # ELBO
    # -----------------------
    def forward(self, y, u, kl_free_bits=0.05):
        """Alias for `forward_elbo` so the module works under a DDP wrapper."""
        return self.forward_elbo(y, u, kl_free_bits=kl_free_bits)

    def forward_elbo(self, y, u, kl_free_bits=0.05):
        """
        Parameters
        ----------
        y : [B, Y, obs_dim] annual means of monthly-standardised fields
        u : [B, Y, u_dim]   annual means of standardised forcing

        Returns (nll, kl), both **normalised per element** — nll per
        (year x obs channel), kl per (year x latent dim). Both are O(1)
        regardless of Y, obs_dim or z_dim.

        `kl_free_bits` is applied per latent dimension. This is the standard
        anti-collapse device and matters here because the emission is
        deliberately weak: without a floor, dimensions of z that are briefly
        unhelpful get driven to the prior and never recover.
        """
        B, T, _ = y.shape
        h, _ = self.gru(torch.cat([y, u], dim=-1))
        mu_q, logvar_q = torch.chunk(self.q_head(h), 2, dim=-1)
        logvar_q = torch.clamp(logvar_q, -12.0, 6.0)

        log_obs_var = torch.clamp(self.log_obs_var, -12.0, 6.0)
        inv_var = torch.exp(-log_obs_var)

        nll = y.new_zeros(B)
        kl = y.new_zeros(B)
        z_prev = None
        for t in range(T):
            z_t = self.sample(mu_q[:, t], logvar_q[:, t])

            resid = y[:, t] - self.emit_mean(z_t, u[:, t])
            nll = nll + 0.5 * (
                resid * resid * inv_var + log_obs_var + math.log(2.0 * math.pi)
            ).sum(dim=-1)

            if t == 0:
                mu_p = torch.zeros_like(mu_q[:, 0])
                logvar_p = torch.zeros_like(logvar_q[:, 0])
            else:
                mu_p, logvar_p = self.transition(z_prev, u[:, t])
            kl_dim = diag_gaussian_kl_per_dim(
                mu_q[:, t], logvar_q[:, t], mu_p, logvar_p)
            kl = kl + torch.clamp(kl_dim, min=kl_free_bits).sum(dim=-1)

            z_prev = z_t

        nll = nll.mean() / (T * self.obs_dim)
        kl = kl.mean() / (T * self.z_dim)
        return nll, kl

    # -----------------------
    # latent trajectories (used by the DiT conditioner)
    # -----------------------
    def posterior(self, y, u):
        """Posterior mean over an observed window. Returns ([B,Y,z], [B,z])."""
        h, _ = self.gru(torch.cat([y, u], dim=-1))
        mu_q, _ = torch.chunk(self.q_head(h), 2, dim=-1)
        return mu_q, mu_q[:, -1]

    def rollout(self, z_last, u_fut, stochastic=True):
        """Prior rollout under future forcing. Returns [B, Y_fut, z]."""
        z = z_last
        out = []
        for k in range(u_fut.shape[1]):
            mu_p, logvar_p = self.transition(z, u_fut[:, k])
            z = self.sample(mu_p, logvar_p) if stochastic else mu_p
            out.append(z)
        return torch.stack(out, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# DiT conditioner
# ─────────────────────────────────────────────────────────────────────────────
class AnnualSSMLatentEncoder(nn.Module):
    """
    Drop-in replacement for `conditioning.SSMLatentEncoder`.

    Same call signature and same return shape ([B, Tc+H, z_dim] at *monthly*
    resolution), so `ConditioningBuilder` and the DiT need no changes. Internally
    it averages to annual, runs the annual SSM, and repeats each year's latent
    across its months.

    As with the monthly encoder, the context half uses the posterior (the
    context is observed at inference too) and the future half uses the prior
    rollout — conditioning on a posterior that saw the future would leak the
    target and collapse at sampling time.
    """

    def __init__(self, ssm, freeze=True):
        super().__init__()
        self.ssm = ssm
        self.z_dim = ssm.z_dim
        self.frozen = bool(freeze)
        if self.frozen:
            for p in self.ssm.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.ssm.eval()
        return self

    def trainable_ssm_parameters(self):
        return [n for n, p in self.ssm.named_parameters() if p.requires_grad]

    def annual_inputs(self, tas_ctx, pr_ctx, u_ctx, u_fut):
        y_ann = to_annual(torch.cat([tas_ctx, pr_ctx], dim=-1))
        u_ctx_ann = to_annual(u_ctx)
        u_fut_ann = to_annual(u_fut)
        return y_ann, u_ctx_ann, u_fut_ann

    def forward(self, tas_ctx, pr_ctx, u_ctx, u_fut, stochastic=True):
        n_ctx, n_fut = tas_ctx.shape[1], u_fut.shape[1]
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            y_ann, u_ctx_ann, u_fut_ann = self.annual_inputs(
                tas_ctx, pr_ctx, u_ctx, u_fut)
            z_ctx_ann, z_last = self.ssm.posterior(y_ann, u_ctx_ann)
            z_fut_ann = self.ssm.rollout(z_last, u_fut_ann, stochastic=stochastic)
            z_ctx = broadcast_to_monthly(z_ctx_ann, n_ctx)
            z_fut = broadcast_to_monthly(z_fut_ann, n_fut)
            z = torch.cat([z_ctx, z_fut], dim=1)
        return z.detach() if self.frozen else z

    def elbo(self, tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut, kl_free_bits=0.05):
        """Auxiliary ELBO over the full window, for end-to-end training."""
        y_ann = to_annual(torch.cat([torch.cat([tas_ctx, tas_fut], dim=1),
                                     torch.cat([pr_ctx, pr_fut], dim=1)], dim=-1))
        u_ann = to_annual(torch.cat([u_ctx, u_fut], dim=1))
        return self.ssm.forward_elbo(y_ann, u_ann, kl_free_bits=kl_free_bits)
