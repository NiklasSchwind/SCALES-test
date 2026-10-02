"""
annual_jepa_kl_dit
===================

`annual_jepa_dit`, with the one-step JEPA loss promoted from a plain MSE to a
KL between a probabilistic online encoder and a prior built from the *target*
(EMA) encoder, instead of from the online encoder itself.

Why this exists
----------------
`ForcedAnnualJEPA` (in `annual_jepa_dit`) trains

    z_pred_t = transition(z_t-1, u_t)
    loss     = || z_pred_t - zbar_t ||^2        zbar_t = target_encoder(y_<=t)

with nothing recursively anchoring the *online* encoder's own behaviour across
a long context — the loss only asks the transition to chase wherever the
encoder (and its slowly-moving EMA shadow) happen to go. `ForcedAnnualSSM`'s
ELBO does not have this gap: its KL term,

    KL( q(z_t | y_<=t, u_<=t) || p(z_t | z_t-1, u_t) )

pulls the posterior back toward the transition's prediction at *every single
year*, context included, which is plausibly why that model stays well-behaved
at context lengths it was never trained on while the plain-MSE JEPA model's
predictions degrade as context length grows (see the `ForcedAnnualJEPAKL`
docstring below for the empirical check this is meant to follow up on).

Copying the ELBO's KL verbatim is not safe here, though: `forward_elbo`'s KL is
self-referential (the prior at year t is built from the *same* posterior's own
year t-1), which is only a non-degenerate objective because a second term —
the emission NLL — externally pins down what z has to mean. Drop the emission
(the point of the JEPA family) and keep only a self-referential KL, and the
loss has a free, trivial optimum: both sides collapse together to a shared
near-constant, near-zero-variance distribution.

`ForcedAnnualJEPAKL` keeps the KL's probabilistic form but builds the prior
from the *target* encoder instead:

    q(z_t)  = N(mu_q_t, diag(exp(logvar_q_t)))         online, gets gradient
    zbar_t-1 = target_encoder(y_<=t-1, u_<=t-1)         EMA copy, no gradient
    p(z_t)  = N(transition_mean(zbar_t-1, u_t), process_var)
    loss    = KL(q(z_t) || p(z_t))

This keeps exactly the anti-collapse protection the plain-MSE JEPA loss
already relies on (the prior's `z_t-1` only moves via the slow EMA, never via
gradients from this loss, so nothing can collapse both sides for free) while
adding the thing the MSE version lacks: a per-step recursive pull of the
online encoder toward what the stable transition predicts, at every context
year, not just a loose "close on average" target.

What is unchanged from `annual_jepa_dit`
-------------------------------------------
The transition (`decay() * z_prev + B(u_t)`, optionally `+ trans_mlp`),
`jepa_rollout_loss` (still a plain MSE between the mean forcing-only rollout
and the target encoder's mean reading of the real future — there is no online,
data-informed distribution over a blind rollout to KL against, so promoting
*that* term to a KL would not be meaningful), and every DiT-side piece
(conditioning, diffusion loss, block-wise sampling, `ForcedAnnualLatentEncoder`)
are exactly `annual_ssm_dit`'s / `annual_jepa_dit`'s. `AnnualJEPAKLOutpaintingDiT`
subclasses `AnnualSSMOutpaintingDiT` and overrides only the auxiliary-loss
method and the weights that pick it, same as `AnnualJEPAOutpaintingDiT`.

z conditioning is still the mean, not a sample
------------------------------------------------
The encoder is now a genuine Gaussian (`q_head` outputs `2 * z_dim`), but
`posterior()` still returns only the mean for the DiT to condition on,
matching `ForcedAnnualSSM`'s own convention (see `annual_ssm_dit`'s docstring,
"Why the conditioning z is the mean, not a sample") — this module adds a
calibrated per-step variance to the *training signal*, it does not change what
the DiT sees at sampling time.

Three degenerate directions, three separate guards — same as `annual_jepa_dit`
-------------------------------------------------------------------------------
Every guard `ForcedAnnualJEPA` needed, this needs too, for the same reasons
(see that module's docstring for the full derivation and the empirical
symptoms each one was found from): `decay()` hard-clamps to `decay_efold_range`
against magnitude collapse, `z_covariance_loss` penalises z-dimensions
collapsing onto redundant copies of each other, and `_standardize` (applied to
`mu`, pooled per dimension over batch and time) removes the per-dimension
rescaling direction that the one-step KL is exactly as blind to as the plain
MSE is — a KL between two Gaussians is just as invariant to a per-dimension
rescale of `mu_q`, `mu_p` and `logvar_q`/`logvar_p` as an MSE is, so nothing
about moving to a KL loss makes this particular direction go away on its own.
(An earlier version of this guard used `nn.LayerNorm(z_dim)` — normalizing
*across* dimensions instead of *per* dimension — and that was wrong for the
same reason it was wrong in `annual_jepa_dit`: seeing it fail there is what
caught it here before it was ever trained on.)

What the KL *does* add beyond the deterministic model, on top of all three
guards above: `log_process_var` is independently clamped to `[-12, 6]` in
log-space, which caps how far the prior's implied variance can grow to chase
a drifting `mu_q`/`mu_p` gap — a second, independent (if looser) ceiling on
the same kind of runaway `_standardize` targets directly. The two are
complementary, not redundant: `_standardize` pins each dimension's own scale
exactly; the process-variance clamp bounds how much spread the model is
allowed to claim around it.
"""

import copy
import math

import torch
import torch.nn as nn

from proto_scales.ssm_dit_model.annual_ssm import diag_gaussian_kl_per_dim, to_annual
from proto_scales.ssm_dit_model.annual_ssm_dit import AnnualSSMOutpaintingDiT


# ─────────────────────────────────────────────────────────────────────────────
# Latent process
# ─────────────────────────────────────────────────────────────────────────────
class ForcedAnnualJEPAKL(nn.Module):
    """
    Annual encoder + transition, no emission model, trained with a KL between
    a probabilistic online encoder and a prior built from the target (EMA)
    encoder — see the module docstring for the full derivation and why this
    differs from both `ForcedAnnualSSM`'s ELBO and `ForcedAnnualJEPA`'s plain
    MSE.

        z_t      ~ q(. | y_<=t, u_<=t)           online GRU encoder, Gaussian
        zbar_t   = target_encoder(y_<=t, u_<=t)  EMA copy, mean only, no grad
        prior_t  = N(transition_mean(zbar_t-1, u_t), process_var)
        loss     = KL(q(z_t) || prior_t)

    `posterior` and `roll_forward` keep the exact signatures `ForcedAnnualSSM`
    / `ForcedAnnualJEPA` have, so `ForcedAnnualLatentEncoder` (imported
    unchanged from `annual_ssm_dit`) needs no changes to condition the DiT on
    this latent process instead.

    Parameters
    ----------
    decay_efold_range : e-folding times in YEARS spanned at initialisation.
                        Same parameterisation as `ForcedAnnualSSM`: a
                        per-dimension contraction through a sigmoid so every
                        eigenvalue is real and in (0, 1) and a long rollout
                        cannot ring or diverge.
    trans_hidden      : optional MLP correction on the transition mean, as in
                        `ForcedAnnualSSM` / `ForcedAnnualJEPA`.
    target_decay      : EMA decay for `target_encoder`. Deliberately lower
                        than the model-wide sampling EMA (`ema_decay`,
                        typically 0.999): an encoder that changes too slowly
                        leaves the predictor chasing a stale target early in
                        training, when the online encoder itself is moving
                        fastest.
    """

    def __init__(
        self,
        obs_dim,
        u_dim,
        z_dim=16,
        rnn_hidden=64,
        trans_hidden=0,
        decay_efold_range=(1.0, 50.0),
        target_decay=0.996,
    ):
        super().__init__()
        lo, hi = decay_efold_range
        if not (0 < lo <= hi):
            raise ValueError("decay_efold_range must satisfy 0 < low <= high (years)")

        self.obs_dim = obs_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.target_decay = float(target_decay)

        # Online encoder: causal read of the field, now a Gaussian.
        self.gru = nn.GRU(obs_dim + u_dim, rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        # Target encoder: architecturally identical, gradient-free, EMA-only.
        self.target_gru = copy.deepcopy(self.gru)
        self.target_q_head = copy.deepcopy(self.q_head)
        for p in list(self.target_gru.parameters()) + list(self.target_q_head.parameters()):
            p.requires_grad_(False)

        # Per-dimension contraction, identical parameterisation to
        # `ForcedAnnualSSM.__init__` — see that module for why a dense `A` is
        # not used.
        tau = torch.logspace(math.log10(lo), math.log10(hi), z_dim)
        d0 = torch.exp(-1.0 / tau).clamp(1e-4, 1 - 1e-4)
        self.logit_decay = nn.Parameter(torch.log(d0 / (1.0 - d0)))

        # `decay()` clamps to exactly this range forever after, not just at
        # init — see `annual_jepa_dit.ForcedAnnualJEPA` for why nothing else
        # stops `logit_decay` drifting to the sigmoid's saturation ceiling
        # without this.
        self.register_buffer("_decay_min", torch.tensor(math.exp(-1.0 / lo)))
        self.register_buffer("_decay_max", torch.tensor(math.exp(-1.0 / hi)))

        self.B = nn.Linear(u_dim, z_dim, bias=False)
        self.trans_mlp = (
            nn.Sequential(nn.Linear(z_dim + u_dim, trans_hidden), nn.SiLU(),
                          nn.Linear(trans_hidden, z_dim))
            if trans_hidden else None
        )

        # Prior variance: a single learned per-dimension parameter, exactly as
        # in `ForcedAnnualSSM` — not something recursively propagated through
        # the rollout, since the ELBO's own prior variance isn't either.
        self.log_process_var = nn.Parameter(torch.zeros(z_dim))

    # -----------------------
    # pieces
    # -----------------------
    def decay(self):
        return torch.sigmoid(self.logit_decay).clamp(self._decay_min, self._decay_max)

    def efolding_years(self):
        """Per-dimension e-folding times in years — the thing to inspect."""
        with torch.no_grad():
            d = self.decay().clamp(1e-6, 1 - 1e-9)
            return (-1.0 / torch.log(d)).cpu()

    def transition_mean(self, z_prev, u_t):
        mu = self.decay() * z_prev + self.B(u_t)
        if self.trans_mlp is not None:
            mu = mu + self.trans_mlp(torch.cat([z_prev, u_t], dim=-1))
        return mu

    def _standardize(self, z):
        """
        Per-dimension standardization, pooled over batch and time. Unlike
        `nn.LayerNorm(z_dim)` (which normalizes *across* dimensions at every
        timestep, forcing all z-dimensions to comparable scale at each
        instant — fights having a few dominant slow modes and several minor
        fast ones), this leaves each dimension's own relative importance
        alone and only removes *that dimension's own* scale drifting freely
        over training: a factor-`k` rescale of one dimension's whole
        trajectory is exactly cancelled by that dimension's own pooled std
        also scaling by `k`. See `annual_jepa_dit.ForcedAnnualJEPA` for why
        the LayerNorm version of this fix was tried first and found wrong.
        """
        mean = z.mean(dim=(0, 1), keepdim=True)
        std = z.std(dim=(0, 1), keepdim=True).clamp(min=1e-4)
        return (z - mean) / std

    def _encode_dist(self, y, u):
        """
        Online causal encoding over a window. Returns (mu, logvar), each
        [B, T, z]. `mu` is standardized per dimension; `logvar` is left
        alone — it already has its own clamp.
        """
        h, _ = self.gru(torch.cat([y, u], dim=-1))
        mu, logvar = torch.chunk(self.q_head(h), 2, dim=-1)
        return self._standardize(mu), torch.clamp(logvar, -12.0, 6.0)

    def encode(self, y, u):
        """Online encoding, mean only. Returns [B, T, z]."""
        mu, _ = self._encode_dist(y, u)
        return mu

    @torch.no_grad()
    def encode_target(self, y, u):
        """
        Target-encoder read of the same window, mean only. Never carries a
        gradient. Only the mean is used anywhere (to build the prior's mean,
        and as the rollout target) — the target distribution's own variance
        is not needed, same as `ForcedAnnualSSM`'s prior variance is a single
        learned parameter rather than something read off any encoder.
        """
        h, _ = self.target_gru(torch.cat([y, u], dim=-1))
        mu, _ = torch.chunk(self.target_q_head(h), 2, dim=-1)
        return self._standardize(mu)

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA the online encoder into the target encoder. Call once per step."""
        d = self.target_decay
        for tp, p in zip(self.target_gru.parameters(), self.gru.parameters()):
            tp.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for tp, p in zip(self.target_q_head.parameters(), self.q_head.parameters()):
            tp.mul_(d).add_(p.detach(), alpha=1.0 - d)

    # -----------------------
    # latent trajectories — same interface as ForcedAnnualSSM / ForcedAnnualJEPA
    # -----------------------
    def posterior(self, y, u):
        """Online encoding over an observed window. Returns ([B,Y,z], [B,z])."""
        z = self.encode(y, u)
        return z, z[:, -1]

    def roll_forward(self, z0, u_fut, stochastic=False):
        """
        Iterate the transition over `u_fut`. Returns [B, Y_fut, z].

        `stochastic` exists only so the shared `ForcedAnnualLatentEncoder` can
        call this exactly as it calls `ForcedAnnualSSM.roll_forward`; the DiT
        still conditions on the mean (see the module docstring), so this must
        be False.
        """
        if stochastic:
            raise ValueError(
                "ForcedAnnualJEPAKL conditions the DiT on the mean rollout "
                "only; call roll_forward with stochastic=False")
        z = z0
        out = []
        for k in range(u_fut.shape[1]):
            z = self.transition_mean(z, u_fut[:, k])
            out.append(z)
        return torch.stack(out, dim=1)

    # -----------------------
    # objectives
    # -----------------------
    def jepa_kl_loss(self, y, u, kl_free_bits=0.05):
        """
        One-step KL loss over a whole sequence: at every year t >= 1, KL the
        online posterior q(z_t) against a prior whose mean is
        `transition_mean(zbar_t-1, u_t)` — the *target* encoder's read of
        year t-1, pushed forward by forcing alone — and whose variance is the
        learned `process_var`. Per-(year x z-dim) average (O(1) regardless of
        T or z_dim), the KL analogue of `ForcedAnnualJEPA.jepa_loss`, built the
        way `ForcedAnnualSSM.forward_elbo`'s KL is built, but anchored to the
        target encoder rather than to the online posterior's own previous
        step — see the module docstring for why that substitution is the
        point.

        `kl_free_bits` clamps the per-dimension KL the same way
        `forward_elbo` does: insurance against the posterior trivially
        matching the prior on dimensions that have nothing to say.
        """
        B, T, _ = y.shape
        mu_q, logvar_q = self._encode_dist(y, u)
        z_target = self.encode_target(y, u)

        mu_p = self.transition_mean(z_target[:, :-1], u[:, 1:])
        logvar_p = torch.clamp(self.log_process_var, -12.0, 6.0)
        logvar_p = logvar_p.view(1, 1, -1).expand(B, T - 1, -1)

        kl_dim = diag_gaussian_kl_per_dim(mu_q[:, 1:], logvar_q[:, 1:], mu_p, logvar_p)
        kl = torch.clamp(kl_dim, min=kl_free_bits).sum(dim=-1).sum(dim=1)
        return kl.mean() / ((T - 1) * self.z_dim)

    def jepa_rollout_loss(self, y_ctx, u_ctx, y_fut, u_fut):
        """
        Multi-step consistency: unchanged from `ForcedAnnualJEPA.jepa_rollout_loss`.
        From the online posterior's mean at the end of the context, roll
        forward on forcing alone for the whole future and require the
        trajectory to track the *target* encoding of the actual future field
        at every lead. There is no online, data-informed distribution over a
        blind forcing-only rollout, so there is nothing meaningful to KL here
        — this stays a plain MSE, exactly as in the plain-MSE JEPA model.
        """
        _, z_last = self.posterior(y_ctx, u_ctx)
        z_traj = self.roll_forward(z_last, u_fut, stochastic=False)
        z_target_fut = self.encode_target(
            torch.cat([y_ctx, y_fut], dim=1),
            torch.cat([u_ctx, u_fut], dim=1),
        )[:, y_ctx.shape[1]:]
        return ((z_traj - z_target_fut) ** 2).mean()

    def z_covariance_loss(self, y, u):
        """
        VICReg-style covariance penalty on the online encoding's mean, pooled
        over batch and time — identical in purpose and implementation to
        `ForcedAnnualJEPA.z_covariance_loss`: guards against z-dimensions
        becoming linearly redundant with each other, which `_standardize` and
        the KL's free-bits floor do not by themselves prevent (a dimension can
        be unit-scale and individually non-degenerate while still being a
        near-copy of another one).
        """
        z = self.encode(y, u).reshape(-1, self.z_dim)
        z = z - z.mean(dim=0, keepdim=True)
        n = z.shape[0]
        cov = (z.T @ z) / max(n - 1, 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        return (off_diag ** 2).sum() / self.z_dim


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AnnualJEPAKLOutpaintingDiT(AnnualSSMOutpaintingDiT):
    """
    `AnnualSSMOutpaintingDiT` with the latent process's ELBO replaced end to
    end by `ForcedAnnualJEPAKL`'s KL loss. `jepa` is a `ForcedAnnualJEPAKL`
    (encoder + transition, no emission model); `ssm_aux_loss` is overridden to
    score the KL between the online posterior and the target-anchored prior,
    plus the same rollout MSE `AnnualJEPAOutpaintingDiT` uses. Everything else
    — DiT, diffusion loss, conditioning, block-wise sampling — is inherited
    unchanged.

    `jepa_weight` / `jepa_rollout_weight` / `jepa_kl_free_bits` replace
    `ssm_elbo_weight` / `ssm_kl_weight` / `ssm_kl_free_bits` /
    `ssm_rollout_weight`: the one-step term is itself already a KL, so there
    is no separate NLL-vs-KL weighting the way the ELBO has — `jepa_weight`
    is the whole one-step term's weight, directly comparable to
    `AnnualJEPAOutpaintingDiT`'s `jepa_weight` for the plain-MSE version.

    `jepa_cov_weight` adds `ForcedAnnualJEPAKL.z_covariance_loss`, same as
    `AnnualJEPAOutpaintingDiT.jepa_cov_weight` — see that class's docstring
    for why the ELBO doesn't need a counterpart for it.
    """

    def __init__(self, jepa, y_dim, u_dim, jepa_weight=1.0,
                 jepa_rollout_weight=1.0, jepa_kl_free_bits=0.05,
                 jepa_cov_weight=1.0, **kwargs):
        super().__init__(
            ssm=jepa, y_dim=y_dim, u_dim=u_dim,
            ssm_elbo_weight=0.0, ssm_kl_weight=0.0, ssm_kl_free_bits=0.0,
            ssm_rollout_weight=0.0, **kwargs,
        )
        self.jepa_weight = float(jepa_weight)
        self.jepa_rollout_weight = float(jepa_rollout_weight)
        self.jepa_kl_free_bits = float(jepa_kl_free_bits)
        self.jepa_cov_weight = float(jepa_cov_weight)
        self.use_ssm_aux = (not self.ssm_encoder.frozen) and (
            self.jepa_weight > 0 or self.jepa_rollout_weight > 0
            or self.jepa_cov_weight > 0)

    def ssm_aux_loss(self, tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut):
        """
        Overrides `AnnualSSMOutpaintingDiT.ssm_aux_loss`; same call site
        (`loss`), same job (shaping z beyond what the diffusion loss alone
        would), different objective. `ssm_encoder.ssm` is the
        `ForcedAnnualJEPAKL` instance — the attribute keeps the `ssm` name so
        the shared `ForcedAnnualLatentEncoder` needs no changes.
        """
        enc = self.ssm_encoder
        y_ctx_ann, u_ctx_ann, u_fut_ann = enc.annual_inputs(
            tas_ctx, pr_ctx, u_ctx, u_fut)
        y_fut_ann = to_annual(torch.cat([tas_fut, pr_fut], dim=-1))
        y_ann = torch.cat([y_ctx_ann, y_fut_ann], dim=1)
        u_ann = torch.cat([u_ctx_ann, u_fut_ann], dim=1)

        parts = {}
        total = torch.zeros((), device=y_ctx_ann.device)

        if self.jepa_weight > 0:
            jepa_kl = enc.ssm.jepa_kl_loss(y_ann, u_ann, kl_free_bits=self.jepa_kl_free_bits)
            total = total + self.jepa_weight * jepa_kl
            parts["jepa_kl"] = jepa_kl.detach()

        if self.jepa_rollout_weight > 0:
            roll = enc.ssm.jepa_rollout_loss(
                y_ctx_ann, u_ctx_ann, y_fut_ann, u_fut_ann)
            total = total + self.jepa_rollout_weight * roll
            parts["jepa_rollout"] = roll.detach()

        if self.jepa_cov_weight > 0:
            cov = enc.ssm.z_covariance_loss(y_ann, u_ann)
            total = total + self.jepa_cov_weight * cov
            parts["jepa_cov"] = cov.detach()

        return total, parts


def build_model(y_dim, u_dim, device="cpu", jepa_weights=None, **kwargs):
    """
    Convenience constructor, mirroring `annual_jepa_dit.build_model`.
    `jepa_weights` optionally warm-starts the latent process from a standalone
    `ForcedAnnualJEPAKL` run; unexpected keys are ignored.
    """
    jepa = ForcedAnnualJEPAKL(
        obs_dim=2 * y_dim,
        u_dim=u_dim,
        z_dim=kwargs.pop("z_dim", 16),
        rnn_hidden=kwargs.pop("rnn_hidden", 64),
        trans_hidden=kwargs.pop("trans_hidden", 0),
        decay_efold_range=kwargs.pop("decay_efold_range", (1.0, 50.0)),
        target_decay=kwargs.pop("target_decay", 0.996),
    )
    if jepa_weights is not None:
        ckpt = torch.load(jepa_weights, map_location="cpu")
        missing, unexpected = jepa.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"[build_model] JEPA-KL missing keys: {missing}")
        if unexpected:
            print(f"[build_model] JEPA-KL unexpected keys ignored: {len(unexpected)}")

    model = AnnualJEPAKLOutpaintingDiT(jepa=jepa, y_dim=y_dim, u_dim=u_dim, **kwargs)
    return model.to(device)
