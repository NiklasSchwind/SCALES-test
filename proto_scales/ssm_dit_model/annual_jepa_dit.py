"""
annual_jepa_dit
===============

`annual_ssm_dit`, with the latent process's ELBO replaced by a JEPA objective.

`ForcedAnnualSSM` (in `annual_ssm_dit`) is an emission-based SSM: an encoder
(the GRU posterior), a transition, and a decoder (`emit`) trained jointly by a
one-step NLL plus a KL against the transition's prior. `ForcedAnnualJEPA` here
keeps only the encoder and the transition — there is no `emit`, no
`log_obs_var`, no KL, no prior to regularise against. It is trained instead by
matching the transition's prediction of z_t against a *target* encoder's own
read of the actual y_t:

    z_t      = encoder(y_<=t, u_<=t)                online, trainable
    zbar_t   = target_encoder(y_<=t, u_<=t)          EMA of the online encoder
    z_pred_t = transition(z_t-1, u_t)                same contraction + forcing
    loss     = || z_pred_t - zbar_t ||^2

This is a Joint-Embedding Predictive Architecture: instead of decoding a
prediction back into observation space and scoring it there (an ELBO, a
reconstruction MSE), the model scores agreement in representation space
directly. The DiT is the only decoder left anywhere in the pipeline.

Why a target encoder, not the online encoder on both sides
------------------------------------------------------------
Minimising `|| transition(z_t-1, u_t) - encoder(y_t) ||` by gradient through
*both* branches has a trivial global optimum: collapse the encoder to a
constant and the predictor matches it for free. `target_encoder` is a
`copy.deepcopy` of the online encoder (`gru` + `enc_head`) that never receives
a gradient; it is instead nudged toward the online weights after every
optimiser step by `ForcedAnnualJEPA.update_target_encoder`, called from the
training loop exactly like the model-wide sampling EMA in
`annual_dit_training.ModelEMA` — the same fix BYOL/I-JEPA use for the same
failure mode. It is cheap here because it only shadows the encoder, not the
whole model.

The DiT itself provides a second, independent guard against collapse: `z` also
has to carry enough information to condition generation, and a constant `z`
cannot do that, so the diffusion loss pulls against collapse from the other
side. The target encoder removes the trivial optimum outright rather than
relying on that alone.

What is unchanged from `annual_ssm_dit`
-----------------------------------------
The transition (`decay() * z_prev + B(u_t)`, optionally `+ trans_mlp`), the
`ForcedAnnualLatentEncoder` wrapper that turns a context posterior plus a
forward roll into the monthly-resolution trajectory the DiT conditions on, and
every DiT-side piece (conditioning, diffusion loss, block-wise sampling) are
exactly `annual_ssm_dit`'s. `AnnualJEPAOutpaintingDiT` subclasses
`AnnualSSMOutpaintingDiT` and overrides only the auxiliary-loss method and the
weights that pick it, since that is the only thing that changed; see that
module's docstring for the DiT-side design and for why the oscillator is gone,
why z is the mean rather than a sample, and why the forcing memory stays
window-local.
"""

import copy
import math

import torch
import torch.nn as nn

from proto_scales.ssm_dit_model.annual_ssm import to_annual
from proto_scales.ssm_dit_model.annual_ssm_dit import AnnualSSMOutpaintingDiT


# ─────────────────────────────────────────────────────────────────────────────
# Latent process
# ─────────────────────────────────────────────────────────────────────────────
class ForcedAnnualJEPA(nn.Module):
    """
    Annual encoder + transition, no emission model, trained with a JEPA loss.

        z_t    = encoder(y_<=t, u_<=t)          online causal GRU encoder
        zbar_t = target_encoder(y_<=t, u_<=t)   EMA copy, gradient-free
        z_pred = d * z_t-1 + B u_t              contraction + forcing

    One step is one year, matching the convention in `annual_ssm_dit`. The
    encoder is deterministic — there is no KL term here to give a variance
    head somewhere to go, so `enc_head` outputs `z` directly rather than a
    Gaussian's parameters.

    `posterior` and `roll_forward` keep the exact signatures
    `ForcedAnnualSSM` has, so `ForcedAnnualLatentEncoder` (imported unchanged
    from `annual_ssm_dit`) needs no changes to condition the DiT on this
    latent process instead.

    Parameters
    ----------
    decay_efold_range : e-folding times in YEARS spanned at initialisation.
                        Same parameterisation as `ForcedAnnualSSM`: a
                        per-dimension contraction through a sigmoid so every
                        eigenvalue is real and in (0, 1) and a long rollout
                        cannot ring or diverge. See that module's docstring
                        for why a dense `A` is deliberately not offered.
    trans_hidden      : optional MLP correction on the transition mean, as in
                        `ForcedAnnualSSM`.
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

        # Online encoder: causal read of the field, deterministic.
        self.gru = nn.GRU(obs_dim + u_dim, rnn_hidden, batch_first=True)
        self.enc_head = nn.Linear(rnn_hidden, z_dim)

        # Target encoder: architecturally identical, gradient-free, EMA-only.
        self.target_gru = copy.deepcopy(self.gru)
        self.target_head = copy.deepcopy(self.enc_head)
        for p in list(self.target_gru.parameters()) + list(self.target_head.parameters()):
            p.requires_grad_(False)

        # Per-dimension contraction, identical parameterisation to
        # `ForcedAnnualSSM.__init__` — see that module for why a dense `A` is
        # not used.
        tau = torch.logspace(math.log10(lo), math.log10(hi), z_dim)
        d0 = torch.exp(-1.0 / tau).clamp(1e-4, 1 - 1e-4)
        self.logit_decay = nn.Parameter(torch.log(d0 / (1.0 - d0)))
        self.B = nn.Linear(u_dim, z_dim, bias=False)
        self.trans_mlp = (
            nn.Sequential(nn.Linear(z_dim + u_dim, trans_hidden), nn.SiLU(),
                          nn.Linear(trans_hidden, z_dim))
            if trans_hidden else None
        )

    # -----------------------
    # pieces
    # -----------------------
    def decay(self):
        return torch.sigmoid(self.logit_decay)

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

    def encode(self, y, u):
        """Online causal encoding over a window. Returns [B, T, z]."""
        h, _ = self.gru(torch.cat([y, u], dim=-1))
        return self.enc_head(h)

    @torch.no_grad()
    def encode_target(self, y, u):
        """Target-encoder read of the same window. Never carries a gradient."""
        h, _ = self.target_gru(torch.cat([y, u], dim=-1))
        return self.target_head(h)

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA the online encoder into the target encoder. Call once per step."""
        d = self.target_decay
        for tp, p in zip(self.target_gru.parameters(), self.gru.parameters()):
            tp.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for tp, p in zip(self.target_head.parameters(), self.enc_head.parameters()):
            tp.mul_(d).add_(p.detach(), alpha=1.0 - d)

    # -----------------------
    # latent trajectories — same interface as ForcedAnnualSSM
    # -----------------------
    def posterior(self, y, u):
        """Online encoding over an observed window. Returns ([B,Y,z], [B,z])."""
        z = self.encode(y, u)
        return z, z[:, -1]

    def roll_forward(self, z0, u_fut, stochastic=False):
        """
        Iterate the transition over `u_fut`. Returns [B, Y_fut, z].

        `stochastic` exists only so the shared `ForcedAnnualLatentEncoder` can
        call this exactly as it calls `ForcedAnnualSSM.roll_forward`; there is
        no process-noise model here to sample from, so it must be False.
        """
        if stochastic:
            raise ValueError(
                "ForcedAnnualJEPA has no process-noise model to sample from; "
                "call roll_forward with stochastic=False")
        z = z0
        out = []
        for k in range(u_fut.shape[1]):
            z = self.transition_mean(z, u_fut[:, k])
            out.append(z)
        return torch.stack(out, dim=1)

    # -----------------------
    # objectives
    # -----------------------
    def jepa_loss(self, y, u):
        """
        One-step predictive loss over a whole sequence: at every year t >= 1,
        match `transition_mean(z_t-1, u_t)` — the online state pushed forward
        by forcing alone — against the target encoder's read of the actual
        y_t. Per-element MSE (O(1) regardless of T or z_dim), the JEPA
        analogue of `ForcedAnnualSSM.forward_elbo`'s one-step-ahead nll.
        """
        z_online = self.encode(y, u)
        z_target = self.encode_target(y, u)
        z_pred = self.transition_mean(z_online[:, :-1], u[:, 1:])
        return ((z_pred - z_target[:, 1:]) ** 2).mean()

    def jepa_rollout_loss(self, y_ctx, u_ctx, y_fut, u_fut):
        """
        Multi-step consistency: the JEPA analogue of
        `ForcedAnnualSSM.rollout_loss`. From the online posterior at the end
        of the context, roll forward on forcing alone for the whole future
        and require the trajectory to track the *target* encoding of the
        actual future field at every lead — the gap the one-step loss above
        leaves open, exactly as `rollout_loss` closes it for the ELBO.
        """
        _, z_last = self.posterior(y_ctx, u_ctx)
        z_traj = self.roll_forward(z_last, u_fut, stochastic=False)
        z_target_fut = self.encode_target(
            torch.cat([y_ctx, y_fut], dim=1),
            torch.cat([u_ctx, u_fut], dim=1),
        )[:, y_ctx.shape[1]:]
        return ((z_traj - z_target_fut) ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AnnualJEPAOutpaintingDiT(AnnualSSMOutpaintingDiT):
    """
    `AnnualSSMOutpaintingDiT` with the latent process's ELBO replaced end to
    end by a JEPA loss. `jepa` is a `ForcedAnnualJEPA` (encoder + transition,
    no emission model); `ssm_aux_loss` is overridden to score its transition's
    prediction against its target encoder instead of an NLL + KL. Everything
    else — DiT, diffusion loss, conditioning, block-wise sampling — is
    inherited unchanged.

    `jepa_weight` / `jepa_rollout_weight` replace `ssm_elbo_weight` /
    `ssm_kl_weight` / `ssm_kl_free_bits` / `ssm_rollout_weight`: there is no KL
    term to weight because there is no prior to regularise against.
    """

    def __init__(self, jepa, y_dim, u_dim, jepa_weight=1.0,
                 jepa_rollout_weight=1.0, **kwargs):
        super().__init__(
            ssm=jepa, y_dim=y_dim, u_dim=u_dim,
            ssm_elbo_weight=0.0, ssm_kl_weight=0.0, ssm_kl_free_bits=0.0,
            ssm_rollout_weight=0.0, **kwargs,
        )
        self.jepa_weight = float(jepa_weight)
        self.jepa_rollout_weight = float(jepa_rollout_weight)
        self.use_ssm_aux = (not self.ssm_encoder.frozen) and (
            self.jepa_weight > 0 or self.jepa_rollout_weight > 0)

    def ssm_aux_loss(self, tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut):
        """
        Overrides `AnnualSSMOutpaintingDiT.ssm_aux_loss`; same call site
        (`loss`), same job (shaping z beyond what the diffusion loss alone
        would), different objective. `ssm_encoder.ssm` is the
        `ForcedAnnualJEPA` instance — the attribute keeps the `ssm` name so
        the shared `ForcedAnnualLatentEncoder` needs no changes.
        """
        enc = self.ssm_encoder
        y_ctx_ann, u_ctx_ann, u_fut_ann = enc.annual_inputs(
            tas_ctx, pr_ctx, u_ctx, u_fut)
        y_fut_ann = to_annual(torch.cat([tas_fut, pr_fut], dim=-1))

        parts = {}
        total = torch.zeros((), device=y_ctx_ann.device)

        if self.jepa_weight > 0:
            y_ann = torch.cat([y_ctx_ann, y_fut_ann], dim=1)
            u_ann = torch.cat([u_ctx_ann, u_fut_ann], dim=1)
            jepa = enc.ssm.jepa_loss(y_ann, u_ann)
            total = total + self.jepa_weight * jepa
            parts["jepa"] = jepa.detach()

        if self.jepa_rollout_weight > 0:
            roll = enc.ssm.jepa_rollout_loss(
                y_ctx_ann, u_ctx_ann, y_fut_ann, u_fut_ann)
            total = total + self.jepa_rollout_weight * roll
            parts["jepa_rollout"] = roll.detach()

        return total, parts


def build_model(y_dim, u_dim, device="cpu", jepa_weights=None, **kwargs):
    """
    Convenience constructor, mirroring `annual_ssm_dit.build_model`.
    `jepa_weights` optionally warm-starts the latent process from a standalone
    `ForcedAnnualJEPA` run; unexpected keys are ignored.
    """
    jepa = ForcedAnnualJEPA(
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
            print(f"[build_model] JEPA missing keys: {missing}")
        if unexpected:
            print(f"[build_model] JEPA unexpected keys ignored: {len(unexpected)}")

    model = AnnualJEPAOutpaintingDiT(jepa=jepa, y_dim=y_dim, u_dim=u_dim, **kwargs)
    return model.to(device)
