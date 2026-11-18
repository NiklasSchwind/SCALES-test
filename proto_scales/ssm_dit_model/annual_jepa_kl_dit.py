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
import glob
import math
import os
import re

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
    normalize_z       : whether `_encode_dist`/`encode_target` apply
                        `_standardize` to `mu`. Default True (the fix
                        described above). False recovers the pre-fix
                        behaviour, for ablating how much of the observed
                        improvement is actually coming from this versus from
                        using a KL loss at all — not something to turn off
                        otherwise.
    gated_decay       : make each dimension's timescale depend on the state,
                        tau_t = tau(z_t-1, u_t), so the slow modes can take
                        up and release heat at different rates (warming vs.
                        cooling asymmetry). Bounded by construction in
                        log-timescale space:

                            log tau = log lo + (log hi - log lo)
                                      * sigmoid(logit_tau + gate(z_prev, u_t))

                        so tau never leaves `decay_efold_range` and long
                        rollouts stay contractive — unlike `trans_hidden`'s
                        additive correction, which can undo the clamp. The
                        bound is smooth, so dimensions near a limit still get
                        gradient (the hard clamp in the linear mode gives none
                        once a dimension is pinned). The gate's last layer is
                        zero-initialised: at step 0 the model is a linear SSM
                        with log-spaced timescales.

                        The gate reads `z_prev` rather than a forcing
                        increment: the slow z-dimensions already integrate
                        the forcing history, so `z_prev` vs. `u_t` carries
                        the warming/cooling signal without a noisy annual
                        delta-GMT, and without `roll_forward` needing the
                        last context forcing. Cost: the transition is no
                        longer linear in z, so the deterministic rollout is
                        an approximation of the conditional mean, exactly as
                        with `trans_hidden`.
    gate_hidden       : hidden width of the decay gate MLP.
    monthly_encoder   : run both GRUs over the *monthly* field and forcing and
                        read z off at each year end (December), instead of
                        over annual means. The transition, rollout and z stay
                        annual. Reason: a GRU started from h0 = 0 has a warm-up
                        transient of some number of *steps*; on annual input a
                        10-year context is only 10 steps, shorter than that
                        transient, so z_last was read mid-warm-up. Monthly
                        input gives 120 steps for the same context, as in the
                        original monthly SSM. Encoder inputs then come from
                        `encoder_inputs(y_monthly, u_monthly)`; every method
                        that encodes takes those, and transition forcing is
                        passed separately in annual units.
    enc_window_years  : 0 (default) -> both GRUs run over the whole window
                        from h0 = 0, so z_t depends on how long the GRU has
                        been running (its "age"), not only on recent field.
                        Training only ever reads the online encoder at ages
                        <= context + horizon and z_last at age = context; a
                        slow z-dim can lock onto age, become a lead-time
                        clock in the rollout target, and go out of range
                        beyond the training horizon. > 0 -> every year is
                        read from a fixed lookback of this many years (a
                        fresh GRU run over just those months), so z_t is a
                        function of (y, u) over [t - L, t] only, identical at
                        every position and in every window. Years with less
                        history than L (the first L-1 of a window) keep the
                        prefix read. Stored as a buffer, so a checkpoint
                        carries its own setting (old checkpoints load as 0).
                        Should not exceed the context, else z_last is a
                        shorter read than the targets.
    annual_emission   : add an annual-mean emission head y_ann = emit(z), used
                        only as a training loss (`emission_loss`), never at
                        sampling, where the DiT still produces the field. It
                        reads z only, not (z, u): with u as an input the
                        forced response could again be carried by f(u)
                        instead of by the dynamics, which is the shortcut
                        this head exists to close. Its job is to make z hold
                        the forced field state, slow modes included, rather
                        than whatever representation is easiest to predict.
    emission_hidden   : 0 -> linear emission; > 0 -> one hidden layer of that
                        width.
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
        normalize_z=True,
        gated_decay=False,
        gate_hidden=32,
        monthly_encoder=False,
        annual_emission=False,
        emission_hidden=0,
        enc_window_years=0,
    ):
        super().__init__()
        lo, hi = decay_efold_range
        if not (0 < lo <= hi):
            raise ValueError("decay_efold_range must satisfy 0 < low <= high (years)")

        self.obs_dim = obs_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.target_decay = float(target_decay)
        self.normalize_z = bool(normalize_z)
        self.gated_decay = bool(gated_decay)
        self.monthly_encoder = bool(monthly_encoder)

        # Online encoder: causal read of the field, now a Gaussian.
        self.gru = nn.GRU(obs_dim + u_dim, rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        # Target encoder: architecturally identical, gradient-free, EMA-only.
        self.target_gru = copy.deepcopy(self.gru)
        self.target_q_head = copy.deepcopy(self.q_head)
        for p in list(self.target_gru.parameters()) + list(self.target_q_head.parameters()):
            p.requires_grad_(False)

        if self.gated_decay:
            # Position of each dimension's timescale in [log lo, log hi],
            # through a sigmoid. Spread log-uniformly but kept off the very
            # ends, where the sigmoid is saturated. Separate parameter name
            # from `logit_decay` so a linear-mode checkpoint can't be loaded
            # into it and silently reinterpreted.
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
            # Per-dimension contraction, identical parameterisation to
            # `ForcedAnnualSSM.__init__` — see that module for why a dense
            # `A` is not used.
            tau = torch.logspace(math.log10(lo), math.log10(hi), z_dim)
            d0 = torch.exp(-1.0 / tau).clamp(1e-4, 1 - 1e-4)
            self.logit_decay = nn.Parameter(torch.log(d0 / (1.0 - d0)))

            # `decay()` clamps to exactly this range forever after, not just
            # at init — see `annual_jepa_dit.ForcedAnnualJEPA` for why nothing
            # else stops `logit_decay` drifting to the sigmoid's saturation
            # ceiling without this.
            self.register_buffer("_decay_min", torch.tensor(math.exp(-1.0 / lo)))
            self.register_buffer("_decay_max", torch.tensor(math.exp(-1.0 / hi)))

        self.B = nn.Linear(u_dim, z_dim, bias=False)
        self.trans_mlp = (
            nn.Sequential(nn.Linear(z_dim + u_dim, trans_hidden), nn.SiLU(),
                          nn.Linear(trans_hidden, z_dim))
            if trans_hidden else None
        )
        if self.trans_mlp is not None:
            # Start from the linear model, as the gate does: the correction
            # grows only as far as the losses ask for it.
            nn.init.zeros_(self.trans_mlp[-1].weight)
            nn.init.zeros_(self.trans_mlp[-1].bias)

        # Prior variance: a single learned per-dimension parameter, exactly as
        # in `ForcedAnnualSSM` — not something recursively propagated through
        # the rollout, since the ELBO's own prior variance isn't either.
        self.log_process_var = nn.Parameter(torch.zeros(z_dim))

        # Optional annual emission from z alone (training loss only). Only
        # created when enabled, so checkpoints without it load unchanged.
        self.annual_emission = bool(annual_emission)
        if self.annual_emission:
            self.emit = (
                nn.Sequential(nn.Linear(z_dim, emission_hidden), nn.SiLU(),
                              nn.Linear(emission_hidden, obs_dim))
                if emission_hidden else nn.Linear(z_dim, obs_dim)
            )

        # Fixed per-dimension standardization stats, fitted once on the
        # training set after training (`fit_z_stats`), used in eval mode.
        # Separate stats for the online encoder (fitted over context-length
        # windows, the frame z_last is read in) and the target encoder (fitted
        # over context+horizon windows, the frame the rollout loss targets).
        # Until fitted, eval mode falls back to batch stats as before.
        self.register_buffer("z_mean", torch.zeros(z_dim))
        self.register_buffer("z_std", torch.ones(z_dim))
        self.register_buffer("z_tgt_mean", torch.zeros(z_dim))
        self.register_buffer("z_tgt_std", torch.ones(z_dim))
        self.register_buffer("z_stats_fitted", torch.tensor(False))
        self.register_buffer("enc_window_years", torch.tensor(int(enc_window_years)))

    _Z_STAT_BUFFERS = ("z_mean", "z_std", "z_tgt_mean", "z_tgt_std", "z_stats_fitted")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints from before the fixed stats existed: keep the defaults
        # (unfitted -> batch stats) instead of failing a strict load.
        for name in self._Z_STAT_BUFFERS:
            state_dict.setdefault(prefix + name, getattr(self, name).clone())
        # Checkpoints from before the fixed-lookback option: whole-window reads.
        state_dict.setdefault(prefix + "enc_window_years", torch.tensor(0))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset_z_stats(self):
        """Back to batch stats, e.g. before continuing training a fitted checkpoint."""
        self.z_mean.zero_(); self.z_std.fill_(1.0)
        self.z_tgt_mean.zero_(); self.z_tgt_std.fill_(1.0)
        self.z_stats_fitted.fill_(False)

    # -----------------------
    # pieces
    # -----------------------
    def _log_tau(self, z_prev=None, u_t=None):
        """Gated mode only. Without inputs, the gate is off (baseline tau)."""
        a = self.logit_tau
        if z_prev is not None:
            a = a + self.decay_gate(torch.cat([z_prev, u_t], dim=-1))
        return self._log_tau_lo + (self._log_tau_hi - self._log_tau_lo) * torch.sigmoid(a)

    def decay(self, z_prev=None, u_t=None):
        """
        Per-dimension contraction. In gated mode, pass `z_prev`/`u_t` for the
        state-dependent decay; without them this returns the baseline (gate
        off), which is what `efolding_years` reports.
        """
        if self.gated_decay:
            return torch.exp(-torch.exp(-self._log_tau(z_prev, u_t)))
        return torch.sigmoid(self.logit_decay).clamp(self._decay_min, self._decay_max)

    def efolding_years(self):
        """
        Per-dimension e-folding times in years — the thing to inspect. In
        gated mode this is the baseline, before the gate; see
        `gated_efolding_years` for what the rollout actually uses.
        """
        with torch.no_grad():
            d = self.decay().clamp(1e-6, 1 - 1e-9)
            return (-1.0 / torch.log(d)).cpu()

    @torch.no_grad()
    def gated_efolding_years(self, z0, u_fut):
        """
        Gated mode only: e-folding times along a mean rollout from `z0`
        under `u_fut` ([B, Y, u]). Returns [B, Y, z] in years — compare
        warming vs. cooling phases to see whether the gate is being used.
        """
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

    def _standardize(self, z, target=False):
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

        A no-op when `normalize_z=False` — gated here, in one place, rather
        than at each call site.

        Training mode (and eval before `fit_z_stats` has run) uses the
        batch's own stats. Those depend on which windows share the batch and
        on the window length, so at inference — one trajectory, any context
        length — they put z_last in a different frame from the one the
        transition and DiT were trained on. Eval mode with fitted stats uses
        the fixed training-set stats instead (BatchNorm-style).
        """
        if not self.normalize_z:
            return z
        if not self.training and bool(self.z_stats_fitted):
            mean, std = (self.z_tgt_mean, self.z_tgt_std) if target else (self.z_mean, self.z_std)
            return (z - mean) / std
        mean = z.mean(dim=(0, 1), keepdim=True)
        std = z.std(dim=(0, 1), keepdim=True).clamp(min=1e-4)
        return (z - mean) / std

    def encoder_inputs(self, y_monthly, u_monthly):
        """
        What the encoders read, from monthly field/forcing [B, T, .]: the
        monthly series themselves with `monthly_encoder`, annual means
        otherwise. Pass the result to `posterior` / `encode` / `encode_target`
        / the losses.
        """
        if self.monthly_encoder:
            return y_monthly, u_monthly
        return to_annual(y_monthly), to_annual(u_monthly)

    def _run_gru(self, gru, ye, ue):
        """
        [B, T_enc, .] -> [B, Y, h]: one state per year (year end if monthly).
        With `enc_window_years` > 0, each year with a full lookback behind it
        is read by a separate GRU run over just that lookback (see the class
        docstring); the windows are batched, so this is one GRU call of
        length L instead of one of length T.
        """
        x = torch.cat([ye, ue], dim=-1)
        T = x.shape[1]
        step = 12 if self.monthly_encoder else 1
        # December of each year; a trailing partial year reads its last month,
        # matching `to_annual`'s partial-year mean.
        idx = torch.arange(step - 1, T + step - 1, step, device=x.device).clamp(max=T - 1)
        L = int(self.enc_window_years) * step
        if L <= 0 or T <= L:
            h, _ = gru(x)
            return h[:, idx] if self.monthly_encoder else h
        full = idx >= L - 1
        h_pre, _ = gru(x[:, :L])                      # prefix reads, first years
        ends = idx[full]
        win = x.unfold(1, L, 1)[:, ends - L + 1]      # [B, nW, F, L]
        Bsz, nW, F, _ = win.shape
        hw, _ = gru(win.permute(0, 1, 3, 2).reshape(Bsz * nW, L, F))
        hw = hw[:, -1].reshape(Bsz, nW, -1)
        return torch.cat([h_pre[:, idx[~full]], hw], dim=1)

    def _raw_online(self, ye, ue):
        return torch.chunk(self.q_head(self._run_gru(self.gru, ye, ue)), 2, dim=-1)

    @torch.no_grad()
    def _raw_target(self, ye, ue):
        return torch.chunk(self.target_q_head(self._run_gru(self.target_gru, ye, ue)), 2, dim=-1)[0]

    @torch.no_grad()
    def fit_z_stats(self, batches):
        """
        Fit the fixed standardization stats once, on the training set.
        `batches` yields (ye_ctx, ue_ctx, ye_full, ue_full) — encoder inputs
        (`encoder_inputs`, monthly or annual by mode): context-length windows
        for the online stats, context+horizon windows for the target stats
        (the frames each is read in during training).
        Pooled over windows and time, per dimension.
        """
        acc = {k: [0.0, 0.0, 0] for k in ("on", "tgt")}

        def add(key, z):
            z = z.reshape(-1, self.z_dim).double()
            a = acc[key]
            a[0] = a[0] + z.sum(0); a[1] = a[1] + (z ** 2).sum(0); a[2] += z.shape[0]

        for y_ctx, u_ctx, y_full, u_full in batches:
            add("on", self._raw_online(y_ctx, u_ctx)[0])
            add("tgt", self._raw_target(y_full, u_full))

        def finish(key):
            s, ss, n = acc[key]
            if n < 2:
                raise ValueError("fit_z_stats: no data")
            mean = s / n
            std = ((ss / n - mean ** 2) * n / (n - 1)).clamp(min=1e-8).sqrt().clamp(min=1e-4)
            return mean.float(), std.float()

        m, s = finish("on"); self.z_mean.copy_(m); self.z_std.copy_(s)
        m, s = finish("tgt"); self.z_tgt_mean.copy_(m); self.z_tgt_std.copy_(s)
        self.z_stats_fitted.fill_(True)

    def _encode_dist(self, y, u):
        """
        Online causal encoding over a window. Returns (mu, logvar), each
        [B, T, z]. `mu` is standardized per dimension; `logvar` is left
        alone — it already has its own clamp.
        """
        mu, logvar = self._raw_online(y, u)
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
        return self._standardize(self._raw_target(y, u), target=True)

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
        """
        Online encoding over an observed window of encoder inputs
        (`encoder_inputs`). Returns annual ([B,Y,z], [B,z]).
        """
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
    def jepa_kl_loss(self, ye, ue, u_ann, kl_free_bits=0.05):
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

        `ye`/`ue` are encoder inputs (`encoder_inputs`); `u_ann` [B, Y, u] is
        the annual forcing the transition is driven by.
        """
        mu_q, logvar_q = self._encode_dist(ye, ue)
        z_target = self.encode_target(ye, ue)
        B, T, _ = mu_q.shape

        mu_p = self.transition_mean(z_target[:, :-1], u_ann[:, 1:])
        logvar_p = torch.clamp(self.log_process_var, -12.0, 6.0)
        logvar_p = logvar_p.view(1, 1, -1).expand(B, T - 1, -1)

        kl_dim = diag_gaussian_kl_per_dim(mu_q[:, 1:], logvar_q[:, 1:], mu_p, logvar_p)
        kl = torch.clamp(kl_dim, min=kl_free_bits).sum(dim=-1).sum(dim=1)
        return kl.mean() / ((T - 1) * self.z_dim)

    def jepa_rollout_loss(self, ye_ctx, ue_ctx, ye_fut, ue_fut, u_fut_ann):
        """
        Multi-step consistency: unchanged from `ForcedAnnualJEPA.jepa_rollout_loss`.
        From the online posterior's mean at the end of the context, roll
        forward on forcing alone for the whole future and require the
        trajectory to track the *target* encoding of the actual future field
        at every lead. There is no online, data-informed distribution over a
        blind forcing-only rollout, so there is nothing meaningful to KL here
        — this stays a plain MSE, exactly as in the plain-MSE JEPA model.

        `ye_*`/`ue_*` are encoder inputs (`encoder_inputs`); `u_fut_ann` is the
        annual future forcing the rollout is driven by.
        """
        z_ctx, z_last = self.posterior(ye_ctx, ue_ctx)
        z_traj = self.roll_forward(z_last, u_fut_ann, stochastic=False)
        z_target_fut = self.encode_target(
            torch.cat([ye_ctx, ye_fut], dim=1),
            torch.cat([ue_ctx, ue_fut], dim=1),
        )[:, z_ctx.shape[1]:]
        return ((z_traj - z_target_fut) ** 2).mean()

    def emission_loss(self, ye_ctx, ue_ctx, y_ctx_ann, y_fut_ann, u_fut_ann):
        """
        Annual-mean emission terms, both through `emit(z)` (z only):

          recon : the online encoding over the context reproduces the
                  context's annual means.
          roll  : the mean rollout from the end of the context, driven by
                  forcing alone (exactly the trajectory the DiT is
                  conditioned on), reproduces the future's annual means at
                  every lead — `ForcedAnnualSSM.rollout_loss`, but with an
                  emission that can't read u.

        `y_*_ann` are annual means of the normalised field [B, Y, obs_dim].
        Returns (recon, roll), per-element MSEs.
        """
        z_ctx = self.encode(ye_ctx, ue_ctx)
        recon = ((self.emit(z_ctx) - y_ctx_ann) ** 2).mean()
        z_traj = self.roll_forward(z_ctx[:, -1], u_fut_ann, stochastic=False)
        roll = ((self.emit(z_traj) - y_fut_ann) ** 2).mean()
        return recon, roll

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
                 jepa_cov_weight=1.0, emission_weight=0.0,
                 emission_rollout_weight=0.0, detach_latents=False, **kwargs):
        super().__init__(
            ssm=jepa, y_dim=y_dim, u_dim=u_dim,
            ssm_elbo_weight=0.0, ssm_kl_weight=0.0, ssm_kl_free_bits=0.0,
            ssm_rollout_weight=0.0, **kwargs,
        )
        self.jepa_weight = float(jepa_weight)
        self.jepa_rollout_weight = float(jepa_rollout_weight)
        self.jepa_kl_free_bits = float(jepa_kl_free_bits)
        self.jepa_cov_weight = float(jepa_cov_weight)
        self.emission_weight = float(emission_weight)
        self.emission_rollout_weight = float(emission_rollout_weight)
        # True: the diffusion loss does not reach the latent process; z is
        # shaped by the JEPA/emission terms only (which encode separately).
        # Without it the DiT can build itself a lead-time clock in z: a dim
        # with B = 0 and a fixed start value decays deterministically along
        # the rollout, at no cost to the JEPA losses.
        self.detach_latents = bool(detach_latents)
        if (self.emission_weight > 0 or self.emission_rollout_weight > 0) \
                and not jepa.annual_emission:
            raise ValueError("emission weights > 0 need ForcedAnnualJEPAKL(annual_emission=True)")
        self.use_ssm_aux = (not self.ssm_encoder.frozen) and (
            self.jepa_weight > 0 or self.jepa_rollout_weight > 0
            or self.jepa_cov_weight > 0 or self.emission_weight > 0
            or self.emission_rollout_weight > 0)

    def latents(self, tas_ctx, pr_ctx, u_ctx, u_fut):
        z = self.ssm_encoder(tas_ctx, pr_ctx, u_ctx, u_fut, stochastic=self.stochastic_z)
        return z.detach() if self.detach_latents else z

    def ssm_aux_loss(self, tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut):
        """
        Overrides `AnnualSSMOutpaintingDiT.ssm_aux_loss`; same call site
        (`loss`), same job (shaping z beyond what the diffusion loss alone
        would), different objective. `ssm_encoder.ssm` is the
        `ForcedAnnualJEPAKL` instance — the attribute keeps the `ssm` name so
        the shared `ForcedAnnualLatentEncoder` can drive it.
        """
        enc = self.ssm_encoder
        jepa = enc.ssm
        ye_ctx, ue_ctx = jepa.encoder_inputs(torch.cat([tas_ctx, pr_ctx], dim=-1), u_ctx)
        ye_fut, ue_fut = jepa.encoder_inputs(torch.cat([tas_fut, pr_fut], dim=-1), u_fut)
        ye = torch.cat([ye_ctx, ye_fut], dim=1)
        ue = torch.cat([ue_ctx, ue_fut], dim=1)
        u_fut_ann = to_annual(u_fut)
        u_ann = torch.cat([to_annual(u_ctx), u_fut_ann], dim=1)

        parts = {}
        total = torch.zeros((), device=ye.device)

        if self.jepa_weight > 0:
            jepa_kl = enc.ssm.jepa_kl_loss(ye, ue, u_ann, kl_free_bits=self.jepa_kl_free_bits)
            total = total + self.jepa_weight * jepa_kl
            parts["jepa_kl"] = jepa_kl.detach()

        if self.jepa_rollout_weight > 0:
            roll = enc.ssm.jepa_rollout_loss(ye_ctx, ue_ctx, ye_fut, ue_fut, u_fut_ann)
            total = total + self.jepa_rollout_weight * roll
            parts["jepa_rollout"] = roll.detach()

        if self.jepa_cov_weight > 0:
            cov = enc.ssm.z_covariance_loss(ye, ue)
            total = total + self.jepa_cov_weight * cov
            parts["jepa_cov"] = cov.detach()

        if self.emission_weight > 0 or self.emission_rollout_weight > 0:
            recon, eroll = jepa.emission_loss(
                ye_ctx, ue_ctx,
                to_annual(torch.cat([tas_ctx, pr_ctx], dim=-1)),
                to_annual(torch.cat([tas_fut, pr_fut], dim=-1)),
                u_fut_ann)
            total = total + self.emission_weight * recon \
                + self.emission_rollout_weight * eroll
            parts["emit_recon"] = recon.detach()
            parts["emit_rollout"] = eroll.detach()

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
        normalize_z=kwargs.pop("normalize_z", True),
        gated_decay=kwargs.pop("gated_decay", False),
        gate_hidden=kwargs.pop("gate_hidden", 32),
        monthly_encoder=kwargs.pop("monthly_encoder", False),
        annual_emission=kwargs.pop("annual_emission", False),
        emission_hidden=kwargs.pop("emission_hidden", 0),
        enc_window_years=kwargs.pop("enc_window_years", 0),
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


_CKPT_EPOCH_RE = re.compile(r"annual_jepa_kl_dit_epoch(\d+)\.pt$")
_EMIT_PREFIX = "ssm_encoder.ssm.emit."


def emission_kwargs_from_state_dict(state_dict):
    """
    `annual_emission` / `emission_hidden` as stored in a state dict: the head
    exists only in runs trained with an emission weight > 0, and its shape
    gives the hidden width (`emit.0.weight` [hidden, z]) or says it is
    linear (`emit.weight`).
    """
    if not any(k.startswith(_EMIT_PREFIX) for k in state_dict):
        return dict(annual_emission=False, emission_hidden=0)
    w0 = state_dict.get(_EMIT_PREFIX + "0.weight")
    return dict(annual_emission=True,
                emission_hidden=0 if w0 is None else int(w0.shape[0]))


def load_jepa_dit_annual(run_dir, y_dim, u_dim=1, device="gpu", ema=True,
                         checkpoint=None, prefer_zstats=True, **model_kwargs):
    """
    Load a trained `AnnualJEPAKLOutpaintingDiT` plus its tas/pr/u scalers from
    a `train_annual_jepa_kl_dit.py` run directory.

    Which weights
    -------------
    `checkpoint=None` or `"latest"`: the highest epoch in
    `run_dir/checkpoints/` if there are any, else `run_dir/model_out`
    (note: the reverse of `eval_jepa_rollout`'s default, kept from the
    original loader). `checkpoint="model_out"` or an epoch number picks one
    explicitly. Epoch checkpoints hold
    `{"model": ..., "ema": ...}` and `ema` picks which; `model_out` is
    already the best-validation EMA shadow, so `ema=False` raises there.
    With `prefer_zstats`, a `*_zstats` copy written by
    `proto_scales.fit_jepa_kl_z_stats` is used instead of the original when it
    exists (epoch checkpoints are saved before the end-of-training z-stat
    fit; `model_out` already has fitted stats).

    Architecture
    ------------
    `model_kwargs` go to `build_model` and must match the run (`config.txt`).
    `annual_emission` / `emission_hidden` are read from the weights unless
    given explicitly, so emission and non-emission runs load the same way.

    `device="gpu"` -> "cuda" (CPU fallback with a warning). Returns
    (model, tas_scaler, pr_scaler, u_scaler), model in eval() mode — needed for
    the fixed z stats to be used.
    """
    from proto_scales.ssm_model.scales_ssm import StandardScaler

    if device in ("gpu", "cuda"):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            print("[load_jepa_dit_annual] no GPU visible, falling back to CPU")
            device = "cpu"

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    epochs = {}
    for f in glob.glob(os.path.join(ckpt_dir, "annual_jepa_kl_dit_epoch*.pt")):
        m = _CKPT_EPOCH_RE.search(f)   # skips the *_zstats.pt copies
        if m:
            epochs[int(m.group(1))] = f
    model_out = os.path.join(run_dir, "model_out")

    if checkpoint is None or checkpoint == "latest":
        if checkpoint == "latest" and not epochs:
            raise FileNotFoundError(f"no epoch checkpoints in {ckpt_dir}")
        checkpoint = max(epochs) if epochs else "model_out"
    if checkpoint == "model_out":
        if not ema:
            raise ValueError(
                f"{model_out} is already the EMA shadow, so ema=False cannot be "
                f"satisfied; pick an epoch checkpoint (--ckpt_every) for raw weights.")
        if not os.path.exists(model_out):
            raise FileNotFoundError(
                f"{model_out} doesn't exist (training not finished?) and there are "
                f"no epoch checkpoints in {ckpt_dir}")
        path = model_out
    else:
        if int(checkpoint) not in epochs:
            raise FileNotFoundError(
                f"no epoch {checkpoint} in {ckpt_dir}; available: {sorted(epochs)}")
        path = epochs[int(checkpoint)]
    if prefer_zstats:
        root, ext = os.path.splitext(path)
        zs = f"{root}_zstats{ext}"
        if os.path.exists(zs):
            path = zs

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
        state_dict = ckpt["ema" if ema else "model"]
        which = "ema" if ema else "model"
    else:
        state_dict, which = ckpt, "ema"
    print(f"[load_jepa_dit_annual] loaded {which} weights from {path}")

    for k, v in emission_kwargs_from_state_dict(state_dict).items():
        model_kwargs.setdefault(k, v)
    if model_kwargs["annual_emission"]:
        h = model_kwargs["emission_hidden"]
        print(f"[load_jepa_dit_annual] annual emission head "
              f"({'linear' if not h else f'hidden {h}'})")

    model = build_model(y_dim, u_dim, device=device, **model_kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    if model.ssm_encoder.ssm.normalize_z and not bool(model.ssm_encoder.ssm.z_stats_fitted):
        opt = "" if checkpoint == "model_out" else f" --checkpoint {checkpoint}"
        print("[load_jepa_dit_annual] WARNING: no fitted z stats in these weights -> "
              "batch-stat standardization. Run python -m proto_scales.fit_jepa_kl_z_stats "
              f"--run_dir {run_dir}{opt}")

    tas_scaler = StandardScaler.from_file(os.path.join(run_dir, "y_scaler.out"))
    pr_scaler = StandardScaler.from_file(os.path.join(run_dir, "pr_scaler.out"))
    u_scaler = StandardScaler.from_file(os.path.join(run_dir, "u_scaler.out"))
    return model, tas_scaler, pr_scaler, u_scaler
