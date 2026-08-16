"""
ssm_dit
=======

SSM-conditioned outpainting diffusion transformer for regional tas/pr.

Division of labour
------------------
    SSM latent z      temporal correlation and oscillatory (ENSO-band) internal
                      variability, rolled forward causally under GSAT
    memory kernels    multi-decadal forced memory, on annual means, causal
    month embedding   the seasonal cycle
    DiT               everything local and hard to write down: the joint
                      tas-pr spatial covariance, non-Gaussian precipitation
                      marginals, and the fine structure within a window

The DiT never sees the whole GSAT trajectory. It sees a short window plus
conditioning signals that are all causal functions of u_{<=t}. Long horizons
are produced by `sample_blocks`, which slides the window forward and recomputes
the conditioning per block. Century-scale memory travels between blocks through
z and the annual-mean kernels, not through the attention window.

This is the structural difference from a DiT conditioned on the entire GSAT
trajectory at once, which can key on the global shape of the forcing path and
therefore does not transfer to trajectory shapes outside the training set.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import proto_scales.ssm_model.scales_ssm_cross_corr_osc_trans as scales_ssm_osc
from proto_scales.ssm_dit_model.annual_ssm import AnnualSSM, AnnualSSMLatentEncoder
from proto_scales.ssm_dit_model.conditioning import ConditioningBuilder, SSMLatentEncoder
from proto_scales.ssm_dit_model.dit import DiT1D
from proto_scales.ssm_dit_model.memory_kernel import MONTHS_PER_YEAR


def cosine_beta_schedule(n_steps, s=0.008):
    """Nichol & Dhariwal cosine schedule."""
    t = torch.linspace(0, n_steps, n_steps + 1, dtype=torch.float64) / n_steps
    f = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-8, 0.999).float()


class SSMConditionedOutpaintingDiT(nn.Module):
    """
    Parameters
    ----------
    ssm       : a constructed `DeepSSMPatternConditioned` (optionally pretrained).
    y_dim     : number of regions. The field has 2*y_dim channels (tas, pr).
    u_dim     : forcing dimension (1 for GSAT).
    freeze_ssm: train the DiT on a fixed latent process. Recommended first.

    Field convention
    ----------------
    Everywhere in this module the field is the concatenation
    `x = cat([tas, pr], dim=-1)` with `2 * y_dim` channels, in *normalised*
    units (apply the same StandardScalers used for the SSM).
    """

    def __init__(
        self,
        ssm,
        y_dim,
        u_dim,
        context_len=120,
        horizon=120,
        freeze_ssm=True,
        cond_dim=256,
        hidden=384,
        depth=8,
        heads=6,
        n_diffusion_steps=1000,
        timescales_years=(1.0, 5.0, 20.0, 100.0),
        learnable_timescales=True,
        field_memory_rank=32,
        parameterization="v",
        # auxiliary SSM objective (end-to-end training)
        ssm_elbo_weight=0.0,
        ssm_kl_weight=5.0,
        ssm_kl_free_bits=0.2,
        ssm_acf_weight=0.0,
        ssm_acf_max_lag=120,
        ssm_rollout_steps=0,
        ssm_rollout_samples=1,
    ):
        super().__init__()
        if parameterization not in ("v", "eps"):
            raise ValueError("parameterization must be 'v' or 'eps'")
        if int(context_len) < 2 * MONTHS_PER_YEAR:
            # With fewer than two complete years of context, every month maps to
            # the zero "year -1" state of the causal annual EMA, so the field
            # memory is identically zero. Its parameters then receive no
            # gradient, which also makes DDP raise on unreduced parameters.
            raise ValueError(
                f"context_len must be >= {2 * MONTHS_PER_YEAR} months so the "
                f"annual-mean field memory has at least one completed year to "
                f"summarise (got {context_len})")
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.n_total = self.context_len + self.horizon
        self.in_channels = 2 * y_dim
        self.n_diffusion_steps = int(n_diffusion_steps)
        # v-prediction by default: eps-prediction is numerically unstable at
        # high noise levels for this data (confirmed empirically on an earlier
        # DiT for the same fields). See `_to_x0_eps` for the mechanism.
        self.parameterization = parameterization

        # The auxiliary ELBO is what gives z an objective of its own. Without it,
        # joint training shapes z purely through the diffusion loss, and nothing
        # stops the DiT from learning to ignore a z that has drifted into noise —
        # in particular the oscillator (omega / log_amp / log_damping) has no
        # pressure at all unless the ACF term is also on.
        self.ssm_elbo_weight = float(ssm_elbo_weight)
        self.ssm_kl_weight = float(ssm_kl_weight)
        self.ssm_kl_free_bits = float(ssm_kl_free_bits)
        self.ssm_acf_weight = float(ssm_acf_weight)
        self.ssm_acf_max_lag = int(ssm_acf_max_lag)
        self.ssm_rollout_steps = int(ssm_rollout_steps)
        self.ssm_rollout_samples = int(ssm_rollout_samples)
        self.use_ssm_aux = (not freeze_ssm) and (
            self.ssm_elbo_weight > 0 or self.ssm_acf_weight > 0)
        if self.use_ssm_aux and freeze_ssm:
            raise ValueError("ssm auxiliary losses require freeze_ssm=False")

        # The two SSMs expose the same encoder interface, so everything
        # downstream (ConditioningBuilder, DiT1D, sampling) is identical.
        self.annual_ssm = isinstance(ssm, AnnualSSM)
        if self.annual_ssm:
            if self.ssm_acf_weight > 0:
                raise ValueError(
                    "AnnualSSM is trained with the ELBO alone; there is no ACF "
                    "term. Set ssm_acf_weight=0.")
            self.ssm_encoder = AnnualSSMLatentEncoder(ssm, freeze=freeze_ssm)
        else:
            self.ssm_encoder = SSMLatentEncoder(
                ssm, freeze=freeze_ssm, train_emission=self.use_ssm_aux)
        self.cond_builder = ConditioningBuilder(
            y_dim=y_dim,
            u_dim=u_dim,
            z_dim=self.ssm_encoder.z_dim,
            cond_dim=cond_dim,
            timescales_years=timescales_years,
            learnable_timescales=learnable_timescales,
            field_memory_rank=field_memory_rank,
        )
        self.dit = DiT1D(
            in_channels=self.in_channels,
            cond_dim=cond_dim,
            max_len=self.n_total,
            hidden=hidden,
            depth=depth,
            heads=heads,
        )

        betas = cosine_beta_schedule(self.n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab", alpha_bar.sqrt())
        self.register_buffer("sqrt_1mab", (1.0 - alpha_bar).sqrt())

    # ------------------------------------------------------------------
    # conditioning
    # ------------------------------------------------------------------
    def build_conditioning(self, tas_ctx, pr_ctx, u_ctx, u_fut, start_month=0,
                           stochastic_z=True):
        """Returns (cond [B, T_tot, cond_dim], field_ctx [B, Tc, 2*y_dim])."""
        field_ctx = torch.cat([tas_ctx, pr_ctx], dim=-1)
        z = self.ssm_encoder(tas_ctx, pr_ctx, u_ctx, u_fut, stochastic=stochastic_z)
        u_full = torch.cat([u_ctx, u_fut], dim=1)
        cond = self.cond_builder(
            z=z,
            u_full=u_full,
            field_ctx=field_ctx,
            n_ctx=tas_ctx.shape[1],
            n_total=tas_ctx.shape[1] + u_fut.shape[1],
            start_month=start_month,
        )
        return cond, field_ctx

    def _mask(self, B, n_ctx, n_total, device):
        """1 on observed context positions, 0 on generated positions."""
        mask = torch.zeros(B, n_total, 1, device=device)
        mask[:, :n_ctx] = 1.0
        return mask

    def _to_x0_eps(self, pred, x_t, t):
        """
        Convert the network output to (x0_hat, eps_hat) for the current
        parameterisation. `t` is a scalar timestep index.

        With v-parameterisation, v = a*eps - s*x0 and a^2 + s^2 = 1, so the
        inverse is the rotation
            x0 = a*x_t - s*v,      eps = s*x_t + a*v
        which stays well conditioned at every noise level. The eps route,
        x0 = (x_t - s*eps)/a, divides by a -> 0 at high noise and amplifies any
        error in eps_hat without bound; that is the instability v-prediction
        removes.
        """
        a = self.sqrt_ab[t]
        s = self.sqrt_1mab[t]
        if self.parameterization == "v":
            x0_hat = a * x_t - s * pred
            eps_hat = s * x_t + a * pred
        else:
            eps_hat = pred
            x0_hat = (x_t - s * eps_hat) / a.clamp(min=1e-8)
        x0_hat = x0_hat.clamp(-10.0, 10.0)
        # Re-derive eps from the clamped x0 so the pair stays self-consistent;
        # an inconsistent (x0, eps) pair injects a bias into the DDIM step.
        eps_hat = (x_t - a * x0_hat) / s.clamp(min=1e-8)
        return x0_hat, eps_hat

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        """
        Alias for `loss` so the module can be called through a DDP wrapper.

        This matters: DDP synchronises gradients via hooks armed by the forward
        pass on the *wrapped* module. Calling an inner method directly (e.g.
        `raw_model.loss(...)`) bypasses the reducer, and each rank then trains
        on its own shard without ever averaging gradients.
        """
        return self.loss(*args, **kwargs)

    def ssm_aux_loss(self, tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut):
        """
        The SSM's own objective, evaluated inside the DiT forward so it is part
        of the same DDP reduction.

        Two terms, with different jobs:
          ELBO  keeps z a genuine latent state of the data rather than whatever
                the diffusion loss happens to find convenient;
          ACF   is the only thing that gives the oscillator (omega, log_amp,
                log_damping) any reason to occupy the ENSO band. Without it the
                oscillatory inductive bias is nominal.

        Returns (aux_total, parts_dict).
        """
        device = tas_ctx.device
        if self.annual_ssm:
            # ELBO only, already normalised per element by AnnualSSM.
            nll, kl = self.ssm_encoder.elbo(
                tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut,
                kl_free_bits=self.ssm_kl_free_bits)
            elbo = nll + self.ssm_kl_weight * kl
            return (self.ssm_elbo_weight * elbo,
                    {"ssm_nll": nll.detach(), "ssm_kl": kl.detach()})

        ssm = self.ssm_encoder.ssm
        y_full = torch.cat([tas_ctx, tas_fut], dim=1)
        pr_full = torch.cat([pr_ctx, pr_fut], dim=1)
        u_full = torch.cat([u_ctx, u_fut], dim=1)

        parts = {}
        total = torch.zeros((), device=device)

        if self.ssm_elbo_weight > 0:
            nll, kl, _ = ssm.forward_elbo(
                y_full, pr_full, u_full, kl_free_bits=self.ssm_kl_free_bits)
            # Normalise to per-element. forward_elbo sums over time and channels,
            # so the raw ELBO is O(10^4) against a diffusion loss of O(1) — and,
            # worse, it rescales whenever T or the number of indicators changes.
            # Per-element normalisation makes ssm_elbo_weight mean the same thing
            # across configurations, which is the whole point of training
            # end-to-end when new indicators are added.
            T = y_full.shape[1]
            nll_n = nll / (T * 2 * self.y_dim)
            kl_n = kl / (T * ssm.z_dim)
            elbo = nll_n + self.ssm_kl_weight * kl_n
            total = total + self.ssm_elbo_weight * elbo
            parts["ssm_nll"] = nll_n.detach()
            parts["ssm_kl"] = kl_n.detach()

        if self.ssm_acf_weight > 0:
            R = min(self.ssm_rollout_steps or u_fut.shape[1], u_fut.shape[1])
            eff_lag = min(self.ssm_acf_max_lag, (R - 1) // 2)
            if eff_lag >= 2:
                tas_s, pr_s = ssm.rollout_samples(
                    tas_ctx, u_ctx, u_fut[:, :R], steps=R,
                    n_samples=self.ssm_rollout_samples)
                B, Dy = tas_ctx.shape[0], self.y_dim
                if ssm.use_linear_model:
                    with torch.no_grad():
                        ctrl = ssm.ctrl_lin(
                            u_fut[:, :R].reshape(-1, self.u_dim)).reshape(B, R, Dy)
                else:
                    ctrl = torch.zeros_like(tas_fut[:, :R])
                acf_true_tas = scales_ssm_osc.batch_acf(tas_fut[:, :R] - ctrl, eff_lag)
                acf_true_pr = scales_ssm_osc.batch_acf(pr_fut[:, :R], eff_lag)
                acf = torch.zeros((), device=device)
                for si in range(tas_s.shape[0]):
                    acf = acf + (
                        ((scales_ssm_osc.batch_acf(tas_s[si] - ctrl, eff_lag)
                          - acf_true_tas) ** 2).mean()
                        + ((scales_ssm_osc.batch_acf(pr_s[si], eff_lag)
                            - acf_true_pr) ** 2).mean())
                acf = acf / tas_s.shape[0]
                total = total + self.ssm_acf_weight * acf
                parts["ssm_acf"] = acf.detach()

        return total, parts

    def loss(self, tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut, start_month=0,
             t_diff=None, return_parts=False):
        """
        Diffusion loss on the generated positions only, in whichever
        parameterisation the model was built with (v by default).

        The context positions are held at their clean values throughout, so the
        denoiser always sees an intact history — that is the outpainting
        condition, and it is what the model will see at sampling time.

        `t_diff` may be supplied to fix the diffusion timesteps (used by the
        validation loop, where a stratified sweep gives a far lower-variance
        estimate than random draws and makes early stopping meaningful).
        """
        B = tas_ctx.shape[0]
        device = tas_ctx.device
        n_ctx = tas_ctx.shape[1]
        n_fut = tas_fut.shape[1]
        n_total = n_ctx + n_fut

        cond, field_ctx = self.build_conditioning(
            tas_ctx, pr_ctx, u_ctx, u_fut, start_month=start_month, stochastic_z=True)

        field_fut = torch.cat([tas_fut, pr_fut], dim=-1)
        x0 = torch.cat([field_ctx, field_fut], dim=1)               # [B, T, C]
        mask = self._mask(B, n_ctx, n_total, device)                # 1 = observed

        if t_diff is None:
            t_diff = torch.randint(0, self.n_diffusion_steps, (B,), device=device)
        eps = torch.randn_like(x0)
        a = self.sqrt_ab[t_diff].view(B, 1, 1)          # sqrt(alpha_bar)
        s = self.sqrt_1mab[t_diff].view(B, 1, 1)        # sqrt(1 - alpha_bar)
        x_noisy = a * x0 + s * eps
        # keep the context clean
        x_in = mask * x0 + (1.0 - mask) * x_noisy

        pred = self.dit(x_in, t_diff, cond, mask)
        target = (a * eps - s * x0) if self.parameterization == "v" else eps

        gen = (1.0 - mask)
        denom = gen.sum() * x0.shape[-1]
        diff_loss = ((pred - target) ** 2 * gen).sum() / denom.clamp(min=1.0)

        parts = {"diffusion": diff_loss.detach()}
        total = diff_loss
        if self.use_ssm_aux:
            aux, aux_parts = self.ssm_aux_loss(
                tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut)
            total = total + aux
            parts.update(aux_parts)
            parts["ssm_aux_weighted"] = aux.detach()

        return (total, parts) if return_parts else total

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, tas_ctx, pr_ctx, u_ctx, u_fut, start_month=0, n_steps=None,
               stochastic_z=True, generator=None):
        """
        Ancestral DDPM sampling of one window. Returns (tas, pr), each
        [B, horizon, y_dim].

        `n_steps` strides the reverse chain (n_steps < n_diffusion_steps gives a
        faster, slightly coarser sampler).
        """
        B = tas_ctx.shape[0]
        device = tas_ctx.device
        n_ctx = tas_ctx.shape[1]
        n_fut = u_fut.shape[1]
        n_total = n_ctx + n_fut

        cond, field_ctx = self.build_conditioning(
            tas_ctx, pr_ctx, u_ctx, u_fut, start_month=start_month,
            stochastic_z=stochastic_z)
        mask = self._mask(B, n_ctx, n_total, device)

        x = torch.randn(B, n_total, self.in_channels, device=device, generator=generator)
        x = mask * F.pad(field_ctx, (0, 0, 0, n_fut)) + (1.0 - mask) * x

        total = self.n_diffusion_steps
        n_steps = total if n_steps is None else int(n_steps)
        step_idx = torch.linspace(total - 1, 0, n_steps, device=device).long()

        for i, t in enumerate(step_idx):
            t_batch = t.repeat(B)
            pred = self.dit(x, t_batch, cond, mask)

            ab_t = self.alpha_bar[t]
            x0_hat, eps_hat = self._to_x0_eps(pred, x, t)

            if i < len(step_idx) - 1:
                t_prev = step_idx[i + 1]
                ab_prev = self.alpha_bar[t_prev]
                # DDIM-style deterministic step with a stochastic component
                sigma = ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
                dir_xt = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_hat
                noise = torch.randn(x.shape, device=device, generator=generator)
                x = ab_prev.sqrt() * x0_hat + dir_xt + sigma * noise
            else:
                x = x0_hat

            # re-impose the observed context at every step
            x = mask * F.pad(field_ctx, (0, 0, 0, n_fut)) + (1.0 - mask) * x

        fut = x[:, n_ctx:]
        return fut[..., : self.y_dim], fut[..., self.y_dim:]

    @torch.no_grad()
    def sample_blocks(self, tas_ctx, pr_ctx, u_ctx, u_fut, start_month=0,
                      n_steps=None, stochastic_z=True, generator=None):
        """
        Long-horizon generation by sliding the window forward.

        `u_fut` may be arbitrarily long. Each block generates `self.horizon`
        months, then the context slides to include what was just generated and
        the conditioning (z rollout, annual-mean memory kernels, month phase) is
        recomputed. Long-range information therefore propagates through the
        conditioning rather than through attention, so cost is linear in horizon
        instead of quadratic, and the model is never asked to attend over a
        1000-month window it was not trained on.

        Returns (tas, pr), each [B, len(u_fut), y_dim].
        """
        H_total = u_fut.shape[1]
        tas_hist, pr_hist, u_hist = tas_ctx, pr_ctx, u_ctx
        month = int(start_month)
        out_tas, out_pr = [], []

        done = 0
        while done < H_total:
            blk = min(self.horizon, H_total - done)
            u_blk = u_fut[:, done:done + blk]

            tas_b, pr_b = self.sample(
                tas_hist[:, -self.context_len:],
                pr_hist[:, -self.context_len:],
                u_hist[:, -self.context_len:],
                u_blk,
                start_month=month,
                n_steps=n_steps,
                stochastic_z=stochastic_z,
                generator=generator,
            )
            out_tas.append(tas_b)
            out_pr.append(pr_b)

            tas_hist = torch.cat([tas_hist, tas_b], dim=1)
            pr_hist = torch.cat([pr_hist, pr_b], dim=1)
            u_hist = torch.cat([u_hist, u_blk], dim=1)
            month = (month + blk) % MONTHS_PER_YEAR
            done += blk

        return torch.cat(out_tas, dim=1), torch.cat(out_pr, dim=1)


def build_model(y_dim, u_dim, z_dim=64, ssm_weights=None, device="cpu", **kwargs):
    """
    Convenience constructor: builds the SSM, optionally loads SSM-only weights,
    and wraps it in the DiT.

    `ssm_weights` should be a checkpoint from the SSM training run
    (`train_scales_ssm_cross_corr.py`). Unexpected keys are ignored, so an SSM
    checkpoint containing emission heads loads cleanly even though the DiT does
    not use them.
    """
    import proto_scales.ssm_model.scales_ssm_cross_corr_osc_trans as ssm_mod

    ssm = ssm_mod.DeepSSMPatternConditioned(
        y_dim=y_dim, u_dim=u_dim, z_dim=z_dim,
        rnn_hidden=kwargs.pop("rnn_hidden", 256),
        emission_uses_u=kwargs.pop("emission_uses_u", True),
        use_linear_model=kwargs.pop("use_linear_model", True),
        cov_rank=kwargs.pop("cov_rank", 8),
    )
    if ssm_weights is not None:
        ckpt = torch.load(ssm_weights, map_location="cpu")
        missing, unexpected = ssm.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"[build_model] SSM missing keys: {missing}")
        if unexpected:
            print(f"[build_model] SSM unexpected keys ignored: {len(unexpected)}")

    model = SSMConditionedOutpaintingDiT(ssm=ssm, y_dim=y_dim, u_dim=u_dim, **kwargs)
    return model.to(device)
