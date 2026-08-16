"""
conditioning
============

Builds the per-timestep conditioning sequence that the outpainting DiT is
conditioned on. Three signals, each responsible for one thing:

  z_t          internal variability with oscillatory (ENSO-like) structure,
               from the SSM latent state
  memory       multi-decadal forced memory of GSAT and of the regional fields,
               from causal exponential kernels on annual means
  month embed  the seasonal cycle, explicitly

The split is deliberate. In the pure-SSM model all three had to be carried by
`z`, which is why a slow reservoir had to be bolted on: a state fast enough for
the seasonal cycle cannot integrate a century of forcing. Here `z` only has to
do the first job.

Train/inference consistency of z
--------------------------------
This is the subtle part. Over the *context* window the SSM posterior
q(z_t | y_{<=t}, u_{<=t}) is available at inference too, so the posterior mean
is used. Over the *future* window it is not, so `z` must come from the prior
rollout under u_fut — during training as well. Conditioning the DiT on a
posterior that saw the future would leak the target and produce a model that
collapses at sampling time, when only the prior is available.
"""

import torch
import torch.nn as nn

from proto_scales.ssm_dit_model.memory_kernel import MultiTimescaleMemory, MONTHS_PER_YEAR


class SSMLatentEncoder(nn.Module):
    """
    Wraps a `scales_ssm_cross_corr_osc_trans.DeepSSMPatternConditioned` and
    exposes only its latent trajectory.

    The emission heads of the SSM are unused here: the DiT replaces them. What
    is kept is the inference GRU (for the context posterior) and the
    oscillatory transition `_osc_transition` (for the future prior rollout),
    which is the part that carries the temporal-correlation inductive bias.

    Parameters
    ----------
    ssm    : a constructed DeepSSMPatternConditioned, optionally with weights
             loaded from an SSM-only training run.
    freeze : if True the SSM parameters are frozen and the DiT is trained on a
             fixed latent process. Recommended for a first run — it makes the
             DiT's job well-posed and stops the SSM from degenerating into a
             constant once the DiT can explain the data on its own.
    """

    # SSM components the DiT does not use: the emission head, the pattern-scaling
    # term and the slow reservoir are all replaced by the DiT and the memory
    # kernels. Only the inference GRU, q_head and the oscillatory transition are
    # on the path from z to the loss.
    UNUSED_PREFIXES = ("emit.", "ctrl_lin.", "u_gru.", "omega_lin.", "log_alpha")

    def __init__(self, ssm, freeze=True, train_emission=False):
        super().__init__()
        self.ssm = ssm
        self.z_dim = ssm.z_dim
        self.frozen = bool(freeze)
        self.train_emission = bool(train_emission)
        if self.frozen:
            for p in self.ssm.parameters():
                p.requires_grad = False
        elif not self.train_emission:
            # Freeze the unused components in joint-training mode. They receive
            # no gradient from the diffusion loss, and DDP raises on parameters
            # that require grad but are never reduced. Semantically they are
            # also undefined here: without the SSM's own ELBO there is no
            # objective that would train them.
            for name, p in self.ssm.named_parameters():
                if name.startswith(self.UNUSED_PREFIXES):
                    p.requires_grad = False
        # else: the auxiliary ELBO is active, so the emission head and reservoir
        # do have an objective and stay trainable. `ctrl_lin` is handled
        # separately (ridge-initialised and frozen, as in SSM training).

    def trainable_ssm_parameters(self):
        return [n for n, p in self.ssm.named_parameters() if p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.ssm.eval()
        return self

    def context_posterior(self, y_ctx, u_ctx):
        """Posterior mean over the context window. Returns ([B,Tc,z], [B,z])."""
        ssm = self.ssm
        h, _ = ssm.gru(torch.cat([y_ctx, u_ctx], dim=-1))
        mu_q, logvar_q = torch.chunk(ssm.q_head(h), 2, dim=-1)
        return mu_q, mu_q[:, -1]

    def future_prior(self, z_last, u_fut, stochastic=True):
        """
        Roll the oscillatory prior forward under u_fut. Returns [B, H, z].

        `stochastic=True` draws z_t from the transition, so different DiT
        samples get different ENSO-phase realisations rather than all being
        conditioned on the same mean trajectory.
        """
        ssm = self.ssm
        z = z_last
        out = []
        for k in range(u_fut.shape[1]):
            mu_p, logvar_p = ssm._osc_transition(z, u_fut[:, k])
            logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
            z = ssm.sample(mu_p, logvar_p) if stochastic else mu_p
            z = torch.clamp(z, -10.0, 10.0)
            out.append(z)
        return torch.stack(out, dim=1)

    def forward(self, y_ctx, pr_ctx, u_ctx, u_fut, stochastic=True):
        """
        Returns z over the whole window, [B, Tc + H, z_dim].

        Posterior over the context, prior rollout over the future — see the
        module docstring on why the future half must not use the posterior.

        `pr_ctx` is accepted but unused: this SSM's inference GRU was built to
        see tas and u only. It is in the signature so that this class and
        `annual_ssm.AnnualSSMLatentEncoder` — which does use pr — are
        interchangeable from the DiT's point of view.
        """
        ctx = torch.enable_grad() if not self.frozen else torch.no_grad()
        with ctx:
            z_ctx, z_last = self.context_posterior(y_ctx, u_ctx)
            z_fut = self.future_prior(z_last, u_fut, stochastic=stochastic)
        z = torch.cat([z_ctx, z_fut], dim=1)
        return z.detach() if self.frozen else z


class ConditioningBuilder(nn.Module):
    """
    Assembles z, memory-kernel features, month-of-year and instantaneous
    forcing into one per-timestep conditioning vector.

    Returns [B, T_tot, cond_dim].
    """

    def __init__(
        self,
        y_dim,
        u_dim,
        z_dim,
        cond_dim=256,
        timescales_years=(1.0, 5.0, 20.0, 100.0),
        learnable_timescales=True,
        field_memory_rank=32,
        month_embed_dim=32,
    ):
        super().__init__()
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        # Separate kernel banks: the forcing memory and the field memory have
        # no reason to share timescales.
        self.mem_u = MultiTimescaleMemory(timescales_years, learnable=learnable_timescales)
        self.mem_y = MultiTimescaleMemory(timescales_years, learnable=learnable_timescales)

        n_tau = self.mem_u.n_tau
        # The field memory is D*n_tau wide (D ~ 50 regions), which is large and
        # highly redundant across regions; project it down before it reaches the
        # conditioning MLP.
        self.field_mem_proj = nn.Linear(2 * y_dim * n_tau, field_memory_rank)

        self.month_embed = nn.Embedding(MONTHS_PER_YEAR, month_embed_dim)

        in_dim = z_dim + u_dim * n_tau + field_memory_rank + month_embed_dim + u_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, z, u_full, field_ctx, n_ctx, n_total, start_month=0):
        """
        Parameters
        ----------
        z         : [B, T_tot, z_dim]   latent trajectory (context + future)
        u_full    : [B, T_tot, u_dim]   forcing over the whole window
        field_ctx : [B, Tc, 2*y_dim]    observed field over the context only
        n_ctx     : int, context length
        n_total   : int, context + horizon
        start_month : calendar month index (0-11) of the first step

        The field memory is computed from the context field only and then held
        constant across the future window. It cannot be computed from the future
        field: that is what is being generated. Holding it fixed is the honest
        choice at a window length where the annual-mean memory barely moves; for
        long horizons, generate in blocks and recompute per block (see
        `ssm_dit.sample_blocks`).
        """
        B = z.shape[0]
        device = z.device

        mem_u = self.mem_u(u_full)                                  # [B,T,u*K]

        mem_y_ctx = self.mem_y(field_ctx)                           # [B,Tc,2D*K]
        last = mem_y_ctx[:, -1:] if n_ctx > 0 else mem_y_ctx.new_zeros(
            B, 1, mem_y_ctx.shape[-1])
        mem_y = torch.cat([mem_y_ctx, last.expand(B, n_total - n_ctx, -1)], dim=1)
        mem_y = self.field_mem_proj(mem_y)                          # [B,T,rank]

        months = (torch.arange(n_total, device=device) + int(start_month)) % MONTHS_PER_YEAR
        m_emb = self.month_embed(months).unsqueeze(0).expand(B, -1, -1)

        c = torch.cat([z, mem_u, mem_y, m_emb, u_full], dim=-1)
        return self.proj(c)
