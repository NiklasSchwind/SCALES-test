import copy
import logging
import math
import numpy as np
import sys
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from proto_scales.ssm_model.scales_ssm_z2tasAndpr_hysteresis import (
    UnifiedWindowDataset,
    DeepSSMPatternConditioned,
    sinh_arcsinh_flow_nll_conditional,
    sinh_arcsinh_forward,
)
from proto_scales.ssm_model.scales_ssm import StandardScaler
import proto_scales.ssm_model.scales_ssm as scales_ssm
import proto_scales.data_prep.prepare_data as prep





# ---------------------------------------------------------------------------
# Build task dictionary
# ---------------------------------------------------------------------------

def build_task_dict(
    esm_data,
    context_len=600,
    horizon=1200,
    stride=1,
    support_frac=0.7,
    start_mode="all",
    run_dir=None,
):
    """
    Build a MAML task dictionary from multi-ESM data, with global cross-ESM scalers.

    Scalers are fitted on the pooled support data from all ESMs so that every ESM
    is normalised consistently.  Fitting only on support (not query) data follows
    the same convention as run_train in scales_ssm_z2tasAndpr_hysteresis.py.

    Parameters
    ----------
    esm_data : dict
        Keys are ESM names (str). Each value is a dict with:
          'y'      : np.ndarray [N_scenarios, T, Dy]  -- regional temperature anomalies
          'pr'     : np.ndarray [N_scenarios, T, Dy]  -- regional precipitation anomalies
          'u'      : np.ndarray [N_scenarios, T, Du]  -- GMT forcing
          'weight' : float                            -- weight in meta-learning
    context_len : int
        Context window length passed to UnifiedWindowDataset.
    horizon : int
        Forecast horizon passed to UnifiedWindowDataset.
    stride : int
        Stride between consecutive window starts.
    support_frac : float
        Fraction of scenarios (by index) used for the support set.
        Remaining scenarios form the query set.
    start_mode : str
        Window sampling mode passed to UnifiedWindowDataset ("all" or "zero").
    run_dir : str or None
        If provided, the three global scalers are saved here as
        y_scaler.out / pr_scaler.out / u_scaler.out.

    Returns
    -------
    tasks : dict
        {
          esm_name: {
            'support' : UnifiedWindowDataset,  (normalised)
            'query'   : UnifiedWindowDataset,  (normalised)
            'weight'  : float,
          },
          ...
        }
    scalers : dict
        {'y': StandardScaler, 'pr': StandardScaler, 'u': StandardScaler}
        Global scalers fitted on the pooled support data.
    """
    # ---- Pass 1: validate shapes and compute support/query splits ----
    raw = {}
    for esm_name, data in esm_data.items():
        y      = np.asarray(data['y'],  dtype=np.float32)
        pr     = np.asarray(data['pr'], dtype=np.float32)
        u      = np.asarray(data['u'],  dtype=np.float32)
        weight = float(data.get('weight', 1.0))

        if y.ndim == 2:
            y  = y[None]
            pr = pr[None]
            u  = u[None]

        N = y.shape[0]
        n_support = max(1, int(np.floor(support_frac * N)))
        n_query   = N - n_support

        if n_query < 1:
            raise ValueError(
                f"ESM '{esm_name}' has only {N} scenario(s); "
                f"support_frac={support_frac} leaves no scenarios for the query set."
            )

        raw[esm_name] = {
            'y': y, 'pr': pr, 'u': u,
            'weight': weight, 'n_support': n_support,
        }

    # ---- Fit global scalers on pooled support data from all ESMs ----
    # Using only support (not query) data mirrors the train-only convention in run_train.
    all_y_sup  = np.concatenate([v['y'][:v['n_support']] for v in raw.values()], axis=0)
    all_pr_sup = np.concatenate([v['pr'][:v['n_support']] for v in raw.values()], axis=0)
    all_u_sup  = np.concatenate([v['u'][:v['n_support']] for v in raw.values()], axis=0)

    y_scaler  = StandardScaler().fit(all_y_sup)
    pr_scaler = StandardScaler().fit(all_pr_sup)
    u_scaler  = StandardScaler().fit(all_u_sup)

    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        y_scaler.save(os.path.join(run_dir,  "y_scaler.out"))
        pr_scaler.save(os.path.join(run_dir, "pr_scaler.out"))
        u_scaler.save(os.path.join(run_dir,  "u_scaler.out"))

    # ---- Pass 2: normalise and build UnifiedWindowDatasets ----
    tasks = {}
    for esm_name, d in raw.items():
        n_sup = d['n_support']

        y_n  = y_scaler.transform(d['y'])
        pr_n = pr_scaler.transform(d['pr'])
        u_n  = u_scaler.transform(d['u'])

        support_ds = UnifiedWindowDataset(
            y_n[:n_sup], pr_n[:n_sup], u_n[:n_sup],
            context_len=context_len,
            horizon=horizon,
            stride=stride,
            start_mode=start_mode,
        )
        query_ds = UnifiedWindowDataset(
            y_n[n_sup:], pr_n[n_sup:], u_n[n_sup:],
            context_len=context_len,
            horizon=horizon,
            stride=stride,
            start_mode=start_mode,
        )

        tasks[esm_name] = {
            'support': support_ds,
            'query':   query_ds,
            'weight':  d['weight'],
        }

    scalers = {'y': y_scaler, 'pr': pr_scaler, 'u': u_scaler}
    return tasks, scalers


# ---------------------------------------------------------------------------
# Loss function — mirrors run_train in scales_ssm_z2tasAndpr_hysteresis.py
# ---------------------------------------------------------------------------

def compute_task_loss(
    model,
    batch,
    device,
    kl_w=5,
    alpha_w=10000,
    omega_w=500,
    gamma_w=80000,
    n_rollout_samples=10,
):
    """
    Compute the full training loss for one batch, matching the loss in run_train.

    Batch tuple: (y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut) — from UnifiedWindowDataset.

    ELBO terms (nll, kl, nll_pr) flow gradients through model parameters.
    Rollout terms (Huber MSE for tas/pr, MSE for yearly tas) are computed via
    forecast_deterministic (@torch.no_grad), so they add to the scalar loss value
    but do not contribute gradient signal — consistent with run_train.
    """
    y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut = [
        t.float().to(device) for t in batch
    ]
    y_full  = torch.cat([y_ctx,  y_fut],  dim=1)
    pr_full = torch.cat([pr_ctx, pr_fut], dim=1)
    u_full  = torch.cat([u_ctx,  u_fut],  dim=1)

    B, _, Dy = y_full.shape
    horizon  = y_fut.shape[1]

    # ELBO: nll (Gaussian, tas), nll_pr (sinh-arcsinh flow, pr), KL
    nll, kl, nll_pr = model.forward_elbo(y_full, pr_full, u_full, kl_free_bits=0.2)

    # Rollout MSE (detached — forecast_deterministic is @torch.no_grad)
    mean, _, _, mean_pr, _, _ = model.forecast_deterministic(
        y_ctx, u_ctx, u_fut, steps=horizon, n_samples=n_rollout_samples
    )
    roll_out_mse    = F.huber_loss(mean,    y_fut,  delta=1.0)
    roll_out_mse_pr = F.huber_loss(mean_pr, pr_fut, delta=1.0)

    # Yearly-average MSE on tas (no warmup, constant gamma)
    n_complete_years = horizon // 12
    if n_complete_years > 0:
        H_yr       = n_complete_years * 12
        mean_yr    = mean[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
        y_fut_yr   = y_fut[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
        mse_yearly = ((mean_yr - y_fut_yr) ** 2).mean()
    else:
        mse_yearly = torch.tensor(0.0, device=device)

    return (
        nll + nll_pr
        + kl_w    * kl
        + alpha_w * roll_out_mse
        + omega_w * roll_out_mse_pr
        + gamma_w * mse_yearly
    ), nll.item(), nll_pr.item(), kl.item(), roll_out_mse.item(), roll_out_mse_pr.item(), mse_yearly.item()


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        dims = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ContextEncoderForCnp(nn.Module):
    """
    Maps a single (forcing, state) pair from ESM-i to a
    fixed-dim representation r. Applied independently to
    every point in the context set — no attention, no ordering.
    Every state is all climate variables e.g. concat([tas,pr],dim=-1)
    """
    def __init__(self, x_dim, y_dim, r_dim):
        super().__init__()
        self.mlp = MLP(
            in_dim  = x_dim + y_dim,   # forcing + ESM state concatenated
            out_dim = r_dim,
            hidden  = [256, 256]
        )

    def forward(self, x_context, y_context):
        # x_context: [B, N_ctx, x_dim]
        # y_context: [B, N_ctx, y_dim]
        xy = torch.cat([x_context, y_context], dim=-1)   # [B, N_ctx, x+y]
        return self.mlp(xy)                               # [B, N_ctx, r_dim]

def aggregate(r_context):
    """
    Collapse N_ctx representations into one, permutation-invariantly.
    Mean is the canonical CNP choice; alternatives below.
    """
    # r_context: [B, N_ctx, r_dim]
    r_agg = r_context.mean(dim=1)    # [B, r_dim]
    return r_agg

class CnpLatentEncoder(nn.Module):
    def __init__(self, r_dim, z_dim):
        super().__init__()
        self.to_mu    = nn.Linear(r_dim, z_dim)
        self.to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r_agg):
        mu    = self.to_mu(r_agg)
        sigma = 0.1 + 0.9 * F.softplus(self.to_sigma(r_agg))  # strictly positive
        z     = mu + sigma * torch.randn_like(sigma)            # reparameterization
        return z, mu, sigma                                     # [B, z_dim] each

class DeepCnpSsmforESM(nn.Module):
    """
    Conditional Neural Process wrapper around DeepSSMPatternConditioned.

    A context set of (u, y, pr) point-pairs from the target ESM is encoded to a
    task-specific latent z_cnp via ContextEncoderForCnp + CnpLatentEncoder.
    z_cnp modulates every SSM emission step via FiLM (Feature-wise Linear Modulation):
        e_in_conditioned = gamma(z_cnp) * e_in + beta(z_cnp)
    applied before emit / emit_pr, allowing the ESM embedding to both rescale and
    shift how the emission networks respond to the current SSM state.

    Training objective:
        nll + nll_pr + kl_ssm_w * kl_ssm + kl_cnp_w * kl_cnp
    where kl_cnp = KL( q(z_cnp | context) || N(0,I) ).
    """

    def __init__(self, ssm_model: DeepSSMPatternConditioned, r_dim, z_cnp_dim):
        super().__init__()
        self.ssm = ssm_model
        self.z_cnp_dim = z_cnp_dim

        # CNP encoder stack
        # x = forcing u, y = [tas, pr] concatenated → state_dim = 2 * y_dim
        x_dim     = ssm_model.u_dim
        state_dim = 2 * ssm_model.y_dim
        self.ctx_encoder = ContextEncoderForCnp(x_dim=x_dim, y_dim=state_dim, r_dim=r_dim)
        self.lat_encoder = CnpLatentEncoder(r_dim=r_dim, z_dim=z_cnp_dim)

        # FiLM layers: project z_cnp to scale (gamma) and shift (beta) for e_in
        if ssm_model.emission_uses_u:
            emit_in_dim = (ssm_model.z_dim
                           + ssm_model.u_rnn_hidden
                           + ssm_model.y_dim * ssm_model.reservoir_dim)
        else:
            emit_in_dim = ssm_model.z_dim
        self.z_cnp_scale = nn.Linear(z_cnp_dim, emit_in_dim)
        self.z_cnp_shift = nn.Linear(z_cnp_dim, emit_in_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_context(self, u_cnp, y_cnp, pr_cnp):
        """
        u_cnp  : [B, N_ctx, u_dim]
        y_cnp  : [B, N_ctx, y_dim]
        pr_cnp : [B, N_ctx, y_dim]
        Returns z_cnp, mu_cnp, sigma_cnp  — each [B, z_cnp_dim]
        """
        state = torch.cat([y_cnp, pr_cnp], dim=-1)   # [B, N_ctx, 2*y_dim]
        r     = self.ctx_encoder(u_cnp, state)        # [B, N_ctx, r_dim]
        r_agg = aggregate(r)                          # [B, r_dim]
        return self.lat_encoder(r_agg)                # z_cnp, mu_cnp, sigma_cnp

    def _kl_cnp(self, mu, sigma):
        """KL( N(mu, sigma^2) || N(0,I) ), summed over z_cnp_dim, mean over batch."""
        return (-0.5 * (1.0 + 2.0 * sigma.log() - mu.pow(2) - sigma.pow(2))
                ).sum(-1).mean()

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------

    def forward_elbo(
        self,
        y, pr, u,
        u_cnp, y_cnp, pr_cnp,
        kl_free_bits=0.5,
        kl_cnp_w=1.0,
    ):
        """
        y, pr, u         : [B, T, *]      — target (query) sequence
        u_cnp, y_cnp, pr_cnp : [B, N_ctx, *] — context set from the same ESM

        Returns nll, kl_ssm, nll_pr, kl_cnp  (all scalar tensors).
        """
        z_cnp, mu_cnp, sigma_cnp = self._encode_context(u_cnp, y_cnp, pr_cnp)
        kl_cnp     = self._kl_cnp(mu_cnp, sigma_cnp)
        film_gamma = self.z_cnp_scale(z_cnp)   # [B, emit_in_dim]
        film_beta  = self.z_cnp_shift(z_cnp)   # [B, emit_in_dim]

        ssm = self.ssm
        B, T, _ = y.shape

        rnn_in  = torch.cat([y, u], dim=-1)
        h, _    = ssm.gru(rnn_in)
        q_params = ssm.q_head(h)
        mu_q, logvar_q = torch.chunk(q_params, 2, dim=-1)
        logvar_q = torch.clamp(logvar_q, -12.0, 6.0)

        mu_p0    = torch.zeros(B, ssm.z_dim, device=y.device)
        logvar_p0 = torch.zeros(B, ssm.z_dim, device=y.device)

        nll    = 0.0
        nll_pr = 0.0
        kl_ssm = 0.0

        if ssm.emission_uses_u:
            uh, _ = ssm.u_gru(u)
            s = ssm.omega_lin(uh[:, 0]).reshape(B, ssm.y_dim, ssm.reservoir_dim).detach()

        z_prev = None
        for t in range(T):
            z_t = ssm.sample(mu_q[:, t], logvar_q[:, t])

            if ssm.use_linear_model:
                ctrl = ssm.ctrl_lin(u[:, t])

            if ssm.emission_uses_u:
                e_in = torch.cat([z_t, uh[:, t], s.reshape(B, -1)], dim=-1)
            else:
                e_in = z_t

            e_in = film_gamma * e_in + film_beta   # FiLM: ESM-specific scale and shift

            emit_out = ssm.emit(e_in)
            res, log_sigma_y_t = torch.chunk(emit_out, 2, dim=-1)
            log_sigma_y_t = torch.clamp(log_sigma_y_t, -8.0, 4.0)
            sigma_y_t = torch.exp(log_sigma_y_t) + ssm.eps
            y_hat = ctrl + res if ssm.use_linear_model else res

            r_res = (y[:, t] - y_hat) / sigma_y_t
            nll_t = (0.5 * r_res**2 + log_sigma_y_t + 0.5 * math.log(2.0 * math.pi)).sum(-1)
            nll   = nll + nll_t

            out = ssm.emit_pr(e_in)
            mu_t, log_sigma_t, eps_skew_t, log_delta_t = torch.chunk(out, 4, dim=-1)
            nll_pr_t = sinh_arcsinh_flow_nll_conditional(
                pr[:, t], mu_t, log_sigma_t, eps_skew_t, log_delta_t, eps=ssm.eps
            )
            nll_pr = nll_pr + nll_pr_t

            if t == 0:
                kl_t = scales_ssm.diag_gaussian_kl(mu_q[:, 0], logvar_q[:, 0], mu_p0, logvar_p0)
            else:
                trans_in = torch.cat([z_prev, u[:, t]], dim=-1)
                mu_p, logvar_p = torch.chunk(ssm.trans(trans_in), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                kl_t = scales_ssm.diag_gaussian_kl(mu_q[:, t], logvar_q[:, t], mu_p, logvar_p)

            kl_ssm = kl_ssm + torch.clamp(kl_t, min=kl_free_bits)

            if ssm.emission_uses_u:
                s = ssm._reservoir_step(s, uh[:, t])
            z_prev = z_t

        nll    = nll.mean()
        nll_pr = nll_pr.mean()
        kl_ssm = kl_ssm.mean()
        return nll, kl_ssm, nll_pr, kl_cnp

    # ------------------------------------------------------------------
    # Forecast (stochastic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forecast(
        self,
        y_ctx, u_ctx, u_fut, steps, n_samples=50,
        pr_ctx=None,
    ):
        """
        y_ctx, u_ctx : [B, Tc, *]   — SSM warm-up context (time series); reused as CNP context.
        u_fut        : [B, steps, u_dim]
        pr_ctx       : [B, Tc, y_dim] — precipitation context for CNP encoding.
            If None, z_cnp = 0 (no ESM-specific adaptation).

        Returns mean, q10, q90 for tas and pr — each [B, steps, y_dim].
        """
        B   = y_ctx.shape[0]
        ssm = self.ssm

        if pr_ctx is not None:
            z_cnp, _, _ = self._encode_context(u_ctx, y_ctx, pr_ctx)
        else:
            z_cnp = torch.zeros(B, self.z_cnp_dim, device=y_ctx.device)
        film_gamma = self.z_cnp_scale(z_cnp)   # [B, emit_in_dim]
        film_beta  = self.z_cnp_shift(z_cnp)   # [B, emit_in_dim]

        rnn_in  = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _    = ssm.gru(rnn_in)
        q_params = ssm.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT = torch.clamp(logvar_qT, -12.0, 6.0)

        if ssm.emission_uses_u:
            uh_ctx, h_u = ssm.u_gru(u_ctx)
            s_ctx = torch.zeros(B, ssm.y_dim, ssm.reservoir_dim, device=u_ctx.device)
            for k in range(u_ctx.shape[1]):
                s_ctx = ssm._reservoir_step(s_ctx, uh_ctx[:, k])

        ysamps, ysamps_pr = [], []
        for _ in range(n_samples):
            z     = ssm.sample(mu_qT, logvar_qT)
            h_u_s = h_u.clone() if ssm.emission_uses_u else None
            s     = s_ctx.clone() if ssm.emission_uses_u else None

            preds, preds_pr = [], []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(ssm.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = ssm.sample(mu_p, logvar_p)

                if ssm.use_linear_model:
                    ctrl = ssm.ctrl_lin(u_t)

                if ssm.emission_uses_u:
                    uh_t, h_u_s = ssm.u_gru(u_t.unsqueeze(1), h_u_s)
                    uh_t = uh_t.squeeze(1)
                    e_in = torch.cat([z, uh_t, s.reshape(B, -1)], dim=-1)
                    s    = ssm._reservoir_step(s, uh_t)
                else:
                    e_in = z

                e_in = film_gamma * e_in + film_beta

                emit_out = ssm.emit(e_in)
                res, log_sigma_y_t = torch.chunk(emit_out, 2, dim=-1)
                log_sigma_y_t = torch.clamp(log_sigma_y_t, -8.0, 4.0)
                sigma_y_t = torch.exp(log_sigma_y_t) + ssm.eps
                y_hat = ctrl + res if ssm.use_linear_model else res

                out = ssm.emit_pr(e_in)
                mu_t, log_sigma_t, eps_skew_t, log_delta_t = torch.chunk(out, 4, dim=-1)
                sigma_pr = torch.exp(log_sigma_t) + ssm.eps
                x_samp   = mu_t + sigma_pr * torch.randn_like(mu_t)

                preds.append(y_hat + sigma_y_t * torch.randn_like(y_hat))
                preds_pr.append(sinh_arcsinh_forward(x_samp, eps_skew_t, log_delta_t, eps=ssm.eps))

            ysamps.append(torch.stack(preds, dim=1))
            ysamps_pr.append(torch.stack(preds_pr, dim=1))

        samp    = torch.stack(ysamps,    dim=0)   # [S, B, steps, Dy]
        samp_pr = torch.stack(ysamps_pr, dim=0)
        return (samp.mean(0),    samp.quantile(0.10, 0),    samp.quantile(0.90, 0),
                samp_pr.mean(0), samp_pr.quantile(0.10, 0), samp_pr.quantile(0.90, 0))

    # ------------------------------------------------------------------
    # Forecast (deterministic mean)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forecast_deterministic(
        self,
        y_ctx, u_ctx, u_fut, steps, n_samples=50,
        pr_ctx=None,
    ):
        """
        Same signature as forecast. Uses mean y_hat (no additive sigma noise for tas).
        Returns mean, q10, q90 for tas and pr — each [B, steps, y_dim].
        """
        B   = y_ctx.shape[0]
        ssm = self.ssm

        if pr_ctx is not None:
            z_cnp, _, _ = self._encode_context(u_ctx, y_ctx, pr_ctx)
        else:
            z_cnp = torch.zeros(B, self.z_cnp_dim, device=y_ctx.device)
        film_gamma = self.z_cnp_scale(z_cnp)
        film_beta  = self.z_cnp_shift(z_cnp)

        rnn_in  = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _    = ssm.gru(rnn_in)
        q_params = ssm.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT = torch.clamp(logvar_qT, -12.0, 6.0)

        if ssm.emission_uses_u:
            uh_ctx, h_u = ssm.u_gru(u_ctx)
            s_ctx = torch.zeros(B, ssm.y_dim, ssm.reservoir_dim, device=u_ctx.device)
            for k in range(u_ctx.shape[1]):
                s_ctx = ssm._reservoir_step(s_ctx, uh_ctx[:, k])

        ysamps, ysamps_pr = [], []
        for _ in range(n_samples):
            z     = ssm.sample(mu_qT, logvar_qT)
            h_u_s = h_u.clone() if ssm.emission_uses_u else None
            s     = s_ctx.clone() if ssm.emission_uses_u else None

            preds, preds_pr = [], []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(ssm.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = ssm.sample(mu_p, logvar_p)

                if ssm.use_linear_model:
                    ctrl = ssm.ctrl_lin(u_t)

                if ssm.emission_uses_u:
                    uh_t, h_u_s = ssm.u_gru(u_t.unsqueeze(1), h_u_s)
                    uh_t = uh_t.squeeze(1)
                    e_in = torch.cat([z, uh_t, s.reshape(B, -1)], dim=-1)
                    s    = ssm._reservoir_step(s, uh_t)
                else:
                    e_in = z

                e_in = film_gamma * e_in + film_beta

                emit_out = ssm.emit(e_in)
                res, _ = torch.chunk(emit_out, 2, dim=-1)
                y_hat  = ctrl + res if ssm.use_linear_model else res

                out = ssm.emit_pr(e_in)
                mu_t, _, eps_skew_t, log_delta_t = torch.chunk(out, 4, dim=-1)

                preds.append(y_hat)
                preds_pr.append(sinh_arcsinh_forward(mu_t, eps_skew_t, log_delta_t, eps=ssm.eps))

            ysamps.append(torch.stack(preds, dim=1))
            ysamps_pr.append(torch.stack(preds_pr, dim=1))

        samp    = torch.stack(ysamps,    dim=0)
        samp_pr = torch.stack(ysamps_pr, dim=0)
        return (samp.mean(0),    samp.quantile(0.10, 0),    samp.quantile(0.90, 0),
                samp_pr.mean(0), samp_pr.quantile(0.10, 0), samp_pr.quantile(0.90, 0))




# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cnp(
    model,
    task_dict,
    num_epochs,
    horizon,
    lr=2e-3,
    batch_size=64,
    kl_w=5,
    alpha_w=10000,
    omega_w=500,
    gamma_w=80000,
    kl_cnp_w=1.0,
    run_dir=None,
    device="cuda",
    patience=15,
    weights_file=None,
    unfreeze_frac=0.3,
    lr_ssm_unfrozen=2e-3,
):
    """
    Train a DeepCnpSsmforESM model on a task dictionary.

    task_dict : {esm_name: {'support': UnifiedWindowDataset, 'query': UnifiedWindowDataset, 'weight': float}}
        support  → training set,  query → validation set  (already normalised by build_task_dict)
    horizon          : forecast horizon in time steps (must match the datasets)
    weights_file     : path to a pretrained DeepSSMPatternConditioned state dict.
        If provided, those weights are loaded into model.ssm and all SSM sub-modules
        are frozen except the emission heads (emit, emit_pr).  The CNP modules
        (ctx_encoder, lat_encoder, z_cnp_scale, z_cnp_shift) are always trainable.
    unfreeze_frac    : fraction of num_epochs after which all SSM weights are unfrozen.
        Only has effect when weights_file is set. Default 0.3 (after 30 % of epochs).
    lr_ssm_unfrozen  : learning rate applied to the newly unfrozen SSM parameters.
        Typically lower than lr to avoid disrupting pretrained weights.

    Loss (mirrors run_train):
        nll + nll_pr + kl_w*kl_ssm + kl_cnp_w*kl_cnp
        + alpha*huber(tas) + omega*huber(pr) + gamma*mse_yearly
    alpha and omega are ramped up over the first 30 % of training steps (same schedule as run_train).
    The warm-up part of each batch (y_ctx, u_ctx, pr_ctx) is reused as the CNP context set.
    """
    model = model.to(device)

    ssm_frozen = False
    unfreeze_epoch = int(unfreeze_frac * num_epochs) + 1

    if weights_file is not None:
        model.ssm.load_state_dict(torch.load(weights_file, map_location=device))
        print(f"Loaded SSM weights from {weights_file}")
        for name, p in model.ssm.named_parameters():
            p.requires_grad = name.startswith("emit")
        print(f"SSM frozen except emit and emit_pr; will unfreeze all at epoch {unfreeze_epoch}.")
        ssm_frozen = True

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    train_loaders = {
        name: DataLoader(task['support'], batch_size=batch_size, shuffle=True, drop_last=True)
        for name, task in task_dict.items()
    }
    val_loaders = {
        name: DataLoader(task['query'], batch_size=batch_size, shuffle=False)
        for name, task in task_dict.items()
    }

    # Inverse-frequency weights: up-weight ESMs with fewer support samples so
    # every ESM contributes equally to the expected loss regardless of dataset size.
    # Normalized so the mean weight across ESMs is 1 (keeps overall loss scale stable).
    n_support    = {name: len(task['support']) for name, task in task_dict.items()}
    max_n        = max(n_support.values())
    raw_weights  = {name: max_n / n for name, n in n_support.items()}
    w_mean       = sum(raw_weights.values()) / len(raw_weights)
    task_weights = {name: w / w_mean for name, w in raw_weights.items()}
    print("Inverse-frequency task weights:",
          {name: f"{w:.2f}" for name, w in task_weights.items()})

    # Balanced iteration: every ESM takes the same number of steps per epoch,
    # matching the richest ESM. Sparse ESMs cycle through their loader multiple times.
    steps_per_esm = max(len(dl) for dl in train_loaders.values())
    total_steps   = num_epochs * steps_per_esm * len(task_dict)
    train_iters   = {name: iter(dl) for name, dl in train_loaders.items()}

    best_val      = float('inf')
    best_state    = None
    patience_left = patience
    global_step   = 0
    alpha = omega = 0.0   # annealed weights — persist into validation

    for epoch in range(1, num_epochs + 1):

        if ssm_frozen and epoch == unfreeze_epoch:
            newly_unfrozen = [p for p in model.ssm.parameters() if not p.requires_grad]
            for p in model.ssm.parameters():
                p.requires_grad = True
            opt.add_param_group({"params": newly_unfrozen, "lr": lr_ssm_unfrozen})
            ssm_frozen = False
            print(f"Epoch {epoch}: unfroze all SSM weights (lr={lr_ssm_unfrozen}).")

        model.train()
        tr_losses = []

        for _ in range(steps_per_esm):
            for esm_name in task_dict:
                weight = task_weights[esm_name]

                # Advance iterator; reshuffle when the loader is exhausted
                try:
                    batch = next(train_iters[esm_name])
                except StopIteration:
                    train_iters[esm_name] = iter(train_loaders[esm_name])
                    batch = next(train_iters[esm_name])

                y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut = [
                    t.float().to(device) for t in batch
                ]

                y_full  = torch.cat([y_ctx,  y_fut],  dim=1)
                pr_full = torch.cat([pr_ctx, pr_fut], dim=1)
                u_full  = torch.cat([u_ctx,  u_fut],  dim=1)

                B, _, Dy = y_full.shape

                global_step += 1
                frac  = min(1.0, global_step / int(0.3 * total_steps))
                alpha = alpha_w * frac
                omega = omega_w * frac

                # ELBO — CNP context reuses warm-up part of the batch
                nll, kl_ssm, nll_pr, kl_cnp = model.forward_elbo(
                    y_full, pr_full, u_full,
                    u_cnp=u_ctx, y_cnp=y_ctx, pr_cnp=pr_ctx,
                    kl_free_bits=0.2,
                )

                # Rollout MSE — forecast_deterministic is @torch.no_grad so these
                # are scalar additions to the loss; gradients flow only through ELBO
                mean, _, _, mean_pr, _, _ = model.forecast_deterministic(
                    y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30, pr_ctx=pr_ctx,
                )
                roll_mse    = F.huber_loss(mean,    y_fut,  delta=1.0)
                roll_mse_pr = F.huber_loss(mean_pr, pr_fut, delta=1.0)

                n_complete_years = horizon // 12
                if n_complete_years > 0:
                    H_yr     = n_complete_years * 12
                    mean_yr  = mean[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
                    y_fut_yr = y_fut[:, :H_yr, :].reshape(B, n_complete_years, 12, Dy).mean(dim=2)
                    roll_mse_yearly = ((mean_yr - y_fut_yr) ** 2).mean()
                else:
                    roll_mse_yearly = torch.tensor(0.0, device=device)

                loss = weight * (
                    nll + nll_pr
                    + kl_w     * kl_ssm
                    + kl_cnp_w * kl_cnp
                    + alpha    * roll_mse
                    + omega    * roll_mse_pr
                    + gamma_w  * roll_mse_yearly
                )

                if global_step % 100 == 0:
                    print(
                        f"[{esm_name}] step {global_step:05d} | "
                        f"nll {nll.item():.3f} | nll_pr {nll_pr.item():.3f} | "
                        f"kl_ssm {kl_ssm.item():.3f} | kl_cnp {kl_cnp.item():.3f} | "
                        f"mse {roll_mse.item():.4f} | mse_pr {roll_mse_pr.item():.4f} | "
                        f"mse_yr {roll_mse_yearly.item():.4f}"
                    )

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                tr_losses.append(loss.item())

        # ---- Validation on query sets ----
        model.eval()
        va_losses = []
        va_mses   = []

        with torch.no_grad():
            for esm_name, loader in val_loaders.items():
                weight = task_dict[esm_name].get('weight', 1.0)

                for y_ctx, pr_ctx, u_ctx, u_fut, pr_fut, y_fut in loader:
                    y_ctx  = y_ctx.float().to(device)
                    pr_ctx = pr_ctx.float().to(device)
                    u_ctx  = u_ctx.float().to(device)
                    u_fut  = u_fut.float().to(device)
                    pr_fut = pr_fut.float().to(device)
                    y_fut  = y_fut.float().to(device)

                    y_full  = torch.cat([y_ctx,  y_fut],  dim=1)
                    pr_full = torch.cat([pr_ctx, pr_fut], dim=1)
                    u_full  = torch.cat([u_ctx,  u_fut],  dim=1)

                    nll, kl_ssm, nll_pr, kl_cnp = model.forward_elbo(
                        y_full, pr_full, u_full,
                        u_cnp=u_ctx, y_cnp=y_ctx, pr_cnp=pr_ctx,
                        kl_free_bits=0.2,
                    )
                    mean, _, _, mean_pr, _, _ = model.forecast(
                        y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30, pr_ctx=pr_ctx,
                    )
                    mse    = ((mean    - y_fut)  ** 2).mean().item()
                    mse_pr = ((mean_pr - pr_fut) ** 2).mean().item()

                    val_loss = (
                        nll + nll_pr
                        + kl_w     * kl_ssm
                        + kl_cnp_w * kl_cnp
                        + alpha    * mse
                        + omega    * mse_pr
                    )
                    va_losses.append(weight * val_loss.item())
                    va_mses.append(mse)

        tr     = float(np.mean(tr_losses))
        va     = float(np.mean(va_losses))
        mse_va = float(np.mean(va_mses))
        print(f"epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | val_mse {mse_va:.4f}")

        if run_dir is not None and epoch % 10 == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"cnp_epoch{epoch:04d}.pt"))

        if va < best_val - 1e-4:
            best_val      = va
            best_state    = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

    if best_state is not None:
        model.load_state_dict(best_state)

    if run_dir is not None:
        torch.save(model.state_dict(), os.path.join(run_dir, "cnp_model_out"))

    return model


if __name__ == "__main__":

    models = ['CanESM5','ACCESS-ESM1-5','MPI-ESM1-2-LR','MIROC6']
    weights = [1.,1.,1.,1.]
    INDICATORS = ['tas','pr']
    TRAIN_SCENARIOS = [ 'ssp585','1pctco2','ssp460','ssp534-over','abrupt-4xco2','flat10zecincspinoff','flat10cdrincspinoff','ssp126','ssp370']
    maml_weights_file = "/home/kainverena/PythonProjects/outputs_ssm_scales/scales_maml_20260528_140312/checkpoints/meta_epoch0100.pt"
    cnp_weights_file = "/home/kainverena/PythonProjects/outputs_ssm_scales/checkpoints/cnp_epoch1950.pt"

    run_dir = os.path.join("outputs_ssm_scales", "scales_cnp_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_scenarios.txt"), "w") as f:
        f.write("\n".join(TRAIN_SCENARIOS))
    

    esm_data = {}

    for i,model in enumerate(models):
        task_data = {}
        model_path = f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{model}'

        u,tas,pr=prep.prepare_ds_data(
            model_path = model_path,
            train_scenarios=TRAIN_SCENARIOS,
            indicators=INDICATORS,
            sample_length=1800,
            n_skip=450,
            )
        task_data['y'] = tas
        task_data['pr'] = pr
        task_data['u'] = u
        task_data['weight'] = weights[i]
        esm_data[model] = task_data
    
    Dy = tas.shape[-1]
    Du = u.shape[-1]
    
    tasks,scalers = build_task_dict(esm_data=esm_data,context_len=600,horizon=1200,run_dir=run_dir)

    zdim = 64
    rnn_hidden=256
    use_linear_model = True
    emission_uses_u =True
    alpha_max = 0.002
    resevoir_dim = 4

    device = "cuda"

    model_ssm = DeepSSMPatternConditioned(y_dim=Dy, u_dim=Du, z_dim=zdim,rnn_hidden=rnn_hidden,use_linear_model=use_linear_model,
                                 emission_uses_u=emission_uses_u,reservoir_dim=resevoir_dim,alpha_max=alpha_max).to(device)

    model = DeepCnpSsmforESM(ssm_model=model_ssm,r_dim = 128, z_cnp_dim=32)
    model.load_state_dict(torch.load(cnp_weights_file, map_location=device))    

    model = train_cnp(model=model,task_dict=tasks,num_epochs=2000,horizon=1200,batch_size=256,run_dir=run_dir,weights_file=maml_weights_file,device=device)

    


    

    
        



