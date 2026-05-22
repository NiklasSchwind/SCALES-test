import copy
import logging
import numpy as np
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from proto_scales.ssm_model.scales_ssm_z2tasAndpr_hysteresis import (
    UnifiedWindowDataset,
    DeepSSMPatternConditioned,
)





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
):
    """
    Build a MAML task dictionary from multi-ESM data.

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

    Returns
    -------
    tasks : dict
        {
          esm_name: {
            'support' : UnifiedWindowDataset,
            'query'   : UnifiedWindowDataset,
            'weight'  : float,
          },
          ...
        }
    """
    tasks = {}
    for esm_name, data in esm_data.items():
        y      = np.asarray(data['y'],  dtype=np.float32)
        pr     = np.asarray(data['pr'], dtype=np.float32)
        u      = np.asarray(data['u'],  dtype=np.float32)
        weight = float(data.get('weight', 1.0))

        # Ensure shape is [N, T, D]
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

        support_ds = UnifiedWindowDataset(
            y[:n_support],
            pr[:n_support],
            u[:n_support],
            context_len=context_len,
            horizon=horizon,
            stride=stride,
            start_mode=start_mode,
        )
        query_ds = UnifiedWindowDataset(
            y[n_support:],
            pr[n_support:],
            u[n_support:],
            context_len=context_len,
            horizon=horizon,
            stride=stride,
            start_mode=start_mode,
        )

        tasks[esm_name] = {
            'support': support_ds,
            'query':   query_ds,
            'weight':  weight,
        }

    return tasks


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


# ---------------------------------------------------------------------------
# MAML inner loop
# ---------------------------------------------------------------------------

def inner_loop(
    w_meta,
    model,
    task,
    alpha,
    num_steps,
    device,
    batch_size=64,
    kl_w=5,
    alpha_w=10000,
    omega_w=500,
    gamma_w=80000,
):
    """
    MAML inner loop: adapt w_meta to the given task's support set.

    w_meta    : dict {param_name: tensor} — meta-initialisation, from model.named_parameters()
    model     : DeepSSMPatternConditioned — defines architecture; params are overwritten per call
    task      : dict with 'support' UnifiedWindowDataset (and 'query')
    alpha     : inner-loop learning rate (SGD)
    num_steps : number of gradient steps on the support set (typically 1–5)

    Returns
    -------
    w_task : dict {param_name: tensor}
        Adapted parameters after num_steps gradient steps on the support set.

    Note: implements first-order MAML (FOMAML). The returned w_task tensors are
    detached from the meta-gradient graph. For second-order MAML, replace the
    deep-copy/SGD approach with torch.func.functional_call and create_graph=True
    through the update steps.
    """
    # Deep-copy model and initialise from meta-parameters
    task_model = copy.deepcopy(model).to(device)
    with torch.no_grad():
        for name, p in task_model.named_parameters():
            if name in w_meta:
                p.copy_(w_meta[name])
    task_model.train()

    inner_opt = torch.optim.SGD(
        [p for p in task_model.parameters() if p.requires_grad],
        lr=alpha,
    )

    loader    = DataLoader(task['support'], batch_size=batch_size, shuffle=True, drop_last=True)
    data_iter = iter(loader)

    for _ in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        loss, *_ = compute_task_loss(
            task_model, batch, device,
            kl_w=kl_w, alpha_w=alpha_w, omega_w=omega_w, gamma_w=gamma_w,
        )
        inner_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(task_model.parameters(), 1.0)
        inner_opt.step()

    return {name: p.detach().clone() for name, p in task_model.named_parameters()}


# ---------------------------------------------------------------------------
# Meta-training outer loop
# ---------------------------------------------------------------------------

def train_meta(
    model,
    task_dict,
    num_epochs,
    alpha,
    beta,
    num_inner_steps,
    device,
    batch_size=64,
    kl_w=5,
    alpha_w=10000,
    omega_w=500,
    gamma_w=80000,
    run_dir=None,
    checkpoint_every=10,
    log_every=1,
):
    """
    Full MAML meta-training outer loop.

    Parameters
    ----------
    model           : DeepSSMPatternConditioned — meta-model (w_meta = model.parameters())
    task_dict       : output of build_task_dict
                      {esm_name: {'support': ..., 'query': ..., 'weight': float}}
    num_epochs      : number of outer meta-update steps
    alpha           : inner-loop SGD learning rate
    beta            : outer-loop AdamW learning rate
    num_inner_steps : gradient steps per task in the inner loop (typically 1–5)
    device          : torch device
    batch_size      : batch size for both inner and query DataLoaders
    kl_w            : KL weight in the ELBO loss
    alpha_w         : tas rollout Huber loss weight
    omega_w         : pr rollout Huber loss weight
    gamma_w         : yearly-average tas MSE weight
    run_dir         : if set, checkpoints are saved here every checkpoint_every epochs
    checkpoint_every: epochs between checkpoint saves (only if run_dir is set)
    log_every       : epochs between stdout log lines

    Returns
    -------
    model : DeepSSMPatternConditioned
        Meta-model restored to the best (lowest meta-loss) checkpoint.
    """
    # Configure logger — outputs to stdout so it appears in cluster job logs
    logger = logging.getLogger("scales_maml")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    meta_opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=beta,
        weight_decay=1e-4,
    )

    best_meta_loss = float("inf")
    best_state     = None

    logger.info(
        f"Starting meta-training: {num_epochs} epochs | "
        f"{len(task_dict)} tasks | alpha={alpha} | beta={beta} | "
        f"inner_steps={num_inner_steps}"
    )

    for epoch in range(1, num_epochs + 1):
        model.train()

        # Snapshot current meta-parameters for inner-loop initialisation
        w_meta = {name: p.detach().clone() for name, p in model.named_parameters()}

        meta_loss_total  = torch.tensor(0.0, device=device)
        per_task_details = {}   # {esm_name: (loss, nll, nll_pr, kl, mse_tas, mse_pr, mse_yr)}

        # ---- Per-task inner loop + query evaluation ----
        for esm_name, task in task_dict.items():
            weight = task.get('weight', 1.0)

            # Adapt to support set
            w_task = inner_loop(
                w_meta, model, task, alpha, num_inner_steps, device,
                batch_size=batch_size,
                kl_w=kl_w, alpha_w=alpha_w, omega_w=omega_w, gamma_w=gamma_w,
            )

            # Build adapted query model
            query_model = copy.deepcopy(model).to(device)
            with torch.no_grad():
                for name, p in query_model.named_parameters():
                    if name in w_task:
                        p.copy_(w_task[name])
            query_model.train()

            query_loader = DataLoader(
                task['query'], batch_size=batch_size, shuffle=True, drop_last=True
            )
            query_batch = next(iter(query_loader))

            loss_query, nll, nll_pr, kl, mse_tas, mse_pr, mse_yr = compute_task_loss(
                query_model, query_batch, device,
                kl_w=kl_w, alpha_w=alpha_w, omega_w=omega_w, gamma_w=gamma_w,
            )

            per_task_details[esm_name] = (
                loss_query.item(), nll, nll_pr, kl, mse_tas, mse_pr, mse_yr
            )
            meta_loss_total = meta_loss_total + weight * loss_query

        # ---- Outer gradient step ----
        meta_opt.zero_grad()
        meta_loss_total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        meta_opt.step()

        meta_loss_val = meta_loss_total.item()

        # ---- Logging ----
        if epoch % log_every == 0:
            task_lines = []
            for esm_name, (loss, nll, nll_pr, kl, mse_tas, mse_pr, mse_yr) in per_task_details.items():
                task_lines.append(
                    f"  {esm_name:<14} loss={loss:10.2f}  "
                    f"nll={nll:8.3f}  nll_pr={nll_pr:8.3f}  kl={kl:7.3f}  "
                    f"mse_tas={mse_tas:.4f}  mse_pr={mse_pr:.4f}  mse_yr={mse_yr:.4f}"
                )
            logger.info(
                f"epoch {epoch:04d}/{num_epochs}  meta_loss={meta_loss_val:10.2f}\n"
                + "\n".join(task_lines)
            )

        # ---- Best-model tracking ----
        if meta_loss_val < best_meta_loss:
            best_meta_loss = meta_loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # ---- Periodic checkpointing ----
        if run_dir is not None and epoch % checkpoint_every == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"meta_epoch{epoch:04d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  checkpoint saved → {ckpt_path}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best meta-model (loss={best_meta_loss:.4f})")

    return model



