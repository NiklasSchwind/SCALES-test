"""
training
========

Dataset, EMA and DDP training loop for the SSM-conditioned outpainting DiT.

Scaler discipline
-----------------
If a pretrained SSM checkpoint is used (`ssm_weights`), the DiT **must** reuse
that run's scalers rather than refitting its own. The SSM's GRU posterior and
oscillatory transition were fitted in a particular normalisation; feeding them
differently-scaled inputs silently degrades `z` with no error anywhere. Pass
`ssm_run_dir` and the scalers are loaded from it. Refitting is only correct when
the SSM is being trained from scratch alongside the DiT.
"""

import copy
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

import proto_scales.ssm_model.scales_ssm as scales_ssm
import proto_scales.ssm_model.scales_ssm_cross_corr_osc_trans as scales_ssm_osc
from proto_scales.ssm_dit_model.memory_kernel import MONTHS_PER_YEAR
from proto_scales.ssm_dit_model.ssm_dit import SSMConditionedOutpaintingDiT


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class DiTWindowDataset(Dataset):
    """
    Context+horizon windows with the calendar phase of each window.

    On `data_start_month`: the month embedding is learned, so a constant offset
    between "index 0" and the true calendar January is harmless — it just shifts
    which embedding row means January. What matters is that windows are phase
    *consistent* with each other, which `(start + data_start_month) % 12`
    guarantees for any offset. Set it correctly anyway if you know it, and use
    the same value at inference.

    Returns per item:
        tas_ctx, pr_ctx, u_ctx, tas_fut, pr_fut, u_fut, start_month
    """

    def __init__(self, tas, pr, u, context_len, horizon, stride=1, data_start_month=0):
        tas = np.asarray(tas, dtype=np.float32)
        pr = np.asarray(pr, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)
        for a in (tas, pr, u):
            if a.ndim != 3:
                raise ValueError("tas, pr and u must be [N, T, D]")
        if not (tas.shape[:2] == pr.shape[:2] == u.shape[:2]):
            raise ValueError("tas, pr and u must agree on [N, T]")
        if tas.shape[2] != pr.shape[2]:
            raise ValueError("tas and pr must have the same number of regions")

        self.tas, self.pr, self.u = tas, pr, u
        self.N, self.T, self.Dy = tas.shape
        self.Tc, self.H = int(context_len), int(horizon)
        self.data_start_month = int(data_start_month) % MONTHS_PER_YEAR
        if self.Tc + self.H > self.T:
            raise ValueError("context_len + horizon must be <= T")

        max_start = self.T - (self.Tc + self.H)
        self.index = [(s, st) for s in range(self.N)
                      for st in range(0, max_start + 1, int(stride))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s, st = self.index[idx]
        Tc, H = self.Tc, self.H
        c = slice(st, st + Tc)
        f = slice(st + Tc, st + Tc + H)
        start_month = (st + self.data_start_month) % MONTHS_PER_YEAR
        return (self.tas[s][c], self.pr[s][c], self.u[s][c],
                self.tas[s][f], self.pr[s][f], self.u[s][f],
                np.int64(start_month))


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────
class ModelEMA:
    """
    Exponential moving average of the trainable weights.

    Not optional in practice for diffusion models: samples drawn from the raw
    weights are markedly worse than from the EMA copy, and the gap does not
    close with longer training.
    """

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.shadow.state_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def _load_scalers(ssm_run_dir):
    S = scales_ssm.StandardScaler
    return (S.from_file(os.path.join(ssm_run_dir, "y_scaler.out")),
            S.from_file(os.path.join(ssm_run_dir, "pr_scaler.out")),
            S.from_file(os.path.join(ssm_run_dir, "u_scaler.out")))


def run_train(
    tas_np, pr_np, u_np,
    context_len=120,
    horizon=120,
    batch_size=32,
    epochs=200,
    lr=1e-4,
    weight_decay=0.0,
    warmup_steps=500,
    # SSM
    ssm_weights=None,
    ssm_run_dir=None,
    freeze_ssm=True,
    z_dim=64,
    rnn_hidden=256,
    cov_rank=8,
    # DiT
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
    # misc
    ema_decay=0.999,
    stride=1,
    data_start_month=0,
    grad_clip=1.0,
    run_dir=None,
    weights_file=None,
    ckpt_every=10,
    seed=0,
):
    """
    Returns (ema_model, tas_scaler, pr_scaler, u_scaler).

    The returned model is the EMA copy — that is the one to sample from.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")
    is_main = local_rank == 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- split -------------------------------------------------------------
    N = tas_np.shape[0]
    idx = np.random.permutation(N)
    n_train = max(1, int(0.8 * N))
    tr, va = idx[:n_train], idx[n_train:]
    if len(va) == 0:
        va = tr[-1:]

    # ---- scalers -----------------------------------------------------------
    if ssm_run_dir is not None:
        tas_scaler, pr_scaler, u_scaler = _load_scalers(ssm_run_dir)
        if is_main:
            print(f"[scalers] reusing SSM scalers from {ssm_run_dir}")
    else:
        if ssm_weights is not None and is_main:
            print("[scalers] WARNING: ssm_weights given without ssm_run_dir; "
                  "refitting scalers. The frozen SSM will see a normalisation it "
                  "was not trained in. Pass ssm_run_dir unless the SSM is untrained.")
        tas_scaler = scales_ssm.StandardScaler().fit(tas_np[tr])
        pr_scaler = scales_ssm.StandardScaler().fit(pr_np[tr])
        u_scaler = scales_ssm.StandardScaler().fit(u_np[tr])

    if run_dir is not None and is_main:
        tas_scaler.save(os.path.join(run_dir, "y_scaler.out"))
        pr_scaler.save(os.path.join(run_dir, "pr_scaler.out"))
        u_scaler.save(os.path.join(run_dir, "u_scaler.out"))

    tas_n = tas_scaler.transform(tas_np)
    pr_n = pr_scaler.transform(pr_np)
    u_n = u_scaler.transform(u_np)

    # ---- data --------------------------------------------------------------
    ds_kw = dict(context_len=context_len, horizon=horizon, stride=stride,
                 data_start_month=data_start_month)
    train_ds = DiTWindowDataset(tas_n[tr], pr_n[tr], u_n[tr], **ds_kw)
    val_ds = DiTWindowDataset(tas_n[va], pr_n[va], u_n[va], **ds_kw)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    if is_main:
        print(f"[data] train windows {len(train_ds)} | val windows {len(val_ds)}")

    # ---- model -------------------------------------------------------------
    Dy = tas_np.shape[-1]
    Du = u_np.shape[-1]

    import proto_scales.ssm_model.scales_ssm_cross_corr_osc_trans as ssm_mod
    ssm = ssm_mod.DeepSSMPatternConditioned(
        y_dim=Dy, u_dim=Du, z_dim=z_dim, rnn_hidden=rnn_hidden,
        emission_uses_u=True, use_linear_model=True, cov_rank=cov_rank,
    )
    if ssm_weights is not None:
        ckpt = torch.load(ssm_weights, map_location="cpu")
        missing, unexpected = ssm.load_state_dict(ckpt, strict=False)
        if is_main:
            print(f"[ssm] loaded {ssm_weights} | missing {len(missing)} | "
                  f"unexpected {len(unexpected)}")
            if missing:
                print(f"[ssm] missing keys: {missing}")
    elif is_main:
        print("[ssm] no checkpoint given — SSM starts from random init")

    raw_model = SSMConditionedOutpaintingDiT(
        ssm=ssm, y_dim=Dy, u_dim=Du,
        context_len=context_len, horizon=horizon, freeze_ssm=freeze_ssm,
        cond_dim=cond_dim, hidden=hidden, depth=depth, heads=heads,
        n_diffusion_steps=n_diffusion_steps,
        timescales_years=timescales_years,
        learnable_timescales=learnable_timescales,
        field_memory_rank=field_memory_rank,
        parameterization=parameterization,
        ssm_elbo_weight=ssm_elbo_weight,
        ssm_kl_weight=ssm_kl_weight,
        ssm_kl_free_bits=ssm_kl_free_bits,
        ssm_acf_weight=ssm_acf_weight,
        ssm_acf_max_lag=ssm_acf_max_lag,
        ssm_rollout_steps=ssm_rollout_steps,
        ssm_rollout_samples=ssm_rollout_samples,
    ).to(device)
    if is_main:
        print(f"[model] diffusion parameterization: {parameterization}")
        if raw_model.use_ssm_aux:
            print(f"[model] SSM auxiliary objective ON "
                  f"(elbo_w={ssm_elbo_weight}, acf_w={ssm_acf_weight})")
        elif not freeze_ssm:
            print("[model] WARNING: SSM is trainable but has no auxiliary "
                  "objective. z is shaped by the diffusion loss alone and the "
                  "oscillator has no ACF pressure; consider --ssm_elbo_weight.")

    # ctrl_lin is the frozen ridge pattern-scaling term. The SSM's own training
    # fits and freezes it; the auxiliary ELBO uses it, so do the same here.
    if raw_model.use_ssm_aux and raw_model.ssm_encoder.ssm.use_linear_model:
        W, b = scales_ssm_osc.fit_ridge_D(
            u_n[tr], tas_n[tr], alpha=1e-2, fit_intercept=True)
        scales_ssm_osc.load_into_ctrl_lin(raw_model.ssm_encoder.ssm, W, b, freeze=True)
        if is_main:
            print("[model] ctrl_lin ridge-initialised and frozen")

    if weights_file is not None:
        ckpt = torch.load(weights_file, map_location=device)
        missing, unexpected = raw_model.load_state_dict(ckpt, strict=False)
        if is_main:
            print(f"[dit] warm start from {weights_file} | missing {len(missing)} | "
                  f"unexpected {len(unexpected)}")

    n_par = sum(p.numel() for p in raw_model.parameters())
    n_tr = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        print(f"[model] params {n_par:,} | trainable {n_tr:,} | frozen {n_par - n_tr:,}")

    ddp_kw = {"device_ids": [local_rank], "output_device": local_rank} if use_cuda else {}
    # NOTE: call `model(...)`, never `raw_model.loss(...)` — the DDP reducer is
    # armed by the forward pass on the wrapper, so bypassing it silently drops
    # cross-rank gradient averaging.
    model = DDP(raw_model, **ddp_kw)

    ema = ModelEMA(raw_model, decay=ema_decay)

    params = [p for p in raw_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    total_steps = max(1, epochs * len(train_dl))

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def to_dev(batch):
        tas_c, pr_c, u_c, tas_f, pr_f, u_f, sm = batch
        return (tas_c.to(device), pr_c.to(device), u_c.to(device),
                tas_f.to(device), pr_f.to(device), u_f.to(device), sm)

    best_val = float("inf")
    best_state = None
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for batch in train_dl:
            tas_c, pr_c, u_c, tas_f, pr_f, u_f, sm = to_dev(batch)
            # Windows in a batch may start in different calendar months. The
            # month embedding is built from a scalar phase, so batch on a single
            # phase: take the first item's. With stride=1 all phases are seen
            # across the epoch anyway.
            start_month = int(sm[0].item())

            loss, parts = model(tas_c, pr_c, u_c, tas_f, pr_f, u_f,
                                start_month=start_month, return_parts=True)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            sched.step()
            ema.update(raw_model)

            tr_losses.append(loss.item())
            global_step += 1
            if is_main and global_step % 100 == 0:
                extra = "".join(f" | {k} {float(v):.4f}" for k, v in parts.items())
                print(f"step {global_step} | loss {np.mean(tr_losses[-100:]):.4f}"
                      f"{extra} | lr {sched.get_last_lr()[0]:.2e}")

        # ---- validation ----------------------------------------------------
        # Stratified diffusion timesteps rather than random draws: the random-t
        # loss has enough variance to swamp the epoch-to-epoch signal and makes
        # early stopping meaningless.
        model.eval()
        va_losses = []
        with torch.no_grad():
            for batch in val_dl:
                tas_c, pr_c, u_c, tas_f, pr_f, u_f, sm = to_dev(batch)
                B = tas_c.shape[0]
                t_strat = torch.linspace(0, n_diffusion_steps - 1, B, device=device).long()
                va_losses.append(raw_model.loss(
                    tas_c, pr_c, u_c, tas_f, pr_f, u_f,
                    start_month=int(sm[0].item()), t_diff=t_strat).item())

        tr_mean = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va_mean = float(np.mean(va_losses)) if va_losses else float("nan")
        if is_main:
            print(f"epoch {epoch:03d} | train {tr_mean:.4f} | val {va_mean:.4f}")

        if va_mean < best_val - 1e-5:
            best_val = va_mean
            best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}

        if run_dir is not None and is_main and epoch % ckpt_every == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"model": raw_model.state_dict(), "ema": ema.state_dict()},
                       os.path.join(ckpt_dir, f"dit_epoch{epoch:04d}.pt"))

    if best_state is not None:
        ema.shadow.load_state_dict(best_state)
    if is_main:
        print(f"[done] best val {best_val:.4f}")

    return ema.shadow, tas_scaler, pr_scaler, u_scaler
