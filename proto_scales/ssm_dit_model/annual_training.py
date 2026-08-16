"""
annual_training
===============

Standalone training for `AnnualSSM`. One loss: the ELBO.

Compare with `scales_ssm_cross_corr_osc_trans.run_train`, which balances an NLL,
a KL, two rollout-MSE terms, a yearly-mean MSE and an ACF term with five
hand-set weights and a warmup schedule. Here:

    loss = nll + kl_weight * kl

with both terms already normalised per element, so the single weight is
interpretable and does not need rescaling when the number of regions, the
sequence length or the latent size changes.

Normalisation
-------------
Scalers are fitted on the **monthly** fields and the model consumes annual means
of the standardised monthly data — the convention documented in `annual_ssm`.
This is what keeps standalone training and DiT-embedded use identical, and it
means the scalers written here are directly reusable by the DiT.
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
from proto_scales.ssm_dit_model.annual_ssm import MONTHS_PER_YEAR, AnnualSSM, to_annual


class AnnualWindowDataset(Dataset):
    """
    Windows of `seq_years` consecutive years.

    Takes monthly arrays [N, T, D], converts once to annual up front (the model
    only ever sees annual data, so there is no reason to redo it per batch).
    """

    def __init__(self, tas, pr, u, seq_years=100, stride=1):
        def ann(x):
            return to_annual(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy()

        y = np.concatenate([ann(tas), ann(pr)], axis=-1)   # [N, Y, 2D]
        uu = ann(u)                                        # [N, Y, u]
        self.y, self.u = y, uu
        self.N, self.Y, self.obs_dim = y.shape
        self.L = int(seq_years)
        if self.L > self.Y:
            raise ValueError(
                f"seq_years={self.L} exceeds the {self.Y} years available")
        self.index = [(s, st) for s in range(self.N)
                      for st in range(0, self.Y - self.L + 1, int(stride))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, st = self.index[i]
        sl = slice(st, st + self.L)
        return self.y[s][sl], self.u[s][sl]


class ModelEMA:
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


def run_train(
    tas_np, pr_np, u_np,
    seq_years=100,
    stride=1,
    batch_size=16,
    epochs=200,
    lr=1e-3,
    weight_decay=0.0,
    warmup_steps=200,
    # model
    z_dim=16,
    rnn_hidden=64,
    emit_hidden=0,
    trans_hidden=0,
    osc_period_range=(2.0, 20.0),
    osc_damping_max=1.0,
    # loss — the only two knobs
    kl_weight=1.0,
    kl_free_bits=0.05,
    # misc
    ema_decay=0.999,
    grad_clip=1.0,
    run_dir=None,
    weights_file=None,
    scaler_dir=None,
    ckpt_every=20,
    seed=0,
):
    """Returns (ema_model, tas_scaler, pr_scaler, u_scaler)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}") if use_cuda else torch.device("cpu")
    is_main = local_rank == 0

    torch.manual_seed(seed)
    np.random.seed(seed)

    N = tas_np.shape[0]
    idx = np.random.permutation(N)
    n_train = max(1, int(0.8 * N))
    tr, va = idx[:n_train], idx[n_train:]
    if len(va) == 0:
        va = tr[-1:]

    # Monthly scalers, per the shared normalisation convention.
    if scaler_dir is not None:
        S = scales_ssm.StandardScaler
        tas_scaler = S.from_file(os.path.join(scaler_dir, "y_scaler.out"))
        pr_scaler = S.from_file(os.path.join(scaler_dir, "pr_scaler.out"))
        u_scaler = S.from_file(os.path.join(scaler_dir, "u_scaler.out"))
        if is_main:
            print(f"[scalers] reusing {scaler_dir}")
    else:
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

    ds_kw = dict(seq_years=seq_years, stride=stride)
    train_ds = AnnualWindowDataset(tas_n[tr], pr_n[tr], u_n[tr], **ds_kw)
    val_ds = AnnualWindowDataset(tas_n[va], pr_n[va], u_n[va], **ds_kw)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    if is_main:
        print(f"[data] {train_ds.Y} years available | train windows "
              f"{len(train_ds)} | val windows {len(val_ds)}")

    raw_model = AnnualSSM(
        obs_dim=train_ds.obs_dim, u_dim=u_np.shape[-1], z_dim=z_dim,
        rnn_hidden=rnn_hidden, emit_hidden=emit_hidden, trans_hidden=trans_hidden,
        osc_period_range=osc_period_range, osc_damping_max=osc_damping_max,
    ).to(device)

    if weights_file is not None:
        ckpt = torch.load(weights_file, map_location=device)
        raw_model.load_state_dict(ckpt)
        if is_main:
            print(f"[model] warm start from {weights_file}")

    if is_main:
        n_par = sum(p.numel() for p in raw_model.parameters())
        print(f"[model] AnnualSSM params {n_par:,} | obs_dim {train_ds.obs_dim} "
              f"| z_dim {z_dim} | emission "
              f"{'linear' if not emit_hidden else f'MLP({emit_hidden})'}")

    ddp_kw = {"device_ids": [local_rank], "output_device": local_rank} if use_cuda else {}
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

    best_val = float("inf")
    best_state = None
    global_step = 0
    n_skipped = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_nll, tr_kl = [], []
        for y, u in train_dl:
            y, u = y.to(device), u.to(device)
            nll, kl = model(y, u, kl_free_bits=kl_free_bits)
            loss = nll + kl_weight * kl

            if not torch.isfinite(loss):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                global_step += 1
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(params, grad_clip)
            if not torch.isfinite(gnorm):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                global_step += 1
                continue

            opt.step()
            sched.step()
            ema.update(raw_model)
            tr_nll.append(nll.item())
            tr_kl.append(kl.item())
            global_step += 1

        model.eval()
        va_nll, va_kl = [], []
        with torch.no_grad():
            for y, u in val_dl:
                y, u = y.to(device), u.to(device)
                nll, kl = raw_model(y, u, kl_free_bits=kl_free_bits)
                va_nll.append(nll.item())
                va_kl.append(kl.item())

        if not tr_nll:
            raise RuntimeError(f"epoch {epoch}: all steps non-finite; lower --lr")

        v = float(np.mean(va_nll)) + kl_weight * float(np.mean(va_kl))
        if is_main:
            # KL is the number to watch: if it collapses toward the free-bits
            # floor, z is being ignored and the emission is too strong.
            print(f"epoch {epoch:03d} | train nll {np.mean(tr_nll):.4f} "
                  f"kl {np.mean(tr_kl):.4f} | val nll {np.mean(va_nll):.4f} "
                  f"kl {np.mean(va_kl):.4f}"
                  + (f" | skipped {n_skipped}" if n_skipped else ""))

        if v < best_val - 1e-6:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in ema.state_dict().items()}

        if run_dir is not None and is_main and epoch % ckpt_every == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"model": raw_model.state_dict(), "ema": ema.state_dict()},
                       os.path.join(ckpt_dir, f"annual_ssm_epoch{epoch:04d}.pt"))

    if best_state is not None:
        ema.shadow.load_state_dict(best_state)

    if is_main:
        m = ema.shadow
        per = m.oscillator_periods().tolist()
        ef = m.oscillator_efolding().tolist()
        print(f"[done] best val {best_val:.4f}")
        print("[oscillators] learned periods (yr): "
              + ", ".join(f"{p:.1f}" for p in sorted(per)))
        print("[oscillators] e-folding times (yr): "
              + ", ".join(f"{e:.1f}" for e in sorted(ef)))

    return ema.shadow, tas_scaler, pr_scaler, u_scaler
