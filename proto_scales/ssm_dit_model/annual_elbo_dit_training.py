"""
Training loop for `annual_elbo_dit.AnnualELBOOutpaintingDiT`.

Same windowed dataset, model-wide sampling EMA, LR schedule, non-finite-step
guards and checkpointing as `annual_jepa_kl_dit_training`. Differences:
  - the frozen pattern-scaling emission is fitted on the training split
    (normalised annual tas on annual u) before training;
  - beta (the KL weight) is warmed up linearly over `kl_warmup_epochs`;
    validation always uses the full beta, so best-checkpoint selection is not
    biased towards warm-up epochs;
  - no target encoder, no z-stat fit.
"""

import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import proto_scales.ssm_model.scales_ssm as scales_ssm
from proto_scales.ssm_dit_model.annual_dit_training import LongHorizonWindowDataset, ModelEMA
from proto_scales.ssm_dit_model.annual_elbo_dit import (
    AnnualELBOOutpaintingDiT, ForcedAnnualELBO, fit_pattern_scaling, load_matching,
    ssm_state_from_file)
from proto_scales.ssm_dit_model.memory_kernel import MONTHS_PER_YEAR


def _load_scalers(run_dir):
    S = scales_ssm.StandardScaler
    return (S.from_file(os.path.join(run_dir, "y_scaler.out")),
            S.from_file(os.path.join(run_dir, "pr_scaler.out")),
            S.from_file(os.path.join(run_dir, "u_scaler.out")))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def run_train(
    tas_np, pr_np, u_np,
    context_len=120,
    enc_context_len=None,
    enc_context_min=None,
    horizon=1200,
    block_len=120,
    blocks_per_step=2,
    batch_size=8,
    epochs=200,
    lr=1e-4,
    weight_decay=1e-4,
    warmup_steps=500,
    # ELBO latent process
    ssm_weights=None,
    scaler_run_dir=None,
    freeze_ssm=False,
    z_dim=16,
    rnn_hidden=64,
    trans_hidden=0,
    decay_efold_range=(1.0, 50.0),
    gated_decay=False,
    gate_hidden=32,
    monthly_encoder=True,
    enc_window_years=0,
    encoder_reads_pr=False,
    emission_hidden=0,
    pr_emission=False,
    pr_emission_hidden=64,
    reservoir_k=0,
    u_rnn_hidden=32,
    reservoir_tau_range=(2.0, 1000.0),
    z_forced=True,
    cond_baseline=False,
    baseline_rank=32,
    pr_weight=1.0,
    pr_detach_z=True,
    dit_residual=False,
    rollout_obs_var=False,
    ssm_only=False,
    ctx_offset_std=0.0,
    nll_weight=1.0,
    kl_weight=5.0,
    kl_warmup_epochs=5,
    kl_free_bits=0.0,
    rollout_weight=1.0,
    detach_latents=True,
    pattern_ridge=1e-3,
    # DiT
    cond_dim=256,
    hidden=384,
    depth=8,
    heads=6,
    n_flow_steps=50,
    timescales_years=(1.0, 5.0, 20.0),
    learnable_timescales=True,
    field_memory_rank=32,
    # misc
    ema_decay=0.999,
    stride=7,
    data_start_month=0,
    grad_clip=1.0,
    run_dir=None,
    weights_file=None,
    weights_key="model",
    ckpt_every=10,
    seed=0,
):
    """
    Returns (ema_model, tas_scaler, pr_scaler, u_scaler).

    The returned model is the model-wide sampling EMA copy — the one to
    sample from.
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
    if scaler_run_dir is not None:
        tas_scaler, pr_scaler, u_scaler = _load_scalers(scaler_run_dir)
        if is_main:
            print(f"[scalers] reusing scalers from {scaler_run_dir}")
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

    enc_context_len = context_len if enc_context_len is None else int(enc_context_len)
    enc_context_min = enc_context_len if enc_context_min is None else int(enc_context_min)
    if (enc_context_min % MONTHS_PER_YEAR or enc_context_min < context_len
            or enc_context_min > enc_context_len):
        raise ValueError(
            f"enc_context_min must be whole years in [context_len, enc_context_len] "
            f"= [{context_len}, {enc_context_len}]; got {enc_context_min}")
    if enc_window_years > 0 and enc_context_min < enc_context_len:
        print("[warn] enc_window_years > 0 caps every read anyway; a random encoder "
              "context length only changes the ELBO window")
    if enc_window_years > 0 and enc_window_years * MONTHS_PER_YEAR > enc_context_len:
        raise ValueError(
            f"enc_window_years={enc_window_years} exceeds the encoder context "
            f"({enc_context_len / MONTHS_PER_YEAR:.0f} yr): z_last would be a shorter "
            f"read than the rollout targets")

    # ---- data --------------------------------------------------------------
    # Windows carry the encoder's (longer) context; the DiT crops its own.
    ds_kw = dict(context_len=enc_context_len, horizon=horizon, stride=stride,
                 data_start_month=data_start_month)
    train_ds = LongHorizonWindowDataset(tas_n[tr], pr_n[tr], u_n[tr], **ds_kw)
    val_ds = LongHorizonWindowDataset(tas_n[va], pr_n[va], u_n[va], **ds_kw)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    if is_main:
        print(f"[data] train windows {len(train_ds)} | val windows {len(val_ds)} | "
              f"super-window {enc_context_len + horizon} months "
              f"({(enc_context_len + horizon) / MONTHS_PER_YEAR:.0f} yr; encoder context "
              f"{enc_context_len // MONTHS_PER_YEAR} yr, DiT context "
              f"{context_len // MONTHS_PER_YEAR} yr)")
        if enc_context_min < enc_context_len:
            print(f"[data] encoder read length drawn per step from "
                  f"{enc_context_min // MONTHS_PER_YEAR}-{enc_context_len // MONTHS_PER_YEAR} "
                  f"yr (last months of each window's context); validation uses the full "
                  f"{enc_context_len // MONTHS_PER_YEAR} yr")
        T_chunk = tas_np.shape[1]
        if T_chunk - (enc_context_len + horizon) < stride:
            print(f"[data] NOTE: chunks are {T_chunk} months, so each chunk yields "
                  f"{(T_chunk - enc_context_len - horizon) // stride + 1} window(s); "
                  f"lower --chunk_skip (coprime with 12, e.g. 67) or raise "
                  f"--chunk_len for more variety")

    # ---- model -------------------------------------------------------------
    Dy = tas_np.shape[-1]
    Du = u_np.shape[-1]

    ssm = ForcedAnnualELBO(
        y_dim=Dy, u_dim=Du, z_dim=z_dim, rnn_hidden=rnn_hidden,
        trans_hidden=trans_hidden, decay_efold_range=decay_efold_range,
        gated_decay=gated_decay, gate_hidden=gate_hidden,
        monthly_encoder=monthly_encoder, enc_window_years=enc_window_years,
        encoder_reads_pr=encoder_reads_pr, emission_hidden=emission_hidden,
        pr_emission=pr_emission, pr_emission_hidden=pr_emission_hidden,
        reservoir_k=reservoir_k, u_rnn_hidden=u_rnn_hidden,
        reservoir_tau_range=tuple(reservoir_tau_range), z_forced=z_forced,
    )
    # Frozen pattern scaling, fitted on the training split only.
    W_ps, b_ps, r2_ps = fit_pattern_scaling(tas_n[tr], u_n[tr], ridge=pattern_ridge)
    ssm.set_pattern_scaling(W_ps, b_ps)
    if is_main:
        print(f"[pattern] annual tas ~ u fitted on {len(tr)} training trajectories "
              f"(ridge {pattern_ridge}) | R^2 median {np.median(r2_ps):.2f}, "
              f"range [{r2_ps.min():.2f}, {r2_ps.max():.2f}]")
    if pr_emission:
        W_pr, b_pr, r2_pr = fit_pattern_scaling(pr_n[tr], u_n[tr], ridge=pattern_ridge)
        ssm.set_pr_pattern_scaling(W_pr, b_pr)
        if is_main:
            print(f"[pattern] annual pr ~ u | R^2 median {np.median(r2_pr):.2f}, "
                  f"range [{r2_pr.min():.2f}, {r2_pr.max():.2f}]")
    if ssm_weights is not None:
        # Bare ELBO state dict, model_out or an epoch checkpoint. Its pattern
        # scalings replace the ones just fitted: the emission residuals in the
        # checkpoint were trained against them, and they differ from a refit
        # whenever the scenarios, chunking or split changed. Scalings the
        # checkpoint lacks (a new pr head) keep the fit above.
        sd = ssm_state_from_file(ssm_weights)
        if is_main and scaler_run_dir is None:
            print("[warn] --ssm_weights without --scaler_run_dir: the latent process "
                  "was trained in another run's normalised units. Pass that run's "
                  "directory unless data and split are identical.")
        if is_main:
            load_matching(ssm, sd, tag=f"ssm {ssm_weights}")
        else:
            ssm.load_state_dict({k: v for k, v in sd.items()
                                 if k in ssm.state_dict()
                                 and ssm.state_dict()[k].shape == v.shape}, strict=False)
    elif is_main:
        print("[ssm] no checkpoint given — latent process starts from random init")

    raw_model = AnnualELBOOutpaintingDiT(
        ssm=ssm, y_dim=Dy, u_dim=Du,
        context_len=context_len, block_len=block_len, freeze_ssm=freeze_ssm,
        cond_dim=cond_dim, hidden=hidden, depth=depth, heads=heads,
        n_flow_steps=n_flow_steps,
        timescales_years=timescales_years,
        learnable_timescales=learnable_timescales,
        field_memory_rank=field_memory_rank,
        nll_weight=nll_weight, kl_weight=kl_weight, kl_free_bits=kl_free_bits,
        rollout_weight=rollout_weight, detach_latents=detach_latents,
        dit_residual=dit_residual, pr_weight=pr_weight, pr_detach_z=pr_detach_z,
        enc_context_len=enc_context_len, rollout_obs_var=rollout_obs_var,
        ssm_only=ssm_only, ctx_offset_std=ctx_offset_std,
        cond_baseline=cond_baseline, baseline_rank=baseline_rank,
    ).to(device)

    n_blocks = raw_model.n_blocks(horizon)
    blocks_per_step = max(1, min(int(blocks_per_step), n_blocks))
    if is_main and ssm_only:
        print("[model] SSM-ONLY: training the latent process alone (ELBO + rollout"
              f"{' + pr head' if pr_emission else ''}); the DiT is frozen and never "
              "run. 'val' is the latent-process loss at full beta. Reuse with "
              "--ssm_weights <run>/model_out --freeze_ssm")
    if is_main and ctx_offset_std > 0 and not ssm_only:
        print(f"[model] context-offset augmentation: per-sample std ~ U(0, "
              f"{ctx_offset_std}) in normalised units, context only, training only")
    if is_main:
        print(f"[model] linear flow matching, {n_flow_steps} sampling steps by default")
        print(f"[model] {n_blocks} DiT blocks per super-window, "
              f"{blocks_per_step} sampled per step "
              f"(effective batch {batch_size * blocks_per_step})")
        print(f"[model] z e-folding times (yr): "
              f"{[round(v, 1) for v in ssm.efolding_years().tolist()]}")
        print(f"[model] encoder reads "
              f"{'tas+pr' if encoder_reads_pr else 'tas'} + u, "
              f"{'monthly, z at each year end' if monthly_encoder else 'annual means'}; "
              f"lookback "
              f"{f'fixed {enc_window_years} yr per year' if enc_window_years > 0 else 'whole window from h0=0'}")
        print(f"[model] decay: "
              f"{f'gated by (z_prev, u_t), gate_hidden={gate_hidden}' if gated_decay else 'linear (hard-clamped)'}")
        print(f"[model] emission: frozen pattern scaling(u) + "
              f"{'linear' if not emission_hidden else f'MLP({emission_hidden})'} residual(z) for tas")
        if pr_emission:
            print(f"[model] pr head: frozen pattern scaling(u) + "
                  f"{'linear' if not pr_emission_hidden else f'MLP({pr_emission_hidden})'} "
                  f"residual(z), Gaussian NLL + rollout MSE, weight {pr_weight}, "
                  f"z {'detached' if pr_detach_z else 'trained by pr too'}")
        if reservoir_k > 0:
            print(f"[model] forcing pathway: u-GRU({u_rnn_hidden}) on annual u + "
                  f"{reservoir_k} reservoirs per region, tau {tuple(reservoir_tau_range)} yr; "
                  f"read by both emissions")
        print(f"[model] transition: decay * z"
              f"{' + B u' if z_forced else ' (no B u: z is the unforced state)'}"
              f"{f' + MLP(z, u) [{trans_hidden}]' if trans_hidden else ''}")
        print(f"[model] DiT conditioning: "
              f"{f'z + baseline (rank {baseline_rank}) + u + month + field memory' if cond_baseline else 'z + GSAT memory kernels + u + month + field memory'}")
        print(f"[model] DiT target: "
              f"{'residual field - SSM baseline (tas + pr)' if dit_residual else 'full field'}")
        if raw_model.use_ssm_aux:
            print(f"[model] ELBO ON: nll_weight={nll_weight}, beta={kl_weight} "
                  f"(warm-up {kl_warmup_epochs} epochs), kl_free_bits={kl_free_bits}, "
                  f"rollout_weight={rollout_weight} "
                  f"({'0.5 r^2 / obs var, NLL scale' if rollout_obs_var else 'plain MSE'})")
        print(f"[model] DiT gradients into z: "
              f"{'OFF (detached)' if detach_latents else 'ON'}")

    if weights_file is not None:
        ckpt = torch.load(weights_file, map_location=device)
        # Epoch checkpoints are saved as {"model": ..., "ema": ...}; the final
        # `model_out` is a bare state dict. Unwrap the former.
        if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
            if weights_key not in ckpt:
                raise KeyError(f"[dit] checkpoint {weights_file} has keys "
                               f"{list(ckpt.keys())}, no '{weights_key}' entry")
            ckpt = ckpt[weights_key]
            src = f"{weights_file} ['{weights_key}']"
        else:
            src = weights_file
        # The checkpoint's pattern scalings replace the fit above (see --ssm_weights).
        n_model = len(raw_model.state_dict())
        loaded, *_ = load_matching(raw_model, ckpt, tag=f"dit warm start {src}")
        if not loaded:
            raise RuntimeError(f"[dit] warm start from {src} matched none of the "
                               f"{n_model} model keys — wrong file or architecture?")

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

    rng = np.random.default_rng(seed + local_rank)
    # Validate on the first and last block: the gap between them is the drift
    # signal. A mean over all blocks would hide exactly what we want to see.
    val_blocks = sorted({0, n_blocks - 1})

    best_val = float("inf")
    best_state = None
    global_step = 0
    n_skipped = 0

    def beta_scale(epoch):
        if kl_warmup_epochs <= 0:
            return 1.0
        return min(1.0, epoch / float(kl_warmup_epochs))

    for epoch in range(1, epochs + 1):
        raw_model.set_kl_beta_scale(beta_scale(epoch))
        model.train()
        tr_losses = []
        for batch in train_dl:
            tas_c, pr_c, u_c, tas_f, pr_f, u_f, sm = to_dev(batch)
            # Windows in a batch may start in different calendar months. The
            # month embedding is built from a scalar phase, so batch on a
            # single phase: take the first item's. Strides coprime with 12
            # mean all phases are seen across an epoch.
            start_month = int(sm[0].item())
            blocks = rng.choice(n_blocks, size=blocks_per_step, replace=False).tolist()
            if enc_context_min < enc_context_len:
                # Random encoder read length, so the GRU (h0 = 0) sees a range
                # of lengths and a longer read at inference is not out of
                # distribution. Whole years are dropped from the front, so
                # start_month is unchanged.
                L = int(rng.integers(enc_context_min // MONTHS_PER_YEAR,
                                     enc_context_len // MONTHS_PER_YEAR + 1)) * MONTHS_PER_YEAR
                tas_c, pr_c, u_c = tas_c[:, -L:], pr_c[:, -L:], u_c[:, -L:]

            loss, parts = model(tas_c, pr_c, u_c, tas_f, pr_f, u_f,
                                block_idx=blocks, start_month=start_month,
                                return_parts=True)

            # A single non-finite loss is enough to destroy a run: the NaN
            # reaches every gradient, the optimiser writes NaN into the
            # weights, and every later forward pass produces NaN. The failure
            # is then reported far from its cause. Skip the step instead.
            if not torch.isfinite(loss):
                n_skipped += 1
                if is_main and n_skipped <= 10:
                    bad = [k for k, v in parts.items() if not torch.isfinite(v)]
                    print(f"[warn] step {global_step}: non-finite loss, skipping. "
                          f"offending terms: {bad or 'total only'}")
                opt.zero_grad(set_to_none=True)
                global_step += 1
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = nn.utils.clip_grad_norm_(params, grad_clip)
            if not torch.isfinite(gnorm):
                # Gradients can be non-finite even when the loss is finite.
                n_skipped += 1
                if is_main and n_skipped <= 10:
                    print(f"[warn] step {global_step}: non-finite grad norm, skipping")
                opt.zero_grad(set_to_none=True)
                global_step += 1
                continue

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
        # Stratified flow times rather than random draws: the random-t loss
        # has enough variance to swamp the epoch-to-epoch signal and makes
        # early stopping meaningless.
        model.eval()
        raw_model.set_kl_beta_scale(1.0)
        va_losses = []
        va_per_block = {f"flow_blk{b}": [] for b in val_blocks}
        with torch.no_grad():
            for batch in val_dl:
                tas_c, pr_c, u_c, tas_f, pr_f, u_f, sm = to_dev(batch)
                B = tas_c.shape[0]
                t_strat = torch.linspace(0.0, 1.0, B, device=device)
                v, p = raw_model.loss(
                    tas_c, pr_c, u_c, tas_f, pr_f, u_f,
                    block_idx=val_blocks, start_month=int(sm[0].item()),
                    t=t_strat, return_parts=True)
                va_losses.append(v.item())
                for k in va_per_block:
                    if k in p:
                        va_per_block[k].append(float(p[k]))

        if not tr_losses:
            raise RuntimeError(
                f"epoch {epoch}: every step was skipped as non-finite. The model "
                f"has diverged; lower --lr, --kl_weight or --rollout_weight.")
        tr_mean = float(np.mean(tr_losses))
        va_mean = float(np.mean(va_losses)) if va_losses else float("nan")
        if is_main:
            skip_note = f" | skipped {n_skipped}" if n_skipped else ""
            blk_note = "".join(f" | {k} {np.mean(v):.4f}"
                               for k, v in va_per_block.items() if v)
            print(f"epoch {epoch:03d} | train {tr_mean:.4f} | val {va_mean:.4f}"
                  f"{blk_note}{skip_note}")

        if va_mean < best_val - 1e-5:
            best_val = va_mean
            best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}

        if run_dir is not None and is_main and epoch % ckpt_every == 0:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"model": raw_model.state_dict(), "ema": ema.state_dict()},
                       os.path.join(ckpt_dir, f"annual_elbo_dit_epoch{epoch:04d}.pt"))

    if best_state is not None:
        ema.shadow.load_state_dict(best_state)
    if is_main:
        print(f"[done] best val {best_val:.4f}")
        print(f"[done] z e-folding times (yr): "
              f"{[round(v, 1) for v in raw_model.ssm_encoder.ssm.efolding_years().tolist()]}")

    return ema.shadow, tas_scaler, pr_scaler, u_scaler
