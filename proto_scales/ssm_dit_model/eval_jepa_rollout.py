"""
Does the forcing-only z rollout of a trained `ForcedAnnualJEPAKL` track what
the (target) encoder reads off the real field?

Run on the cluster from the repo root, e.g.

    python -m proto_scales.ssm_dit_model.eval_jepa_rollout \
        --run_dir outputs_ssm_scales/annual_jepa_kl_dit_20261105_120000 \
        --cluster IIASA --scenarios ssp126 ssp245 --out_dir eval_rollout

Only the latent process is built (no DiT): weights are taken from the
`ssm_encoder.ssm.*` entries of `model_out` (the sampling-EMA copy, best val)
or of an epoch checkpoint (`--weights ... --weights_key ema|model`).

Diagnostics, all in annual z space
----------------------------------
1. Lead-time error, training frame (`lead_error_*.png`)
   Windows of `context_len + horizon` slid along each trajectory, scored
   exactly like `jepa_rollout_loss`: z_last = posterior(ctx), rollout on
   forcing alone, target = encode_target(ctx + fut)[ctx:]. Reported per lead
   year, overall and per z-dim, next to two baselines:
     - persistence (hold z_last), which the rollout must beat;
     - one-step (transition_mean(z_tgt[t-1], u_t) vs z_tgt[t]), the floor that
       compounding starts from. If one-step error is small but rollout error
       grows fast, the per-step bias compounds; if one-step error is already
       large, the transition itself does not fit the target.
   Per-dim errors are also divided by the target's variance (1 - R^2), so a
   dimension with a tiny dynamic range doesn't look good just for being small.

2. Normalisation-frame sensitivity
   Batch-stat standardization (what training uses, and what eval used before
   `fit_z_stats`) makes z_last depend on which windows share the batch and on
   the window length. Each window is scored with batch stats alone (B=1, what
   single-trajectory inference did) and pooled with every other window of the
   scenario, and — if the checkpoint has fitted stats (`model_out` from new
   runs, or `fit_jepa_kl_z_stats` for old ones) — with the fixed stats. The
   rollout error under each and ||z_last(fixed) - z_last(batch, B=1)|| are
   reported. Plots/long rollout use fixed stats when available.

3. Long rollout (`long_rollout_*.png`)
   One rollout from the first `context_len` of each trajectory to its end,
   against encode_target over the whole trajectory. Dims are sorted by
   e-folding time and annotated with ||B_row||. Caveat: the target here is
   standardized over the whole trajectory (e.g. 290 yr) rather than the
   110 yr the rollout loss sees, so its frame differs from training (that is
   point 2 again). With --gated_decay, the gated e-folding time along the
   rollout and along the target trajectory is plotted too (`gated_tau_*.png`).

4. Emission rollout (`emission_*.png`, runs with --emission_* weights only)
   The annual emission emit(z) along the same long rollout, in physical
   units, against the trajectory's annual means, for the regions given by
   --emit_regions (default SOO). Compare with the DiT's forecast for the same
   region: if the emission is right and the DiT is biased, the DiT is not
   taking the forced response from z; if the emission is biased too, the
   dynamics are.

5. Restarts (`restarts_*.png`, with --restart_years)
   Is the state the rollout carries through the decades the state the encoder
   reads? For each restart year the online encoder reads the *whole* history
   from the trajectory start up to that year (as the forecasts do), then
   rolls out on forcing alone to the end. The z trajectories from all
   restarts are overlaid with the target encoding. Printed per dim (sorted by
   e-folding time): the gap between the rollout from the first restart and
   the encoder's read at each later restart year — a consistent state model
   has gaps near zero; a gap that grows with time on the slow dims means the
   rollout reaches the later year in a different state than the data say.
   With an emission head, the same for emit(z) in the --emit_regions.

Everything is also saved to `<out_dir>/eval_jepa_rollout.npz`.
"""

import argparse
import ast
import os
import re
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import proto_scales.data_prep.prepare_data as prep
import proto_scales.ssm_model.scales_ssm as scales_ssm
from proto_scales.ssm_dit_model.annual_jepa_kl_dit import ForcedAnnualJEPAKL
from proto_scales.ssm_dit_model.annual_ssm import to_annual

MONTHS = 12
MODEL_PATHS = {
    "IIASA": "/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{model}",
    "ASC": "/gpfs/data/fs73093/kain/CMIP6/cmip6-ng-inc-oceans/{model}",
}


# ─────────────────────────────────────────────────────────────────────────────
# loading
# ─────────────────────────────────────────────────────────────────────────────
def read_config(run_dir):
    cfg = {}
    path = os.path.join(run_dir, "config.txt")
    if not os.path.exists(path):
        print(f"[config] no {path}; using defaults / CLI overrides")
        return cfg
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.rstrip("\n").split("=", 1)
            try:
                cfg[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                cfg[k] = v
    return cfg


CKPT_RE = re.compile(r"annual_(?:jepa_kl|elbo)_dit_epoch(\d+)\.pt$")


def zstats_path(path):
    """Where `fit_jepa_kl_z_stats` writes the fitted copy of `path`."""
    root, ext = os.path.splitext(path)
    return f"{root}_zstats{ext}" if ext == ".pt" else f"{path}_zstats"


def list_checkpoints(run_dir):
    """{epoch: path} of the epoch checkpoints in <run_dir>/checkpoints."""
    d = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in os.listdir(d):
        m = CKPT_RE.match(f)
        if m:
            out[int(m.group(1))] = os.path.join(d, f)
    return out


def resolve_weights(run_dir, weights=None, checkpoint=None, prefer_zstats=True):
    """
    Pick the weights file. Order: explicit `weights` path; `checkpoint`
    ('latest' or an epoch number) from <run_dir>/checkpoints; otherwise
    <run_dir>/model_out, or the latest epoch checkpoint if training hasn't
    finished yet. With `prefer_zstats`, a fitted copy written by
    `fit_jepa_kl_z_stats` next to the chosen file is used if it exists.
    Returns (path, tag) — tag names the source, e.g. 'epoch0160' / 'model_out'.
    """
    if weights:
        path = weights
    else:
        ckpts = list_checkpoints(run_dir)
        final = os.path.join(run_dir, "model_out")
        if checkpoint is None and os.path.exists(final):
            path = final
        else:
            if not ckpts:
                raise SystemExit(f"no model_out and no epoch checkpoints in {run_dir}")
            if checkpoint in (None, "latest"):
                ep = max(ckpts)
                if checkpoint is None:
                    print(f"[weights] no model_out yet -> latest checkpoint, epoch {ep}")
            else:
                ep = int(checkpoint)
                if ep not in ckpts:
                    raise SystemExit(f"no checkpoint for epoch {ep}; have {sorted(ckpts)}")
            path = ckpts[ep]
    base = os.path.basename(path)
    for suf in ("_zstats.pt", "_zstats", ".pt"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    m = re.search(r"epoch(\d+)", base)
    tag = f"epoch{m.group(1)}" if m else base
    if prefer_zstats and not path.endswith(("_zstats", "_zstats.pt")):
        zp = zstats_path(path)
        if os.path.exists(zp):
            print(f"[weights] using fitted copy {zp}")
            path = zp
    return path, tag


def is_elbo(m):
    """`annual_elbo_dit.ForcedAnnualELBO` (vs `ForcedAnnualJEPAKL`)."""
    return hasattr(m, "emit_mean")


def ref_encode(m, ye, ue):
    """Reference read of the real field: target encoder (JEPA), posterior mean (ELBO)."""
    return m.posterior(ye, ue)[0] if is_elbo(m) else m.encode_target(ye, ue)


def has_emission(m):
    return is_elbo(m) or bool(getattr(m, "annual_emission", False))


def emit_ann(m, z, u_ann):
    """
    Annual emission [..., 2R] in normalised units. ELBO: pattern scaling(u) +
    residual(z), tas only, pr filled with NaN so the plots/units code is shared.
    """
    if is_elbo(m):
        tas = m.emit_mean(z, u_ann)
        return torch.cat([tas, torch.full_like(tas, float("nan"))], dim=-1)
    return m.emit(z)


def uses_z_stats(m):
    return bool(getattr(m, "normalize_z", False))


def load_jepa(run_dir, weights, weights_key, cfg, y_dim, u_dim):
    """
    The latent process alone (`ForcedAnnualJEPAKL`, or `ForcedAnnualELBO` when
    the weights carry the frozen pattern scaling `ps_weight`), in eval mode.
    """
    path = weights or os.path.join(run_dir, "model_out")
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and ("model" in sd or "ema" in sd):
        sd = sd[weights_key]
    if "ssm_encoder.ssm.ps_weight" in sd:
        from proto_scales.ssm_dit_model.annual_elbo_dit import (
            ForcedAnnualELBO, ssm_kwargs_from_state_dict)
        jepa = ForcedAnnualELBO(
            y_dim=y_dim, u_dim=u_dim,
            decay_efold_range=tuple(cfg.get("decay_efold", (1.0, 50.0))),
            monthly_encoder=cfg.get("monthly_encoder", True),
            **ssm_kwargs_from_state_dict(sd, y_dim, u_dim))
        print("[model] ELBO latent process (annual_elbo_dit)")
    else:
        jepa = ForcedAnnualJEPAKL(
            obs_dim=2 * y_dim, u_dim=u_dim,
            z_dim=cfg.get("z_dim", 16),
            rnn_hidden=cfg.get("rnn_hidden", 64),
            trans_hidden=cfg.get("trans_hidden", 0),
            decay_efold_range=tuple(cfg.get("decay_efold", (1.0, 50.0))),
            target_decay=cfg.get("target_decay", 0.996),
            normalize_z=cfg.get("normalize_z", True),
            gated_decay=cfg.get("gated_decay", False),
            monthly_encoder=cfg.get("monthly_encoder", False),
            gate_hidden=cfg.get("gate_hidden", 32),
            annual_emission=(cfg.get("emission_weight", 0.0) > 0
                             or cfg.get("emission_rollout_weight", 0.0) > 0),
            emission_hidden=cfg.get("emission_hidden", 0),
            enc_window_years=cfg.get("enc_window_years", 0),
        )
    src = f"{path} ['{weights_key}']"
    prefix = "ssm_encoder.ssm."
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not sub:
        raise RuntimeError(f"no '{prefix}*' keys in {src}")
    missing, unexpected = jepa.load_state_dict(sub, strict=False)
    print(f"[weights] {src} | missing {missing} | unexpected {unexpected}")
    return jepa.eval()


def load_trajectories(model_path, scenarios, monthly=True):
    """
    Full-length trajectories, one per file (no 150-yr chunking), aligned the
    way `prep.prepare_train_data` aligns them: GMT is shifted one month ahead
    of the field. Returns list of dicts {name, tas [T,R], pr [T,R], u [T,1]}
    and the region names.
    """
    files = prep.get_all_files_(model_path)
    out, regions = [], None
    for scen in scenarios:
        tas_files = prep.filter_climate_files(files=files, scenarios=[scen], indicators=["tas"])
        for f in tas_files:
            base = prep.get_baseline_filename(filename=f, files=files)
            dfs = {}
            for ind in ("tas", "pr"):
                b, e = base.replace("tas", ind), f.replace("tas", ind)
                gmt, reg = prep.process_scenarios(
                    experiment_scenario_path=f"{model_path}/{e}", simulation_name=e,
                    baseline_scenario_path=f"{model_path}/{b}", delete_first_years=0,
                    monthly_trend=monthly, smoothed=False)
                dfs[ind] = (gmt, reg.add_suffix(f"_{ind}"))
            reg_df = pd.concat([dfs["tas"][1], dfs["pr"][1]], axis=1)
            arr = prep.process_gmt_and_regions_into_array(
                (dfs["tas"][0], reg_df), weighted_linear_smoothing=False,
                ramp_down_corrected_ps=False)
            R = (arr.shape[0] - 1) // 2
            if regions is None:
                regions = [c[:-4] for c in reg_df.columns[:R]]
            out.append(dict(name=f"{scen}:{os.path.basename(f)}",
                            u=arr[0, 1:, None], tas=arr[1:1 + R, :-1].T,
                            pr=arr[1 + R:, :-1].T))
            print(f"[data] {out[-1]['name']} | {arr.shape[1] / MONTHS:.0f} yr")
    return out, regions


# ─────────────────────────────────────────────────────────────────────────────
# diagnostics
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_windows(jepa, y_m, u_m, ctx, H, starts, pooled):
    """
    y_m [T, 2R], u_m [T, 1] monthly, for one trajectory. Each start s (years)
    gives ctx = years [s, s+ctx), fut = [s+ctx, s+ctx+H). The encoders read
    `jepa.encoder_inputs` of each slice (monthly or annual by mode); the
    rollout is driven by annual forcing. Returns dict of arrays
    [n_win, H, z]. `pooled` encodes all windows as one batch (shared
    standardization stats), otherwise one window at a time.
    """
    M = MONTHS
    Yc = torch.stack([y_m[s * M:(s + ctx) * M] for s in starts])
    Uc = torch.stack([u_m[s * M:(s + ctx) * M] for s in starts])
    Yf = torch.stack([y_m[(s + ctx) * M:(s + ctx + H) * M] for s in starts])
    Uf = torch.stack([u_m[(s + ctx) * M:(s + ctx + H) * M] for s in starts])

    def run(yc, uc, yf, uf):
        ye_c, ue_c = jepa.encoder_inputs(yc, uc)
        ye_f, ue_f = jepa.encoder_inputs(yf, uf)
        uf_ann = to_annual(uf)
        _, z_last = jepa.posterior(ye_c, ue_c)
        z_roll = jepa.roll_forward(z_last, uf_ann)
        z_tgt_all = ref_encode(jepa, torch.cat([ye_c, ye_f], 1), torch.cat([ue_c, ue_f], 1))
        z_tgt = z_tgt_all[:, ctx:]
        z_prev = z_tgt_all[:, ctx - 1:-1]
        z_one = jepa.transition_mean(z_prev, uf_ann)
        return z_last, z_roll, z_tgt, z_one

    if pooled:
        z_last, z_roll, z_tgt, z_one = run(Yc, Uc, Yf, Uf)
    else:
        parts = [run(Yc[i:i + 1], Uc[i:i + 1], Yf[i:i + 1], Uf[i:i + 1])
                 for i in range(len(starts))]
        z_last, z_roll, z_tgt, z_one = (torch.cat(p, 0) for p in zip(*parts))

    return dict(
        z_last=z_last.numpy(), z_roll=z_roll.numpy(), z_tgt=z_tgt.numpy(),
        err_roll=((z_roll - z_tgt) ** 2).numpy(),
        err_pers=((z_last[:, None] - z_tgt) ** 2).numpy(),
        err_one=((z_one - z_tgt) ** 2).numpy(),
    )


@torch.no_grad()
def long_rollout(jepa, y_m, u_m, ctx):
    """One rollout from the first `ctx` years to the end; y_m/u_m monthly [T, .]."""
    y, u = y_m[None], u_m[None]
    u_ann = to_annual(u)
    _, z_last = jepa.posterior(*jepa.encoder_inputs(y[:, :ctx * MONTHS], u[:, :ctx * MONTHS]))
    z_roll = jepa.roll_forward(z_last, u_ann[:, ctx:])
    z_tgt_all = ref_encode(jepa, *jepa.encoder_inputs(y, u))
    res = dict(z_roll=z_roll[0].numpy(), z_tgt=z_tgt_all[0, ctx:].numpy(),
               u_ann=u_ann[0].numpy())
    if has_emission(jepa):
        # normalised units; converted to physical in main()
        res["y_emit"] = emit_ann(jepa, z_roll, u_ann[:, ctx:])[0].numpy()
        res["y_ann"] = to_annual(y)[0, ctx:ctx + z_roll.shape[1]].numpy()
    if jepa.gated_decay:
        res["tau_roll"] = jepa.gated_efolding_years(z_last, u_ann[:, ctx:])[0].numpy()
        res["tau_tgt"] = torch.exp(jepa._log_tau(z_tgt_all[:, ctx - 1:-1], u_ann[:, ctx:]))[0].numpy()
    return res


@torch.no_grad()
def restart_rollouts(jepa, y_m, u_m, restarts):
    """
    `restarts`: years since trajectory start (each >= 2). For each r: encode
    the whole history [0, r) (monthly or annual per the encoder mode), roll
    out on annual forcing over [r, end). Returns {r: z_roll [Y_end - r, z]},
    {r: z_last [z]} (the encoder's read of year r-1).
    """
    y, u = y_m[None], u_m[None]
    u_ann = to_annual(u)
    rolls, lasts = {}, {}
    for r in restarts:
        M = r * MONTHS
        _, z_last = jepa.posterior(*jepa.encoder_inputs(y[:, :M], u[:, :M]))
        rolls[r] = jepa.roll_forward(z_last, u_ann[:, r:])[0].numpy()
        lasts[r] = z_last[0].numpy()
    return rolls, lasts


def restart_gaps(rolls, lasts, restarts):
    """
    Gap at each later restart r_k between the rollout from the first restart
    r_0 and the encoder's read there: z_roll_0[year r_k - 1] - z_last(r_k).
    Returns [len(restarts) - 1, z].
    """
    r0 = restarts[0]
    return np.stack([rolls[r0][rk - 1 - r0] - lasts[rk] for rk in restarts[1:]])


@contextmanager
def batch_stats(jepa):
    """Temporarily standardize with batch stats, as before `fit_z_stats`."""
    if not hasattr(jepa, "z_stats_fitted"):
        yield
        return
    was = bool(jepa.z_stats_fitted)
    jepa.z_stats_fitted.fill_(False)
    try:
        yield
    finally:
        jepa.z_stats_fitted.fill_(was)


# ─────────────────────────────────────────────────────────────────────────────
# plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_emission(y_emit, y_ann, regions, sel, ctx, tag, out_dir):
    """y_emit/y_ann [Y, 2R] physical annual means (tas then pr)."""
    R = len(regions)
    years = np.arange(ctx, ctx + y_emit.shape[0])
    fig, axs = plt.subplots(2, len(sel), figsize=(4.5 * len(sel), 5.5),
                            sharex=True, squeeze=False)
    for j, name in enumerate(sel):
        r = regions.index(name)
        for i, (off, lab) in enumerate([(0, "tas [K]"), (R, "pr")]):
            ax = axs[i, j]
            ax.plot(years, y_ann[:, off + r], color="tab:red", lw=1, label="ESM")
            ax.plot(years, y_emit[:, off + r], color="tab:blue", lw=1.2,
                    label="emit(z rollout)")
            ax.set_title(f"{name} {lab}", fontsize=9)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"{tag}: emit(z rollout)\nx = years after start", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"emission_{tag}.png"), dpi=120)
    plt.close(fig)


def plot_restarts(rolls, z_tgt, restarts, order, efold, gain, year0, tag, out_dir):
    """z_tgt [Y, z] target encoding over the whole trajectory."""
    n = len(order)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.3 * nrow), sharex=True)
    axs = axs.ravel()
    years = year0 + np.arange(z_tgt.shape[0])
    cmap = plt.get_cmap("viridis")
    for j, d in enumerate(order):
        ax = axs[j]
        ax.plot(years, z_tgt[:, d], color="tab:red", lw=0.8, label="target enc")
        for k, r in enumerate(restarts):
            c = cmap(k / max(1, len(restarts) - 1))
            yr = year0 + r + np.arange(rolls[r].shape[0])
            ax.plot(yr, rolls[r][:, d], color=c, lw=1.2, label=f"from {year0 + r}")
            ax.axvline(year0 + r, color=c, lw=0.5, ls=":")
        ax.set_title(f"d{d} | τ={efold[d]:.1f}y | gain={gain[d]:.2f}", fontsize=8)
    for k in range(n, len(axs)):
        axs[k].axis("off")
    axs[0].legend(fontsize=6)
    fig.suptitle(f"{tag}: rollouts restarted from the encoder's read of the full history",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"restarts_{tag}.png"), dpi=120)
    plt.close(fig)


def plot_restarts_emission(e_rolls, y_ann, restarts, regions, sel, year0, tag, out_dir):
    """e_rolls {r: [Y - r, 2R]}, y_ann [Y, 2R], physical units; tas only."""
    fig, axs = plt.subplots(1, len(sel), figsize=(5 * len(sel), 3.5), squeeze=False)
    years = year0 + np.arange(y_ann.shape[0])
    cmap = plt.get_cmap("viridis")
    for j, name in enumerate(sel):
        ri = regions.index(name)
        ax = axs[0, j]
        ax.plot(years, y_ann[:, ri], color="tab:red", lw=0.8, label="ESM")
        for k, r in enumerate(restarts):
            ax.plot(year0 + r + np.arange(e_rolls[r].shape[0]), e_rolls[r][:, ri],
                    color=cmap(k / max(1, len(restarts) - 1)), lw=1.2,
                    label=f"from {year0 + r}")
        ax.set_title(f"{name} tas [K], emit(z rollout)", fontsize=9)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"restarts_emission_{tag}.png"), dpi=120)
    plt.close(fig)


def plot_lead(res, order, efold, tag, out_dir):
    lead = np.arange(1, res["err_roll"].shape[1] + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, lab in [("err_roll", "rollout"), ("err_pers", "persistence"), ("err_one", "one-step")]:
        ax[0].plot(lead, res[key].mean(axis=(0, 2)), label=lab)
    ax[0].set(xlabel="lead [yr]", ylabel="MSE (mean over windows, dims)",
              title=f"{tag}: rollout vs target encoder")
    ax[0].legend()
    var = res["z_tgt"].reshape(-1, res["z_tgt"].shape[-1]).var(0) + 1e-8
    nerr = res["err_roll"].mean(0) / var  # [H, z]
    cmap = plt.get_cmap("viridis")
    for j, d in enumerate(order):
        ax[1].plot(lead, nerr[:, d], color=cmap(j / max(1, len(order) - 1)),
                   label=f"d{d} ({efold[d]:.0f}y)")
    ax[1].set(xlabel="lead [yr]", ylabel="MSE / Var(target)  (1 - R²)",
              title="per dim, coloured by e-folding time", yscale="log")
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"lead_error_{tag}.png"), dpi=130)
    plt.close(fig)


def plot_long(res, u, order, efold, bnorm, ctx, tag, out_dir, key_a="z_roll",
              key_b="z_tgt", fname="long_rollout", ylab="z"):
    a, b = res[key_a], res[key_b]
    years = np.arange(ctx, ctx + a.shape[0])
    n = len(order)
    ncol = 4
    nrow = int(np.ceil(n / ncol)) + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.3 * nrow), sharex=True)
    axs = axs.ravel()
    axs[0].plot(years, u[ctx:ctx + a.shape[0], 0], color="tab:green")
    axs[0].set_title("GSAT forcing (scaled)", fontsize=9)
    for i in range(1, ncol):
        axs[i].axis("off")
    for j, d in enumerate(order):
        ax = axs[ncol + j]
        ax.plot(years, b[:, d], color="tab:red", lw=1, label="target enc")
        ax.plot(years, a[:, d], color="tab:blue", lw=1.2, label="rollout")
        ax.set_title(f"d{d} | τ={efold[d]:.1f}y | ‖B‖={bnorm[d]:.3f}", fontsize=8)
        if ylab == "tau":
            ax.set_yscale("log")
    for k in range(ncol + n, len(axs)):
        axs[k].axis("off")
    axs[ncol].legend(fontsize=7)
    fig.suptitle(f"{tag} ({ylab}); x = years after trajectory start", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fname}_{tag}.png"), dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--weights", default=None,
                   help="explicit weights file; overrides --checkpoint")
    p.add_argument("--checkpoint", default=None,
                   help="'latest' or an epoch number from <run_dir>/checkpoints. Default: "
                        "model_out, or the latest checkpoint if training hasn't finished. "
                        "A fitted *_zstats copy next to it is used if present")
    p.add_argument("--weights_key", default="ema", choices=["ema", "model"],
                   help="which weights of an epoch checkpoint {model, ema}")
    p.add_argument("--fit_z_stats", action="store_true",
                   help="fit the z stats in memory before evaluating (loads the training "
                        "data, as fit_jepa_kl_z_stats does) instead of needing a *_zstats file")
    p.add_argument("--cluster", default="IIASA", choices=list(MODEL_PATHS))
    p.add_argument("--esm", default="ACCESS-ESM1-5")
    p.add_argument("--scenarios", nargs="+", default=["ssp126"])
    p.add_argument("--context_years", type=int, default=None,
                   help="default: context_len from config.txt")
    p.add_argument("--horizon_years", type=int, default=None,
                   help="default: horizon from config.txt (training rollout length)")
    p.add_argument("--window_stride_years", type=int, default=5)
    p.add_argument("--out_dir", default=None, help="default: <run_dir>/eval_rollout")
    p.add_argument("--restart_years", nargs="+", type=int, default=None,
                   help="calendar years to restart the rollout from, each using the whole "
                        "history before it as context, e.g. 1860 1950 2010 (see 5. above)")
    p.add_argument("--traj_start_year", type=int, default=1850,
                   help="calendar year of the first month of each trajectory")
    p.add_argument("--emit_regions", nargs="+", default=["SOO"],
                   help="regions to plot the annual emission for (emission runs only)")
    args = p.parse_args()

    cfg = read_config(args.run_dir)
    ctx = args.context_years or cfg.get("context_len", 120) // MONTHS
    H = args.horizon_years or cfg.get("horizon", 1200) // MONTHS
    weights, wtag = resolve_weights(args.run_dir, args.weights, args.checkpoint)
    out_dir = args.out_dir or os.path.join(args.run_dir, f"eval_rollout_{wtag}")
    os.makedirs(out_dir, exist_ok=True)

    S = scales_ssm.StandardScaler
    tas_sc = S.from_file(os.path.join(args.run_dir, "y_scaler.out"))
    pr_sc = S.from_file(os.path.join(args.run_dir, "pr_scaler.out"))
    u_sc = S.from_file(os.path.join(args.run_dir, "u_scaler.out"))

    trajs, regions = load_trajectories(
        MODEL_PATHS[args.cluster].format(model=args.esm), args.scenarios)
    if not trajs:
        raise SystemExit(f"no files found for {args.scenarios}")
    y_dim = trajs[0]["tas"].shape[1]
    jepa = load_jepa(args.run_dir, weights, args.weights_key, cfg, y_dim, 1)
    if args.fit_z_stats and not uses_z_stats(jepa):
        print("[z-stats] this latent process has no z standardization; --fit_z_stats ignored")
    elif args.fit_z_stats:
        from proto_scales.fit_jepa_kl_z_stats import fit_on_training_set
        fit_on_training_set(jepa, args.run_dir, cfg, args.cluster)

    efold = jepa.efolding_years().numpy()
    bnorm = jepa.B.weight.detach().norm(dim=1).numpy()
    # long-run response of each dim to constant forcing, ||B_row|| / (1 - d)
    # with the baseline decay (gate off)
    with torch.no_grad():
        gain = bnorm / (1.0 - jepa.decay().numpy())
    order = np.argsort(efold)
    print(f"[model] ctx {ctx} yr | horizon {H} yr | gated_decay {jepa.gated_decay} | "
          f"monthly_encoder {jepa.monthly_encoder} | "
          f"normalize_z {uses_z_stats(jepa)} | trans_mlp {jepa.trans_mlp is not None}")

    # no standardization at all (ELBO) -> the frame is fixed by construction
    fitted = (not uses_z_stats(jepa)) or bool(jepa.z_stats_fitted)
    if uses_z_stats(jepa) and not fitted:
        print("[warn] checkpoint has no fitted z stats -> batch stats everywhere. "
              "Run proto_scales.fit_jepa_kl_z_stats and pass --weights <...>_zstats")
    saved = {"efold": efold, "bnorm": bnorm}
    rows = []
    for i, tr in enumerate(trajs):
        tag = f"{i:02d}_{tr['name'].split(':')[0]}"
        tas = tas_sc.transform(tr["tas"][None])[0]
        pr = pr_sc.transform(tr["pr"][None])[0]
        u = u_sc.transform(tr["u"][None])[0]
        y_m = torch.tensor(np.concatenate([tas, pr], -1), dtype=torch.float32)
        u_m = torch.tensor(u, dtype=torch.float32)
        Y = y_m.shape[0] // MONTHS  # whole years available for windows

        # --- 1 + 2: training-frame windows, fixed vs batch stats -----------
        starts = list(range(0, Y - ctx - H + 1, args.window_stride_years))
        if starts:
            runs = {}
            if fitted:
                runs["fixed"] = score_windows(jepa, y_m, u_m, ctx, H, starts, pooled=False)
            if uses_z_stats(jepa):
                with batch_stats(jepa):
                    runs["batchB1"] = score_windows(jepa, y_m, u_m, ctx, H, starts, pooled=False)
                    runs["batchPooled"] = score_windows(jepa, y_m, u_m, ctx, H, starts, pooled=True)
            main_key = "fixed" if fitted else "batchB1"
            for k, r in runs.items():
                plot_lead(r, order, efold, f"{tag}_{k}", out_dir)
                for kk, v in r.items():
                    saved[f"{tag}/{k}/{kk}"] = v
            ref = runs[main_key]
            row = dict(traj=tr["name"], windows=len(starts))
            for k, r in runs.items():
                row[f"roll_{k}"] = r["err_roll"].mean()
            row.update(pers=ref["err_pers"].mean(), one_step=ref["err_one"].mean(),
                       roll_last10=ref["err_roll"][:, -10:].mean())
            if uses_z_stats(jepa):
                other = runs["batchPooled"] if main_key == "batchB1" else runs["batchB1"]
                dz = np.linalg.norm(ref["z_last"] - other["z_last"], axis=-1)
                row["zlast_rel_diff"] = float((dz / (np.linalg.norm(ref["z_last"], axis=-1) + 1e-8)).mean())
            rows.append(row)
        else:
            print(f"[skip windows] {tr['name']}: {Y} yr < ctx+horizon {ctx + H}")

        # --- 3: long rollout -----------------------------------------------
        lr = long_rollout(jepa, y_m, u_m, ctx)
        u_np = lr["u_ann"]
        plot_long(lr, u_np, order, efold, bnorm, ctx, tag, out_dir)
        if jepa.gated_decay:
            plot_long(lr, u_np, order, efold, bnorm, ctx, tag, out_dir,
                      key_a="tau_roll", key_b="tau_tgt", fname="gated_tau", ylab="tau")
        if has_emission(jepa):
            R = y_dim
            def phys(a):
                return np.concatenate([
                    tas_sc.inverse_transform(a[None, :, :R])[0],
                    pr_sc.inverse_transform(a[None, :, R:])[0]], -1)
            lr["y_emit"], lr["y_ann"] = phys(lr["y_emit"]), phys(lr["y_ann"])
            sel = [r for r in args.emit_regions if r in regions]
            missing = sorted(set(args.emit_regions) - set(sel))
            if missing and i == 0:
                print(f"[warn] --emit_regions not found: {missing}; available: {regions}")
            if sel:
                plot_emission(lr["y_emit"], lr["y_ann"], regions, sel, ctx, tag, out_dir)
            if rows and rows[-1]["traj"] == tr["name"]:
                err = (lr["y_emit"] - lr["y_ann"])[:, :R]
                rows[-1]["emit_tas_bias_last100"] = float(err[-100:].mean())
                for r in sel:
                    rows[-1][f"emit_{r}_bias_last100"] = float(err[-100:, regions.index(r)].mean())
        for k, v in lr.items():
            saved[f"{tag}/long/{k}"] = v

        # --- 5: restarts ---------------------------------------------------
        if args.restart_years:
            rs = sorted({yr - args.traj_start_year for yr in args.restart_years})
            rs = [r for r in rs if 2 <= r < Y]
            if len(rs) < len(set(args.restart_years)):
                print(f"[restarts] {tr['name']}: kept {[args.traj_start_year + r for r in rs]} "
                      f"(trajectory {args.traj_start_year}-{args.traj_start_year + Y - 1})")
            if rs:
                rolls, lasts = restart_rollouts(jepa, y_m, u_m, rs)
                with torch.no_grad():
                    z_tgt_full = ref_encode(jepa, *jepa.encoder_inputs(y_m[None], u_m[None]))[0].numpy()
                plot_restarts(rolls, z_tgt_full, rs, order, efold, gain,
                              args.traj_start_year, tag, out_dir)
                for r in rs:
                    saved[f"{tag}/restart/{args.traj_start_year + r}/z_roll"] = rolls[r]
                    saved[f"{tag}/restart/{args.traj_start_year + r}/z_last"] = lasts[r]
                if len(rs) > 1:
                    gaps = restart_gaps(rolls, lasts, rs)
                    saved[f"{tag}/restart/gaps"] = gaps
                    y0 = args.traj_start_year
                    print(f"\n[restarts] {tr['name']}: z_roll(from {y0 + rs[0]}) - "
                          f"z_encoder(read at year), per dim")
                    print("   dim   efold    gain  " + "".join(f"{y0 + r:>9d}" for r in rs[1:]))
                    for d in order:
                        print(f"   d{d:<3d}{efold[d]:7.1f} {gain[d]:7.3f}  "
                              + "".join(f"{g:9.3f}" for g in gaps[:, d]))
                if has_emission(jepa):
                    sel = [r for r in args.emit_regions if r in regions]
                    if sel:
                        with torch.no_grad():
                            u_ann_full = to_annual(u_m[None])[0]
                            e_rolls = {r: phys(emit_ann(
                                jepa, torch.tensor(rolls[r]),
                                u_ann_full[r:r + rolls[r].shape[0]]).numpy())
                                       for r in rs}
                        y_ann_full = phys(to_annual(y_m[None])[0].numpy())
                        plot_restarts_emission(e_rolls, y_ann_full, rs, regions, sel,
                                               args.traj_start_year, tag, out_dir)

    if rows:
        df = pd.DataFrame(rows)
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print("\nMSE in z (training frame). roll_* should be well below pers. "
                  "zlast_rel_diff = ||z_last(fixed) - z_last(batch,B=1)|| / ||z_last(fixed)|| "
                  "(without fitted stats: B=1 vs pooled)")
            print(df.to_string(index=False))
        df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print("\nper-dim (sorted by e-folding): efold [yr] | ||B_row|| | long-run gain ||B||/(1-d)")
    for d in order:
        print(f"  d{d:2d}  {efold[d]:7.1f}  {bnorm[d]:.4f}  {gain[d]:.3f}")
    np.savez_compressed(os.path.join(out_dir, "eval_jepa_rollout.npz"), **saved)
    print(f"\nwrote plots + summary.csv + eval_jepa_rollout.npz to {out_dir}")


if __name__ == "__main__":
    main()
