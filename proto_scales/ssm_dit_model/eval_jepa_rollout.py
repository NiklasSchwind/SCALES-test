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


CKPT_RE = re.compile(r"annual_jepa_kl_dit_epoch(\d+)\.pt$")


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


def load_jepa(run_dir, weights, weights_key, cfg, y_dim, u_dim):
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
    )
    path = weights or os.path.join(run_dir, "model_out")
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and ("model" in sd or "ema" in sd):
        sd = sd[weights_key]
        src = f"{path} ['{weights_key}']"
    else:
        src = path
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
        z_tgt_all = jepa.encode_target(torch.cat([ye_c, ye_f], 1), torch.cat([ue_c, ue_f], 1))
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
    z_tgt_all = jepa.encode_target(*jepa.encoder_inputs(y, u))
    res = dict(z_roll=z_roll[0].numpy(), z_tgt=z_tgt_all[0, ctx:].numpy(),
               u_ann=u_ann[0].numpy())
    if jepa.gated_decay:
        res["tau_roll"] = jepa.gated_efolding_years(z_last, u_ann[:, ctx:])[0].numpy()
        res["tau_tgt"] = torch.exp(jepa._log_tau(z_tgt_all[:, ctx - 1:-1], u_ann[:, ctx:]))[0].numpy()
    return res


@contextmanager
def batch_stats(jepa):
    """Temporarily standardize with batch stats, as before `fit_z_stats`."""
    was = bool(jepa.z_stats_fitted)
    jepa.z_stats_fitted.fill_(False)
    try:
        yield
    finally:
        jepa.z_stats_fitted.fill_(was)


# ─────────────────────────────────────────────────────────────────────────────
# plots
# ─────────────────────────────────────────────────────────────────────────────
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
    if args.fit_z_stats:
        from proto_scales.fit_jepa_kl_z_stats import fit_on_training_set
        fit_on_training_set(jepa, args.run_dir, cfg, args.cluster)

    efold = jepa.efolding_years().numpy()
    bnorm = jepa.B.weight.detach().norm(dim=1).numpy()
    order = np.argsort(efold)
    print(f"[model] ctx {ctx} yr | horizon {H} yr | gated_decay {jepa.gated_decay} | "
          f"monthly_encoder {jepa.monthly_encoder} | "
          f"normalize_z {jepa.normalize_z} | trans_mlp {jepa.trans_mlp is not None}")

    fitted = bool(jepa.z_stats_fitted) and jepa.normalize_z
    if jepa.normalize_z and not fitted:
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
        for k, v in lr.items():
            saved[f"{tag}/long/{k}"] = v

    if rows:
        df = pd.DataFrame(rows)
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print("\nMSE in z (training frame). roll_* should be well below pers. "
                  "zlast_rel_diff = ||z_last(fixed) - z_last(batch,B=1)|| / ||z_last(fixed)|| "
                  "(without fitted stats: B=1 vs pooled)")
            print(df.to_string(index=False))
        df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print("\nper-dim (sorted by e-folding): efold [yr] | ||B_row||")
    for d in order:
        print(f"  d{d:2d}  {efold[d]:7.1f}  {bnorm[d]:.4f}")
    np.savez_compressed(os.path.join(out_dir, "eval_jepa_rollout.npz"), **saved)
    print(f"\nwrote plots + summary.csv + eval_jepa_rollout.npz to {out_dir}")


if __name__ == "__main__":
    main()
