"""
Where does the long-horizon warm bias of an `annual_elbo_dit` forecast come
from: the latent rollout, the DiT's z -> tas mapping, or the DiT feeding on
its own output?

Run on the cluster from the repo root, e.g.

    python -m proto_scales.ssm_dit_model.eval_elbo_dit_bias \
        --run_dir outputs_ssm_scales/annual_elbo_dit_2026... \
        --scenarios ssp126 --start_years 1860 2010 --n_samples 10

For each trajectory and start year S, the DiT gets `context_len` months of
true field before S and generates S..end. Curves are annual, in K, for the
unweighted regional mean and for single regions (--regions plus the
--auto_regions worst ones). The regional mean is NOT a useful bias test on
its own: pattern scaling of u (the ESM's own GSAT) pins it to the ESM, so a
forced-response bias shows up only in regions that lag GSAT (oceans, SOO).
`region_bias_<traj>_<S>.csv` ranks every region by its bias over years 51+.

Curves (`bias_<traj>_<S>_<target>.png`, top: levels, bottom: minus ESM)
----------------------------------------------------------------
    ESM           the trajectory itself
    SSM emit      ssm.emit_mean(z_roll, u): pattern scaling + residual(z),
                  along the z rollout the DiT is conditioned on. Deterministic.
    SSM emit(full) same, but z_last read from the whole history since the
                  trajectory start (what the restart eval uses) instead of
                  only the `context_len` window (what `sample_blocks` uses).
    DiT free      `sample_blocks` equivalent: each block's field context is the
                  DiT's own previous output. Mean over --n_samples. This is
                  the forecast.
    DiT free(full) as DiT free, z from the full-history read.
    DiT tf        teacher-forced: z as in DiT free, but each block's field
                  context is the TRUE ESM field. Mean over --n_samples.

Reading it
----------
    SSM emit biased                   -> latent rollout / emission is the problem
    SSM emit ok, DiT tf biased        -> DiT does not take the forced response
                                         from z (mapping z,u -> tas is off)
    DiT tf ok, DiT free biased        -> exposure bias: the DiT drifts on its
                                         own generated context
    (full) curves clearly better      -> 10 yr of context is too little to read
                                         the slow z dims; z_last is the problem

Printed and saved to summary.csv: mean bias over first block, years 1-50,
51-100, 101-200, 201+ for every curve, plus per-block bias of DiT free / tf.
Arrays go to `bias.npz`.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from proto_scales.ssm_dit_model.annual_elbo_dit import load_elbo_dit_annual
from proto_scales.ssm_dit_model.annual_ssm import broadcast_to_monthly, to_annual
from proto_scales.ssm_dit_model.eval_jepa_rollout import MODEL_PATHS, load_trajectories

MONTHS = 12


# ─────────────────────────────────────────────────────────────────────────────
# latents
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def latent_trajectory(ssm, y_hist, u_hist, u_fut, n_ctx):
    """
    z for the DiT, [B, n_ctx + n_fut, z] monthly, built like
    `ForcedAnnualLatentEncoder.forward`: posterior over `y_hist` (its last
    `n_ctx` months form the context half), then the mean rollout over u_fut.
    `y_hist` may be longer than n_ctx (full-history read). Also returns the
    annual rollout [B, Y_fut, z].
    """
    z_hist_ann, z_last = ssm.posterior(*ssm.encoder_inputs(y_hist, u_hist))
    z_fut_ann = ssm.roll_forward(z_last, to_annual(u_fut))
    n_ctx_yr = n_ctx // MONTHS
    z_ctx = broadcast_to_monthly(z_hist_ann[:, -n_ctx_yr:], n_ctx)
    z_fut = broadcast_to_monthly(z_fut_ann, u_fut.shape[1])
    return torch.cat([z_ctx, z_fut], dim=1), z_fut_ann


# ─────────────────────────────────────────────────────────────────────────────
# sampling
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample(model, z_all, u_all, field_ctx, field_true_fut, n_steps, generator,
           teacher_forced):
    """
    Block-wise sampling over the whole future with a given z trajectory.
    Same loop as `AnnualSSMOutpaintingDiT.sample_blocks`, except that z is
    passed in and, if `teacher_forced`, each block's field context comes from
    the true field instead of the generated one. Assumes the window starts in
    January (trajectories do; start years are whole years).
    Returns the generated future field [B, n_fut, 2R].
    """
    n_fut = u_all.shape[1] - model.context_len
    hist_gen = field_ctx
    hist_true = torch.cat([field_ctx, field_true_fut], dim=1)
    out, done = [], 0
    while done < n_fut:
        w0, bs, nb = model.block_bounds(done // model.block_len, n_fut)
        win = slice(w0, bs + nb)
        ctx = (hist_true if teacher_forced else hist_gen)[:, w0:bs]
        cond = model.window_conditioning(
            z_win=z_all[:, win], u_win=u_all[:, win], field_ctx=ctx,
            n_block=nb, start_month=w0 % MONTHS)
        blk = model._sample_window(cond, ctx, nb, n_steps, generator)
        hist_gen = torch.cat([hist_gen, blk], dim=1)
        out.append(blk)
        done += nb
    return torch.cat(out, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def to_K(tas_norm_ann, tas_sc):
    """[..., Y, R] normalised annual tas -> [..., Y, R] in K."""
    a = np.asarray(tas_norm_ann)
    shp = a.shape
    return tas_sc.inverse_transform(a.reshape(1, -1, shp[-1]))[0].reshape(shp)


def select(a, target, regions):
    """[..., R] -> [...]: unweighted regional mean ('mean') or one region."""
    return a.mean(-1) if target == "mean" else a[..., regions.index(target)]


def lead_bins(Y, block_years):
    bins = {f"blk0 (1-{block_years})": slice(0, block_years),
            "1-50": slice(0, 50), "51-100": slice(50, 100),
            "101-200": slice(100, 200), "201+": slice(200, None)}
    return {k: s for k, s in bins.items() if (s.start or 0) < Y}


def plot(years, curves, esm, title, path):
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                            gridspec_kw=dict(height_ratios=[3, 2]))
    axs[0].plot(years, esm, color="tab:red", lw=1.2, label="ESM")
    styles = {
        "SSM emit": dict(color="black", lw=1.5),
        "SSM emit(full)": dict(color="black", lw=1.2, ls="--"),
        "DiT free": dict(color="tab:blue", lw=1.3),
        "DiT free(full)": dict(color="tab:blue", lw=1.1, ls="--"),
        "DiT tf": dict(color="tab:green", lw=1.3),
    }
    for k, v in curves.items():
        axs[0].plot(years, v, label=k, **styles.get(k, {}))
        axs[1].plot(years, v - esm, label=k, **styles.get(k, {}))
    axs[1].axhline(0, color="tab:red", lw=0.8)
    axs[0].set_ylabel("regional-mean tas [K]")
    axs[1].set_ylabel("minus ESM [K]")
    axs[1].set_xlabel("year")
    axs[0].legend(fontsize=8, ncol=3)
    axs[0].set_title(title, fontsize=10)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="None/'latest' = highest epoch checkpoint, 'model_out', or an epoch number")
    p.add_argument("--raw_weights", action="store_true",
                   help="use the raw (non-EMA) weights of an epoch checkpoint")
    p.add_argument("--cluster", default="IIASA", choices=list(MODEL_PATHS))
    p.add_argument("--esm", default="ACCESS-ESM1-5")
    p.add_argument("--scenarios", nargs="+", default=["ssp126"])
    p.add_argument("--max_trajs", type=int, default=1,
                   help="trajectories per run (each costs n_samples x 3 DiT rollouts)")
    p.add_argument("--traj_start_year", type=int, default=1850)
    p.add_argument("--start_years", nargs="+", type=int, default=[2010],
                   help="first generated year; needs context_len years of history before it")
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--n_steps", type=int, default=None, help="flow steps (default: model's)")
    p.add_argument("--no_full_history", action="store_true",
                   help="skip the full-history-z variants (saves one DiT ensemble)")
    p.add_argument("--regions", nargs="+", default=["SOO"],
                   help="regions to plot besides the regional mean")
    p.add_argument("--auto_regions", type=int, default=3,
                   help="also plot the N regions with the largest DiT-free bias")
    p.add_argument("--top_regions", type=int, default=15,
                   help="rows of the printed per-region bias ranking")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="gpu")
    p.add_argument("--out_dir", default=None, help="default: <run_dir>/eval_bias_<ckpt>")
    args = p.parse_args()

    ckpt = args.checkpoint
    if ckpt not in (None, "latest", "model_out"):
        ckpt = int(ckpt)

    # y_dim is needed to build the model; read it off the data first
    trajs, regions = load_trajectories(
        MODEL_PATHS[args.cluster].format(model=args.esm), args.scenarios)
    if not trajs:
        raise SystemExit(f"no files found for {args.scenarios}")
    trajs = trajs[:args.max_trajs]
    y_dim = trajs[0]["tas"].shape[1]

    model, tas_sc, pr_sc, u_sc = load_elbo_dit_annual(
        args.run_dir, y_dim, u_dim=1, device=args.device,
        ema=not args.raw_weights, checkpoint=ckpt)
    dev = next(model.parameters()).device
    ssm = model.ssm_encoder.ssm
    n_ctx = model.context_len
    ctx_yr = n_ctx // MONTHS
    block_yr = model.block_len // MONTHS
    print(f"[model] context {ctx_yr} yr | block {block_yr} yr | "
          f"enc_window_years {int(ssm.enc_window_years)} | device {dev} | "
          f"flow steps {args.n_steps or model.n_flow_steps}"
          f"{'' if args.n_steps else ' (model default)'}")

    tag_ck = "latest" if ckpt in (None, "latest") else str(ckpt)
    out_dir = args.out_dir or os.path.join(args.run_dir, f"eval_bias_{tag_ck}")
    os.makedirs(out_dir, exist_ok=True)

    gen = torch.Generator(device=dev)
    rows, saved = [], {}
    for i, tr in enumerate(trajs):
        name = tr["name"].split(":")[0]
        Y = tr["tas"].shape[0] // MONTHS
        T = Y * MONTHS
        tas = tas_sc.transform(tr["tas"][None, :T])[0]
        pr = pr_sc.transform(tr["pr"][None, :T])[0]
        u = u_sc.transform(tr["u"][None, :T])[0]
        field = torch.tensor(np.concatenate([tas, pr], -1), dtype=torch.float32, device=dev)
        u_t = torch.tensor(u, dtype=torch.float32, device=dev)
        esm_ann_all = to_annual(field[None, :, :y_dim])[0].cpu().numpy()

        for S in args.start_years:
            s = S - args.traj_start_year
            if s < ctx_yr or s >= Y - 1:
                print(f"[skip] {name} start {S}: need {ctx_yr} yr history and >1 yr future")
                continue
            tag = f"{i:02d}_{name}_{S}"
            B = args.n_samples
            m0, m1 = (s - ctx_yr) * MONTHS, s * MONTHS
            field_ctx = field[None, m0:m1].expand(B, -1, -1).contiguous()
            field_fut = field[None, m1:].expand(B, -1, -1).contiguous()
            u_ctx = u_t[None, m0:m1].expand(B, -1, -1)
            u_fut = u_t[None, m1:].expand(B, -1, -1)
            u_all = torch.cat([u_ctx, u_fut], dim=1)
            u_fut_ann = to_annual(u_fut)

            variants = {"ctx": (field[None, m0:m1].expand(B, -1, -1),
                                u_t[None, m0:m1].expand(B, -1, -1))}
            if not args.no_full_history and m0 > 0:
                variants["full"] = (field[None, :m1].expand(B, -1, -1),
                                    u_t[None, :m1].expand(B, -1, -1))

            years = S + np.arange(Y - s)
            esm = to_K(esm_ann_all[s:], tas_sc)                            # [Y, R]
            curves = {}                                                    # name -> [Y, R]
            for vk, (y_hist, u_hist) in variants.items():
                sfx = "" if vk == "ctx" else "(full)"
                z_all, z_fut_ann = latent_trajectory(ssm, y_hist, u_hist, u_fut, n_ctx)
                with torch.no_grad():
                    emit = ssm.emit_mean(z_fut_ann[:1], u_fut_ann[:1])[0].cpu().numpy()
                curves[f"SSM emit{sfx}"] = to_K(emit, tas_sc)

                modes = [("free", False)] + ([("tf", True)] if vk == "ctx" else [])
                for mk, tf in modes:
                    gen.manual_seed(args.seed)
                    print(f"[sample] {tag} z={vk} {mk}: {B} samples x "
                          f"{model.n_blocks(u_fut.shape[1])} blocks")
                    fut = sample(model, z_all, u_all, field_ctx, field_fut,
                                 args.n_steps, gen, teacher_forced=tf)
                    tas_ann = to_annual(fut[..., :y_dim]).cpu().numpy()   # [B, Y, R]
                    g = to_K(tas_ann, tas_sc)                              # [B, Y, R]
                    curves[f"DiT {mk}{sfx}"] = g.mean(0)
                    saved[f"{tag}/dit_{mk}_{vk}_samples"] = g

            for k, v in curves.items():
                saved[f"{tag}/{k}"] = v
            saved[f"{tag}/esm"] = esm
            saved[f"{tag}/years"] = years

            # Per-region ranking: the regional mean is dominated by pattern
            # scaling of u (the ESM's own GSAT), so a bias only shows up in
            # regions whose response lags GSAT (oceans, SOO, ...).
            late = slice(min(50, len(years) - 1), None)
            rb = pd.DataFrame({k: (v - esm)[late].mean(0) for k, v in curves.items()},
                              index=regions)
            rb.index.name = "region"
            rb.to_csv(os.path.join(out_dir, f"region_bias_{tag}.csv"))
            sort_key = "DiT free" if "DiT free" in rb else rb.columns[0]
            rb_sorted = rb.reindex(rb[sort_key].abs().sort_values(ascending=False).index)
            with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
                print(f"\n[{tag}] mean bias over years 51+ [K], top {args.top_regions} "
                      f"regions by |{sort_key}|")
                print(rb_sorted.head(args.top_regions).to_string())

            targets = ["mean"] + [r for r in args.regions if r in regions]
            targets += [r for r in rb_sorted.index[:args.auto_regions] if r not in targets]
            missing = sorted(set(args.regions) - set(regions))
            if missing:
                print(f"[warn] --regions not found: {missing}; available: {regions}")

            bins = lead_bins(len(years), block_yr)
            nb = int(np.ceil(len(years) / block_yr))
            for tgt in targets:
                e = select(esm, tgt, regions)
                cv = {k: select(v, tgt, regions) for k, v in curves.items()}
                lab = "unweighted regional mean" if tgt == "mean" else f"region {tgt}"
                plot(years, cv, e,
                     f"{tr['name']} | start {S} | ctx {ctx_yr} yr | "
                     f"DiT = mean of {B} samples | {lab}",
                     os.path.join(out_dir, f"bias_{tag}_{tgt}.png"))
                for k, v in cv.items():
                    row = dict(traj=name, start=S, target=tgt, curve=k)
                    for bk, sl in bins.items():
                        row[bk] = float((v - e)[sl].mean())
                    rows.append(row)

                # per-block bias: does it step up block by block?
                print(f"\n[{tag} | {tgt}] bias per DiT block [K] (block = {block_yr} yr)")
                print("   block  " + "".join(f"{k:>16s}" for k in cv))
                for b in range(nb):
                    sl = slice(b * block_yr, (b + 1) * block_yr)
                    print(f"   {b:5d}  " + "".join(f"{(v - e)[sl].mean():16.3f}"
                                                 for v in cv.values()))

    if rows:
        df = pd.DataFrame(rows)
        with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
            print("\nmean bias (model - ESM) [K], unweighted regional mean, by lead years")
            print(df.to_string(index=False))
        df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "bias.npz"), **saved)
    print(f"\nwrote plots + summary.csv + bias.npz to {out_dir}")


if __name__ == "__main__":
    main()
