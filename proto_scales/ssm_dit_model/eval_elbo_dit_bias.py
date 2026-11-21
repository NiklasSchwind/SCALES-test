"""
Where does the long-horizon bias of an `annual_elbo_dit` forecast come from:
the latent rollout, the DiT's z -> field mapping, or the DiT feeding on its
own output? For tas and pr.

Run on the cluster from the repo root, e.g.

    python -m proto_scales.ssm_dit_model.eval_elbo_dit_bias \
        --run_dir outputs_ssm_scales/annual_elbo_dit_2026... \
        --scenarios ssp126 --start_years 1860 2010 --n_samples 10

For each trajectory and start year S, the DiT gets `context_len` months of
true field before S and generates S..end. Curves are annual means in
physical units, per variable, for the unweighted regional mean and for
single regions (--regions plus the --auto_regions worst ones per variable).
The regional mean is NOT a useful tas bias test on its own: pattern scaling
of u (the ESM's own GSAT) pins it to the ESM, so a forced-response bias shows
up only in regions that lag GSAT (oceans, SOO).
`region_bias_<var>_<traj>_<S>.csv` ranks every region by its bias over
years 51+.

Works for full-field and residual (`dit_residual`) models: sampling goes
through the model's own `sample_blocks`, which adds the baseline back.

Curves (`bias_<var>_<traj>_<S>_<target>.png`, top: levels, bottom: minus ESM)
------------------------------------------------------------------------------
    ESM            the trajectory itself
    SSM emit       the latent process's annual mean along the z rollout the
                   DiT is conditioned on: emit_mean (tas) / emit_mean_pr (pr,
                   only with a pr head). In residual mode this IS the DiT's
                   baseline. Deterministic.
    SSM emit(full) same, z_last read from the whole history since the
                   trajectory start instead of the last `enc_context_len`
                   months (what `sample_blocks` reads; = `context_len` for
                   runs without --enc_context_len).
    DiT free       the forecast: each block's context is the DiT's own
                   previous output. Mean over --n_samples.
    DiT free(full) as DiT free, z from the full-history read.
    DiT tf         teacher-forced: z as in DiT free, but each block's context
                   is the TRUE ESM field. Mean over --n_samples.

Reading it
----------
    SSM emit biased                   -> latent rollout / emission is the problem
    SSM emit ok, DiT tf biased        -> DiT does not take the forced response
                                         from z (mapping z,u -> field is off)
    DiT tf ok, DiT free biased        -> exposure bias: the DiT drifts on its
                                         own generated context
    (full) curves clearly better      -> 10 yr of context is too little to read
                                         the slow z dims; z_last is the problem
In residual mode DiT free - SSM emit is the mean of the generated residual,
which should stay near 0 at every lead.

Printed and saved to summary.csv: mean bias over the first block, years 1-50,
51-100, 101-200, 201+ for every curve, plus per-block bias. Arrays (incl.
every sample's regional field) go to `bias.npz`.
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
from proto_scales.ssm_dit_model.annual_ssm import to_annual
from proto_scales.ssm_dit_model.eval_jepa_rollout import MODEL_PATHS, load_trajectories

MONTHS = 12
VARS = ("tas", "pr")
UNITS = {"tas": "K", "pr": "pr units"}


# ─────────────────────────────────────────────────────────────────────────────
# latents
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def to_phys(a_norm, sc):
    """[..., Y, R] normalised -> physical units."""
    a = np.asarray(a_norm)
    shp = a.shape
    return sc.inverse_transform(a.reshape(1, -1, shp[-1]))[0].reshape(shp)


def select(a, target, regions):
    """[..., R] -> [...]: unweighted regional mean ('mean') or one region."""
    return a.mean(-1) if target == "mean" else a[..., regions.index(target)]


def lead_bins(Y, block_years):
    bins = {f"blk0 (1-{block_years})": slice(0, block_years),
            "1-50": slice(0, 50), "51-100": slice(50, 100),
            "101-200": slice(100, 200), "201+": slice(200, None)}
    return {k: s for k, s in bins.items() if (s.start or 0) < Y}


STYLES = {
    "SSM emit": dict(color="black", lw=1.5),
    "SSM emit(full)": dict(color="black", lw=1.2, ls="--"),
    "DiT free": dict(color="tab:blue", lw=1.3),
    "DiT free(full)": dict(color="tab:blue", lw=1.1, ls="--"),
    "DiT tf": dict(color="tab:green", lw=1.3),
}


def plot(years, curves, esm, title, ylab, path):
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                            gridspec_kw=dict(height_ratios=[3, 2]))
    axs[0].plot(years, esm, color="tab:red", lw=1.2, label="ESM")
    for k, v in curves.items():
        axs[0].plot(years, v, label=k, **STYLES.get(k, {}))
        axs[1].plot(years, v - esm, label=k, **STYLES.get(k, {}))
    axs[1].axhline(0, color="tab:red", lw=0.8)
    axs[0].set_ylabel(ylab)
    axs[1].set_ylabel("minus ESM")
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
    p.add_argument("--ssm_only", action="store_true",
                   help="only the SSM emission curves, no DiT sampling (fast; for --ssm_only training runs)")
    p.add_argument("--vars", nargs="+", default=list(VARS), choices=list(VARS))
    p.add_argument("--regions", nargs="+", default=["SOO"],
                   help="regions to plot besides the regional mean")
    p.add_argument("--auto_regions", type=int, default=3,
                   help="also plot the N regions with the largest DiT-free bias, per variable")
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
    R = trajs[0]["tas"].shape[1]

    model, tas_sc, pr_sc, u_sc = load_elbo_dit_annual(
        args.run_dir, R, u_dim=1, device=args.device,
        ema=not args.raw_weights, checkpoint=ckpt)
    scalers = {"tas": tas_sc, "pr": pr_sc}
    dev = next(model.parameters()).device
    ssm = model.ssm_encoder.ssm
    has_pr_head = bool(getattr(ssm, "pr_emission", False))
    n_ctx = model.context_len
    ctx_yr = n_ctx // MONTHS
    block_yr = model.block_len // MONTHS
    enc_yr = getattr(model, "enc_context_len", n_ctx) // MONTHS
    print(f"[model] DiT context {ctx_yr} yr | encoder context {enc_yr} yr | "
          f"block {block_yr} yr | "
          f"enc_window_years {int(ssm.enc_window_years)} | device {dev} | "
          f"flow steps {args.n_steps or model.n_flow_steps}"
          f"{'' if args.n_steps else ' (model default)'} | "
          f"DiT {'residual' if model.dit_residual else 'full field'} | "
          f"pr head {'yes' if has_pr_head else 'no'}")

    tag_ck = "latest" if ckpt in (None, "latest") else str(ckpt)
    out_dir = args.out_dir or os.path.join(args.run_dir, f"eval_bias_{tag_ck}")
    os.makedirs(out_dir, exist_ok=True)

    gen = torch.Generator(device=dev)
    rows, saved = [], {}
    for i, tr in enumerate(trajs):
        name = tr["name"].split(":")[0]
        Y = tr["tas"].shape[0] // MONTHS
        T = Y * MONTHS
        tas = torch.tensor(tas_sc.transform(tr["tas"][None, :T])[0], dtype=torch.float32, device=dev)
        pr = torch.tensor(pr_sc.transform(tr["pr"][None, :T])[0], dtype=torch.float32, device=dev)
        u = torch.tensor(u_sc.transform(tr["u"][None, :T])[0], dtype=torch.float32, device=dev)
        field = torch.cat([tas, pr], dim=-1)
        esm_ann_all = {"tas": to_annual(tas[None])[0].cpu().numpy(),
                       "pr": to_annual(pr[None])[0].cpu().numpy()}

        for S in args.start_years:
            s = S - args.traj_start_year
            if s < ctx_yr or s >= Y - 1:
                print(f"[skip] {name} start {S}: need {ctx_yr} yr history and >1 yr future")
                continue
            tag = f"{i:02d}_{name}_{S}"
            B = args.n_samples
            m0, m1 = (s - ctx_yr) * MONTHS, s * MONTHS
            me = max(0, s - enc_yr) * MONTHS       # encoder history start (forecast)

            def rep(x):
                return x[None].expand(B, *x.shape).contiguous()

            tas_fut, pr_fut, u_fut = rep(tas[m1:]), rep(pr[m1:]), rep(u[m1:])

            # History start per variant: what the forecast reads (the last
            # enc_context_len months) and everything since the trajectory
            # start. The model reads all history it is given (encoder and
            # forcing pathway alike).
            variants = {"ctx": me}
            if not args.no_full_history and me > 0:
                variants["full"] = 0

            years = S + np.arange(Y - s)
            esm = {v: to_phys(esm_ann_all[v][s:], scalers[v]) for v in VARS}  # [Y, R]
            curves = {v: {} for v in VARS}                                     # name -> [Y, R]
            for vk, h0 in variants.items():
                sfx = "" if vk == "ctx" else "(full)"
                tas_h, pr_h, u_h = rep(tas[h0:m1]), rep(pr[h0:m1]), rep(u[h0:m1])
                tas_m, pr_m = model.ssm_mean(tas_h[:1], pr_h[:1], u_h[:1], u_fut[:1],
                                             enc_months="all")
                curves["tas"][f"SSM emit{sfx}"] = to_phys(tas_m[0].cpu().numpy(), tas_sc)
                if pr_m is not None:
                    curves["pr"][f"SSM emit{sfx}"] = to_phys(pr_m[0].cpu().numpy(), pr_sc)

                modes = [("free", False)] + ([("tf", True)] if vk == "ctx" else [])
                if args.ssm_only:
                    modes = []
                for mk, tf in modes:
                    gen.manual_seed(args.seed)
                    print(f"[sample] {tag} z={vk} {mk}: {B} samples x "
                          f"{model.n_blocks(u_fut.shape[1])} blocks")
                    tas_g, pr_g = model.sample_blocks(
                        tas_h, pr_h, u_h, u_fut, start_month=0,
                        n_steps=args.n_steps, generator=gen, enc_months="all",
                        tas_fut_true=tas_fut if tf else None,
                        pr_fut_true=pr_fut if tf else None)
                    for v, g in (("tas", tas_g), ("pr", pr_g)):
                        ga = to_phys(to_annual(g).cpu().numpy(), scalers[v])  # [B, Y, R]
                        curves[v][f"DiT {mk}{sfx}"] = ga.mean(0)
                        saved[f"{tag}/{v}/dit_{mk}_{vk}_samples"] = ga

            saved[f"{tag}/years"] = years
            bins = lead_bins(len(years), block_yr)
            nb = int(np.ceil(len(years) / block_yr))
            for v in args.vars:
                cv_all, e_all = curves[v], esm[v]
                if not cv_all:
                    print(f"[{tag} | {v}] no curves (no pr head / --ssm_only); skipped")
                    continue
                for k, a in cv_all.items():
                    saved[f"{tag}/{v}/{k}"] = a
                saved[f"{tag}/{v}/esm"] = e_all

                # Per-region ranking over years 51+.
                late = slice(min(50, len(years) - 1), None)
                rb = pd.DataFrame({k: (a - e_all)[late].mean(0) for k, a in cv_all.items()},
                                  index=regions)
                rb.index.name = "region"
                rb.to_csv(os.path.join(out_dir, f"region_bias_{v}_{tag}.csv"))
                sort_key = "DiT free" if "DiT free" in rb else rb.columns[0]
                rb_sorted = rb.reindex(rb[sort_key].abs().sort_values(ascending=False).index)
                with pd.option_context("display.width", 200,
                                       "display.float_format", "{:.3f}".format):
                    print(f"\n[{tag} | {v}] mean bias over years 51+ [{UNITS[v]}], "
                          f"top {args.top_regions} regions by |{sort_key}|")
                    print(rb_sorted.head(args.top_regions).to_string())

                targets = ["mean"] + [r for r in args.regions if r in regions]
                targets += [r for r in rb_sorted.index[:args.auto_regions] if r not in targets]
                missing = sorted(set(args.regions) - set(regions))
                if missing:
                    print(f"[warn] --regions not found: {missing}; available: {regions}")

                for tgt in targets:
                    e = select(e_all, tgt, regions)
                    cv = {k: select(a, tgt, regions) for k, a in cv_all.items()}
                    lab = "unweighted regional mean" if tgt == "mean" else f"region {tgt}"
                    plot(years, cv, e,
                         f"{tr['name']} | {v} | start {S} | ctx {ctx_yr}/{enc_yr} yr | "
                         f"DiT = mean of {B} samples | {lab}",
                         f"{v} [{UNITS[v]}]",
                         os.path.join(out_dir, f"bias_{v}_{tag}_{tgt}.png"))
                    for k, a in cv.items():
                        row = dict(var=v, traj=name, start=S, target=tgt, curve=k)
                        for bk, sl in bins.items():
                            row[bk] = float((a - e)[sl].mean())
                        rows.append(row)

                    print(f"\n[{tag} | {v} | {tgt}] bias per DiT block "
                          f"[{UNITS[v]}] (block = {block_yr} yr)")
                    print("   block  " + "".join(f"{k:>16s}" for k in cv))
                    for b in range(nb):
                        sl = slice(b * block_yr, (b + 1) * block_yr)
                        print(f"   {b:5d}  " + "".join(f"{(a - e)[sl].mean():16.3f}"
                                                     for a in cv.values()))

    if rows:
        df = pd.DataFrame(rows)
        with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
            print("\nmean bias (model - ESM), by lead years")
            print(df.to_string(index=False))
        df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "bias.npz"), **saved)
    print(f"\nwrote plots + summary.csv + region_bias_*.csv + bias.npz to {out_dir}")


if __name__ == "__main__":
    main()
