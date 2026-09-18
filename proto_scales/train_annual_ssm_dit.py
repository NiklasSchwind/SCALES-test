"""
Train the annual-SSM-conditioned outpainting DiT (`annual_ssm_dit`).

Data loading is identical to `train_ssm_dit.py` / `train_scales_ssm_cross_corr.py`,
so all these models are trained on the same arrays.

Typical use — the latent process is trained jointly with the DiT, so there is no
separate SSM pretraining step to run first:

    torchrun --nproc_per_node=1 -m proto_scales.train_annual_ssm_dit --cluster ASC

The defaults train on 100-year futures with two DiT blocks scored per step. The
super-window is `context_len + horizon` months and trajectories are only
`N` months long (1800 = 150 yr), which is the hard ceiling on `--horizon`.

--scaler_run_dir only reuses scalers from an earlier run; the latent process here
is a `ForcedAnnualSSM` and is not checkpoint-compatible with either the monthly
SSM or the oscillatory `AnnualSSM`.
"""

import argparse
import faulthandler
import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import proto_scales.data_prep.prepare_data as prep
import proto_scales.ssm_dit_model.annual_dit_training as dit_training

MODEL = 'ACCESS-ESM1-5'
INDICATORS = ['tas', 'pr']
TEST_SCENARIOS = ['ssp245']
TRAIN_SCENARIOS = ['ssp585', 'esm-1pct-brch-1000pgc-from-025', 'esm-1pct-brch-750pgc-from-025',
                   'esm-1pct-brch-2000pgc-from-025', 'ssp460', 'ssp245', 'ssp534-over',
                   'abrupt-4xco2', 'flat10-cdr-from-025', 'flat10-zec-from-025',
                   'flat10-from-025', 'ssp370']
N = 1800
n_skip = 450
monthly_flag = True
use_smoothing = False
RAMP_DOWN_CORRECTED_PS = False

MODEL_PATH_IIASA = f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}'
MODEL_PATH_ASC = f'/gpfs/data/fs73093/kain/CMIP6/cmip6-ng-inc-oceans/{MODEL}'


def load_data(model_path, train_scenarios):
    """Identical to the other training scripts; returns (tas, pr, u) as [N, T, D]."""
    potential_files = prep.get_all_files_(model_path)

    train_files_tas = prep.filter_climate_files(
        files=potential_files, scenarios=train_scenarios, indicators=['tas'])
    train_files_with_baseline_tas = [
        (prep.get_baseline_filename(filename=f, files=potential_files), f)
        for f in train_files_tas]

    train_data_df_tas = [
        prep.process_scenarios(
            experiment_scenario_path=f'{model_path}/{experiment}',
            simulation_name=experiment,
            baseline_scenario_path=f'{model_path}/{baseline}',
            delete_first_years=0, monthly_trend=monthly_flag, smoothed=use_smoothing)
        for (baseline, experiment) in train_files_with_baseline_tas]

    train_data_gmt = [d[0] for d in train_data_df_tas]
    train_data_regional_temps = [d[1].add_suffix("_tas") for d in train_data_df_tas]

    regional_averages_indicators_train = []
    for indicator in INDICATORS:
        if indicator == 'tas':
            regional_averages_indicators_train.append(train_data_regional_temps)
        else:
            files_ind = [(b.replace('tas', indicator), e.replace('tas', indicator))
                         for (b, e) in train_files_with_baseline_tas]
            df_ind = [
                prep.process_scenarios(
                    experiment_scenario_path=f'{model_path}/{experiment}',
                    simulation_name=experiment,
                    baseline_scenario_path=f'{model_path}/{baseline}',
                    delete_first_years=0, monthly_trend=monthly_flag,
                    smoothed=use_smoothing)
                for (baseline, experiment) in files_ind]
            regional_averages_indicators_train.append(
                [d[1].add_suffix(f"_{indicator}") for d in df_ind])

    train_data_df = [
        (train_data_gmt[i],
         pd.concat([r[i] for r in regional_averages_indicators_train], axis=1))
        for i in range(len(train_data_gmt))]

    train_data_np = [
        prep.process_gmt_and_regions_into_array(
            t, weighted_linear_smoothing=False,
            ramp_down_corrected_ps=RAMP_DOWN_CORRECTED_PS)
        for t in train_data_df]

    train_data_input, train_data_output = prep.prepare_all_train_data(
        train_data_np, n=N, n_skip=n_skip)
    shuffled = prep.shuffle_train_data(train_data_input, train_data_output, random_state=42)

    u = shuffled[0][0, :].T[..., None]
    y = np.transpose(shuffled[0], (2, 1, 0))[:, :, 1:]
    regions = int(y.shape[-1] / 2)
    return y[:, :, :regions], y[:, :, -regions:], u


if __name__ == "__main__":
    faulthandler.enable()
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))

    p = argparse.ArgumentParser(
        description="Annual-SSM-conditioned outpainting DiT for climate projections")
    p.add_argument("--cluster", type=str, help="Cluster name ASC or IIASA")
    p.add_argument("--add_data", nargs="+", type=str, default=[])
    p.add_argument("--rm_data", nargs="+", type=str, default=[])

    # latent process
    p.add_argument("--ssm_weights", type=str, default=None,
                   help="Warm-start weights for the ForcedAnnualSSM only")
    p.add_argument("--scaler_run_dir", type=str, default=None,
                   help="Reuse y/pr/u scalers from an earlier run directory")
    p.add_argument("--freeze_ssm", action="store_true",
                   help="Train the DiT on a fixed latent process. Only sensible with --ssm_weights")
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--rnn_hidden", type=int, default=64)
    p.add_argument("--emit_hidden", type=int, default=0)
    p.add_argument("--trans_hidden", type=int, default=0,
                   help="MLP correction on the transition mean. Non-zero buys nonlinearity in u (hysteresis) but makes the mean propagation the DiT conditions on approximate rather than exact")
    p.add_argument("--decay_efold", nargs=2, type=float, default=[1.0, 50.0],
                   metavar=("LOW", "HIGH"),
                   help="Range of latent e-folding times in YEARS spanned at init")
    p.add_argument("--ssm_elbo_weight", type=float, default=1.0,
                   help="One-step-ahead ELBO. Normalised per element, so on the same scale as the diffusion loss")
    p.add_argument("--ssm_kl_weight", type=float, default=5.0)
    p.add_argument("--ssm_kl_free_bits", type=float, default=0.2)
    p.add_argument("--ssm_rollout_weight", type=float, default=1.0,
                   help="Multi-step rollout consistency. This is the only term that scores the many-step operator the sampler iterates; at 0, century rollouts are an unsupervised extrapolation from a next-year fit")
    p.add_argument("--stochastic_z", action="store_true",
                   help="Condition on a sampled rollout instead of the mean. Off by default: accumulated process noise swamps the forced signal at long lead and the DiT learns to ignore z")

    # DiT
    p.add_argument("--context_len", type=int, default=120,
                   help="Context window in months; must be a whole number of years")
    p.add_argument("--horizon", type=int, default=1200,
                   help="Future length per training item in months. The z rollout spans all of it; DiT blocks are sampled from inside it")
    p.add_argument("--block_len", type=int, default=120,
                   help="Months generated per DiT block; must be a whole number of years")
    p.add_argument("--blocks_per_step", type=int, default=2,
                   help="DiT blocks scored per item. Blocks share one z rollout, so each extra offset costs a DiT pass and no extra SSM work — but the effective batch is batch_size * blocks_per_step")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--n_diffusion_steps", type=int, default=1000)
    p.add_argument("--parameterization", type=str, default="v", choices=["v", "eps"],
                   help="Diffusion target. 'v' is the default: eps-prediction is unstable at high noise for these fields")
    p.add_argument("--field_memory_rank", type=int, default=32)
    p.add_argument("--timescales", nargs="+", type=float, default=[1.0, 5.0, 20.0],
                   help="Memory-kernel relaxation times in YEARS. The kernels are window-local, so anything much beyond the window length is dead weight; century-scale memory is z's job")
    p.add_argument("--fixed_timescales", action="store_true",
                   help="Keep the memory timescales fixed instead of learning them")

    # optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--stride", type=int, default=7,
                   help="Window stride in months. Must be coprime with 12, otherwise the month embedding never sees every calendar phase")
    p.add_argument("--data_start_month", type=int, default=0,
                   help="Calendar month (0-11) of time index 0; use the same value at inference")
    p.add_argument("--weights_file", type=str, default=None, help="Warm-start full-model weights")
    p.add_argument("--ckpt_every", type=int, default=10)

    args = p.parse_args()

    MODEL_PATH = MODEL_PATH_ASC if args.cluster == "ASC" else MODEL_PATH_IIASA
    TRAIN_SCENARIOS.extend(args.add_data)
    TRAIN_SCENARIOS = [s for s in TRAIN_SCENARIOS if s not in args.rm_data]
    print("MODEL_PATH", MODEL_PATH)

    tas, pr, u = load_data(MODEL_PATH, TRAIN_SCENARIOS)
    print("tas", tas.shape, "pr", pr.shape, "GMT", u.shape)

    run_dir = os.path.join("outputs_ssm_scales",
                           "annual_ssm_dit_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_scenarios.txt"), "w") as f:
        f.write("\n".join(TRAIN_SCENARIOS))
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")

    try:
        model, tas_scaler, pr_scaler, u_scaler = dit_training.run_train(
            tas, pr, u,
            context_len=args.context_len, horizon=args.horizon,
            block_len=args.block_len, blocks_per_step=args.blocks_per_step,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
            ssm_weights=args.ssm_weights, scaler_run_dir=args.scaler_run_dir,
            freeze_ssm=args.freeze_ssm,
            z_dim=args.z_dim, rnn_hidden=args.rnn_hidden,
            emit_hidden=args.emit_hidden, trans_hidden=args.trans_hidden,
            decay_efold_range=tuple(args.decay_efold),
            ssm_elbo_weight=args.ssm_elbo_weight,
            ssm_kl_weight=args.ssm_kl_weight,
            ssm_kl_free_bits=args.ssm_kl_free_bits,
            ssm_rollout_weight=args.ssm_rollout_weight,
            stochastic_z=args.stochastic_z,
            cond_dim=args.cond_dim, hidden=args.hidden, depth=args.depth,
            heads=args.heads, n_diffusion_steps=args.n_diffusion_steps,
            timescales_years=tuple(args.timescales),
            learnable_timescales=not args.fixed_timescales,
            field_memory_rank=args.field_memory_rank,
            parameterization=args.parameterization,
            ema_decay=args.ema_decay, stride=args.stride,
            data_start_month=args.data_start_month,
            run_dir=run_dir, weights_file=args.weights_file,
            ckpt_every=args.ckpt_every,
        )
    except Exception:
        print(f"[Rank {_local_rank}] run_train failed:", flush=True)
        traceback.print_exc()
        raise

    torch.save(model.state_dict(), os.path.join(run_dir, "model_out"))
    tas_scaler.save(os.path.join(run_dir, "y_scaler.out"))
    pr_scaler.save(os.path.join(run_dir, "pr_scaler.out"))
    u_scaler.save(os.path.join(run_dir, "u_scaler.out"))
    print("saved to", run_dir)
