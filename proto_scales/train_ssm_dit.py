"""
Train the SSM-conditioned outpainting DiT.

Data loading mirrors train_scales_ssm_cross_corr.py exactly, so the two models
are trained on identical arrays.

Typical use — train the SSM first, then the DiT on top of it:

    torchrun --nproc_per_node=1 -m proto_scales.train_scales_ssm_cross_corr --cluster ASC
    torchrun --nproc_per_node=1 -m proto_scales.train_ssm_dit --cluster ASC \
        --ssm_run_dir outputs_ssm_scales/scales_cross_corr_YYYYMMDD_HHMMSS

--ssm_run_dir supplies both the SSM weights (model_out) and its scalers. Passing
weights without the scalers puts the frozen SSM in a normalisation it was never
trained in, which degrades z silently, so prefer --ssm_run_dir over
--ssm_weights.
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
import proto_scales.ssm_dit_model.training as dit_training

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
    """Identical to the SSM training script; returns (tas, pr, u) as [N, T, D]."""
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
        description="SSM-conditioned outpainting DiT for climate projections")
    p.add_argument("--cluster", type=str, help="Cluster name ASC or IIASA")
    p.add_argument("--add_data", nargs="+", type=str, default=[])
    p.add_argument("--rm_data", nargs="+", type=str, default=[])

    # SSM
    p.add_argument("--ssm_run_dir", type=str, default=None,
                   help="SSM run directory; supplies both weights (model_out) and scalers. Preferred over --ssm_weights")
    p.add_argument("--ssm_weights", type=str, default=None,
                   help="SSM checkpoint path. Only use without --ssm_run_dir if the SSM is untrained")
    p.add_argument("--train_ssm", action="store_true",
                   help="Unfreeze the SSM and train it jointly with the DiT")
    p.add_argument("--end_to_end", action="store_true",
                   help="Shorthand for --train_ssm with the SSM's own ELBO and ACF losses enabled. Recommended over bare --train_ssm: without them z is shaped by the diffusion loss alone and the oscillator gets no ACF pressure")
    p.add_argument("--ssm_elbo_weight", type=float, default=1.0,
                   help="Weight on the SSM ELBO when training end-to-end (0 disables). The ELBO is normalised per-element, so this is on the same scale as the diffusion loss and does not need rescaling when indicators are added")
    p.add_argument("--ssm_kl_weight", type=float, default=5.0)
    p.add_argument("--ssm_kl_free_bits", type=float, default=0.2)
    p.add_argument("--ssm_acf_weight", type=float, default=10.0,
                   help="Weight on the SSM ACF loss when training end-to-end (0 disables). NOTE: not comparable to the 5000 used in train_scales_ssm_cross_corr.py, which competes with a raw summed NLL of order 1e5; here it competes with an O(1) diffusion loss")
    p.add_argument("--ssm_acf_max_lag", type=int, default=120)
    p.add_argument("--ssm_rollout_steps", type=int, default=0,
                   help="Differentiable rollout length for the ACF term; 0 uses the full horizon")
    p.add_argument("--ssm_rollout_samples", type=int, default=1)
    p.add_argument("--annual_ssm", action="store_true",
                   help="Use the simplified ELBO-only AnnualSSM instead of the monthly SSM")
    p.add_argument("--annual_ssm_run_dir", type=str, default=None,
                   help="AnnualSSM run directory; implies --annual_ssm and supplies weights + scalers")
    p.add_argument("--annual_z_dim", type=int, default=16)
    p.add_argument("--annual_rnn_hidden", type=int, default=64)
    p.add_argument("--annual_emit_hidden", type=int, default=0)
    p.add_argument("--annual_trans_hidden", type=int, default=0)
    p.add_argument("--z_dim", type=int, default=64)
    p.add_argument("--rnn_hidden", type=int, default=256)
    p.add_argument("--cov_rank", type=int, default=8)

    # DiT
    p.add_argument("--context_len", type=int, default=120, help="Context window in months")
    p.add_argument("--horizon", type=int, default=120, help="Generated block length in months")
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--cond_dim", type=int, default=256)
    p.add_argument("--n_diffusion_steps", type=int, default=1000)
    p.add_argument("--parameterization", type=str, default="v", choices=["v", "eps"],
                   help="Diffusion target. 'v' is the default: eps-prediction is unstable at high noise for these fields")
    p.add_argument("--field_memory_rank", type=int, default=32)
    p.add_argument("--timescales", nargs="+", type=float, default=[1.0, 5.0, 20.0, 100.0],
                   help="Memory-kernel relaxation times in YEARS")
    p.add_argument("--fixed_timescales", action="store_true",
                   help="Keep the memory timescales fixed instead of learning them")

    # optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--data_start_month", type=int, default=0,
                   help="Calendar month (0-11) of time index 0; use the same value at inference")
    p.add_argument("--weights_file", type=str, default=None, help="Warm-start DiT weights")
    p.add_argument("--ckpt_every", type=int, default=10)

    args = p.parse_args()

    MODEL_PATH = MODEL_PATH_ASC if args.cluster == "ASC" else MODEL_PATH_IIASA
    TRAIN_SCENARIOS.extend(args.add_data)
    TRAIN_SCENARIOS = [s for s in TRAIN_SCENARIOS if s not in args.rm_data]
    print("MODEL_PATH", MODEL_PATH)

    use_annual = args.annual_ssm or args.annual_ssm_run_dir is not None
    if args.annual_ssm_run_dir is not None:
        args.ssm_run_dir = args.annual_ssm_run_dir

    ssm_weights = args.ssm_weights
    if args.ssm_run_dir is not None:
        candidate = os.path.join(args.ssm_run_dir, "model_out")
        if os.path.exists(candidate):
            ssm_weights = candidate
        elif ssm_weights is None:
            raise FileNotFoundError(
                f"No model_out in {args.ssm_run_dir}; pass --ssm_weights explicitly")

    tas, pr, u = load_data(MODEL_PATH, TRAIN_SCENARIOS)
    print("tas", tas.shape, "pr", pr.shape, "GMT", u.shape)

    run_dir = os.path.join("outputs_ssm_scales",
                           "ssm_dit_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_scenarios.txt"), "w") as f:
        f.write("\n".join(TRAIN_SCENARIOS))
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")
        f.write(f"resolved_ssm_weights={ssm_weights}\n")

    try:
        model, tas_scaler, pr_scaler, u_scaler = dit_training.run_train(
            tas, pr, u,
            context_len=args.context_len, horizon=args.horizon,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
            ssm_weights=ssm_weights, ssm_run_dir=args.ssm_run_dir,
            freeze_ssm=not (args.train_ssm or args.end_to_end),
            use_annual_ssm=use_annual,
            annual_z_dim=args.annual_z_dim,
            annual_rnn_hidden=args.annual_rnn_hidden,
            annual_emit_hidden=args.annual_emit_hidden,
            annual_trans_hidden=args.annual_trans_hidden,
            ssm_elbo_weight=args.ssm_elbo_weight if args.end_to_end else 0.0,
            ssm_kl_weight=args.ssm_kl_weight,
            ssm_kl_free_bits=args.ssm_kl_free_bits,
            ssm_acf_weight=(0.0 if use_annual
                            else (args.ssm_acf_weight if args.end_to_end else 0.0)),
            ssm_acf_max_lag=args.ssm_acf_max_lag,
            ssm_rollout_steps=args.ssm_rollout_steps,
            ssm_rollout_samples=args.ssm_rollout_samples,
            z_dim=args.z_dim, rnn_hidden=args.rnn_hidden, cov_rank=args.cov_rank,
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
