"""
Train the ELBO-latent outpainting DiT (`annual_elbo_dit`).

Data loading is identical to `train_annual_jepa_kl_dit.py`, so both models
train on the same arrays.

    torchrun --nproc_per_node=1 -m proto_scales.train_annual_elbo_dit --cluster ASC \
        --gated_decay --decay_efold 0.5 500 --enc_window_years 10

The latent process is a tas-only ELBO state-space model with a frozen
pattern-scaling emission (fitted on the training split) plus a residual from
z; beta defaults to 5 as in the original monthly SSM. The DiT's diffusion loss
does not reach z unless --no_detach_latents.
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
import proto_scales.ssm_dit_model.annual_elbo_dit_training as dit_training

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
        description="ELBO-latent outpainting DiT for climate projections")
    p.add_argument("--cluster", type=str, help="Cluster name ASC or IIASA")
    p.add_argument("--add_data", nargs="+", type=str, default=[])
    p.add_argument("--rm_data", nargs="+", type=str, default=[])

    # latent process
    p.add_argument("--ssm_weights", type=str, default=None,
                   help="Warm-start weights for the ForcedAnnualELBO only: a bare ELBO state dict, a model_out or an epoch checkpoint (EMA weights). Shape-mismatched tensors (e.g. a new --emission_hidden) stay at init")
    p.add_argument("--scaler_run_dir", type=str, default=None,
                   help="Reuse y/pr/u scalers from an earlier run directory")
    p.add_argument("--freeze_ssm", action="store_true",
                   help="Train the DiT on a fixed latent process. Only sensible with --ssm_weights")
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--rnn_hidden", type=int, default=64)
    p.add_argument("--trans_hidden", type=int, default=0,
                   help="MLP correction on the transition mean (zero-initialised)")
    p.add_argument("--decay_efold", nargs=2, type=float, default=[1.0, 50.0],
                   metavar=("LOW", "HIGH"),
                   help="Range of latent e-folding times in YEARS")
    p.add_argument("--gated_decay", action="store_true",
                   help="State-dependent timescales tau(z_t-1, u_t), smoothly bounded to --decay_efold")
    p.add_argument("--gate_hidden", type=int, default=32)
    p.add_argument("--annual_encoder", action="store_false", dest="monthly_encoder",
                   help="Encoder GRU over annual means instead of monthly input (monthly is the default here)")
    p.add_argument("--enc_window_years", type=int, default=0,
                   help="0: encoder GRU reads the whole window from h0=0. > 0: each year is encoded from a fixed lookback of this many years only. Must not exceed --context_len in years; 10 with the default context")
    p.add_argument("--encoder_reads_pr", action="store_true",
                   help="Encoder also reads pr (default: tas + u, as in the original SSM)")
    p.add_argument("--emission_hidden", type=int, default=0,
                   help="0: linear residual emission from z; > 0: one hidden layer of this width")
    p.add_argument("--pr_emission", action="store_true",
                   help="Add an annual pr head: frozen pattern scaling(u) + residual(z). Trained on detached z unless --pr_trains_z; required by --dit_residual")
    p.add_argument("--pr_emission_hidden", type=int, default=64,
                   help="Hidden width of the pr head's residual MLP (0: linear)")
    p.add_argument("--pr_weight", type=float, default=1.0,
                   help="Weight of the pr head loss (Gaussian NLL on the posterior z + rollout MSE)")
    p.add_argument("--pr_trains_z", action="store_false", dest="pr_detach_z",
                   help="Let the pr head's loss train the latent process too")
    p.add_argument("--dit_residual", action="store_true",
                   help="DiT generates field - SSM baseline (tas and pr means from z, u) and the baseline is added back at sampling. Needs --pr_emission")
    p.add_argument("--pattern_ridge", type=float, default=1e-3,
                   help="Ridge penalty for the frozen pattern-scaling fit (relative to the number of training years)")
    p.add_argument("--nll_weight", type=float, default=1.0,
                   help="Weight of the ELBO (NLL + beta*KL); 0 disables both")
    p.add_argument("--kl_weight", type=float, default=5.0,
                   help="beta. KL and NLL are both summed over their dims and divided by the same (years x regions), so this matches the original monthly SSM's kl_w")
    p.add_argument("--kl_warmup_epochs", type=int, default=5,
                   help="Linear beta warm-up from 0; validation always uses the full beta")
    p.add_argument("--kl_free_bits", type=float, default=0.0,
                   help="Floor on each year's total KL (nats); 0 lets unused dims collapse onto the prior")
    p.add_argument("--rollout_weight", type=float, default=1.0,
                   help="MSE of the emission mean along the forcing-only mean rollout over the horizon")
    p.add_argument("--no_detach_latents", action="store_false", dest="detach_latents",
                   help="Let the DiT's diffusion loss train the latent process too")

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
    p.add_argument("--n_flow_steps", type=int, default=50,
                   help="Default number of Euler steps for the flow-matching ODE at sampling time; overridable per call via sample_blocks(n_steps=...). Linear flow paths need far fewer steps than a DDPM chain")
    p.add_argument("--field_memory_rank", type=int, default=32)
    p.add_argument("--timescales", nargs="+", type=float, default=[1.0, 5.0, 20.0],
                   help="Memory-kernel relaxation times in YEARS. The kernels are window-local, so anything much beyond the window length is dead weight; century-scale memory is z's job")
    p.add_argument("--fixed_timescales", action="store_true",
                   help="Keep the memory timescales fixed instead of learning them")

    # optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--stride", type=int, default=7,
                   help="Window stride in months. Must be coprime with 12, otherwise the month embedding never sees every calendar phase")
    p.add_argument("--data_start_month", type=int, default=0,
                   help="Calendar month (0-11) of time index 0; use the same value at inference")
    p.add_argument("--weights_file", type=str, default=None, help="Warm-start full-model weights")
    p.add_argument("--weights_key", type=str, default="model", choices=["model", "ema"],
                   help="Which weights to take from an epoch checkpoint {model, ema}; ignored for a bare state dict such as model_out")
    p.add_argument("--ckpt_every", type=int, default=10)

    args = p.parse_args()
    if args.dit_residual and not args.pr_emission:
        p.error("--dit_residual needs --pr_emission (the pr baseline comes from the pr head)")

    MODEL_PATH = MODEL_PATH_ASC if args.cluster == "ASC" else MODEL_PATH_IIASA
    TRAIN_SCENARIOS.extend(args.add_data)
    TRAIN_SCENARIOS = [s for s in TRAIN_SCENARIOS if s not in args.rm_data]
    print("MODEL_PATH", MODEL_PATH)

    tas, pr, u = load_data(MODEL_PATH, TRAIN_SCENARIOS)
    print("tas", tas.shape, "pr", pr.shape, "GMT", u.shape)

    run_dir = os.path.join("outputs_ssm_scales",
                           "annual_elbo_dit_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
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
            trans_hidden=args.trans_hidden,
            decay_efold_range=tuple(args.decay_efold),
            gated_decay=args.gated_decay, gate_hidden=args.gate_hidden,
            monthly_encoder=args.monthly_encoder,
            enc_window_years=args.enc_window_years,
            encoder_reads_pr=args.encoder_reads_pr,
            emission_hidden=args.emission_hidden,
            pr_emission=args.pr_emission,
            pr_emission_hidden=args.pr_emission_hidden,
            pr_weight=args.pr_weight, pr_detach_z=args.pr_detach_z,
            dit_residual=args.dit_residual,
            nll_weight=args.nll_weight, kl_weight=args.kl_weight,
            kl_warmup_epochs=args.kl_warmup_epochs,
            kl_free_bits=args.kl_free_bits, rollout_weight=args.rollout_weight,
            detach_latents=args.detach_latents, pattern_ridge=args.pattern_ridge,
            cond_dim=args.cond_dim, hidden=args.hidden, depth=args.depth,
            heads=args.heads, n_flow_steps=args.n_flow_steps,
            timescales_years=tuple(args.timescales),
            learnable_timescales=not args.fixed_timescales,
            field_memory_rank=args.field_memory_rank,
            ema_decay=args.ema_decay, stride=args.stride,
            data_start_month=args.data_start_month,
            run_dir=run_dir, weights_file=args.weights_file,
            weights_key=args.weights_key,
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
