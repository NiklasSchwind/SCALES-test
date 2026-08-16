
import os
import traceback
import faulthandler
import proto_scales.data_prep.prepare_data as prep
import torch
import proto_scales.ssm_model.scales_ssm_cross_corr_osc_trans as scales_ssm_z2pr
import numpy as np
import pickle
import argparse
import pandas as pd
from datetime import datetime

MODEL = 'ACCESS-ESM1-5'
INDICATORS = ['tas','pr']
TEST_SCENARIOS = ['ssp245']
TRAIN_SCENARIOS = [ 'ssp585','esm-1pct-brch-1000pgc-from-025', 'esm-1pct-brch-750pgc-from-025', 'esm-1pct-brch-2000pgc-from-025','ssp460','ssp245','ssp534-over','abrupt-4xco2','flat10-cdr-from-025', 'flat10-zec-from-025', 'flat10-from-025','ssp370']#'ssp126'#,"ssp370",'ssp534-over','flat10zecincspinoff'',''flat10cdrincspinoff'','abrupt-4xco2','ssp119','ssp460','ssp370']
N = 1800
n_skip = 450
ML_MODEL = 'feed_forward'
PATTERN_SCALING_RESIDUALS = False
RAMP_DOWN_CORRECTED_PS = False
monthly_flag = True
use_smoothing = False
train_pattern_scaling_name = 'ssp585'
epochs = 500
use_linear_model = True

MODEL_PATH_IIASA = f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}'
MODEL_PATH_ASC = f'/gpfs/data/fs73093/kain/CMIP6/cmip6-ng-inc-oceans/{MODEL}'

if __name__ == "__main__":

    faulthandler.enable()
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))

    parser = argparse.ArgumentParser(description="SSM model for climate projections (cross-corr + oscillatory transition)")

    parser.add_argument("--cluster", type=str, help="Cluster name ASC or IIASA")
    parser.add_argument("--add_data", nargs="+", type=str, default=[], help="Additional scenario names to add to training data")
    parser.add_argument("--rm_data", nargs="+", type=str, default=[], help="Scenario names to remove from training data")
    parser.add_argument("--reservoir", type=int, default=2, help="reservoir_dim for the slow reservoir")
    parser.add_argument("--alpha_max", type=float, default=0.02, help="Maximum alpha for slow reservoir (controls minimum time constant)")
    parser.add_argument("--cov_rank", type=int, default=8, help="Rank of the low-rank component of the joint (2*y_dim) emission covariance")
    parser.add_argument("--weights_file", type=str, default=None, help="Path to existing model weights file to initialise training from (must match cross-corr architecture)")
    parser.add_argument("--acf_max_lag", type=int, default=120, help="Max lag (months) for ACF loss; 0 disables it")
    parser.add_argument("--acf_weight", type=float, default=5000.0, help="Weight for ACF loss term")
    parser.add_argument("--rollout_steps", type=int, default=120, help="Length (months) of the differentiable rollout used for the rollout-MSE and ACF losses; caps the longest resolvable ACF lag at (rollout_steps-1)//2")
    parser.add_argument("--rollout_samples", type=int, default=1, help="Samples per differentiable rollout; memory scales linearly")

    args = parser.parse_args()


    if(args.cluster == "ASC"):
        MODEL_PATH = MODEL_PATH_ASC
    else:
        MODEL_PATH = MODEL_PATH_IIASA

    TRAIN_SCENARIOS.extend(args.add_data)
    TRAIN_SCENARIOS = [s for s in TRAIN_SCENARIOS if s not in args.rm_data]

    print("MODEL_PATH", MODEL_PATH)

    potential_files = prep.get_all_files_(MODEL_PATH)

    train_files_tas = prep.filter_climate_files(files = potential_files, scenarios = TRAIN_SCENARIOS, indicators = ['tas'])
    test_files_tas = prep.filter_climate_files(files = potential_files, scenarios = TEST_SCENARIOS, indicators = ['tas'])

    train_files_with_baseline_tas = [(prep.get_baseline_filename(filename=filename, files= potential_files), filename) for filename in train_files_tas]
    test_files_with_baseline_tas = [(prep.get_baseline_filename(filename=filename, files= potential_files), filename) for filename in test_files_tas]

    train_data_df_tas = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in train_files_with_baseline_tas]
    test_data_df_tas = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in test_files_with_baseline_tas]

    train_data_gmt = [train_data_df_tas[i][0] for i in range(len(train_data_df_tas))]
    test_data_gmt = [test_data_df_tas[i][0] for i in range(len(test_data_df_tas))]
    train_data_regional_temps = [train_data_df_tas[i][1].add_suffix("_tas") for i in range(len(train_data_df_tas))]
    test_data_regional_temps = [test_data_df_tas[i][1].add_suffix("_tas") for i in range(len(test_data_df_tas))]

    regional_averages_indicators_train = []
    regional_averages_indicators_test = []

    for indicator in INDICATORS:
        if indicator == 'tas':
            regional_averages_indicators_train.append(train_data_regional_temps)
        else:
            train_files_with_baseline_indicator = [(baseline_file.replace('tas', indicator), experiment_file.replace('tas', indicator)) for (baseline_file, experiment_file) in train_files_with_baseline_tas]
            train_data_df_indicator = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in train_files_with_baseline_indicator]
            train_data_regional_indicator = [train_data_df_indicator[i][1].add_suffix(f"_{indicator}") for i in range(len(train_data_df_indicator))]
            regional_averages_indicators_train.append(train_data_regional_indicator)
    print("created first training andlists ")

    train_data_df = [(train_data_gmt[i],pd.concat([regional_averages_indicator[i] for regional_averages_indicator in regional_averages_indicators_train], axis = 1)) for i in range(len(train_data_gmt))]

    print("created pandas data frame")

    if PATTERN_SCALING_RESIDUALS:
        flat10cdr_index = [i for i, f in enumerate(train_files_tas) if train_pattern_scaling_name in f][0]
        regional_regression_slopes_intersepts = prep.process_gmt_and_regions_into_array(train_data_df[flat10cdr_index], weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS)
        train_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS, slope_intercept = regional_regression_slopes_intersepts,ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS) for GMT_regional_values_tuple in train_data_df]
    else:
        train_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False,ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS) for GMT_regional_values_tuple in train_data_df]

    print("created train_data_np")
    train_data_input, train_data_output = prep.prepare_all_train_data(train_data_np, n=N, n_skip=n_skip)
    print("created train_input_data")

    train_data_shuffeled = prep.shuffle_train_data(train_data_input, train_data_output, random_state=42)

    u = train_data_shuffeled[0][0,:].T[..., None]
    y = np.transpose(train_data_shuffeled[0],(2, 1, 0))
    y = y[:,:,1:]
    regions = int(y.shape[-1]/2)
    print(regions)
    tas = y[:,:,:regions]
    pr = y[:,:,-regions:]
    print("tas ",tas.shape)
    print("GMT ",u.shape)
    print("pr ",pr.shape)

    device = "cuda"

    run_dir = os.path.join("outputs_ssm_scales", "scales_cross_corr_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_scenarios.txt"), "w") as f:
        f.write("\n".join(TRAIN_SCENARIOS))
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"reservoir_dim={args.reservoir}\n")
        f.write(f"alpha_max={args.alpha_max}\n")
        f.write(f"cov_rank={args.cov_rank}\n")
        f.write(f"acf_max_lag={args.acf_max_lag}\n")
        f.write(f"acf_weight={args.acf_weight}\n")
        f.write(f"rollout_steps={args.rollout_steps}\n")
        f.write(f"rollout_samples={args.rollout_samples}\n")

    try:
        model, y_scaler, u_scaler,pr_scaler = scales_ssm_z2pr.run_train(tas, pr,u, context_len=600, z_dim=64,rnn_hidden=256,horizon=1200,
            use_linear_model=use_linear_model,epochs=epochs,batch_size=250,alpha_max=args.alpha_max, resevoir_dim=args.reservoir,
            cov_rank=args.cov_rank, run_dir=run_dir, weights_file=args.weights_file,
            acf_max_lag=args.acf_max_lag, acf_weight=args.acf_weight,
            rollout_steps=args.rollout_steps, rollout_samples=args.rollout_samples)
    except Exception:
        print(f"[Rank {_local_rank}] run_train failed:", flush=True)
        traceback.print_exc()
        raise
    torch.save(model.state_dict(), os.path.join(run_dir, "model_out"))
    y_scaler.save(os.path.join(run_dir, "y_scaler.out"))
    u_scaler.save(os.path.join(run_dir, "u_scaler.out"))
    pr_scaler.save(os.path.join(run_dir, "pr_scaler.out"))
