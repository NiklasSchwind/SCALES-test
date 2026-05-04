
import os
import traceback
import faulthandler
import proto_scales.data_prep.prepare_data as prep
import torch
import proto_scales.ssm_model.scales_ssm_z2tasAndpr_laggingGMT_slowHiddenRes as scales_ssm_z2pr
import numpy as np
import pickle
import argparse
import pandas as pd
from datetime import datetime

MODEL = 'ACCESS-ESM1-5'
INDICATORS = ['tas','pr']
TEST_SCENARIOS = ['ssp245']
TRAIN_SCENARIOS = [ 'ssp585','1pctco2','ssp460','ssp245','ssp534-over','abrupt-4xco2','flat10zecincspinoff','flat10cdrincspinoff']#'ssp126'#,"ssp370",'ssp534-over','flat10zecincspinoff'',''flat10cdrincspinoff'','abrupt-4xco2','ssp119','ssp460','ssp370']
N = 600
n_skip = 150
ML_MODEL = 'feed_forward' 
PATTERN_SCALING_RESIDUALS = False
RAMP_DOWN_CORRECTED_PS = False
monthly_flag = True
use_smoothing = False
train_pattern_scaling_name = 'ssp585'
epochs = 1000
use_linear_model = True

MODEL_PATH_IIASA = f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}'
MODEL_PATH_ASC = f'/gpfs/data/fs73093/kain/CMIP6/cmip6-ng-inc-oceans/{MODEL}'

if __name__ == "__main__":

    faulthandler.enable()
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))

    parser = argparse.ArgumentParser(description="SSM model for climate projections")
    
    # Define command-line arguments
    # parser.add_argument("operation", choices=["add", "sub", "mul", "div"],
    #                     help="The operation to perform.")
    # parser.add_argument("x", type=float, help="The first number.")
    parser.add_argument("--cluster", type=str, help="Cluster name ASC or IIASA")
    parser.add_argument("--add_data", nargs="+", type=str, default=[], help="Additional scenario names to add to training data")
    parser.add_argument("--rm_data", nargs="+", type=str, default=[], help="Scenario names to remove from training data")
    parser.add_argument("--reservoir", type=int, default=2, help="reservoir_dim for the slow reservoir")

    args = parser.parse_args()

   
    if(args.cluster == "ASC"):
        MODEL_PATH = MODEL_PATH_ASC
    else:
        MODEL_PATH = MODEL_PATH_IIASA

    TRAIN_SCENARIOS.extend(args.add_data)
    TRAIN_SCENARIOS = [s for s in TRAIN_SCENARIOS if s not in args.rm_data]
    
    print("MODEL_PATH", MODEL_PATH)

    # get all files available for the model
    potential_files = prep.get_all_files_(MODEL_PATH)
    #print(potential_files)

    # get all files for tas indicator
    train_files_tas = prep.filter_climate_files(files = potential_files, scenarios = TRAIN_SCENARIOS, indicators = ['tas'])
    test_files_tas = prep.filter_climate_files(files = potential_files, scenarios = TEST_SCENARIOS, indicators = ['tas'])

    # including baseline
    train_files_with_baseline_tas = [(prep.get_baseline_filename(filename=filename, files= potential_files), filename) for filename in train_files_tas]
    test_files_with_baseline_tas = [(prep.get_baseline_filename(filename=filename, files= potential_files), filename) for filename in test_files_tas]

    # process regional averages etc for tas files
    train_data_df_tas = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in train_files_with_baseline_tas]
    test_data_df_tas = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in test_files_with_baseline_tas]

    # extract gmt and regional averages from the processed tas data
    train_data_gmt = [train_data_df_tas[i][0] for i in range(len(train_data_df_tas))]
    test_data_gmt = [test_data_df_tas[i][0] for i in range(len(test_data_df_tas))]
    train_data_regional_temps = [train_data_df_tas[i][1].add_suffix("_tas") for i in range(len(train_data_df_tas))]
    test_data_regional_temps = [test_data_df_tas[i][1].add_suffix("_tas") for i in range(len(test_data_df_tas))]

    # save the regional averages for each indicator in this list 
    regional_averages_indicators_train = []
    regional_averages_indicators_test = []

    # process remaining indicators regional averages in order of the INDICATORS list
    for indicator in INDICATORS:
        if indicator == 'tas': 
    
            regional_averages_indicators_train.append(train_data_regional_temps)
            #regional_averages_indicators_test.append(test_data_regional_temps)
        else: 
            train_files_with_baseline_indicator = [(baseline_file.replace('tas', indicator), experiment_file.replace('tas', indicator)) for (baseline_file, experiment_file) in train_files_with_baseline_tas]
            #test_files_with_baseline_indicator = [(baseline_file.replace('tas', indicator), experiment_file.replace('tas', indicator)) for (baseline_file, experiment_file) in test_files_with_baseline_tas]
            train_data_df_indicator = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in train_files_with_baseline_indicator]
            #test_data_df_indicator = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in test_files_with_baseline_indicator]
            train_data_regional_indicator = [train_data_df_indicator[i][1].add_suffix(f"_{indicator}") for i in range(len(train_data_df_indicator))]
            #test_data_regional_indicator = [test_data_df_indicator[i][1].add_suffix(f"_{indicator}") for i in range(len(test_data_df_indicator))]
            regional_averages_indicators_train.append(train_data_regional_indicator)
            #regional_averages_indicators_test.append(test_data_regional_indicator)
    print("created first training andlists ")

    # this yields a list of processed simulations as tuples of their GMT values in a dataframe
    # and their regional indicator values in a dataframe with columns ordered like this: 
    # region_1_INDICATOR[0], region_N_INDICATOR[0], ..., region_N_INDICATOR[0], region_1_INDICATOR[1], ..., region_N_INDICATOR[M-1]
    # when we have N regions and M indicators
    # whereas regions are alphabetically ordered and the order of indicators follows the INDICATORS list given in the config above
    train_data_df = [(train_data_gmt[i],pd.concat([regional_averages_indicator[i] for regional_averages_indicator in regional_averages_indicators_train], axis = 1)) for i in range(len(train_data_gmt))]
    #test_data_df = [(test_data_gmt[i],pd.concat([regional_averages_indicator[i] for regional_averages_indicator in regional_averages_indicators_test], axis = 1)) for i in range(len(test_data_gmt))]

    print("created pandas data frame")
    
    if PATTERN_SCALING_RESIDUALS:
        flat10cdr_index = [i for i, f in enumerate(train_files_tas) if train_pattern_scaling_name in f][0]
        regional_regression_slopes_intersepts = prep.process_gmt_and_regions_into_array(train_data_df[flat10cdr_index], weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS)
        train_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS, slope_intercept = regional_regression_slopes_intersepts,ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS) for GMT_regional_values_tuple in train_data_df]
        #test_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS, slope_intercept = regional_regression_slopes_intersepts, ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS) for GMT_regional_values_tuple in test_data_df]
    else:
        #flat10cdr_index = [i for i, f in enumerate(train_files_tas) if train_pattern_scaling_name in f][0]
        #regional_regression_slopes_intersepts = prep.process_gmt_and_regions_into_array(train_data_df[flat10cdr_index], weighted_linear_smoothing = False, pattern_scaling_residuals=True,ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS)
        train_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False,ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS) for GMT_regional_values_tuple in train_data_df]
        #test_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS) for GMT_regional_values_tuple in test_data_df]
    
    print("created train_data_np")
    train_data_input, train_data_output = prep.prepare_all_train_data(train_data_np, n=N, n_skip=n_skip)
    #test_data_input, test_data_output = prep.prepare_all_train_data(test_data_np, n=N)
    print("created train_input_data")
    
    
    #test_data_for_autoregression_input, test_data_for_autoregression_output = prep.prepare_train_data(test_data_np[0],N)
    
    #gmt_autoregressive_test = test_data_np[0][0,:]
    #autoregressive_test_groudtruth = test_data_np[0][1:,:]
    
    #gmt_train = train_data_np[0][0,:]
    #train_data_regional_temperature = train_data_np[0][1:,:]
    # else:
      
    #     pickle_filename = "/hdrive/all_users/kainverena/formatted_data/input_data.pkl"
    #     with open(pickle_filename, 'rb') as f:
    #         train_data_input = pickle.load(f) 
    #         train_data_output = pickle.load(f)
    #         test_data_for_autoregression_input = pickle.load(f)
    #     f.close()


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

    run_dir = os.path.join("outputs_ssm_scales", "scales_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_scenarios.txt"), "w") as f:
        f.write("\n".join(TRAIN_SCENARIOS))
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"reservoir_dim={args.reservoir}\n")

    try:
        model, y_scaler, u_scaler,pr_scaler = scales_ssm_z2pr.run_train(tas, pr,u, context_len=100, z_dim=64,rnn_hidden=256,horizon=500,
            use_linear_model=use_linear_model,epochs=epochs,batch_size=250,alpha_max=0.02, resevoir_dim=args.reservoir,
            run_dir=run_dir)
    except Exception:
        print(f"[Rank {_local_rank}] run_train failed:", flush=True)
        traceback.print_exc()
        raise
    torch.save(model.state_dict(), os.path.join(run_dir, "model_out"))
    y_scaler.save(os.path.join(run_dir, "y_scaler.out"))
    u_scaler.save(os.path.join(run_dir, "u_scaler.out"))
    pr_scaler.save(os.path.join(run_dir, "pr_scaler.out"))
    
