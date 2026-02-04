
import os
import proto_scales.data_prep.prepare_data as prep
import torch
import proto_scales.ssm_model.scales_ssm as scales_ssm

MODEL = 'ACCESS-ESM1-5'
INDICATOR = 'tas'
TEST_SCENARIOS = ['flat10cdrincspinoff']
TRAIN_SCENARIOS = [ 'ssp534-over','flat10cdrincspinoff','ssp585','1pctco2','ssp245']#,'ssp126','flat10zecincspinoff', 'flat10cdrincspinoff']#,'abrupt4xco2','ssp119','ssp460','ssp370']
N = 200
ML_MODEL = 'feed_forward' 
PATTERN_SCALING_RESIDUALS = False
RAMP_DOWN_CORRECTED_PS = False
monthly_flag = False
use_smoothing = True
train_pattern_scaling_name = 'flat10cdrincspinoff'

MODEL_PATH = f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}'


if __name__ == "__main__":
    potential_files = prep.get_all_files_(MODEL_PATH)
    
    train_files = prep.filter_climate_files(files = potential_files, scenarios = TRAIN_SCENARIOS, indicators = [INDICATOR])
    test_files = prep.filter_climate_files(files = potential_files, scenarios = TEST_SCENARIOS, indicators = [INDICATOR])
    
    train_files_with_baseline = [(prep.get_baseline_filename(filename=filename, files= potential_files), filename) for filename in train_files]
    test_files_with_baseline = [(prep.get_baseline_filename(filename=filename, files= potential_files), filename) for filename in test_files]
    
    for (base,exp) in test_files_with_baseline: 
        print(base, exp)
    
    train_data_df = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in train_files_with_baseline]
    test_data_df = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in test_files_with_baseline]
    
    train_data_df = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in train_files_with_baseline]
    test_data_df = [prep.process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = monthly_flag, smoothed = use_smoothing) for (baseline,experiment) in test_files_with_baseline]
    
    if PATTERN_SCALING_RESIDUALS:
        flat10cdr_index = [i for i, f in enumerate(train_files) if train_pattern_scaling_name in f][0]
        regional_regression_slopes_intersepts = prep.process_gmt_and_regions_into_array(train_data_df[flat10cdr_index], weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS)
        train_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS, slope_intercept = regional_regression_slopes_intersepts,ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS) for GMT_regional_values_tuple in train_data_df]
        test_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS, slope_intercept = regional_regression_slopes_intersepts, ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS) for GMT_regional_values_tuple in test_data_df]
    else:
        flat10cdr_index = [i for i, f in enumerate(train_files) if train_pattern_scaling_name in f][0]
        regional_regression_slopes_intersepts = prep.process_gmt_and_regions_into_array(train_data_df[flat10cdr_index], weighted_linear_smoothing = False, pattern_scaling_residuals=True,ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS)
        train_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False,ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS) for GMT_regional_values_tuple in train_data_df]
        test_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, ramp_down_corrected_ps = RAMP_DOWN_CORRECTED_PS) for GMT_regional_values_tuple in test_data_df]
    
    train_data_input, train_data_output = prep.prepare_all_train_data(train_data_np, n=N)
    test_data_input, test_data_output = prep.prepare_all_train_data(test_data_np, n=N)
    
    train_data_shuffeled = prep.shuffle_train_data(train_data_input, train_data_output, random_state=42)
    
    test_data_for_autoregression_input, test_data_for_autoregression_output = prep.prepare_train_data(test_data_np[0],N)
    
    gmt_autoregressive_test = test_data_np[0][0,:]
    autoregressive_test_groudtruth = test_data_np[0][1:,:]
    
    gmt_train = train_data_np[0][0,:]
    train_data_regional_temperature = train_data_np[0][1:,:]



    u = train_data_shuffeled[0][0,:].T[..., None]
    #y = np.transpose(train_data_shuffeled[0][44:48,:],(2, 1, 0))
    y = np.transpose(train_data_shuffeled[0],(2, 1, 0))
    print(u.shape)
    print(y.shape)

    device = "cuda"

    model, y_scaler, u_scaler = scales_ssm.run_train(y, u, context_len=50, z_dim=24,horizon=80, device=device,epochs=10)
    os.makedirs("outputs_ssm_scales", exist_ok=True)
    torch.save(model.state_dict(),"outputs_ssm_scales/model_out")
    y_scaler.save("outputs_ssm_scales/y_scaler.out")
    u_scaler.save("outputs_ssm_scales/u_scaler.out")
    
