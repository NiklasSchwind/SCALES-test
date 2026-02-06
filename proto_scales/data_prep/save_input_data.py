import xarray as xr
import scipy
import numpy as np
import pandas as pd
import copy
from datetime import datetime
import matplotlib.pyplot as plt
import os
import re
from typing import List, Optional
import pandas as pd
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import os
import re
from typing import List, Optional
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.linear_model import LinearRegression
import proto_scales.data_prep.prepare_data as prep
import pickle

import warnings
#warnings.filterwarnings("ignore")

MODEL = 'ACCESS-ESM1-5'
INDICATOR = 'tas'#'pr' #'tas'
TEST_SCENARIOS = ['abrupt-4xco2']
#TRAIN_SCENARIOS = ['ssp585','1pctco2','ssp245'] # for "pr"
monthly_flag = False
use_smoothing = True
TRAIN_SCENARIOS = ['ssp534-over','flat10cdrincspinoff','ssp585','1pctco2','ssp245']#,'ssp126','flat10zecincspinoff']#, 'flat10cdrincspinoff','abrupt-4xco2','ssp119','ssp460','ssp370']
N = 200
ML_MODEL = 'feed_forward'
PATTERN_SCALING_RESIDUALS = False
RAMP_DOWN_CORRECTED_PS = False
train_pattern_scaling_name = 'flat10cdrincspinoff'#'ssp585'#'flat10cdrincspinoff'

MODEL_PATH = f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}'



#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
   

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


pickle_filename = "/hdrive/all_users/kainverena/formatted_data/input_data.pkl"

with open(pickle_filename, 'wb') as file:
    pickle.dump(train_data_input, file)
    pickle.dump(train_data_output, file)
    pickle.dump(test_data_input, file)
    pickle.dump(test_data_output, file)



