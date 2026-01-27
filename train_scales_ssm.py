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
import torch
from sklearn.linear_model import LinearRegression

MODEL = 'ACCESS-ESM1-5'
INDICATOR = 'tas'
TEST_SCENARIOS = ['flat10cdrincspinoff']
TRAIN_SCENARIOS = [ 'ssp534-over','flat10cdrincspinoff']#'ssp585','1pctco2','ssp245','ssp126','flat10zecincspinoff', 'flat10cdrincspinoff']#,'abrupt4xco2','ssp119','ssp460','ssp370']
N = 200
ML_MODEL = 'feed_forward'
PATTERN_SCALING_RESIDUALS = False
RAMP_DOWN_CORRECTED_PS = False

MODEL_PATH = f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}'

def weights_calculate(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau ** 2)))

def local_weighted_regression(x0, X, Y, tau):
    # add bias term
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]

    # weighted least squares
    xw = X.T * weights_calculate(x0, X, tau)
    theta = np.linalg.pinv(xw @ X) @ xw @ Y
    return x0 @ theta
def local_weighted_regression_slopes(x0, X, Y, tau):
    """
    Same weighted regression as `local_weighted_regression`,
    but returns only the slope coefficients (excluding intercept).
    """
    # Add bias term consistently (same as original function)
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]

    # Weighted least squares (same as original)
    xw = X.T * weights_calculate(x0, X, tau)
    theta = np.linalg.pinv(xw @ X) @ xw @ Y

    # Return only slopes (excluding intercept)
    return np.squeeze(theta[1:])

def parse_filename(filename: str) -> Optional[dict]:
    """
    Parses a climate model filename into its components.
    
    Args:
        filename: The filename to parse
    
    Returns:
        Dictionary with keys: model, scenario, ensemble, indicator, or None if parsing fails
    """
    pattern = r'^(.+?)_(.+?)_(.+?)_ipcc-regions_latweight\.csv$'
    match = re.match(pattern, filename)
    
    if match:
        model, scenario_ensemble, indicator = match.groups()
        scenario = '-'.join(scenario_ensemble.split('-')[0:-1])
        ensemble = scenario_ensemble.split('-')[-1]
        return {
            'model': model,
            'scenario': scenario,
            'ensemble': ensemble,
            'indicator': indicator
        }
    return None

def smooth_regional_indicator_timeseries(regional_indicator, bandwidth=20, is_monthly=False):
    """
    Smooth each row of a (N, M) or (N, M*12) array.
    If is_monthly=True, apply smoothing separately for each month across years.

    Parameters:
        regional_indicator : np.ndarray
            Input array, shape (N, M) normally, or (N, M*12) if monthly.
        
        bandwidth : int
            Bandwidth parameter for local_weighted_regression.
        
        is_monthly : bool
            Whether the input data is monthly (N, M*12) and should be smoothed by month.

    Returns:
        np.ndarray
            Smoothed array with same shape as input.
    """

    N, total_cols = regional_indicator.shape

    # Case 1: Regular (non-monthly) smoothing
    if not is_monthly:
        array_x = np.arange(total_cols)
        smoothed = np.zeros_like(regional_indicator)

        for i in range(N):
            array_y = regional_indicator[i]
            smoothed[i] = np.array([
                local_weighted_regression(x0, array_x, array_y, bandwidth)
                for x0 in array_x
            ])
        return smoothed

    # Case 2: Monthly-aware smoothing (total_cols must be multiple of 12)
    if total_cols % 12 != 0:
        raise ValueError("For monthly smoothing, number of columns must be divisible by 12.")

    M = total_cols // 12  # Number of years

    smoothed = np.zeros_like(regional_indicator)
    array_x = np.arange(M)  # x-values: 0...M-1 for each month's yearly data

    for i in range(N):  # Loop over regions
        for month in range(12):  # Loop over months
            y_month = regional_indicator[i, month::12]  # Extract all years of this month

            smoothed_month_values = np.array([
                local_weighted_regression(x0, array_x, y_month, bandwidth)
                for x0 in array_x
            ])

            # Place smoothed values back in correct positions
            smoothed[i, month::12] = smoothed_month_values

    return smoothed

def get_baseline_filename(filename: str, path: str = ".") -> Optional[str]:
    """
    Finds the baseline file for a given scenario file.
    - For SSP scenarios: returns historical file with same ensemble and indicator
    - For non-SSP scenarios: returns piControl file with same ensemble and indicator
    - If no exact match exists for non-SSP: returns piControl with different ensemble but same indicator
    
    Args:
        filename: The scenario filename
        path: Directory path where files are located
    
    Returns:
        The baseline filename if found, None otherwise
    """
    # Parse the input filename
    components = parse_filename(filename)
    
    if not components:
        print(f"Error: Could not parse filename '{filename}'")
        return None
    
    # Check if the scenario contains 'ssp'
    if 'ssp' in components['scenario'].lower():
        # For SSP scenarios, use historical
        baseline_scenario = 'historical'
    else:
        # For non-SSP scenarios, use piControl
        baseline_scenario = 'picontrol'
    
    # Construct the baseline filename with same ensemble
    baseline_filename = (
        f"{components['model']}_{baseline_scenario}-{components['ensemble']}_"
        f"{components['indicator']}_ipcc-regions_latweight.csv"
    )
    
    # Check if the baseline file exists
    baseline_path = os.path.join(path, baseline_filename)
    if os.path.exists(baseline_path):
        return baseline_filename
    
    # If not found and it's piControl, try to find one with a different ensemble
    if baseline_scenario == 'picontrol':
        try:
            all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            
            # Look for any piControl file with the same model and indicator
            pattern = rf'^{re.escape(components["model"])}_picontrol-(.+?)_{re.escape(components["indicator"])}_ipcc-regions_latweight\.csv$'
            
            for file in all_files:
                match = re.match(pattern, file)
                if match:
                    print(f"Info: Exact ensemble match not found. Using '{file}' with different ensemble")
                    return file
            
            print(f"Warning: No piControl baseline file found for indicator '{components['indicator']}' in '{path}'")
            return None
            
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error accessing path: {e}")
            return None
    else:
        print(f"Warning: Baseline file '{baseline_filename}' not found in '{path}'")
        return None



def get_all_files(path: str) -> List[str]:
    """
    Retrieves all filenames from a given directory.

    Args:
        path: Directory path to search for files.

    Returns:
        List of filenames.
    """
    try:
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing path: {e}")
        return []


def filter_climate_files(
    files: List[str],
    scenarios: Optional[List[str]] = None,
    ensembles: Optional[List[str]] = None,
    indicators: Optional[List[str]] = None,
    models: Optional[List[str]] = None
) -> List[str]:
    """
    Filters files matching the pattern:
    {model}_{scenario}-{ensemble}_{indicator}_ipcc-regions_latweight.csv

    Args:
        files: List of filenames to filter.
        scenarios: List of scenarios to filter (e.g., ['ssp370', 'ssp245'])
        ensembles: List of ensembles to filter (e.g., ['r4i1p1f1', 'r1i1p1f1'])
        indicators: List of indicators to filter (e.g., ['pr', 'tas'])
        models: List of models to filter (e.g., ['access-cm2', 'cesm2'])

    Returns:
        List of matching filenames.
    """
    pattern = r'^(.+?)_(.+?)_(.+?)_ipcc-regions_latweight\.csv$'
    matching_files = []

    for filename in files:
        match = re.match(pattern, filename)
        if match:
            model, scenario_ensemble, indicator = match.groups()
            scenario = '-'.join(scenario_ensemble.split('-')[0:-1])
            ensemble = scenario_ensemble.split('-')[-1]
            if scenarios and scenario not in scenarios:
                continue
            if ensembles and ensemble not in ensembles:
                continue
            if indicators and indicator not in indicators:
                continue
            if models and model not in models:
                continue

            matching_files.append(filename)

    return matching_files

def get_baseline_filename(filename: str, files: List[str]) -> Optional[str]:
    """
    Finds the baseline file for a given scenario file.
    - For SSP scenarios: returns historical file with same ensemble and indicator
    - For non-SSP scenarios: returns piControl file with same ensemble and indicator
    - If no exact match exists for non-SSP: returns piControl with different ensemble but same indicator

    Args:
        filename: The scenario filename
        files: List of available filenames to search from

    Returns:
        The baseline filename if found, None otherwise
    """
    # Parse the input filename
    components = parse_filename(filename)

    if not components:
        print(f"Error: Could not parse filename '{filename}'")
        return None

    # Determine baseline scenario
    if 'ssp' in components['scenario'].lower():
        baseline_scenario = 'historical'
    else:
        baseline_scenario = 'picontrol'

    # Construct expected baseline filename (same ensemble & indicator)
    baseline_filename = (
        f"{components['model']}_{baseline_scenario}-{components['ensemble']}_"
        f"{components['indicator']}_ipcc-regions_latweight.csv"
    )

    # Check if exact baseline exists in provided files list
    if baseline_filename in files:
        return baseline_filename

    # If not found and baseline is piControl, try fuzzy match (other ensemble)
    if baseline_scenario == 'picontrol':
        pattern = rf'^{re.escape(components["model"])}_picontrol-(.+?)_' \
                  rf'{re.escape(components["indicator"])}_ipcc-regions_latweight\.csv$'

        for file in files:
            if re.match(pattern, file):
                print(f"Info: Exact ensemble match not found. Using '{file}' with different ensemble")
                return file

        print(f"Warning: No piControl baseline file found for indicator '{components['indicator']}'")
        return None
    else:
        print(f"Warning: Baseline file '{baseline_filename}' not found in file list")
        return None 
    
def reorder_columns(df):
    # Ensure required columns exist
    if 'time' not in df.columns or 'GLOBAL' not in df.columns:
        raise ValueError("DataFrame must contain 'time' and 'GLOBAL' columns")

    # Extract all columns except time and GLOBAL
    middle_cols = [col for col in df.columns if col not in ['time', 'GLOBAL']]
    
    # Sort the region columns alphabetically
    middle_cols_sorted = sorted(middle_cols)

    # Reconstruct column order
    new_column_order = ['time'] + middle_cols_sorted + ['GLOBAL']

    # Reorder dataframe
    return df[new_column_order]


def process_scenarios(experiment_scenario_path, simulation_name, baseline_scenario_path = None, delete_first_years = 0, monthly_trend = False, smoothed = True):
    """
    Load CMIP6 baseline (piControl) and abrupt4xco2 scenario data,
    compute anomalies relative to the baseline scenario,
    and return:
        (1) global anomaly timeseries (annual, 21-year rolling mean)
        (2) regional anomaly timeseries (monthly, 21-year rolling mean)
    """

   

    df_experiment = pd.read_csv(experiment_scenario_path, parse_dates=["time"])
    df_experiment = reorder_columns(df_experiment)
    df_experiment['time'] = df_experiment['time'].astype(str)
    df_experiment['time'] = df_experiment['time'].apply(lambda x: datetime.strptime(x.split('.')[0], "%Y-%m-%d %H:%M:%S"))

    #if 'historical' in baseline_scenario_path:
    df_baseline = pd.read_csv(baseline_scenario_path, parse_dates=["time"])
    df_baseline = reorder_columns(df_baseline)
    df_baseline['time'] = df_baseline['time'].astype(str)
    df_baseline['time'] = df_baseline['time'].apply(lambda x: datetime.strptime(x.split('.')[0], "%Y-%m-%d %H:%M:%S"))
  

    first_year_experiment = df_experiment['time'][0].year
    
    if first_year_experiment < 1000: 
        first_year_experiment_shift = 1850 - first_year_experiment
        df_experiment.time = df_experiment.time.map(lambda dt: dt.replace(year=dt.year + first_year_experiment_shift))
        first_year_experiment = df_experiment['time'][0].year

    
    last_year_baseline = df_baseline['time'].iloc[-1].year
    year_shift = (first_year_experiment - 1)  - int(last_year_baseline)
    df_baseline.time = df_baseline.time.map(lambda dt: dt.replace(year=dt.year + year_shift))
    #df_baseline['time'] = pd.to_datetime(df_experiment['time'])
    
    df_experiment = pd.concat([df_baseline, df_experiment]).sort_values('time').reset_index(drop=True)

    if (df_experiment['time'].iloc[-1].year - df_experiment['time'][0].year) > 450: 
        delete_additional_years = (df_experiment['time'].iloc[-1].year - df_experiment['time'][0].year) - 450
        df_experiment = df_experiment[
            [dt.year >= min(dt.year for dt in df_experiment['time']) + delete_additional_years for dt in df_experiment['time']]
                ].reset_index(drop=True)

        # Shift remaining years so first year becomes start_year
        year_shift = 1750 -  df_experiment['time'][0].year
        df_experiment['time'] = df_experiment['time'].apply(lambda dt: dt.replace(year=dt.year + year_shift))

    #df_experiment['time'] = df_experiment['time'].apply(
    #lambda x: f'{str(int(x[0])+1)}{x[1:]}' if int(x[0:3]) < 1500 else x
    #)
    #df_experiment['time'] = pd.to_datetime(df_experiment['time'])

    first_year = df_experiment['time'][0].year
    year_shift = 1750 - int(first_year)
    df_experiment.time = df_experiment.time.map(lambda dt: dt.replace(year=dt.year + year_shift))
    df_experiment['time'] = pd.to_datetime(df_experiment['time'])
    
    
    region_cols = [col for col in df_experiment.columns if col.lower() != 'time']
    
    # calculate baseline
    #if 'historical' not in baseline_scenario_path:
    #    df_baseline = pd.read_csv(baseline_scenario_path, parse_dates=["time"])
    #    df_baseline = df_baseline.set_index("time")
    #    baseline_means = df_baseline[region_cols].mean()
    #else: 
    df_baseline = copy.deepcopy(df_experiment)
    df_baseline['time'] = df_baseline['time'].astype(str)
    df_baseline['time'] = df_baseline['time'].apply(lambda x: datetime.strptime(x.split('.')[0], "%Y-%m-%d %H:%M:%S"))
    #df_baseline['time'] = pd.to_datetime(df_baseline['time'])
    df_baseline = df_baseline[(df_baseline['time'].dt.year >= 1850) & (df_baseline['time'].dt.year <= 1900)]
    baseline_means = df_baseline[region_cols].mean()

    if delete_first_years != 0: 
        df_experiment = df_experiment[df_experiment['time'].dt.year >= df_experiment['time'].dt.year.min() + delete_first_years].reset_index(drop=True)
        # Shift remaining years so first year becomes start_year
        year_shift = 1850 - df_experiment['time'].dt.year.min()
        df_experiment['time'] = df_experiment['time'].apply(lambda dt: dt.replace(year=dt.year + year_shift))

    df_experiment = df_experiment.set_index("time")

    # (B) Compute regional anomalies and apply 21-year (252-month) rolling mean
    df_regional_anomaly = df_experiment[region_cols] - baseline_means
    #df_regional_smoothed = df_regional_anomaly
    if monthly_trend:
        if smoothed:
            df_regional_smoothed = (
                df_regional_anomaly
                .groupby(df_regional_anomaly.index.month)
                .apply(lambda x: x.rolling(window=21, center=True).mean())
                .reset_index(level=0, drop=True))  
        else: 
            df_regional_smoothed = (
                df_regional_anomaly
                .groupby(df_regional_anomaly.index.month)
                .apply(lambda x: x)  # just keep values as they are
                .reset_index(level=0, drop=True)
            )

        df_global = pd.DataFrame({
        'time': df_experiment.index,
        'GMT': df_regional_smoothed.GLOBAL#df_regional_smoothed.mean(axis=1)
            }).set_index('time')
    
        
    else:
        #df_regional_smoothed = df_regional_anomaly.rolling(window=21*12, center = True).mean()
        df_annual = df_regional_anomaly.resample('Y').mean()
        if smoothed:
            df_regional_smoothed = df_annual.rolling(window=21, center=True).mean()
        else: 
            df_regional_smoothed = df_annual.copy()
        
        df_global = pd.DataFrame({
        'time': df_regional_smoothed.index,
        'GMT': df_regional_smoothed.GLOBAL#df_regional_smoothed.mean(axis=1)
    }).set_index('time')
    
   
    # Convert monthly to annual mean
    df_global['year'] = df_global.index.astype(str).str[:4].astype(int)
    df_global_annual = df_global.groupby('year')['GMT'].mean().reset_index()
    #df_global_annual = df_global_annual.set_index("year")
    
    # Apply 21-year annual rolling mean
    #print(df_global_annual['GMT'])
    #df_global_annual['GMT'] = df_global_annual['GMT'].rolling(window=21, center = True).mean()

    # Add simulation name
    df_global_annual.insert(0, 'simulation_name', simulation_name)
    
    # Remove GMT from regional temperature timeseries
    
    df_regional_smoothed.drop(columns=['GLOBAL'], inplace = True)
   
    return df_global_annual, df_regional_smoothed

def detect_is_monthly(df, gmt_df):
    idx = df.index

    # Case 1: Index is datetime → use infer_freq
    if isinstance(idx, pd.DatetimeIndex):
        freq = pd.infer_freq(idx)
        return freq in ['M', 'MS']

    # Case 2: Index is integer years → assume annual
    if np.issubdtype(idx.dtype, np.integer):
        # Annual data should have roughly same length as GMT series
        return len(df) > len(gmt_df)

    # Case 3: Index is string like '2000-01', '1999-12'
    try:
        idx_dt = pd.to_datetime(idx, format='%Y-%m')
        freq = pd.infer_freq(idx_dt)
        return freq in ['M', 'MS']
    except Exception:
        pass

    # Default fallback
    return False

import pandas as pd

def expand_annual_to_monthly(series: pd.Series) -> pd.Series:
    """
    Take annual data indexed by year (DatetimeIndex or year ints)
    and return monthly data via linear interpolation.
    Ensures first = Jan, last = Dec.
    """

    s = series.copy()

    # ---- 1. Make the index timezone-naive and normalized ----
    if isinstance(s.index, pd.DatetimeIndex):

        # If tz-aware → remove timezone
        if s.index.tz is not None:
            idx = s.index.tz_convert(None)
        else:
            idx = s.index  # already tz-naive

        # Normalize (remove hour/min/sec)
        s.index = idx.normalize()

    else:
        # e.g. Int64Index of years → convert to Timestamp
        s.index = pd.to_datetime(s.index.astype(str) + "-01-01")

    # ---- 2. Add Dec-31 entry for last year ----
    first_year = s.index.year.min()
    last_year = s.index.year.max()

    start = pd.Timestamp(f"{first_year}-01-01")
    end = pd.Timestamp(f"{last_year}-12-01")

    # Ensure last year has a December timestamp
    if end not in s.index:
        s.loc[end] = s.loc[pd.Timestamp(f"{last_year}-01-01")]
        s = s.drop(pd.Timestamp(f"{last_year}-01-01"))
    # ---- 3. Create complete monthly index ----
    full_monthly_index = pd.date_range(start=start, end=end, freq="MS")

    # ---- 4. Reindex & interpolate monthly ----
    s = s.reindex(full_monthly_index).interpolate("linear")

    return s

#def predict_regional_temperatures(global_test_series, slopes, intercepts, same_shape = False):
#    """
#    Uses stored regression parameters to predict regional temps.
#    Returns:
#        predictions: array of shape (len(global_test_series), n_regions)
#    """
#    original_shape = np.asarray(global_test_series).shape
#    global_test_series = np.asarray(global_test_series).reshape(-1, 1)
#    predictions = global_test_series * slopes + intercepts
#
#    if same_shape:
#        # Collapse regional dimension (e.g., mean or first region)
#        # Here: return full region predictions but reshaped like input
#        # so we return an array where the last dimension = n_regions.
#        return predictions.reshape(*original_shape, -1)
#    
#    return predictions

#def fit_regional_regressions(global_series, regional_series):
#    """
#    Fits one linear regression per region: region = a + b * global
#    Returns:
#        slopes:   (46,) array of regression coefficients
#       intercepts: (46,) array of intercepts
#    """
#    global_series = np.asarray(global_series).reshape(-1, 1)
#    regional_series = np.asarray(regional_series)
#    
#    n_regions = regional_series.shape[1]
#    slopes = np.zeros(n_regions)
#    intercepts = np.zeros(n_regions)
#
#    for i in range(n_regions):
#        model = LinearRegression()
#        model.fit(global_series, regional_series[:, i])
#        slopes[i] = model.coef_[0]
#        intercepts[i] = model.intercept_
#
#    return slopes, intercepts
import numpy as np

def predict_regional_temperatures(
    global_test_series,
    slopes,
    intercepts,
    slopes_down=None,
    intercepts_down=None,
    same_shape=False,
):
    """
    Predict regional temperatures with optional ramp-down correction.
    Peak is detected from the provided global_test_series.

    Parameters
    ----------
    global_test_series : array-like
        Global temperature time series
    slopes : (n_regions,)
        Ramp-up slopes
    intercepts : (n_regions,)
        Ramp-up intercepts
    slopes_down : (n_regions,), optional
        Ramp-down residual slopes
    intercepts_down : (n_regions,), optional
        Ramp-down residual intercepts
    same_shape : bool
        Match output shape to input

    Returns
    -------
    predictions : ndarray
        Shape (T, n_regions) or reshaped if same_shape=True
    """
    original_shape = np.asarray(global_test_series).shape
    g = np.asarray(global_test_series).reshape(-1, 1)

    # Base ramp-up prediction everywhere
    predictions = g * slopes + intercepts

    # Apply ramp-down correction if available
    if slopes_down is not None and intercepts_down is not None:
        # Detect peak from test scenario
        peak = np.argmax(g[:, 0])

        if peak + 1 < len(g):
            correction = (
                (g[peak] - g[peak + 1 :]) * slopes_down + intercepts_down
            )
            predictions[peak + 1 :] += correction

    if same_shape:
        return predictions.reshape(*original_shape, -1)

    return predictions


def fit_regional_regressions(
    global_series,
    regional_series,
    train_ramp_down=False,
):
    """
    Fits linear regressions per region.

    If train_ramp_down=False:
        region = a + b * global

    If train_ramp_down=True:
        - Fit regression on ramp-up phase
        - Fit regression on residuals during ramp-down

    Returns
    -------
    result : dict with keys
        slopes_up        (n_regions,)
        intercepts_up    (n_regions,)
        slopes_down      (n_regions,) or None
        intercepts_down  (n_regions,) or None
    """
    global_series = np.asarray(global_series).reshape(-1, 1)
    regional_series = np.asarray(regional_series)

    n_regions = regional_series.shape[1]
    n_time = len(global_series)

    # --- identify ramp-up / ramp-down ---
    peak_idx = np.argmax(global_series[:, 0])
    ramp_up_idx = np.arange(0, peak_idx + 1)
    ramp_down_idx = np.arange(peak_idx + 1, n_time)

    slopes_up = np.zeros(n_regions)
    intercepts_up = np.zeros(n_regions)

    slopes_down = np.zeros(n_regions) if train_ramp_down else None
    intercepts_down = np.zeros(n_regions) if train_ramp_down else None

    # --- ramp-up regression ---
    for i in range(n_regions):
        model_up = LinearRegression()
        model_up.fit(
            global_series[ramp_up_idx],
            regional_series[ramp_up_idx, i],
        )
        slopes_up[i] = model_up.coef_[0]
        intercepts_up[i] = model_up.intercept_

        if train_ramp_down and len(ramp_down_idx) > 0:
            # Predict ramp-up relationship during ramp-down
            pred_up = (
                intercepts_up[i]
                + slopes_up[i] * global_series[ramp_down_idx, 0]
            )

            residuals = (
                regional_series[ramp_down_idx, i] - pred_up
            )

            model_down = LinearRegression(fit_intercept=False)
            model_down.fit(
                global_series[peak_idx] - global_series[ramp_down_idx],
                residuals,
            )
            slopes_down[i] = model_down.coef_[0]
            intercepts_down[i] = model_down.intercept_

    return {
        "slopes_up": slopes_up,
        "intercepts_up": intercepts_up,
        "slopes_down": slopes_down,
        "intercepts_down": intercepts_down,
    }

def process_gmt_and_regions_into_array(data_tuple: Tuple[pd.DataFrame, pd.DataFrame], weighted_linear_smoothing = False, pattern_scaling_residuals = False, slope_intercept = None) -> np.ndarray:
    """
    Processes a single tuple of (GMT DataFrame, regional DataFrame) into
    a numpy array of shape (1 + number_regions) x number_timesteps.

    Steps:
    1. Remove NaN values
    2. If regional data is monthly -> interpolate GMT to monthly
       If regional data is annual -> keep GMT as is
    3. Align data by time
    4. Convert to numpy array: first row GMT, rest regional
    """
    gmt_df, region_df = data_tuple

    # --- 1. Clean NaN values ---
    gmt_df = gmt_df.dropna(subset=['GMT'])
    region_df = region_df.dropna(axis=0, how='all')  # Remove rows where all regions are NaN
    # --- 2. Determine if regional data is monthly or annual ---
    #freq = pd.infer_freq(region_df.index)
    is_monthly = detect_is_monthly(region_df, gmt_df)#len(region_df.index) > len(gmt_df) or freq in ['M', 'MS'] # detect_is_monthly(region_df, gmt_df)

    # --- 3. Prepare GMT time series ---
    # Convert year to datetime for consistency
    gmt_df['time'] = pd.to_datetime(gmt_df['year'], format='%Y')
    gmt_ts = gmt_df.set_index('time')['GMT']
    

    if is_monthly:
        # Create monthly index from the regional dataframe
        monthly_index = region_df.index

        # Interpolate GMT to monthly
        gmt_monthly = expand_annual_to_monthly(gmt_ts)#gmt_ts.reindex(
            #pd.date_range(gmt_ts.index.min(), gmt_ts.index.max(), freq='MS')
        #).interpolate('linear')
    
        # Align GMT to regional timestamps (forward/backward fill allowed)
        gmt_aligned = gmt_monthly#.reindex(monthly_index)
    else:
        # Assume annual data, map directly by year
        region_df = region_df.copy()
        region_df['year'] = region_df.index.year
        gmt_aligned = region_df['year'].map(dict(zip(gmt_df['year'], gmt_df['GMT']))) 


    # --- 4. Merge GMT and regional data ---
    # Ensure no NaN in GMT after alignment
    gmt_vals = np.array(gmt_aligned.dropna()).reshape(1, -1)
  
    # Drop non-regional columns and convert to numpy
    region_clean = region_df.drop(columns=[col for col in ['year'] if col in region_df.columns])
    regional_vals = region_clean.T.values
    
    gmt_vals = gmt_vals[:,:regional_vals.shape[1]]
    # --- 5. Combine GMT + Regional into final array ---
    result_array = np.vstack([gmt_vals, regional_vals])

    if weighted_linear_smoothing:
        return smooth_regional_indicator_timeseries(regional_indicator=result_array, bandwidth=20, is_monthly=is_monthly)
    
    if pattern_scaling_residuals:
        if slope_intercept == None: 
            if weighted_linear_smoothing:
                gmt_vals = smooth_regional_indicator_timeseries(regional_indicator=gmt_vals, bandwidth=20, is_monthly=is_monthly)
            #slopes, intercepts = fit_regional_regressions(gmt_vals, regional_vals.transpose(1,0))
            regional_regression_slopes_intersepts = fit_regional_regressions(gmt_vals, regional_vals.transpose(1,0), RAMP_DOWN_CORRECTED_PS)    
            return regional_regression_slopes_intersepts
        else: 
            slopes_up, intercepts_up, slopes_down, intercepts_down  = slope_intercept["slopes_up"], slope_intercept["intercepts_up"], slope_intercept["slopes_down"], slope_intercept["intercepts_down"]

            if weighted_linear_smoothing:
                regional_vals = smooth_regional_indicator_timeseries(regional_indicator=regional_vals, bandwidth=20, is_monthly=is_monthly)
                gmt_vals = smooth_regional_indicator_timeseries(regional_indicator=gmt_vals, bandwidth=20, is_monthly=is_monthly)

            gmt_vals_preds = copy.deepcopy(gmt_vals)
            ps_prediction = np.squeeze(predict_regional_temperatures(gmt_vals_preds, slopes_up, intercepts_up,slopes_down, intercepts_down, same_shape = True)).T
            
            regional_vals = ps_prediction - regional_vals
            
            # (58, 231)
            # (58, 231)
            # (58, 231)
            # (1, 231)
            result_array_residuals = np.vstack([gmt_vals, regional_vals])
            
            return result_array_residuals
    
    return result_array
    



def prepare_train_data(data,n):
    """ 
    data: array with shape (1 + number_regions, number_timesteps)
    n: window size length
    """
    regions, timesteps = data.shape
    first_region = data[0]   # shape: (timesteps,)
    other_regions = data[1:] # shape: (number_regions, timesteps)

    # Number of valid training windows
    num_samples = timesteps - n - 1

    # X shape target: (regions, n, num_samples)
    X = np.zeros((regions, n, num_samples))

    for i in range(num_samples):  # x goes from n to timesteps-1
        # First region: t[x-n] ... t[x]  (length n)
        X[0, :, i] = first_region[(i+1):(i+n+1)]

        # Other regions: t[x-n-1] ... t[x-1] (also length n)
        X[1:, :, i] = other_regions[:, i:(i+n)]

    # Y shape target: (number_regions, num_samples)
    Y = np.zeros((regions - 1, num_samples))

    # At time t[x], take all region values except the first one
    Y[:, :] = other_regions[:, (n+1):]  # n to end: timesteps - n samples

    return X, Y

def prepare_all_train_data(data_arrays, n):
    """
    data_arrays: list of arrays, each shape (1 + number_regions, number_timesteps)
    n: window size length
    """

    X_list = []
    Y_list = []

    for data in data_arrays:
        
        X, Y = prepare_train_data(data,n)

        X_list.append(X)
        Y_list.append(Y)

    # Combine multiple samples along the third dimension
    X_combined = np.concatenate(X_list, axis=2)  
    Y_combined = np.concatenate(Y_list, axis=1)

    return X_combined, Y_combined

def shuffle_train_data(X, Y, random_state=None):
    """
    Shufflestraining data together so their sample alignment stays correct.

    X shape: (features, n, samples)
    Y shape: (targets, samples)

    Returns: shuffled X and Y
    """
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = X.shape[2]
    indices = np.random.permutation(num_samples)

    X_shuffled = X[:, :, indices]   # shuffle along last axis
    Y_shuffled = Y[:, indices]      # shuffle along last axis

    return X_shuffled, Y_shuffled


potential_files = get_all_files(MODEL_PATH)

train_files = filter_climate_files(files = potential_files, scenarios = TRAIN_SCENARIOS, indicators = [INDICATOR])
test_files = filter_climate_files(files = potential_files, scenarios = TEST_SCENARIOS, indicators = [INDICATOR])

train_files_with_baseline = [(get_baseline_filename(filename=filename, files= potential_files), filename) for filename in train_files]
test_files_with_baseline = [(get_baseline_filename(filename=filename, files= potential_files), filename) for filename in test_files]

for (base,exp) in test_files_with_baseline: 
    print(base, exp)

train_data_df = [process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = False, smoothed = True) for (baseline,experiment) in train_files_with_baseline]
test_data_df = [process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = False, smoothed = True) for (baseline,experiment) in test_files_with_baseline]

train_data_df = [process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = False, smoothed = True) for (baseline,experiment) in train_files_with_baseline]
test_data_df = [process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = False, smoothed = True) for (baseline,experiment) in test_files_with_baseline]

if PATTERN_SCALING_RESIDUALS:
    flat10cdr_index = [i for i, f in enumerate(train_files) if 'flat10cdrincspinoff' in f][0]
    regional_regression_slopes_intersepts = process_gmt_and_regions_into_array(train_data_df[flat10cdr_index], weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS)
    train_data_np = [process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS, slope_intercept = regional_regression_slopes_intersepts) for GMT_regional_values_tuple in train_data_df]
    test_data_np = [process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS, slope_intercept = regional_regression_slopes_intersepts) for GMT_regional_values_tuple in test_data_df]
else: 
    flat10cdr_index = [i for i, f in enumerate(train_files) if 'flat10cdrincspinoff' in f][0]
    regional_regression_slopes_intersepts = process_gmt_and_regions_into_array(train_data_df[flat10cdr_index], weighted_linear_smoothing = False, pattern_scaling_residuals=True)
    train_data_np = [process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False) for GMT_regional_values_tuple in train_data_df]
    test_data_np = [process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False) for GMT_regional_values_tuple in test_data_df]

train_data_input, train_data_output = prepare_all_train_data(train_data_np, n=N)
test_data_input, test_data_output = prepare_all_train_data(test_data_np, n=N)

train_data_shuffeled = shuffle_train_data(train_data_input, train_data_output, random_state=42)

test_data_for_autoregression_input, test_data_for_autoregression_output = prepare_train_data(test_data_np[0],N)

gmt_autoregressive_test = test_data_np[0][0,:]
autoregressive_test_groudtruth = test_data_np[0][1:,:]

gmt_train = train_data_np[0][0,:]
train_data_regional_temperature = train_data_np[0][1:,:]


import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys


class SlidingWindowDataset(Dataset):
    """
    does not have same signature as WindowDataset, discard
    """

    def __init__(self, y, u, context_len, horizon):
        self.y = y.astype(np.float32)
        self.u = u.astype(np.float32)
        self.T = y.shape[0]
        self.Tc = context_len
        self.H = horizon

    def __len__(self):
        return self.T - self.Tc - self.H + 1

    def __getitem__(self, i):
        y_ctx = self.y[i:i+self.Tc]
        u_ctx = self.u[i:i+self.Tc]
        u_fut = self.u[i+self.Tc:i+self.Tc+self.H]
        y_fut = self.y[i+self.Tc:i+self.Tc+self.H]
        return y_ctx, u_ctx, u_fut, y_fut

import numpy as np
from torch.utils.data import Dataset
import pickle

class UnifiedWindowDataset(Dataset):
    """
    Works for:
      - one long series: y [T, Dy], u [T, Du]
      - many series:     y [N, T, Dy], u [N, T, Du]

    Each dataset item corresponds to a window defined by (series_idx, start).
    It returns:
      y_ctx: [Tc, Dy]
      u_ctx: [Tc, Du]
      u_fut: [H,  Du]
      y_fut: [H,  Dy]
    """
    def __init__(self, y, u, context_len=40, horizon=12, stride=1, start_mode="all"):
        """
        stride: step between consecutive window starts (reduces overlap if >1)
        start_mode:
          - "all": generate all valid starts for each series...like sliding window
          - "zero": only start at 0 for each series (behaves like your WindowDataset)
        """
        y = np.asarray(y, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)

        # Normalize shapes to [N, T, D]
        if y.ndim == 2:
            y = y[None, ...]  # [1, T, Dy]
        if u.ndim == 2:
            u = u[None, ...]  # [1, T, Du]

        assert y.shape[0] == u.shape[0] and y.shape[1] == u.shape[1]
        self.y = y
        self.u = u
        self.N, self.T, self.Dy = y.shape
        self.Du = u.shape[2]
        self.Tc = int(context_len)
        self.H = int(horizon)
        self.stride = int(stride)

        if self.Tc + self.H > self.T:
            raise ValueError("context_len + horizon must be <= T")

        # Build an index map: each item maps to (series_idx, start)
        self.index = []
        max_start = self.T - (self.Tc + self.H)
        for s in range(self.N):
            if start_mode == "zero":
                self.index.append((s, 0))
            elif start_mode == "all":
                for start in range(0, max_start + 1, self.stride):
                    self.index.append((s, start))
            else:
                raise ValueError("start_mode must be 'all' or 'zero'")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s, start = self.index[idx]
        Tc, H = self.Tc, self.H

        y = self.y[s]  # [T, Dy]
        u = self.u[s]  # [T, Du]

        y_ctx = y[start : start + Tc]
        u_ctx = u[start : start + Tc]
        u_fut = u[start + Tc : start + Tc + H]
        y_fut = y[start + Tc : start + Tc + H]

        return y_ctx, u_ctx, u_fut, y_fut



# -------------------------
# Dataset (fixed-length windows)
# -------------------------
class WindowDataset(Dataset):
    """
    Expects arrays:
      y: [N, T, Dy]
      u: [N, T, Du]
    Returns:
      y_ctx, u_ctx, u_fut, y_fut
    """
    def __init__(self, y, u, context_len=40, horizon=12):
        assert y.shape[0] == u.shape[0] and y.shape[1] == u.shape[1]
        self.y = y.astype(np.float32)
        self.u = u.astype(np.float32)
        self.context_len = context_len
        self.horizon = horizon
        self.T = y.shape[1]
        assert context_len + horizon <= self.T

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        y = self.y[idx]
        u = self.u[idx]
        Tc = self.context_len
        H = self.horizon
        y_ctx = y[:Tc]
        u_ctx = u[:Tc]
        u_fut = u[Tc:Tc+H]
        y_fut = y[Tc:Tc+H]
        return y_ctx, u_ctx, u_fut, y_fut


# -------------------------
# Utils: normalization
# -------------------------
class StandardScaler:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        # x: [N, T, D]
        mean = x.mean(axis=(0, 1), keepdims=True)
        std = x.std(axis=(0, 1), keepdims=True)
        self.mean_ = mean
        self.std_ = np.maximum(std, self.eps)
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x):
        return x * self.std_ + self.mean_


    def save(self, filepath):
        assert self.mean_ is not None and self.std_ is not None, \
            "Cannot save an unfitted StandardScaler"

        data = {
            "eps": self.eps,
            "mean_": self.mean_,
            "std_": self.std_,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def from_file(cls, filepath):
        assert os.path.exists(filepath), f"File does not exist: {filepath}"

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Basic integrity checks
        assert isinstance(data, dict), "Saved scaler data must be a dictionary"
        assert "eps" in data and "mean_" in data and "std_" in data, \
            "Saved scaler file is missing required keys"

        scaler = cls(eps=data["eps"])
        scaler.mean_ = data["mean_"]
        scaler.std_ = data["std_"]

        # Shape and value checks
        assert scaler.mean_.shape == scaler.std_.shape, \
            "mean_ and std_ must have the same shape"
        assert np.all(scaler.std_ > 0), \
            "All std_ values must be positive"

        return scaler





# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def diag_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    # KL(Nq||Np) for diagonal Gaussians; returns [B]
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0).sum(-1)

class DeepSSMConditioned(nn.Module):
    """
    q(z_t | y_{1:t}, u_{1:t}) via GRU on [y,u]
    p(z_t | z_{t-1}, u_t) via MLP([z_{t-1},u_t]) -> (mu, logvar)
    p(y_t | z_t) via MLP(z_t) -> (mu_y) with learned global sigma_y
    """
    def __init__(self, y_dim, u_dim, z_dim=16, rnn_hidden=62, mlp_hidden=128, emission_uses_u=False):
        super().__init__()
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.emission_uses_u = emission_uses_u

        self.gru = nn.GRU(input_size=y_dim + u_dim, hidden_size=rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        self.trans = MLP(z_dim + u_dim, 2 * z_dim, hidden=mlp_hidden)

        emit_in = z_dim + (u_dim if emission_uses_u else 0)
        self.emit = MLP(emit_in, y_dim, hidden=mlp_hidden)

        # global observation noise (log sigma); initialized modestly
        self.log_sigma_y = nn.Parameter(torch.tensor(-0.2))

    def sample(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward_elbo(self, y, u, kl_free_bits=0.5):
        """
        y,u: [B, T, Dy/Du]
        Returns loss, stats.
        """
        B, T, _ = y.shape

        rnn_in = torch.cat([y, u], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h)
        mu_q, logvar_q = torch.chunk(q_params, 2, dim=-1)

        # stabilize logvar range...numbers not tuned
        logvar_q = torch.clamp(logvar_q, -12.0, 6.0)

        # prior for z0
        mu_p0 = torch.zeros(B, self.z_dim, device=y.device)
        logvar_p0 = torch.zeros(B, self.z_dim, device=y.device)

        sigma_y = torch.exp(torch.clamp(self.log_sigma_y, -6.0, 3.0))

        nll = 0.0
        kl = 0.0

        z_prev = None
        for t in range(T):
            z_t = self.sample(mu_q[:, t], logvar_q[:, t])

            if self.emission_uses_u:
                y_hat = self.emit(torch.cat([z_t, u[:, t]], dim=-1))
            else:
                y_hat = self.emit(z_t)

            # Gaussian NLL (includes log sigma term)
            nll_t = 0.5 * (((y[:, t] - y_hat) / sigma_y) ** 2).sum(-1) + self.y_dim * torch.log(sigma_y)
            nll = nll + nll_t

            if t == 0:
                kl_t = diag_gaussian_kl(mu_q[:, 0], logvar_q[:, 0], mu_p0, logvar_p0)
            else:
                trans_in = torch.cat([z_prev, u[:, t]], dim=-1)
                mu_p, logvar_p = torch.chunk(self.trans(trans_in), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                kl_t = diag_gaussian_kl(mu_q[:, t], logvar_q[:, t], mu_p, logvar_p)

            # free-bits: don't over-penalize small KL; helps avoid posterior collapse: from Claude
            # Applied per-sample
            kl = kl + torch.clamp(kl_t, min=kl_free_bits)

            z_prev = z_t

        # per-batch means
        nll = nll.mean()
        kl = kl.mean()
        return nll, kl

    @torch.no_grad()
    def forecast(self, y_ctx, u_ctx, u_fut, steps, n_samples=50):
        B, Tc, _ = y_ctx.shape

        rnn_in = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT = torch.clamp(logvar_qT, -12.0, 6.0)

        sigma_y = torch.exp(torch.clamp(self.log_sigma_y, -6.0, 3.0))

        ysamps = []
        for _ in range(n_samples):
            z = self.sample(mu_qT, logvar_qT)

            preds = []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(self.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = self.sample(mu_p, logvar_p)

                if self.emission_uses_u:
                    y_hat = self.emit(torch.cat([z, u_t], dim=-1))
                else:
                    y_hat = self.emit(z)

                y_s = y_hat + sigma_y * torch.randn_like(y_hat)
                preds.append(y_s)

            ysamps.append(torch.stack(preds, dim=1))  # [B,steps,Dy]

        samp = torch.stack(ysamps, dim=0)  # [S,B,steps,Dy]
        return samp.mean(0), samp.quantile(0.10, 0), samp.quantile(0.90, 0)

    @torch.no_grad()
    def forecast_deterministic(self, y_ctx, u_ctx, u_fut, steps, n_samples=50):
        B, Tc, _ = y_ctx.shape

        rnn_in = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT = torch.clamp(logvar_qT, -12.0, 6.0)

        sigma_y = torch.exp(torch.clamp(self.log_sigma_y, -6.0, 3.0))

        ysamps = []
        for _ in range(n_samples):
            z = self.sample(mu_qT, logvar_qT)

            preds = []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(self.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = self.sample(mu_p, logvar_p)

                if self.emission_uses_u:
                    y_hat = self.emit(torch.cat([z, u_t], dim=-1))
                else:
                    y_hat = self.emit(z)

                y_s = y_hat #+ sigma_y * torch.randn_like(y_hat)
                preds.append(y_s)

            ysamps.append(torch.stack(preds, dim=1))  # [B,steps,Dy]

        samp = torch.stack(ysamps, dim=0)  # [S,B,steps,Dy]
        return samp.mean(0), samp.quantile(0.10, 0), samp.quantile(0.90, 0)


# -------------------------
# Train / eval loop with early stopping
# -------------------------
def run_train(
    y_np, u_np,
    context_len=40, horizon=12,
    batch_size=64,
    epochs=50,
    lr=2e-3,
    z_dim=16,
    device="cpu",
):
    # split
    N = y_np.shape[0]
    idx = np.random.permutation(N)
    n_train = int(0.8 * N)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    y_tr, u_tr = y_np[tr_idx], u_np[tr_idx]
    y_va, u_va = y_np[va_idx], u_np[va_idx]

    # normalize (fit on train only)
    y_scaler = StandardScaler().fit(y_tr)
    u_scaler = StandardScaler().fit(u_tr)
    y_trn = y_scaler.transform(y_tr)
    y_van = y_scaler.transform(y_va)
    u_trn = u_scaler.transform(u_tr)
    u_van = u_scaler.transform(u_va)
   
    train_ds = UnifiedWindowDataset(y_trn, u_trn, context_len=context_len, horizon=horizon)
    val_ds   = UnifiedWindowDataset(y_van, u_van, context_len=context_len, horizon=horizon)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    Dy = y_np.shape[-1]
    Du = u_np.shape[-1]
    model = DeepSSMConditioned(y_dim=Dy, u_dim=Du, z_dim=z_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None
    patience, patience_left = 15, 15
  
    # KL annealing schedule: ramp from 0 -> 1 over first ~30% of training
    total_steps = epochs * len(train_dl)
    global_step = 0

    for epoch in range(1, epochs + 1):
       
        model.train()
        
        tr_loss = []

        for y_ctx, u_ctx, u_fut, y_fut in train_dl:
           
            y_ctx = torch.tensor(y_ctx, device=device)
            u_ctx = torch.tensor(u_ctx, device=device)
            u_fut = torch.tensor(u_fut, device=device)
            y_fut = torch.tensor(y_fut, device=device)
          
            # We train on full (context+horizon) to teach dynamics across the boundary:
            y_full = torch.cat([y_ctx, y_fut], dim=1)
            u_full = torch.cat([u_ctx, u_fut], dim=1)
       
            nll, kl = model.forward_elbo(y_full, u_full, kl_free_bits=0.2)
            mean, _, _ = model.forecast_deterministic(y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
            roll_out_mse =((mean - y_fut) ** 2).mean()
           
            # anneal KL weight
            global_step += 1
            frac = min(1.0, global_step / int(0.3 * total_steps))
            kl_w = frac  # 0->1
            kl_w = 2
            alpha = 100
         
            loss = nll + kl_w * kl + alpha*roll_out_mse

            if(global_step%100==0):
                print("loss: ",nll.item(),kl_w,kl.item(),roll_out_mse.item())
            
            opt.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           
            opt.step()
           

            tr_loss.append(loss.item())

        # validation: one-step objective + forecast MSE on horizon
        model.eval()
        
        va_loss = []
        va_mse = []

        with torch.no_grad():
            for y_ctx, u_ctx, u_fut, y_fut in val_dl:
                
                y_ctx = torch.tensor(y_ctx, device=device)
                u_ctx = torch.tensor(u_ctx, device=device)
                u_fut = torch.tensor(u_fut, device=device)
                y_fut = torch.tensor(y_fut, device=device)

                y_full = torch.cat([y_ctx, y_fut], dim=1)
                u_full = torch.cat([u_ctx, u_fut], dim=1)

                nll, kl = model.forward_elbo(y_full, u_full, kl_free_bits=0.2)
                loss = nll + 1.0 * kl
                va_loss.append(loss.item())

                mean, _, _ = model.forecast(y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
                mse = ((mean - y_fut) ** 2).mean().item()
                va_mse.append(mse)

        tr = float(np.mean(tr_loss))
        va = float(np.mean(va_loss))
        mse = float(np.mean(va_mse))
        print(f"epoch {epoch:03d} | train {tr:.4f} | val_elbo {va:.4f} | val_mse {mse:.4f}")
        os.makedirs("outputs_ssm_scales", exist_ok=True)
        torch.save(model.state_dict(),"outputs_ssm_scales/model_out")

        # early stopping on val_elbo
        if va < best_val - 1e-4:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, y_scaler, u_scaler


# -------------------------
# Example usage with synthetic data
# -------------------------
def make_synth(N=500, T=80, Dy=1, Du=2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)

    u = np.zeros((N, T, Du), dtype=np.float32)
    u[..., 0] = np.sin(0.7 *t-rng.standard_normal()*0.5)[None, :] + 0.2 * rng.standard_normal((N, T))
    u[..., 1] = np.cos(0.3 * t)[None, :] + 0.2 * rng.standard_normal((N, T))

    z = np.zeros((N, T), dtype=np.float32)
    z[:, 0] = 0.5 * rng.standard_normal(N)

    A = np.array([0.5, -0.3], dtype=np.float32)
    drift = 0.05 * np.sin(0.2 * t).astype(np.float32)

    for i in range(1, T):
        z[:, i] = z[:, i-1] + drift[i] + (u[:, i] * A).sum(-1) + 0.15 * rng.standard_normal(N)

    y = (z ** 3).astype(np.float32) + 0.3 * rng.standard_normal((N, T)).astype(np.float32)
    y = y[..., None]  # [N,T,1]
    return y, u.astype(np.float32)


if __name__ == "__main__":
    u = train_data_shuffeled[0][0,:].T[..., None]
    y = np.transpose(train_data_shuffeled[0][44:48,:],(2, 1, 0))
    print(u.shape)
    print(y.shape)

    device = "cuda"

    model, y_scaler, u_scaler = run_train(y, u, context_len=50, horizon=25, device=device,epochs=10)
    os.makedirs("outputs_ssm_scales", exist_ok=True)
    torch.save(model.state_dict(),"outputs_ssm_scales/model_out")
    y_scaler.save("outputs_ssm_scales/y_scaler.out")
    u_scaler.save("outputs_ssm_scales/u_scaler.out")
    
