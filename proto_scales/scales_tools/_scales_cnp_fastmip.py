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
import proto_scales.data_prep.prepare_data as prep
from proto_scales.ssm_model.scales_ssm import StandardScaler
from proto_scales.multi_esm_model.scales_cnp import DeepCnpSsmforESM,DeepSSMPatternConditioned


@torch.no_grad()
def forecast(model, y_ctx, u_ctx, pr_ctx, u_fut,n_samples = 1):
   
    device = y_ctx.device
    B, H, Du = u_fut.shape

    # DeepSSM mean forecast (IMPORTANT: mean, no observation noise)
    y_ssm, _, _,pr_ssm,_,_ = model.forecast(y_ctx, u_ctx, u_fut, steps=H, n_samples=n_samples,pr_ctx=pr_ctx)  # [B,H,Dy]
    # If your forecast adds obs noise, remove it for MSE evaluation.

    y_lin = model.ssm.ctrl_lin(u_fut.reshape(-1, Du)).reshape(B, H, -1)

    return y_ssm, y_lin, pr_ssm


def get_fastmip_gmt(filename):
    df = pd.read_csv(filename)
    # Expand yearly GMT projections to monthly by repeating each year's value 12 times
    gmt_cols = [c for c in df.columns if c != 'year']
    
    df_monthly = pd.DataFrame({
        'year':  np.repeat(df['year'].values, 12),
        'month': np.tile(np.arange(1, 13), len(df)),
        **{col: np.repeat(df[col].values, 12) for col in gmt_cols}
    })
    
    # u shape expected by the model: [n_scenarios, T, 1]
    # Each column of df is one GMT projection / calibration
    u_fastmip = df_monthly[gmt_cols].values.T[:, :, np.newaxis]  # [n_projections, T_monthly, 1]
    return u_fastmip,df_monthly,gmt_cols

def create_all_model_test_dict(model_list, indicators, test_scenarios):
    model_data = {}
    for model in model_list:
        MODEL_PATH = f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{model}'
        
        test_data_np = prep.fetch_and_massage_data(
        model_path=MODEL_PATH,
        indicators=INDICATORS,
        train_scenarios=TEST_SCENARIOS,
        monthly_flag=True,
        use_smoothing=False,
        train_pattern_scaling_name='ss585',
        pattern_scaling_residuals=False,
        ramp_down_corrected_ps=False,
        )
        model_data[model] = test_data_np

    return model_data

def prepare_data_for_inference(t_horizon,t_context,raw_data):
    test_data_np = np.array(raw_data)
    test_gmt = test_data_np[:,0,:]
    test_data_np = test_data_np[:,1:,:]
    regions = int(test_data_np.shape[1]/2)
    tas_test = test_data_np[:,:regions,:]
    tas_test = np.transpose(tas_test,(0, 2, 1))
    pr_test = test_data_np[:,-regions:,:]
    pr_test = np.transpose(pr_test,(0, 2, 1))
    y_past = tas_test[:,:t_context,:]
    pr_past =pr_test[:,:t_context,:]
    u_past = np.expand_dims(test_gmt[:,:t_context],axis=2)
    u_future = np.expand_dims(test_gmt[:,t_context:t_context+t_horizon],axis=2)
    return y_past, pr_past, u_past, u_future,tas_test,pr_test

def project_for_ESM(esm_name,esm_member_index,gmt_future, n_ensemble,cnp_model):
    rng = np.random.default_rng(seed=42)
    #idx = rng.choice(len(U_future_access), size=30, replace=False)
    idx = range(n_ensemble)
    

    MODEL = esm_name
    test_data_np = model_data[MODEL]

    Tc_test = 600
    H =2100

    y_past, pr_past, u_past, u_future,_,_ = prepare_data_for_inference(H,Tc_test,test_data_np)
    u_future = gmt_future[idx]
    B = u_future.shape[0]
    
    # Use only the first ESM member for context, repeated B times
    y_past_n  = np.repeat(y_scaler.transform(y_past[[esm_member_index]]),   B, axis=0)
    pr_past_n = np.repeat(pr_scaler.transform(pr_past[[esm_member_index]]), B, axis=0)
    u_past_n  = np.repeat(u_scaler.transform(u_past[[esm_member_index]]),   B, axis=0)
    u_future_n = u_scaler.transform(u_future)
    
    y_past_n   = torch.tensor(y_past_n,   device=device, dtype=torch.float32)
    u_past_n   = torch.tensor(u_past_n,   device=device, dtype=torch.float32)
    u_future_n = torch.tensor(u_future_n, device=device, dtype=torch.float32)
    pr_past_n  = torch.tensor(pr_past_n,  device=device, dtype=torch.float32)
    
    with torch.no_grad():
        y_ssm, y_lin, pr_ssm = forecast(cnp_model, y_past_n, u_past_n, pr_past_n, u_future_n, n_samples=1)
    y_pred     = y_scaler.inverse_transform(y_ssm.cpu().numpy())
    pr_pred    = pr_scaler.inverse_transform(pr_ssm.cpu().numpy())
    y_pred_lin = y_scaler.inverse_transform(y_lin.cpu().numpy())
    
    y_pred_mean = np.mean(y_pred,axis=0)
    y_pred_lin_mean = np.mean(y_pred_lin,axis=0)
    pr_pred_mean = np.mean(pr_pred,axis=0)
    
    y_yearly_mean     = [np.mean(y_pred_mean[i-12:i])     for i in np.arange(12,len(y_pred_mean),12)]
    y_yearly_mean_lin = [np.mean(y_pred_lin_mean[i-12:i]) for i in np.arange(12,len(y_pred_lin_mean),12)]
    pr_yearly_mean = [np.mean(pr_pred_mean[i-12:i])     for i in np.arange(12,len(pr_pred_mean),12)]
    output_data ={}
    output_data["tas"] = y_pred
    output_data["pr"] = pr_pred
    output_data["pattern_scaling_tas"] = y_pred_lin
    output_data["tas_yearly"] = y_yearly_mean
    output_data["pr_yearly"] = pr_yearly_mean
    output_data["tas_lin_yearly"] = y_yearly_mean_lin
    output_data["calibration_indices"] = idx
    return output_data


device = "cpu"
path = "/pdrive/projects/icigroup/projects/FastMIP/scenarios/"
scenarios = os.listdir('/pdrive/projects/icigroup/projects/FastMIP/scenarios/')
scenario_index = 5
SCENARIO = scenarios[scenario_index]


files = [path+scenario for scenario in scenarios]
u_fastmip,df_monthly,gmt_cols = get_fastmip_gmt(files[scenario_index])


scaler_path ="/home/kainverena/PythonProjects/outputs_ssm_scales/scales_cnp_20260724_130848/"
model_filename = scaler_path+"checkpoints/cnp_epoch1000.pt" 
zdim = 64
rnn_hidden=256
use_linear_model = True
emission_uses_u =True
alpha_max = 0.002
resevoir_dim = 4



scaler_pickle_y_scaler_filename = scaler_path+"y_scaler.out"
scaler_pickle_u_scaler_filename = scaler_path+"u_scaler.out"
scaler_pickle_pr_scaler_filename = scaler_path+"pr_scaler.out"

y_scaler = StandardScaler.from_file(scaler_pickle_y_scaler_filename)
u_scaler = StandardScaler.from_file(scaler_pickle_u_scaler_filename)
pr_scaler = StandardScaler.from_file(scaler_pickle_pr_scaler_filename)

models = ['MPI-ESM1-2-LR','ACCESS-ESM1-5','CanESM5','MIROC6','IPSL-CM6A-LR']
INDICATORS = ['tas','pr']
TEST_SCENARIOS = ['ssp245']
#TRAIN_SCENARIOS = [ 'ssp585','1pctco2']#,'ssp534-over','flat10cdrincspinoff','ssp126','flat10zecincspinoff', 'flat10cdrincspinoff']#,'abrupt4xco2','ssp119','ssp460','ssp370']

model_data = create_all_model_test_dict(models,INDICATORS,TEST_SCENARIOS)

device = "cpu"
Dy = 58#tas_test.shape[-1]
Du = 1#test_gmt.shape[-1]

# model = DeepSSMPatternConditioned(y_dim=Dy, u_dim=Du, z_dim=zdim,rnn_hidden=rnn_hidden,use_linear_model=use_linear_model,
#                                  emission_uses_u=emission_uses_u).to(device)
model_ssm = DeepSSMPatternConditioned(y_dim=Dy, u_dim=Du, z_dim=zdim,rnn_hidden=rnn_hidden,use_linear_model=use_linear_model,
                                 emission_uses_u=emission_uses_u,reservoir_dim=resevoir_dim,alpha_max=alpha_max).to(device)


model = DeepCnpSsmforESM(ssm_model=model_ssm,r_dim = 128, z_cnp_dim=32)  
model.load_state_dict(torch.load(model_filename, map_location=torch.device(device)))
model.eval()

ds_list = []

for esm_name in models:
    data = project_for_ESM(esm_name=esm_name, esm_member_index=0,
                           gmt_future=u_fastmip, n_ensemble=100,cnp_model=model)
    y_pred   = data["tas"]
    pr_pred  = data["pr"]
    idx      = data["calibration_indices"]

    # ── Time coordinate (cftime handles years beyond 2262) ────────────────────
    H_pred     = y_pred.shape[1]
    time_df    = df_monthly.iloc[:H_pred].reset_index(drop=True)
    time_index = [
        cftime.DatetimeGregorian(int(r["year"]), int(r["month"]), 1)
        for _, r in time_df.iterrows()
    ]

    proj_names = np.array(gmt_cols)[idx]   # [n_realisations]

    ds_esm = xr.Dataset(
        data_vars={
            "tas": (
                ["realisation", "time", "region"],
                y_pred,
                {"long_name": "Near-surface air temperature anomaly", "units": "K"},
            ),
            "pr": (
                ["realisation", "time", "region"],
                pr_pred,
                {"long_name": "Precipitation flux", "units": "kg m-2 s-1"},
            ),
        },
        coords={
            "realisation": ("realisation", proj_names),
            "time":        ("time",        time_index),
            "region":      ("region",      region_names),
            "esm":         esm_name,
        },
        attrs={"scenario": SCENARIO,
               "description": "SCALES monthly emulator output — FastMIP GMT projections"},
    )
    ds_list.append(ds_esm)
    print(f"  {esm_name}: {y_pred.shape}")

ds = xr.concat(ds_list, dim="esm")
print(ds)

#── Save ──────────────────────────────────────────────────────────────────────
out_dir = "/pdrive/projects/icigroup/SCALES-MESH/SCALES/emulator/fastMIP"
os.makedirs(out_dir, exist_ok=True)
out_nc  = f"{out_dir}/{SCENARIO.replace('.csv','')}_tas_pr_monthly_scales.nc"
ds.to_netcdf(out_nc)
print(f"Saved -> {out_nc}")