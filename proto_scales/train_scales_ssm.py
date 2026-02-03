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
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys

MODEL = 'ACCESS-ESM1-5'
INDICATOR = 'tas'
TEST_SCENARIOS = ['flat10cdrincspinoff']
TRAIN_SCENARIOS = [ 'ssp534-over','flat10cdrincspinoff','ssp585','1pctco2','ssp245']#,'ssp126','flat10zecincspinoff', 'flat10cdrincspinoff']#,'abrupt4xco2','ssp119','ssp460','ssp370']
N = 200
ML_MODEL = 'feed_forward' 
PATTERN_SCALING_RESIDUALS = False
RAMP_DOWN_CORRECTED_PS = False

MODEL_PATH = f'/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{MODEL}'


potential_files = prep.get_all_files_(MODEL_PATH)

train_files = prep.filter_climate_files(files = potential_files, scenarios = TRAIN_SCENARIOS, indicators = [INDICATOR])
test_files = prep.filter_climate_files(files = potential_files, scenarios = TEST_SCENARIOS, indicators = [INDICATOR])

train_files_with_baseline = [(prep.get_baseline_filename(filename=filename, files= potential_files), filename) for filename in train_files]
test_files_with_baseline = [(prep.get_baseline_filename(filename=filename, files= potential_files), filename) for filename in test_files]

for (base,exp) in test_files_with_baseline: 
    print(base, exp)

train_data_df = [process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = False, smoothed = True) for (baseline,experiment) in train_files_with_baseline]
test_data_df = [process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = False, smoothed = True) for (baseline,experiment) in test_files_with_baseline]

train_data_df = [process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = False, smoothed = True) for (baseline,experiment) in train_files_with_baseline]
test_data_df = [process_scenarios(experiment_scenario_path = f'{MODEL_PATH}/{experiment}', simulation_name = experiment, baseline_scenario_path = f'{MODEL_PATH}/{baseline}', delete_first_years = 0, monthly_trend = False, smoothed = True) for (baseline,experiment) in test_files_with_baseline]

if PATTERN_SCALING_RESIDUALS:
    flat10cdr_index = [i for i, f in enumerate(train_files) if 'flat10cdrincspinoff' in f][0]
    regional_regression_slopes_intersepts = prep.process_gmt_and_regions_into_array(train_data_df[flat10cdr_index], weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS)
    train_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS, slope_intercept = regional_regression_slopes_intersepts) for GMT_regional_values_tuple in train_data_df]
    test_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False, pattern_scaling_residuals=PATTERN_SCALING_RESIDUALS, slope_intercept = regional_regression_slopes_intersepts) for GMT_regional_values_tuple in test_data_df]
else: 
    flat10cdr_index = [i for i, f in enumerate(train_files) if 'flat10cdrincspinoff' in f][0]
    regional_regression_slopes_intersepts = prep.process_gmt_and_regions_into_array(train_data_df[flat10cdr_index], weighted_linear_smoothing = False, pattern_scaling_residuals=True)
    train_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False) for GMT_regional_values_tuple in train_data_df]
    test_data_np = [prep.process_gmt_and_regions_into_array(GMT_regional_values_tuple, weighted_linear_smoothing = False) for GMT_regional_values_tuple in test_data_df]

train_data_input, train_data_output = prep.prepare_all_train_data(train_data_np, n=N)
test_data_input, test_data_output = prep.prepare_all_train_data(test_data_np, n=N)

train_data_shuffeled = prep.shuffle_train_data(train_data_input, train_data_output, random_state=42)

test_data_for_autoregression_input, test_data_for_autoregression_output = prep.prepare_train_data(test_data_np[0],N)

gmt_autoregressive_test = test_data_np[0][0,:]
autoregressive_test_groudtruth = test_data_np[0][1:,:]

gmt_train = train_data_np[0][0,:]
train_data_regional_temperature = train_data_np[0][1:,:]




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
            kl_w = 5
            alpha = 1000
         
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
    #y = np.transpose(train_data_shuffeled[0][44:48,:],(2, 1, 0))
    y = np.transpose(train_data_shuffeled[0],(2, 1, 0))
    print(u.shape)
    print(y.shape)

    device = "cuda"

    model, y_scaler, u_scaler = run_train(y, u, context_len=50, z_dim=24,horizon=80, device=device,epochs=10)
    os.makedirs("outputs_ssm_scales", exist_ok=True)
    torch.save(model.state_dict(),"outputs_ssm_scales/model_out")
    y_scaler.save("outputs_ssm_scales/y_scaler.out")
    u_scaler.save("outputs_ssm_scales/u_scaler.out")
    
