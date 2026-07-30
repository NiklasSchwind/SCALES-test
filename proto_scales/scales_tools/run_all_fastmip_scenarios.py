import argparse
import os
import cftime
import matplotlib
matplotlib.use("Agg")   # no display needed on cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

import proto_scales.data_prep.prepare_data as prep
from proto_scales.ssm_model.scales_ssm import StandardScaler
from proto_scales.multi_esm_model.scales_cnp import DeepCnpSsmforESM, DeepSSMPatternConditioned

# ── Config ────────────────────────────────────────────────────────────────────
SCALER_PATH    = "/home/kainverena/PythonProjects/outputs_ssm_scales/scales_cnp_20260728_144949/"
MODEL_FILENAME = SCALER_PATH + "checkpoints/cnp_epoch0420.pt"
SCENARIO_DIR   = "/pdrive/projects/icigroup/projects/FastMIP/scenarios/"
OUT_DIR        = "/pdrive/projects/icigroup/SCALES-MESH/SCALES/emulator/fastMIP"
MODELS         = ["MPI-ESM1-2-LR", "ACCESS-ESM1-5", "CanESM5", "MIROC6", "IPSL-CM6A-LR"]
INDICATORS     = ["tas", "pr"]
TEST_SCENARIOS = ["ssp245"]
N_ENSEMBLE     = 841
ZDim           = 64
RNN_HIDDEN     = 256
ALPHA_MAX      = 0.002
RESERVOIR_DIM  = 4

REGION_NAMES = [
    "ARO", "ARP", "ARS", "BOB", "CAF", "CAR", "CAU", "CNA", "EAN", "EAO",
    "EAS", "EAU", "ECA", "EEU", "EIO", "ENA", "EPO", "ESAF", "ESB", "GIC",
    "MDG", "MED", "NAO", "NAU", "NCA", "NEAF", "NEN", "NES", "NEU", "NPO",
    "NSA", "NWN", "NWS", "NZ", "RAR", "RFE", "SAH", "SAM", "SAO", "SAS",
    "SAU", "SCA", "SEA", "SEAF", "SES", "SIO", "SOO", "SPO", "SSA", "SWS",
    "TIB", "WAF", "WAN", "WCA", "WCE", "WNA", "WSAF", "WSB",
]

# ── Helper functions ──────────────────────────────────────────────────────────

def get_fastmip_gmt(filename):
    df = pd.read_csv(filename)
    gmt_cols = [c for c in df.columns if c != "year"]
    df_monthly = pd.DataFrame({
        "year":  np.repeat(df["year"].values, 12),
        "month": np.tile(np.arange(1, 13), len(df)),
        **{col: np.repeat(df[col].values, 12) for col in gmt_cols},
    })
    u_fastmip = df_monthly[gmt_cols].values.T[:, :, np.newaxis]
    return u_fastmip, df_monthly, gmt_cols


def create_all_model_test_dict(model_list):
    model_data = {}
    for model in model_list:
        model_path = f"/projects/icigroup/CMIP6/cmip6-ng-inc-oceans/{model}"
        test_data_np = prep.fetch_and_massage_data(
            model_path=model_path,
            indicators=INDICATORS,
            train_scenarios=TEST_SCENARIOS,
            monthly_flag=True,
            use_smoothing=False,
            train_pattern_scaling_name="ssp585",
            pattern_scaling_residuals=False,
            ramp_down_corrected_ps=False,
        )
        model_data[model] = test_data_np
    return model_data


def prepare_data_for_inference(t_horizon, t_context, raw_data):
    test_data_np = np.array(raw_data)
    test_gmt     = test_data_np[:, 0, :]
    test_data_np = test_data_np[:, 1:, :]
    regions      = int(test_data_np.shape[1] / 2)
    tas_test     = np.transpose(test_data_np[:, :regions, :],  (0, 2, 1))
    pr_test      = np.transpose(test_data_np[:, -regions:, :], (0, 2, 1))
    y_past       = tas_test[:, :t_context, :]
    pr_past      = pr_test[:, :t_context, :]
    u_past       = np.expand_dims(test_gmt[:, :t_context], axis=2)
    u_future     = np.expand_dims(test_gmt[:, t_context:t_context + t_horizon], axis=2)
    return y_past, pr_past, u_past, u_future, tas_test, pr_test


@torch.no_grad()
def forecast_fn(model, y_ctx, u_ctx, pr_ctx, u_fut, n_samples=1):
    B, H, Du = u_fut.shape
    y_ssm, _, _, pr_ssm, _, _ = model.forecast(
        y_ctx, u_ctx, u_fut, steps=H, n_samples=n_samples, pr_ctx=pr_ctx
    )
    y_lin = model.ssm.ctrl_lin(u_fut.reshape(-1, Du)).reshape(B, H, -1)
    return y_ssm, y_lin, pr_ssm


def project_for_ESM(esm_name, esm_member_index, gmt_future, n_ensemble,
                    model, y_scaler, u_scaler, pr_scaler, model_data, device):
    idx    = range(n_ensemble)
    Tc     = 600
    H      = 2100
    y_past, pr_past, u_past, u_future, _, _ = prepare_data_for_inference(
        H, Tc, model_data[esm_name]
    )
    u_future = gmt_future[idx]
    B        = u_future.shape[0]

    y_past_n   = np.repeat(y_scaler.transform(y_past[[esm_member_index]]),   B, axis=0)
    pr_past_n  = np.repeat(pr_scaler.transform(pr_past[[esm_member_index]]), B, axis=0)
    u_past_n   = np.repeat(u_scaler.transform(u_past[[esm_member_index]]),   B, axis=0)
    u_future_n = u_scaler.transform(u_future)

    y_past_n   = torch.tensor(y_past_n,   device=device, dtype=torch.float32)
    u_past_n   = torch.tensor(u_past_n,   device=device, dtype=torch.float32)
    u_future_n = torch.tensor(u_future_n, device=device, dtype=torch.float32)
    pr_past_n  = torch.tensor(pr_past_n,  device=device, dtype=torch.float32)

    y_ssm, y_lin, pr_ssm = forecast_fn(model, y_past_n, u_past_n, pr_past_n, u_future_n)
    y_pred     = y_scaler.inverse_transform(y_ssm.cpu().numpy())
    pr_pred    = pr_scaler.inverse_transform(pr_ssm.cpu().numpy())
    y_pred_lin = y_scaler.inverse_transform(y_lin.cpu().numpy())

    return {
        "tas":                  y_pred,
        "pr":                   pr_pred,
        "pattern_scaling_tas":  y_pred_lin,
        "calibration_indices":  idx,
    }


# ── Pipeline functions ────────────────────────────────────────────────────────

def run_projections_and_save(scenario_index, files, scenarios, model,
                              y_scaler, u_scaler, pr_scaler, model_data, device,
                              n_ensemble=N_ENSEMBLE):
    SCENARIO = scenarios[scenario_index]
    u_fastmip, df_monthly, gmt_cols = get_fastmip_gmt(files[scenario_index])

    ds_list = []
    for esm_name in MODELS:
        data    = project_for_ESM(esm_name, 0, u_fastmip, n_ensemble,
                                  model, y_scaler, u_scaler, pr_scaler, model_data, device)
        y_pred  = data["tas"]
        pr_pred = data["pr"]
        idx     = data["calibration_indices"]

        H_pred     = y_pred.shape[1]
        time_df    = df_monthly.iloc[:H_pred].reset_index(drop=True)
        time_index = [
            cftime.DatetimeGregorian(int(r["year"]), int(r["month"]), 1)
            for _, r in time_df.iterrows()
        ]
        proj_names = np.array(gmt_cols)[idx]

        ds_esm = xr.Dataset(
            data_vars={
                "tas": (["realisation", "time", "region"], y_pred,
                        {"long_name": "Near-surface air temperature anomaly", "units": "K"}),
                "pr":  (["realisation", "time", "region"], pr_pred,
                        {"long_name": "Precipitation flux", "units": "kg m-2 s-1"}),
            },
            coords={
                "realisation": ("realisation", proj_names),
                "time":        ("time",        time_index),
                "region":      ("region",      REGION_NAMES),
                "esm":         esm_name,
            },
            attrs={"scenario": SCENARIO,
                   "description": "SCALES monthly emulator output — FastMIP GMT projections"},
        )
        ds_list.append(ds_esm)
        print(f"  {esm_name}: {y_pred.shape}", flush=True)

    ds = xr.concat(ds_list, dim="esm")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_nc = f"{OUT_DIR}/{SCENARIO.replace('.csv', '')}_tas_pr_monthly_scales.nc"
    ds.to_netcdf(out_nc)
    print(f"Saved -> {out_nc}", flush=True)
    return out_nc


def compute_yearly_stats(out_nc):
    QUANTILES = [0.01, 0.025, 0.05, 0.33, 0.67, 0.95, 0.975, 0.99]

    ds_full   = xr.open_dataset(out_nc, use_cftime=True)
    ds_yearly = ds_full.assign_coords(year=ds_full.time.dt.year).groupby("year").mean("time")

    stats_vars = {}
    for var in ["tas", "pr"]:
        da = ds_yearly[var]
        stats_vars[f"{var}_mean"]   = da.mean("realisation")
        stats_vars[f"{var}_median"] = da.median("realisation")
        for q in QUANTILES:
            qname = f"{var}_q{int(q * 1000):04d}"
            stats_vars[qname] = da.quantile(q, dim="realisation").drop_vars("quantile")

    ds_stats = xr.Dataset(stats_vars)
    ds_stats.attrs = ds_full.attrs
    ds_full.close()

    out_stats_nc = out_nc.replace(".nc", "_yearly_stats.nc")
    ds_stats.to_netcdf(out_stats_nc)
    print(f"Saved -> {out_stats_nc}", flush=True)
    return out_stats_nc


def plot_yearly_stats(out_stats_nc, region_sel="NZ", save_dir=None):
    esm_colors = {
        "ACCESS-ESM1-5":  "#1f77b4",
        "CanESM5":        "#d62728",
        "MPI-ESM1-2-LR":  "#2ca02c",
        "MIROC6":         "#9467bd",
    }
    ds_s  = xr.open_dataset(out_stats_nc, use_cftime=True)
    years = ds_s.year.values

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=150, sharex=True)
    for esm_name, color in esm_colors.items():
        if esm_name not in ds_s.esm.values:
            continue
        sel = dict(esm=esm_name, region=region_sel)
        for var, ax in zip(["tas", "pr"], axes):
            mean   = ds_s[f"{var}_mean"].sel(**sel).values
            median = ds_s[f"{var}_median"].sel(**sel).values
            p05    = ds_s[f"{var}_q0050"].sel(**sel).values
            p95    = ds_s[f"{var}_q0950"].sel(**sel).values
            ax.fill_between(years, p05, p95, alpha=0.15, color=color)
            ax.plot(years, median, linewidth=1.4, linestyle="--", color=color, alpha=0.8)
            ax.plot(years, mean,   linewidth=2.0, color=color, label=esm_name)

    axes[0].set_ylabel("tas anomaly [K]", fontsize=12)
    axes[1].set_ylabel("pr [kg m⁻² s⁻¹]", fontsize=12)
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axes[1].set_xlabel("Year", fontsize=12)
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=11)
    axes[0].set_title(
        f"Yearly statistics  |  region: {region_sel}  |  scenario: {ds_s.attrs['scenario']}\n"
        f"solid=mean  dashed=median  shading=5–95%",
        fontsize=11,
    )
    handles = [plt.Line2D([0], [0], color=c, linewidth=2.5) for c in esm_colors.values()]
    axes[0].legend(handles, list(esm_colors.keys()), fontsize=10, frameon=False, ncol=2)
    plt.tight_layout()

    if save_dir is not None:
        scenario_tag = os.path.basename(out_stats_nc).replace("_yearly_stats.nc", "")
        fig_path = os.path.join(save_dir, f"{scenario_tag}_{region_sel}_yearly_stats.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Plot saved -> {fig_path}", flush=True)
    plt.close(fig)
    ds_s.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SCALES FastMIP projections for all scenarios")
    parser.add_argument("--start", type=int, default=0,  help="First scenario index (inclusive)")
    parser.add_argument("--end",   type=int, default=-1, help="Last scenario index (exclusive, -1 = all)")
    parser.add_argument("--n_ensemble", type=int, default=N_ENSEMBLE, help="Number of GMT realisations")
    parser.add_argument("--plot_region", type=str, default="NZ", help="Region for summary plots")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Load scalers
    y_scaler  = StandardScaler.from_file(SCALER_PATH + "y_scaler.out")
    u_scaler  = StandardScaler.from_file(SCALER_PATH + "u_scaler.out")
    pr_scaler = StandardScaler.from_file(SCALER_PATH + "pr_scaler.out")

    # Build model
    model_ssm = DeepSSMPatternConditioned(
        y_dim=58, u_dim=1, z_dim=ZDim, rnn_hidden=RNN_HIDDEN,
        use_linear_model=True, emission_uses_u=True,
        reservoir_dim=RESERVOIR_DIM, alpha_max=ALPHA_MAX,
    ).to(device)
    model = DeepCnpSsmforESM(ssm_model=model_ssm, r_dim=128, z_cnp_dim=32).to(device)
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
    model.eval()
    print("Model loaded.", flush=True)

    # Load ESM context data once
    model_data = create_all_model_test_dict(MODELS)
    print("ESM data loaded.", flush=True)

    # Scenario file list
    scenarios = sorted(os.listdir(SCENARIO_DIR))
    files     = [os.path.join(SCENARIO_DIR, s) for s in scenarios if scenario.startswith("SSP")]

    end = args.end if args.end >= 0 else len(files)
    plot_dir = OUT_DIR  # save plots alongside netcdf files

    for scenario_index in range(args.start, end):
        print(f"\n=== Scenario {scenario_index}/{end - 1}: {scenarios[scenario_index]} ===", flush=True)
        out_nc       = run_projections_and_save(
            scenario_index, files, scenarios, model,
            y_scaler, u_scaler, pr_scaler, model_data, device,
            n_ensemble=args.n_ensemble,
        )
        out_stats_nc = compute_yearly_stats(out_nc)
        plot_yearly_stats(out_stats_nc, region_sel=args.plot_region, save_dir=plot_dir)

    print("\nAll scenarios done.", flush=True)
