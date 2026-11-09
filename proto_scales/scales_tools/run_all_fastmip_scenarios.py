import argparse
import json
import os
import cftime
import matplotlib
matplotlib.use("Agg")   # no display needed on cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
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


def select_gmt_cols_from_subselection(gmt_cols, key,
                                      csv_path="downloads/subselection_forced_v20260325.csv"):
    """Return the entries of gmt_cols whose indices are listed under `key` in the
    subselection CSV (column A is a row index and is ignored; columns B-H are the keys)."""
    df = pd.read_csv(csv_path, index_col=0)
    subselection = {col: df[col].dropna().astype(int).tolist() for col in df.columns}
    if key not in subselection:
        raise KeyError(f"'{key}' not in subselection keys: {list(subselection)}")
    gmt_cols = np.asarray(gmt_cols)
    return gmt_cols[subselection[key]].tolist()


def filter_monthly_projections_and_save(out_nc, start_year, end_year, proj_names):
    """Resave the monthly projections restricted to [start_year, end_year] (inclusive)
    and to the realisations listed in proj_names."""
    ds_full = xr.open_dataset(out_nc, use_cftime=True)

    years = ds_full.time.dt.year
    ds_sel = ds_full.sel(
        time=(years >= start_year) & (years <= end_year),
        realisation=list(proj_names),
    ).load()
    ds_sel.attrs = ds_full.attrs
    ds_full.close()

    out_filtered_nc = out_nc.replace(".nc", f"_{start_year}_{end_year}_filtered.nc")
    ds_sel.to_netcdf(out_filtered_nc)
    print(f"Saved -> {out_filtered_nc}", flush=True)
    return out_filtered_nc


def baseline_correct_and_save(out_filtered_nc, baseline_dir, variables=("tas", "pr")):
    """Shift each variable/region of the filtered monthly projections by the historical
    'mean' found in <baseline_dir>/<esm lower case>_<var>_historical.json."""
    ds = xr.open_dataset(out_filtered_nc, use_cftime=True).load()

    for var in variables:
        offsets = []
        for esm_name in ds.esm.values:
            json_path = os.path.join(baseline_dir, f"{str(esm_name).lower()}_{var}_historical.json")
            with open(json_path) as f:
                baseline = json.load(f)
            offsets.append([baseline[str(r)]["mean"] for r in ds.region.values])
        offset = xr.DataArray(offsets, dims=["esm", "region"],
                              coords={"esm": ds.esm, "region": ds.region})
        attrs = ds[var].attrs
        ds[var] = ds[var] + offset
        ds[var].attrs = attrs

    ds.attrs["baseline_correction"] = f"historical mean added from {baseline_dir}"
    out_bc_nc = out_filtered_nc.replace(".nc", "_baseline_corrected.nc")
    ds.to_netcdf(out_bc_nc)
    print(f"Saved -> {out_bc_nc}", flush=True)
    return out_bc_nc


def baseline_corrected_to_numpy(out_bc_nc, esm=None, variables=("tas", "pr")):
    """Return the monthly data of one ESM as an array of shape
    (N months, n realisations, C channels, R regions), with regions ordered like
    regionmask.defined_regions.ar6.all.

    esm may be omitted only if the file holds exactly one ESM."""
    regions_xr = regionmask.defined_regions.ar6.all

    ds = xr.open_dataset(out_bc_nc, use_cftime=True)
    if esm is not None:
        ds = ds.sel(esm=esm)
    elif ds.sizes["esm"] == 1:
        ds = ds.isel(esm=0)
    else:
        raise ValueError(f"File has {ds.sizes['esm']} 'esm' entries; pass esm=... to select one")

    # dataset regions are the AR6 abbreviations; ar6.all.abbrevs has the same order as .names
    ds = ds.sel(region=list(regions_xr.abbrevs))
    arr = np.stack(
        [ds[var].transpose("time", "realisation", "region").values for var in variables], axis=2
    )
    ds.close()
    return arr


def plot_yearly_stats(out_stats_nc, region_sel="EAS", save_dir=None):
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


# ── Multi-scenario peak-warming analysis ─────────────────────────────────────

SCENARIO_COLORS = {
    "VL": "#1a1aa6", "LN": "#33d6e0", "L": "#1f9e5a", "ML": "#e6c229",
    "M": "#f07f13", "HL": "#f03cf0", "H": "#b01c1c",
}


_AR6_AREA_WEIGHTS = None


def _ar6_area_weights(res=0.5):
    """Relative surface area of every AR6 region ({abbrev: weight}), from a
    cos(lat)-weighted regular grid. Cached after the first call."""
    global _AR6_AREA_WEIGHTS
    if _AR6_AREA_WEIGHTS is None:
        ar6 = regionmask.defined_regions.ar6.all
        lon = np.arange(-180 + res / 2, 180, res)
        lat = np.arange(-90 + res / 2, 90, res)
        mask = ar6.mask(lon, lat).values
        cell = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], mask.shape)
        _AR6_AREA_WEIGHTS = {
            r.abbrev: float(cell[mask == r.number].sum()) for r in ar6
        }
    return _AR6_AREA_WEIGHTS


def _resolve_region(region, available):
    """
    Map the `region` argument to a list of region abbrevs.

    - an AR6 abbrev (e.g. "EAS"), or any other entry present in the file
      (e.g. a synthetic "GLOBAL" mask entry): that single region;
    - "GLOBAL" (when the file has no such entry): all AR6 regions in the file;
    - "LAND": all AR6 land regions in the file.
    """
    key = region.upper()
    if region in available:
        return [region]
    if key == "GLOBAL":
        regions = [r for r in regionmask.defined_regions.ar6.all.abbrevs if r in available]
    elif key == "LAND":
        regions = [r for r in regionmask.defined_regions.ar6.land.abbrevs if r in available]
    else:
        raise KeyError(f"Region '{region}' not found; use one of {available}, 'GLOBAL' or 'LAND'")
    if not regions:
        raise KeyError(f"No AR6 regions for '{region}' found in {available}")
    return regions


def _load_yearly_realisations(monthly_nc, region="GLOBAL", esm=None, var="tas",
                              area_weighted=True, batch_size=100):
    """
    Read a monthly projection file written by `run_projections_and_save`
    (dims esm, realisation, time, region) and reduce it to annual means,
    one series per (ESM, realisation).

    `region` is a single region spec or a list of them. A spec is a region
    abbrev, "GLOBAL" (all AR6 regions in the file) or "LAND" (AR6 land
    regions), see `_resolve_region`; averages are area-weighted unless
    `area_weighted=False`. `esm=None` pools the realisations of all ESMs; a
    string (or list) selects ESM(s).

    Returns (years, data) with data of shape (n_series, n_years) for a single
    spec, or (years, {spec: data}) for a list. The file is read once, in
    batches of realisations, so many regions can be extracted cheaply.
    """
    single = isinstance(region, str)
    specs = [region] if single else list(region)

    ds = xr.open_dataset(monthly_nc, use_cftime=True)
    try:
        available = [str(r) for r in ds["region"].values]
        plan = {}
        for spec in specs:
            regs = _resolve_region(spec, available)
            w = np.array([_ar6_area_weights()[r] if (area_weighted and len(regs) > 1) else 1.0
                          for r in regs])
            plan[spec] = ([available.index(r) for r in regs], w / w.sum())

        da = ds[var]
        if "esm" not in da.dims:
            da = da.expand_dims("esm")
        if esm is not None:
            da = da.sel(esm=[esm] if isinstance(esm, str) else list(esm))
        da = da.transpose("esm", "realisation", "time", "region")

        years_t = np.asarray(ds["time"].dt.year.values, dtype=int)
        if np.any(np.diff(years_t) < 0):
            raise ValueError(f"Time axis of {monthly_nc} is not sorted")
        years, starts, counts = np.unique(years_t, return_index=True, return_counts=True)

        out = {spec: [] for spec in specs}
        for e in range(da.sizes["esm"]):
            for r0 in range(0, da.sizes["realisation"], batch_size):
                block = da.isel(esm=e, realisation=slice(r0, r0 + batch_size)).values
                yearly = (np.add.reduceat(block, starts, axis=1, dtype=np.float64)
                          / counts[None, :, None])                       # (b, year, region)
                for spec, (idx, w) in plan.items():
                    out[spec].append(yearly[:, :, idx] @ w)
    finally:
        ds.close()

    data = {spec: np.concatenate(v, axis=0) for spec, v in out.items()}
    return (years, data[region]) if single else (years, data)


def _trend_per_decade(years, data, y0, y1):
    """Least-squares linear trend (per decade) of every row of `data` over [y0, y1]."""
    sel = (years >= y0) & (years <= y1)
    if sel.sum() < 2:
        return np.full(data.shape[0], np.nan)
    x = years[sel] - years[sel].mean()
    return 10 * (data[:, sel] @ x) / (x @ x)


def _quantile_levels(bands):
    return sorted({0.5, *[q for pair in bands for q in pair]})


def _scenario_stats(years, data, qs, smooth_years, start_year, split_year,
                    rate_windows, rate_change_windows):
    """Per-realisation derived quantities of one scenario/region, summarised as
    {quantile: value} dicts (see `plot_peak_warming_analysis`)."""
    def _quantiles(x, axis=0):
        vals = np.nanquantile(x, qs, axis=axis)
        return {q: v for q, v in zip(qs, vals)}

    if smooth_years > 1:
        smooth = (pd.DataFrame(data.T)
                  .rolling(smooth_years, center=True, min_periods=1).mean().values.T)
    else:
        smooth = data

    def _at(year):
        idx = np.where(years == year)[0]
        return smooth[:, idx[0]] if len(idx) else np.full(data.shape[0], np.nan)

    (r0y0, r0y1), (r1y0, r1y1) = rate_change_windows
    rate_ref = _trend_per_decade(years, smooth, r0y0, r0y1)
    rate_tgt = _trend_per_decade(years, smooth, r1y0, r1y1)
    valid = rate_ref > 0
    rate_change = np.where(valid, 100 * (rate_tgt - rate_ref) / np.where(valid, rate_ref, 1), np.nan)

    after = years >= start_year
    return dict(
        years=years,
        q=_quantiles(data, axis=0),                              # trajectories
        peak=_quantiles(np.nanmax(smooth[:, after], axis=1)),
        peak_year=_quantiles(years[after][np.nanargmax(smooth[:, after], axis=1)].astype(float)),
        split=_quantiles(_at(split_year)),
        end=_quantiles(smooth[:, -1]),
        rates=[_quantiles(_trend_per_decade(years, smooth, y0, y1)) for y0, y1 in rate_windows],
        rate_change=_quantiles(rate_change) if valid.any() else {q: np.nan for q in qs},
        rate_change_excluded=int((~valid).sum()), n_series=data.shape[0],
    )


def _compute_peak_warming_stats(monthly_files, scenario_names, regions, esm, var,
                                area_weighted, bands, smooth_years, start_year,
                                split_year, rate_windows, rate_change_windows):
    """
    Read every scenario file once and return {region: [scenario stats, ...]}
    (scenarios in the order of `monthly_files`). Only the small per-region
    summaries are kept in memory, never more than one file's yearly data.
    """
    qs = _quantile_levels(bands)
    stats = {r: [] for r in regions}
    for path in monthly_files:
        name = scenario_names.get(path, scenario_names.get(os.path.basename(path)))
        if name is None:
            raise KeyError(f"No scenario name given for {path}")
        years, data = _load_yearly_realisations(path, region=list(regions), esm=esm, var=var,
                                                area_weighted=area_weighted)
        for r in regions:
            s = _scenario_stats(years, data[r], qs, smooth_years, start_year, split_year,
                                rate_windows, rate_change_windows)
            s["name"] = name
            stats[r].append(s)
        print(f"  {name}: {next(iter(data.values())).shape[0]} series, "
              f"{years[0]}-{years[-1]}, {len(regions)} region(s)", flush=True)
        del data
    return stats


def plot_peak_warming_analysis(monthly_files, scenario_names, region="GLOBAL", esm=None,
                               var="tas", bands=((0.05, 0.95), (0.17, 0.83), (0.33, 0.67)),
                               rate_windows=((2016, 2025), (2026, 2035), (2031, 2040),
                                             (2036, 2045), (2041, 2050)),
                               split_year=2100, high_scenarios=None, observed=None,
                               colors=None, area_weighted=True, smooth_years=1,
                               start_year=2000, rate_change_windows=((2016, 2025), (2036, 2045)),
                               rate_change_focus="VL", only_rate_change=False,
                               save_path=None, scenario_stats=None):
    """
    Multi-scenario summary figure (warming trajectories, peak year, rate of
    warming, peak warming, warming in `split_year` / at the final year and the
    percentage change of the warming rate between two windows).

    Parameters
    ----------
    monthly_files : list of str
        Monthly projection netCDF files written by `run_projections_and_save`
        (one per scenario). Plot order follows this list.
    scenario_names : dict
        {filename: short scenario name}, e.g.
        {"SSP1---Very-Low-Emissions_..._tas_pr_monthly_scales.nc": "VL"}.
        Keys may be full paths or basenames.
    region : str
        What to analyse:
        - a single region abbrev, e.g. "EAS";
        - "GLOBAL": average over all AR6 regions in the file (land + ocean);
        - "LAND":   average over the 46 AR6 land regions.
    area_weighted : bool
        Weight the "GLOBAL"/"LAND" averages by region area (default) instead
        of giving every region the same weight.
    esm : str, list of str or None
        ESM(s) to use; None pools the realisations of all ESMs.
    bands : sequence of (lo, hi) quantile pairs
        Uncertainty boxes / shading, outermost first (the outermost band is
        the shading in panels a, b).
    smooth_years : int
        Width of a centred running mean (in years) applied to every annual
        series before computing peak year, peak warming and the warming in
        `split_year` / the final year, and the warming rates (panels e, i).
        1 = no smoothing; the per-realisation values then include the full
        interannual variability, which dominates 10-year regional trends.
        The trajectories in panels a, b always use the unsmoothed annual means.
    start_year : int
        First year shown in panels a and c; peaks are searched from this year on.
    rate_change_windows : ((y0, y1), (y0, y1))
        Reference and target window for panel i, the percentage change of the
        warming rate, 100 * (rate_target - rate_ref) / rate_ref, computed per
        realisation. Realisations with rate_ref <= 0 are excluded (their
        count is printed under the box).
    rate_change_focus : str or None
        Scenario highlighted in panel i (others are drawn faded); None = no highlight.
    only_rate_change : bool
        If True, only panel i is plotted (as a small stand-alone figure).
    split_year : int
        Boundary between the near-term (a, c) and long-term (b, d) panels,
        and the year shown in panel g.
    high_scenarios : list of str or None
        Short names shown on the separate right-hand y-axis in panels f-h.
        None: scenarios whose median peak exceeds twice the smallest median peak.
    observed : dict or None
        Optional {label: (years, values)} reference series drawn in panel a
        (first one solid with markers, others dashed, all in black).
    colors : dict or None
        {short name: color}; defaults to SCENARIO_COLORS, then the tab10 cycle.
    save_path : str or None
        If given, the figure is saved there.
    scenario_stats : list or None
        Pre-computed statistics for this region (used internally by
        `plot_peak_warming_analysis_by_region`); if given, no files are read.

    Notes
    -----
    Every (ESM, realisation) series is first reduced to annual means of the
    (area-weighted) regional average; peak year, peak warming, warming rates
    etc. are computed per series and the quantiles are taken across series.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    if scenario_stats is None:
        scenario_stats = _compute_peak_warming_stats(
            monthly_files, scenario_names, [region], esm, var, area_weighted, bands,
            smooth_years, start_year, split_year, rate_windows, rate_change_windows)[region]

    colors = {**SCENARIO_COLORS, **(colors or {})}
    fallback = iter(plt.cm.tab10.colors)
    scen = [dict(s, color=colors[s["name"]] if s["name"] in colors else next(fallback))
            for s in scenario_stats]
    band_pairs = list(bands)
    alphas = np.linspace(0.3, 0.75, len(band_pairs))
    band_txt = ", ".join(f"{int(lo*100)}–{int(hi*100)}%" for lo, hi in band_pairs)
    esm_txt = esm if esm is not None else "all ESMs pooled"
    footer = f"Region: {region}  |  ESM: {esm_txt}  |  boxes: {band_txt}, line = median"

    # ── Drawing helpers ─────────────────────────────────────────────────────
    def _box(ax, x, stats, color, width=0.6, horizontal=False, alpha_scale=1.0):
        for (qlo, qhi), a in zip(band_pairs, alphas * alpha_scale):
            lo, hi = sorted((stats[qlo], stats[qhi]))
            if np.isnan(lo) or np.isnan(hi):
                continue
            if horizontal:
                ax.barh(x, hi - lo, left=lo, height=width, color=color, alpha=a,
                        edgecolor="k", linewidth=0.6)
            else:
                ax.bar(x, hi - lo, bottom=lo, width=width, color=color, alpha=a,
                       edgecolor="k", linewidth=0.4)
        med = stats[0.5]
        if horizontal:
            ax.vlines(med, x - width / 2, x + width / 2, color="k", linewidth=1.0)
        else:
            if not np.isnan(med):
                ax.hlines(med, x - width / 2, x + width / 2, color="w", linewidth=1.5)

    def _style(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.3)

    def _draw_rate_change(ax, title_fontsize=9, title_prefix=""):
        (r0y0, r0y1), (r1y0, r1y1) = rate_change_windows
        for i, s in enumerate(scen):
            focus = rate_change_focus is None or s["name"] == rate_change_focus
            _box(ax, i, s["rate_change"], s["color"], width=0.7,
                 alpha_scale=1.0 if focus else 0.4)
            if s["rate_change_excluded"]:
                ax.annotate(f"{s['rate_change_excluded']}/{s['n_series']}\nexcl.",
                            (i, 0), xycoords=("data", "axes fraction"), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=5.5, color="0.4")
            if rate_change_focus is not None and s["name"] == rate_change_focus \
                    and not np.isnan(s["rate_change"][0.5]):
                ax.annotate(f"{s['rate_change'][0.5]:+.0f}%", (i + 0.35, s["rate_change"][0.5]),
                            xytext=(2, 0), textcoords="offset points", va="center",
                            fontsize=8, fontweight="bold", color=s["color"])
        ax.set_xticks(range(len(scen)))
        ax.set_xticklabels([s["name"] for s in scen], fontsize=8)
        ax.set_xlim(-0.6, len(scen) - 0.4)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_ylabel("Change in rate of warming (%)")
        ax.set_title(f"{title_prefix}{r1y0}–{r1y1} vs\n{r0y0}–{r0y1}", fontsize=title_fontsize, loc="left")
        _style(ax)

    def _save(fig):
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved -> {save_path}", flush=True)
        return fig

    # ── Panel i only ────────────────────────────────────────────────────────
    if only_rate_change:
        fig, ax_i = plt.subplots(figsize=(4.5, 4.5), dpi=150)
        _draw_rate_change(ax_i, title_fontsize=10, title_prefix=f"{region}: ")
        fig.text(0.5, -0.02, footer, ha="center", va="top", fontsize=6.5, color="0.4")
        return _save(fig)

    end_year = int(min(s["years"].max() for s in scen))
    if high_scenarios is None:
        min_peak = min(s["peak"][0.5] for s in scen)
        high_scenarios = [s["name"] for s in scen if s["peak"][0.5] > 2 * max(min_peak, 1e-6)]
    low = [s for s in scen if s["name"] not in high_scenarios]
    high = [s for s in scen if s["name"] in high_scenarios]

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 15), dpi=150)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[3, 1.8, 2.4, 2.4],
                  hspace=0.35, wspace=0.12, top=0.94, bottom=0.05)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_c)
    gs_mid = gs[2, :].subgridspec(1, 2, width_ratios=[4, 1.1], wspace=0.22)
    ax_e = fig.add_subplot(gs_mid[0, 0])
    ax_i = fig.add_subplot(gs_mid[0, 1])
    gs_bot = gs[3, :].subgridspec(1, 3, wspace=0.3)

    # a, b — trajectories
    qlo, qhi = band_pairs[0]
    for ax, (x0, x1) in ((ax_a, (start_year, split_year)), (ax_b, (split_year, end_year))):
        for s in scen:
            sel = (s["years"] >= x0) & (s["years"] <= x1)
            ax.fill_between(s["years"][sel], s["q"][qlo][sel], s["q"][qhi][sel],
                            color=s["color"], alpha=0.18, linewidth=0)
            ax.plot(s["years"][sel], s["q"][0.5][sel], color=s["color"], linewidth=1.6)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_xlim(x0, x1)
        _style(ax)
    if observed:
        for i, (label, (oy, ov)) in enumerate(observed.items()):
            oy, ov = np.asarray(oy), np.asarray(ov)
            sel = oy >= start_year
            ax_a.plot(oy[sel], ov[sel], color="k", linewidth=1.0,
                      linestyle="-" if i == 0 else "--",
                      marker="o" if i == 0 else None, markersize=2, label=label)
        ax_a.legend(loc="upper left", fontsize=8, frameon=False)
    ax_a.set_ylabel(f"Warming ({var}, {region}) [K]")

    # c, d — peak year
    ypos = {s["name"]: i for i, s in enumerate(scen)}
    for ax, (x0, x1) in ((ax_c, (start_year, split_year)), (ax_d, (split_year, end_year))):
        for s in scen:
            _box(ax, ypos[s["name"]], s["peak_year"], s["color"], width=0.6, horizontal=True)
        ax.set_xlim(x0, x1)
        ax.set_xlabel("Year")
        ax.grid(True, axis="x", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_c.set_yticks(range(len(scen)))
    ax_c.set_yticklabels([s["name"] for s in scen])
    ax_c.invert_yaxis()
    ax_c.set_ylabel("Peak year")
    plt.setp(ax_d.get_yticklabels(), visible=False)

    # e — rate of warming
    n = len(scen)
    group_w = n + 1.5
    for w, (y0, y1) in enumerate(rate_windows):
        for i, s in enumerate(scen):
            _box(ax_e, w * group_w + i, s["rates"][w], s["color"], width=0.75)
        if w:
            ax_e.axvline(w * group_w - 1.25, color="0.8", linewidth=0.6)
    ax_e.set_xticks([w * group_w + (n - 1) / 2 for w in range(len(rate_windows))])
    ax_e.set_xticklabels([f"{y0}–{y1}" for y0, y1 in rate_windows])
    ax_e.set_xlim(-1, len(rate_windows) * group_w - 1.5)
    ax_e.axhline(0, color="k", linewidth=0.6)
    ax_e.set_ylabel("Rate of warming (K / decade)")
    _style(ax_e)

    # i — percentage change of the warming rate between two windows
    _draw_rate_change(ax_i)

    # f, g, h — peak warming, warming in split_year, warming at end
    bottom = [("peak", f"Peak warming (to {end_year})"),
              ("split", f"Warming in {split_year}"),
              ("end", f"Warming in {end_year}")]
    bottom_axes = []
    for k, (key, title) in enumerate(bottom):
        groups = [g for g in (low, high) if g]
        sub = gs_bot[0, k].subgridspec(1, len(groups), wspace=0.35,
                                       width_ratios=[len(g) + 0.5 for g in groups])
        for j, group in enumerate(groups):
            ax = fig.add_subplot(sub[0, j])
            for i, s in enumerate(group):
                _box(ax, i, s[key], s["color"], width=0.7)
            ax.set_xticks(range(len(group)))
            ax.set_xticklabels([s["name"] for s in group])
            ax.set_xlim(-0.6, len(group) - 0.4)
            ax.axhline(0, color="k", linewidth=0.6)
            _style(ax)
            if j == 0:
                ax.set_title(title, fontsize=10, loc="left")
            bottom_axes.append(ax)
            if k == 0 and j == 0:
                ax.set_ylabel("Warming (K)")

    # Panel letters
    letter_axes = [ax_a, ax_b, ax_c, ax_d, ax_e] + \
        [bottom_axes[i] for i in range(0, len(bottom_axes), max(1, len(bottom_axes) // 3))]
    letter_axes.append(ax_i)
    for letter, ax in zip("abcdefghi", letter_axes):
        ax.text(-0.02, 1.02, f"{letter})", transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="right", va="bottom")

    # Scenario legend
    handles = [Line2D([0], [0], color=s["color"], linewidth=2.5) for s in scen]
    fig.legend(handles, [s["name"] for s in scen], loc="upper center", ncol=len(scen),
               frameon=False, title="Scenario", bbox_to_anchor=(0.5, 0.995))
    fig.text(0.5, 0.01, footer, ha="center", fontsize=8, color="0.4")
    return _save(fig)


def plot_peak_warming_analysis_by_region(monthly_files, scenario_names, out_dir,
                                         regions=None, file_prefix="peak_warming",
                                         file_format="png", **kwargs):
    """
    Run `plot_peak_warming_analysis` for every region in `regions` (default:
    the 46 AR6 land regions) and save each figure to
    <out_dir>/<file_prefix>_<region>.<file_format>
    (with "_rate_change" appended when only_rate_change=True).

    All other keyword arguments (esm, smooth_years, only_rate_change, ...) are
    passed on to `plot_peak_warming_analysis`. Each scenario file is read only
    once for all regions. Returns {region: saved path}.
    """
    import inspect

    if regions is None:
        regions = list(regionmask.defined_regions.ar6.land.abbrevs)
    for bad in ("region", "save_path", "scenario_stats"):
        if bad in kwargs:
            raise TypeError(f"'{bad}' is set per region and cannot be passed")

    defaults = {k: p.default for k, p in
                inspect.signature(plot_peak_warming_analysis).parameters.items()
                if p.default is not inspect.Parameter.empty}
    opts = {**defaults, **kwargs}
    stats = _compute_peak_warming_stats(
        monthly_files, scenario_names, regions, opts["esm"], opts["var"],
        opts["area_weighted"], opts["bands"], opts["smooth_years"], opts["start_year"],
        opts["split_year"], opts["rate_windows"], opts["rate_change_windows"])

    os.makedirs(out_dir, exist_ok=True)
    suffix = "_rate_change" if opts["only_rate_change"] else ""
    saved = {}
    for region in regions:
        path = os.path.join(out_dir, f"{file_prefix}_{region}{suffix}.{file_format}")
        fig = plot_peak_warming_analysis(monthly_files, scenario_names, region=region,
                                         save_path=path, scenario_stats=stats[region], **kwargs)
        plt.close(fig)
        saved[region] = path
    return saved


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
    files     = [os.path.join(SCENARIO_DIR, s) for s in scenarios if s.startswith("SSP")]

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
