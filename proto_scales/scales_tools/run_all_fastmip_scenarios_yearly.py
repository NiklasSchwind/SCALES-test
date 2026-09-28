import argparse
import os
import re

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")   # no display needed on cluster
import matplotlib.pyplot as plt
import numpy as np
import regionmask
import torch
import xarray as xr

from proto_scales.scales_tools.run_all_fastmip_scenarios import (
    ALPHA_MAX,
    MODEL_FILENAME,
    MODELS,
    N_ENSEMBLE,
    OUT_DIR,
    RESERVOIR_DIM,
    RNN_HIDDEN,
    SCALER_PATH,
    SCENARIO_DIR,
    ZDim,
    create_all_model_test_dict,
    run_projections_and_save,
)
from proto_scales.ssm_model.scales_ssm import StandardScaler
from proto_scales.multi_esm_model.scales_cnp import DeepCnpSsmforESM, DeepSSMPatternConditioned

ALT_NAME_ABBREV = {
    "Very-Low-Emissions":  "VL",
    "Low-Emissions":       "L",
    "Low-Overshoot":       "LO",
    "Medium-Emissions":    "M",
    "Medium-Low-Emissions": "ML",
    "High-Emissions":      "H",
}

QUANTILES = [0.01, 0.025, 0.05, 0.33, 0.5, 0.67, 0.95, 0.975, 0.99]

# The "mask" coordinate in the yearly-stats files holds AR6 region abbrevs
# (plus a synthetic "GLOBAL" entry with no polygon of its own, produced
# alongside the real AR6 regions elsewhere in the pipeline; see
# create_mask.py). `ar6.all` is the same region set + abbrev convention
# used there.
AR6_REGIONS = regionmask.defined_regions.ar6.all

def _parse_scenario_tag(out_nc):
    """
    Derive the short scenario tag used in output filenames (e.g. "SSP5-ML")
    from an input filename of the form "...SSP<n>---<Alternative Name>_tas_pr_monthly_scales.nc".
    """
    basename = os.path.basename(out_nc)
    print(basename)
    scenario_raw = basename.replace("_fair_GMT_data_inc_variability_tas_pr_monthly_scales", "").replace(".nc", "")
    scenario_raw = scenario_raw.replace("_fair_GMT_data_tas_pr_monthly_scales", "").replace(".nc", "").replace("_a","")
    print("After stripping ", scenario_raw)

    match = re.match(r"^(SSP[^-]+)---(.+)$", scenario_raw)
    print(match)
    # if not match:
    #     raise ValueError(f"Could not parse '<SSP>---<Alternative Name>' from filename: {basename}")
    ssp_part, alt_name_raw = match.groups()
    print(ssp_part,alt_name_raw)

    alt_key = alt_name_raw#alt_name_raw.replace("_", "-").lower()
    abbrev = ALT_NAME_ABBREV.get(alt_key)
    if abbrev is None:
        raise ValueError(f"Unknown scenario alternative name '{alt_name_raw}' (from {basename})")

    return f"{ssp_part}-{abbrev}"


def compute_yearly_stats_for_fastmip(out_nc):
    """
    Aggregate the monthly (esm, realisation, time, region) SCALES output written
    by `run_projections_and_save` into yearly per-ESM statistics, one netCDF
    file per variable (tas, pr), laid out as (quantile, esm_calibration, year,
    mask) — the same convention used for the monthly regional_quantiles-by-ESM
    products, but with "month" dropped since the values are already annual
    means.
    """
    scenario_tag = _parse_scenario_tag(out_nc)

    ds_full = xr.open_dataset(out_nc, use_cftime=True)
    ds_yearly = ds_full.assign_coords(year=ds_full.time.dt.year).groupby("year").mean("time")

    out_dir = os.path.dirname(out_nc)
    out_paths = {}
    for var in ["tas", "pr"]:
        da = ds_yearly[var]

        da_mean = da.mean("realisation").rename({"esm": "esm_calibration", "region": "mask"})
        da_std = da.std("realisation").rename({"esm": "esm_calibration", "region": "mask"})
        da_q = da.quantile(QUANTILES, dim="realisation").rename({"esm": "esm_calibration", "region": "mask"})

        da_mean = da_mean.transpose("esm_calibration", "year", "mask")
        da_std = da_std.transpose("esm_calibration", "year", "mask")
        da_q = da_q.transpose("quantile", "esm_calibration", "year", "mask")

        ds_stats = xr.Dataset(
            data_vars={
                var: da_q,
                f"{var}_mean": da_mean,
                f"{var}_std": da_std,
            },
            attrs={"scenario": scenario_tag,
                   "description": "SCALES yearly trends emulator output — FastMIP GMT projections"},
        )
        ds_stats["year"] = ds_stats["year"].astype("int64")
        #ds_stats.attrs = ds_full.attrs

        out_filename = f"/{var}/{var}_{scenario_tag}_SCALES_regional_quantiles-by-ESM.nc"
        out_path = os.path.join(out_dir, out_filename)
        #print("before writing", out_path,out_dir)
        ds_stats.to_netcdf(out_dir+out_path)
        out_paths[var] = out_path
        print(f"Saved -> {out_path}", flush=True)

    ds_full.close()
    return out_paths["tas"], out_paths["pr"]


def compute_monthly_stats_for_fastmip(out_nc):
    """
    Aggregate the monthly (esm, realisation, time, region) SCALES output written
    by `run_projections_and_save` into monthly per-ESM statistics, one netCDF
    file per variable (tas, pr), laid out as (quantile, esm_calibration, year,
    month, mask) — the same convention used for the yearly regional_quantiles-by-ESM
    products, but keeping each calendar month (1-12) instead of aggregating over the year.
    """
    scenario_tag = _parse_scenario_tag(out_nc)

    ds_full = xr.open_dataset(out_nc, use_cftime=True)
    ds_monthly = ds_full.assign_coords(
        year=ds_full.time.dt.year, month=ds_full.time.dt.month
    ).set_index(time=("year", "month")).unstack("time")

    out_dir = os.path.dirname(out_nc)
    out_paths = {}
    for var in ["tas", "pr"]:
        da = ds_monthly[var]

        da_mean = da.mean("realisation").rename({"esm": "esm_calibration", "region": "mask"})
        da_std = da.std("realisation").rename({"esm": "esm_calibration", "region": "mask"})
        da_q = da.quantile(QUANTILES, dim="realisation").rename({"esm": "esm_calibration", "region": "mask"})

        da_mean = da_mean.transpose("esm_calibration", "year", "month", "mask")
        da_std = da_std.transpose("esm_calibration", "year", "month", "mask")
        da_q = da_q.transpose("quantile", "esm_calibration", "year", "month", "mask")

        ds_stats = xr.Dataset(
            data_vars={
                var: da_q,
                f"{var}_mean": da_mean,
                f"{var}_std": da_std,
            },
            attrs={"scenario": scenario_tag,
                   "description": "SCALES monthly trends emulator output — FastMIP GMT projections"},
        )
        ds_stats["year"] = ds_stats["year"].astype("int64")
        ds_stats["month"] = ds_stats["month"].astype("int64")

        out_filename = f"/{var}/{var}_{scenario_tag}_SCALES_regional_quantiles-by-ESM.nc"
        out_path = os.path.join(out_dir, out_filename)
        ds_stats.to_netcdf(out_dir+out_path)
        out_paths[var] = out_path
        print(f"Saved -> {out_path}", flush=True)

    ds_full.close()
    return out_paths["tas"], out_paths["pr"]


def compute_yearly_stats_across_esm_for_fastmip(out_nc):
    """
    Same as `compute_yearly_stats_for_fastmip`, but pools realisations across
    all ESMs to represent the multi-model spread, so the output has no
    esm_calibration dimension.
    """
    scenario_tag = _parse_scenario_tag(out_nc)

    ds_full = xr.open_dataset(out_nc, use_cftime=True)
    ds_yearly = ds_full.assign_coords(year=ds_full.time.dt.year).groupby("year").mean("time")

    out_dir = os.path.dirname(out_nc)
    out_paths = {}
    for var in ["tas", "pr"]:
        da = ds_yearly[var].rename({"region": "mask"})

        da_mean = da.mean(["esm", "realisation"])
        da_std = da.std(["esm", "realisation"])
        da_q = da.quantile(QUANTILES, dim=["esm", "realisation"])

        da_mean = da_mean.transpose("year", "mask")
        da_std = da_std.transpose("year", "mask")
        da_q = da_q.transpose("quantile", "year", "mask")

        ds_stats = xr.Dataset(
            data_vars={
                var: da_q,
                f"{var}_mean": da_mean,
                f"{var}_std": da_std,
            },
            attrs={"scenario": scenario_tag,
                   "description": "SCALES yearly trends emulator output — FastMIP GMT projections, multi-model (across-ESM) spread"},
        )
        ds_stats["year"] = ds_stats["year"].astype("int64")

        out_filename = f"{var}_{scenario_tag}_SCALES_regional_quantiles-across-ESM.nc"
        out_path = os.path.join(out_dir, out_filename)
        ds_stats.to_netcdf(out_path)
        out_paths[var] = out_path
        print(f"Saved -> {out_path}", flush=True)

    ds_full.close()
    return out_paths["tas"], out_paths["pr"]


def compute_monthly_stats_across_esm_for_fastmip(out_nc):
    """
    Same as `compute_monthly_stats_for_fastmip`, but pools realisations across
    all ESMs to represent the multi-model spread, so the output has no
    esm_calibration dimension.
    """
    scenario_tag = _parse_scenario_tag(out_nc)

    ds_full = xr.open_dataset(out_nc, use_cftime=True)
    ds_monthly = ds_full.assign_coords(
        year=ds_full.time.dt.year, month=ds_full.time.dt.month
    ).set_index(time=("year", "month")).unstack("time")

    out_dir = os.path.dirname(out_nc)
    out_paths = {}
    for var in ["tas", "pr"]:
        da = ds_monthly[var].rename({"region": "mask"})

        da_mean = da.mean(["esm", "realisation"])
        da_std = da.std(["esm", "realisation"])
        da_q = da.quantile(QUANTILES, dim=["esm", "realisation"])

        da_mean = da_mean.transpose("year", "month", "mask")
        da_std = da_std.transpose("year", "month", "mask")
        da_q = da_q.transpose("quantile", "year", "month", "mask")

        ds_stats = xr.Dataset(
            data_vars={
                var: da_q,
                f"{var}_mean": da_mean,
                f"{var}_std": da_std,
            },
            attrs={"scenario": scenario_tag,
                   "description": "SCALES monthly trends emulator output — FastMIP GMT projections, multi-model (across-ESM) spread"},
        )
        ds_stats["year"] = ds_stats["year"].astype("int64")
        ds_stats["month"] = ds_stats["month"].astype("int64")

        out_filename = f"{var}_{scenario_tag}_SCALES_regional_quantiles-across-ESM.nc"
        out_path = os.path.join(out_dir, out_filename)
        ds_stats.to_netcdf(out_path)
        out_paths[var] = out_path
        print(f"Saved -> {out_path}", flush=True)

    ds_full.close()
    return out_paths["tas"], out_paths["pr"]


def plot_yearly_stats_fastmip(out_tas_nc, out_pr_nc, region_sel="EAS", save_dir=None):
    esm_colors = {
        "ACCESS-ESM1-5":  "#1f77b4",
        "CanESM5":        "#d62728",
        "MPI-ESM1-2-LR":  "#2ca02c",
        "MIROC6":         "#9467bd",
    }
    ds_tas = xr.open_dataset(out_tas_nc)
    ds_pr  = xr.open_dataset(out_pr_nc)
    years  = ds_tas.year.values

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=150, sharex=True)
    for esm_name, color in esm_colors.items():
        if esm_name not in ds_tas.esm_calibration.values:
            continue
        for ds_s, var, ax in zip([ds_tas, ds_pr], ["tas", "pr"], axes):
            sel    = dict(esm_calibration=esm_name, mask=region_sel)
            mean   = ds_s[f"{var}_mean"].sel(**sel).values
            median = ds_s[var].sel(quantile=0.5,  **sel).values
            p05    = ds_s[var].sel(quantile=0.05, **sel).values
            p95    = ds_s[var].sel(quantile=0.95, **sel).values
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
        f"Yearly statistics  |  region: {region_sel}  |  scenario: {ds_tas.attrs.get('scenario', '')}\n"
        f"solid=mean  dashed=median  shading=5–95%",
        fontsize=11,
    )
    handles = [plt.Line2D([0], [0], color=c, linewidth=2.5) for c in esm_colors.values()]
    axes[0].legend(handles, list(esm_colors.keys()), fontsize=10, frameon=False, ncol=2)
    plt.tight_layout()

    if save_dir is not None:
        scenario_tag = os.path.basename(out_tas_nc).replace("_tas_regional_quantiles-by-ESM_yearly.nc", "")
        fig_path = os.path.join(save_dir, f"{scenario_tag}_{region_sel}_yearly_stats.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Plot saved -> {fig_path}", flush=True)
    #plt.close(fig)
    ds_tas.close()
    ds_pr.close()


def _region_value_grid(region_values, lon_res=1.0, lat_res=1.0):
    """
    Rasterise a {abbrev: value} dict over the AR6 regions onto a lon/lat grid,
    for plotting as a choropleth with `pcolormesh`.

    A region whose abbrev is not a key of `region_values` (chiefly the
    synthetic "GLOBAL" mask entry, which has no AR6 polygon) is left as NaN,
    same as everywhere outside every AR6 region.

    Returns (lon, lat, grid) with `grid.shape == (len(lat), len(lon))`.
    """
    lon = np.arange(-179.5, 180.0, lon_res)
    lat = np.arange(-89.5, 90.0, lat_res)
    number_grid = AR6_REGIONS.mask(lon, lat).values

    grid = np.full(number_grid.shape, np.nan)
    for region in AR6_REGIONS:
        value = region_values.get(region.abbrev)
        if value is not None:
            grid[number_grid == region.number] = value
    return lon, lat, grid


def plot_average_climate_change(out_tas_nc, out_pr_nc, year_range,
                                 esm_calibration=None, save_dir=None):
    """
    Two AR6-region maps (tas, pr) side by side, each region filled with its
    `*_mean` averaged over `year_range` — the same field `plot_yearly_stats_fastmip`
    plots as a time series, collapsed here to one number per region instead of
    one curve per ESM. Both fields are already anomalies, so this map is
    directly "average climate change over `year_range`", not a value needing a
    separate baseline subtracted.

    `out_tas_nc`/`out_pr_nc` are `compute_yearly_stats_for_fastmip`'s by-ESM
    output. `esm_calibration` selects which ESM calibration to show; defaults
    to whichever one appears first in the file. `year_range` is `(start, end)`,
    inclusive on both ends.
    """
    ds_tas = xr.open_dataset(out_tas_nc)
    ds_pr = xr.open_dataset(out_pr_nc)

    if esm_calibration is None:
        esm_calibration = str(ds_tas.esm_calibration.values[0])

    year_start, year_end = year_range
    year_sel = slice(year_start, year_end)
    tas_avg = ds_tas["tas_mean"].sel(esm_calibration=esm_calibration, year=year_sel).mean("year")
    pr_avg = ds_pr["pr_mean"].sel(esm_calibration=esm_calibration, year=year_sel).mean("year")

    tas_values = dict(zip(tas_avg.mask.values.tolist(), tas_avg.values.tolist()))
    pr_values = dict(zip(pr_avg.mask.values.tolist(), pr_avg.values.tolist()))

    lon, lat, tas_grid = _region_value_grid(tas_values)
    _, _, pr_grid = _region_value_grid(pr_values)

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 5.5), dpi=150,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    panels = [
        (axes[0], tas_grid, "RdBu_r", "tas anomaly [K]"),
        (axes[1], pr_grid, "BrBG", "pr anomaly [kg m⁻² s⁻¹]"),
    ]
    # cartopy fetches Natural Earth shapefiles lazily, on first draw, from the
    # network — so on a compute node with no outbound internet access (e.g.
    # the cluster this pipeline actually runs on) `ax.coastlines()` renders
    # fine here but raises inside `savefig`. Probe once, eagerly, so a missing
    # cache/network degrades to "no coastlines" instead of failing the plot.
    try:
        list(cfeature.COASTLINE.with_scale("110m").geometries())
        have_natural_earth = True
    except Exception as e:
        have_natural_earth = False
        print(f"[plot_average_climate_change] Natural Earth coastlines/borders "
              f"unavailable (no cache, no network?): {e}. Plotting regions "
              f"without a coastline overlay.")

    for ax, grid, cmap, label in panels:
        vmax = np.nanmax(np.abs(grid))
        im = ax.pcolormesh(lon, lat, grid, transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
        if have_natural_earth:
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.set_global()
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        cb.set_label(label, fontsize=11)

    fig.suptitle(
        f"Average climate change {year_start}–{year_end}  |  {esm_calibration}  |  "
        f"scenario: {ds_tas.attrs.get('scenario', '')}",
        fontsize=12,
    )
    plt.tight_layout()

    if save_dir is not None:
        scenario_tag = os.path.basename(out_tas_nc).replace("_tas_regional_quantiles-by-ESM_yearly.nc", "")
        fig_path = os.path.join(
            save_dir,
            f"{scenario_tag}_{esm_calibration}_{year_start}-{year_end}_avg_change_map.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Plot saved -> {fig_path}", flush=True)
    plt.close(fig)
    ds_tas.close()
    ds_pr.close()


def plot_monthly_stats_fastmip(out_tas_nc, out_pr_nc, region_sel="EAS", save_dir=None):
    esm_colors = {
        "ACCESS-ESM1-5":  "#1f77b4",
        "CanESM5":        "#d62728",
        "MPI-ESM1-2-LR":  "#2ca02c",
        "MIROC6":         "#9467bd",
    }
    ds_tas = xr.open_dataset(out_tas_nc)
    ds_pr  = xr.open_dataset(out_pr_nc)

    ds_tas = ds_tas.stack(time=("year", "month")).dropna("time", how="all")
    ds_pr  = ds_pr.stack(time=("year", "month")).dropna("time", how="all")
    time_vals = ds_tas.year.values + (ds_tas.month.values - 1) / 12

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=150, sharex=True)
    for esm_name, color in esm_colors.items():
        if esm_name not in ds_tas.esm_calibration.values:
            continue
        for ds_s, var, ax in zip([ds_tas, ds_pr], ["tas", "pr"], axes):
            sel    = dict(esm_calibration=esm_name, mask=region_sel)
            mean   = ds_s[f"{var}_mean"].sel(**sel).values
            median = ds_s[var].sel(quantile=0.5,  **sel).values
            p05    = ds_s[var].sel(quantile=0.05, **sel).values
            p95    = ds_s[var].sel(quantile=0.95, **sel).values
            ax.fill_between(time_vals, p05, p95, alpha=0.15, color=color)
            ax.plot(time_vals, median, linewidth=1.4, linestyle="--", color=color, alpha=0.8)
            ax.plot(time_vals, mean,   linewidth=2.0, color=color, label=esm_name)

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
        f"Monthly statistics  |  region: {region_sel}  |  scenario: {ds_tas.attrs.get('scenario', '')}\n"
        f"solid=mean  dashed=median  shading=5–95%",
        fontsize=11,
    )
    handles = [plt.Line2D([0], [0], color=c, linewidth=2.5) for c in esm_colors.values()]
    axes[0].legend(handles, list(esm_colors.keys()), fontsize=10, frameon=False, ncol=2)
    plt.tight_layout()

    if save_dir is not None:
        scenario_tag = os.path.basename(out_tas_nc).replace("_tas_regional_quantiles-by-ESM_yearly.nc", "")
        fig_path = os.path.join(save_dir, f"{scenario_tag}_{region_sel}_monthly_stats.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Plot saved -> {fig_path}", flush=True)
    #plt.close(fig)
    ds_tas.close()
    ds_pr.close()


def plot_yearly_stats_across_esm_fastmip(out_tas_nc, out_pr_nc, region_sel="EAS", save_dir=None):
    color = "#1f77b4"
    ds_tas = xr.open_dataset(out_tas_nc)
    ds_pr = xr.open_dataset(out_pr_nc)
    years = ds_tas.year.values

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=150, sharex=True)
    for ds_s, var, ax in zip([ds_tas, ds_pr], ["tas", "pr"], axes):
        sel    = dict(mask=region_sel)
        mean   = ds_s[f"{var}_mean"].sel(**sel).values
        median = ds_s[var].sel(quantile=0.5,  **sel).values
        p05    = ds_s[var].sel(quantile=0.05, **sel).values
        p95    = ds_s[var].sel(quantile=0.95, **sel).values
        ax.fill_between(years, p05, p95, alpha=0.15, color=color, label="5–95% (across-ESM)")
        ax.plot(years, median, linewidth=1.4, linestyle="--", color=color, alpha=0.8, label="median")
        ax.plot(years, mean,   linewidth=2.0, color=color, label="mean")

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
        f"Yearly statistics (multi-model spread, across-ESM)  |  region: {region_sel}  |  "
        f"scenario: {ds_tas.attrs.get('scenario', '')}\n"
        f"solid=mean  dashed=median  shading=5–95%",
        fontsize=11,
    )
    axes[0].legend(fontsize=10, frameon=False, ncol=3)
    plt.tight_layout()

    if save_dir is not None:
        scenario_tag = (
            os.path.basename(out_tas_nc)
            .removeprefix("tas_")
            .replace("_SCALES_regional_quantiles-across-ESM.nc", "")
        )
        fig_path = os.path.join(save_dir, f"{scenario_tag}_{region_sel}_yearly_stats_across_ESM.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Plot saved -> {fig_path}", flush=True)
    plt.close(fig)
    ds_tas.close()
    ds_pr.close()


def plot_monthly_stats_across_esm_fastmip(out_tas_nc, out_pr_nc, region_sel="EAS", save_dir=None):
    color = "#1f77b4"
    ds_tas = xr.open_dataset(out_tas_nc)
    ds_pr = xr.open_dataset(out_pr_nc)

    ds_tas = ds_tas.stack(time=("year", "month")).dropna("time", how="all")
    ds_pr  = ds_pr.stack(time=("year", "month")).dropna("time", how="all")
    time_vals = ds_tas.year.values + (ds_tas.month.values - 1) / 12

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=150, sharex=True)
    for ds_s, var, ax in zip([ds_tas, ds_pr], ["tas", "pr"], axes):
        sel    = dict(mask=region_sel)
        mean   = ds_s[f"{var}_mean"].sel(**sel).values
        median = ds_s[var].sel(quantile=0.5,  **sel).values
        p05    = ds_s[var].sel(quantile=0.05, **sel).values
        p95    = ds_s[var].sel(quantile=0.95, **sel).values
        ax.fill_between(time_vals, p05, p95, alpha=0.15, color=color, label="5–95% (across-ESM)")
        ax.plot(time_vals, median, linewidth=1.4, linestyle="--", color=color, alpha=0.8, label="median")
        ax.plot(time_vals, mean,   linewidth=2.0, color=color, label="mean")

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
        f"Monthly statistics (multi-model spread, across-ESM)  |  region: {region_sel}  |  "
        f"scenario: {ds_tas.attrs.get('scenario', '')}\n"
        f"solid=mean  dashed=median  shading=5–95%",
        fontsize=11,
    )
    axes[0].legend(fontsize=10, frameon=False, ncol=3)
    plt.tight_layout()

    if save_dir is not None:
        scenario_tag = (
            os.path.basename(out_tas_nc)
            .removeprefix("tas_")
            .replace("_SCALES_regional_quantiles-across-ESM.nc", "")
        )
        fig_path = os.path.join(save_dir, f"{scenario_tag}_{region_sel}_monthly_stats_across_ESM.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Plot saved -> {fig_path}", flush=True)
    plt.close(fig)
    ds_tas.close()
    ds_pr.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SCALES FastMIP projections for all scenarios and summarize as yearly stats"
    )
    parser.add_argument("--start", type=int, default=0, help="First scenario index (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="Last scenario index (exclusive, -1 = all)")
    parser.add_argument("--n_ensemble", type=int, default=N_ENSEMBLE, help="Number of GMT realisations")
    parser.add_argument("--plot_region", type=str, default="NZ", help="Region for summary plots")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Load scalers
    y_scaler = StandardScaler.from_file(SCALER_PATH + "y_scaler.out")
    u_scaler = StandardScaler.from_file(SCALER_PATH + "u_scaler.out")
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
    files = [os.path.join(SCENARIO_DIR, s) for s in scenarios if s.startswith("SSP")]

    end = args.end if args.end >= 0 else len(files)
    plot_dir = OUT_DIR  # save plots alongside netcdf files

    for scenario_index in range(args.start, end):
        print(f"\n=== Scenario {scenario_index}/{end - 1}: {scenarios[scenario_index]} ===", flush=True)
        out_nc = run_projections_and_save(
            scenario_index, files, scenarios, model,
            y_scaler, u_scaler, pr_scaler, model_data, device,
            n_ensemble=args.n_ensemble,
        )
        out_tas_nc, out_pr_nc = compute_yearly_stats(out_nc)
        plot_yearly_stats(out_tas_nc, out_pr_nc, region_sel=args.plot_region, save_dir=plot_dir)

    print("\nAll scenarios done.", flush=True)
