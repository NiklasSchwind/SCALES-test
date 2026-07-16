"""
Load ESM climate data, run a SCALES CNP forecast, and save predictions.

The pipeline is: load ESM data → load model → forecast immediately → save results.
Raw ESM arrays are never written to disk; only the predictions are persisted.

Usage
-----
python load_esm_and_model.py \
    --esm MIROC6 \
    --scenario ssp245 \
    --scaler_path /home/kainverena/PythonProjects/outputs_ssm_scales/scales_cnp_20260615_150246/ \
    --model_filename checkpoints/cnp_epoch1030.pt \
    --output /tmp/miroc6_ssp245_forecast.pkl

On success  → prints one JSON line to stdout, exits 0.
On failure  → prints one JSON line with "error" key to stdout, exits 1.

Importable API
--------------
    from proto_scales.scales_tools.load_esm_and_model import (
        load_esm_data, load_model, run_forecast, forecast
    )
"""

import argparse
import json
import pickle
import sys
import os
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Allow running as a script from any working directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.join(_HERE, "..", "..")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import proto_scales.data_prep.prepare_data as prep
from proto_scales.multi_esm_model.scales_cnp import DeepCnpSsmforESM, DeepSSMPatternConditioned
from proto_scales.ssm_model.scales_ssm import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_ESMS = ["MIROC6", "MPI-ESM1-2-LR", "ACCESS-ESM1-5", "CanESM5"]

REGION_NAMES = [
    "ARO", "ARP", "ARS", "BOB", "CAF", "CAR", "CAU",
    "CNA", "EAN", "EAO", "EAS", "EAU", "ECA", "EEU", "EIO", "ENA", "EPO",
    "ESAF", "ESB", "GIC", "MDG", "MED", "NAO", "NAU", "NCA", "NEAF", "NEN",
    "NES", "NEU", "NPO", "NSA", "NWN", "NWS", "NZ", "RAR", "RFE", "SAH",
    "SAM", "SAO", "SAS", "SAU", "SCA", "SEA", "SEAF", "SES", "SIO", "SOO",
    "SPO", "SSA", "SWS", "TIB", "WAF", "WAN", "WCA", "WCE", "WNA", "WSAF",
    "WSB",
]

BASE_DATA_PATH = "/projects/icigroup/CMIP6/cmip6-ng-inc-oceans"

_DEFAULT_ZDIM = 64
_DEFAULT_RNN_HIDDEN = 256
_DEFAULT_USE_LINEAR_MODEL = True
_DEFAULT_EMISSION_USES_U = True
_DEFAULT_ALPHA_MAX = 0.002
_DEFAULT_RESERVOIR_DIM = 4
_DEFAULT_R_DIM = 128
_DEFAULT_Z_CNP_DIM = 32
_DEFAULT_CONTEXT_LEN = 600
_DEFAULT_HORIZON = 2100
_DEFAULT_N_SAMPLES = 1


# ---------------------------------------------------------------------------
# forecast helper (cell 2 of the notebook)
# ---------------------------------------------------------------------------

@torch.no_grad()
def forecast(model, y_ctx, u_ctx, pr_ctx, u_fut, n_samples=1):
    """
    Run the CNP-SSM forecast.

    Parameters
    ----------
    y_ctx   : Tensor [B, Tc, n_regions]   tas context (normalised)
    u_ctx   : Tensor [B, Tc, 1]           GMT context (normalised)
    pr_ctx  : Tensor [B, Tc, n_regions]   pr context (normalised)
    u_fut   : Tensor [B, H, 1]            GMT future (normalised)

    Returns
    -------
    y_ssm  : Tensor [B, H, n_regions]
    y_lin  : Tensor [B, H, n_regions]
    pr_ssm : Tensor [B, H, n_regions]
    """
    B, H, Du = u_fut.shape
    y_ssm, _, _, pr_ssm, _, _ = model.forecast(
        y_ctx, u_ctx, u_fut, steps=H, n_samples=n_samples, pr_ctx=pr_ctx
    )
    y_lin = model.ssm.ctrl_lin(u_fut.reshape(-1, Du)).reshape(B, H, -1)
    return y_ssm, y_lin, pr_ssm


# ---------------------------------------------------------------------------
# Data loading (cell 6 of the notebook)
# ---------------------------------------------------------------------------

def load_esm_data(
    esm: str,
    scenario: str,
    indicators=None,
    monthly_flag: bool = True,
    use_smoothing: bool = False,
    pattern_scaling_residuals: bool = False,
    ramp_down_corrected_ps: bool = False,
    train_pattern_scaling_name: str = "ssp585",
    base_data_path: str = BASE_DATA_PATH,
):
    """
    Load and pre-process ESM data for a single scenario.

    Returns
    -------
    dict:
        'tas'      : np.ndarray [n_scenarios, T, n_regions]
        'pr'       : np.ndarray [n_scenarios, T, n_regions]
        'gmt'      : np.ndarray [n_scenarios, T]
        'regions'  : list[str]
        'esm'      : str
        'scenario' : str
    """
    if indicators is None:
        indicators = ["tas", "pr"]
    if esm not in SUPPORTED_ESMS:
        raise ValueError(f"ESM '{esm}' not supported. Choose from: {SUPPORTED_ESMS}")

    model_path = os.path.join(base_data_path, esm)

    raw = prep.fetch_and_massage_data(
        model_path=model_path,
        indicators=indicators,
        train_scenarios=[scenario],
        monthly_flag=monthly_flag,
        use_smoothing=use_smoothing,
        train_pattern_scaling_name=train_pattern_scaling_name,
        pattern_scaling_residuals=pattern_scaling_residuals,
        ramp_down_corrected_ps=ramp_down_corrected_ps,
    )

    data_np = np.array(raw)          # [n_scenarios, n_channels, T]
    gmt = data_np[:, 0, :]           # [n_scenarios, T]
    data_np = data_np[:, 1:, :]      # [n_scenarios, n_indicators*n_regions, T]

    n_regions = int(data_np.shape[1] / 2)
    tas = np.transpose(data_np[:, :n_regions, :], (0, 2, 1))   # [n_scenarios, T, n_regions]
    pr  = np.transpose(data_np[:, n_regions:, :], (0, 2, 1))   # [n_scenarios, T, n_regions]

    return {"tas": tas, "pr": pr, "gmt": gmt, "regions": REGION_NAMES,
            "esm": esm, "scenario": scenario}


# ---------------------------------------------------------------------------
# Model loading (cells 3 + 8 of the notebook)
# ---------------------------------------------------------------------------

def load_model(
    scaler_path: str,
    model_filename: str,
    n_regions: int,
    zdim: int = _DEFAULT_ZDIM,
    rnn_hidden: int = _DEFAULT_RNN_HIDDEN,
    use_linear_model: bool = _DEFAULT_USE_LINEAR_MODEL,
    emission_uses_u: bool = _DEFAULT_EMISSION_USES_U,
    alpha_max: float = _DEFAULT_ALPHA_MAX,
    reservoir_dim: int = _DEFAULT_RESERVOIR_DIM,
    r_dim: int = _DEFAULT_R_DIM,
    z_cnp_dim: int = _DEFAULT_Z_CNP_DIM,
    device: str = "cpu",
):
    """
    Instantiate and load a pretrained DeepCnpSsmforESM model.

    Returns
    -------
    dict:
        'model'     : DeepCnpSsmforESM (eval mode)
        'y_scaler'  : StandardScaler
        'u_scaler'  : StandardScaler
        'pr_scaler' : StandardScaler
        'device'    : str
    """
    if not os.path.isabs(model_filename):
        model_filename = os.path.join(scaler_path, model_filename)

    y_scaler  = StandardScaler.from_file(os.path.join(scaler_path, "y_scaler.out"))
    u_scaler  = StandardScaler.from_file(os.path.join(scaler_path, "u_scaler.out"))
    pr_scaler = StandardScaler.from_file(os.path.join(scaler_path, "pr_scaler.out"))

    model_ssm = DeepSSMPatternConditioned(
        y_dim=n_regions, u_dim=1,
        z_dim=zdim, rnn_hidden=rnn_hidden,
        use_linear_model=use_linear_model,
        emission_uses_u=emission_uses_u,
        reservoir_dim=reservoir_dim,
        alpha_max=alpha_max,
    ).to(device)

    model = DeepCnpSsmforESM(ssm_model=model_ssm, r_dim=r_dim, z_cnp_dim=z_cnp_dim)
    model.load_state_dict(torch.load(model_filename, map_location=torch.device(device)))
    model.eval()

    return {"model": model, "y_scaler": y_scaler, "u_scaler": u_scaler,
            "pr_scaler": pr_scaler, "device": device}


# ---------------------------------------------------------------------------
# Full pipeline: load → forecast immediately
# ---------------------------------------------------------------------------

def run_forecast(
    esm: str,
    scenario: str,
    scaler_path: str,
    model_filename: str,
    context_len: int = _DEFAULT_CONTEXT_LEN,
    horizon: int = _DEFAULT_HORIZON,
    start: int = 0,
    n_samples: int = _DEFAULT_N_SAMPLES,
    indicators=None,
    device: str = "cpu",
    base_data_path: str = BASE_DATA_PATH,
    **model_kwargs,
):
    """
    End-to-end: load ESM data, load model, run forecast, return predictions.

    The context window begins at time step `start` and spans `context_len`
    months. The forecast horizon starts immediately after and runs for
    `horizon` months.

        y_ctx  = tas[:, start:start+context_len, :]
        u_ctx  = gmt[:, start:start+context_len, None]
        pr_ctx = pr[:, start:start+context_len, :]
        u_fut  = gmt[:, start+context_len:start+context_len+horizon, None]

    Parameters
    ----------
    start : int
        First time step (month index) of the context window. Default 0.

    Returns
    -------
    dict:
        'tas_pred'   : np.ndarray [n_scenarios, H, n_regions]
        'pr_pred'    : np.ndarray [n_scenarios, H, n_regions]
        'tas_lin'    : np.ndarray [n_scenarios, H, n_regions]  linear baseline
        'tas_gt'     : np.ndarray [n_scenarios, H, n_regions]  ground truth
        'pr_gt'      : np.ndarray [n_scenarios, H, n_regions]
        'gmt_ctx'    : np.ndarray [n_scenarios, Tc]
        'gmt_fut'    : np.ndarray [n_scenarios, H]
        'regions'    : list[str]
        'esm'        : str
        'scenario'   : str
        'start'      : int
        'context_len': int
        'horizon'    : int
    """
    data = load_esm_data(esm=esm, scenario=scenario, indicators=indicators,
                         base_data_path=base_data_path)
    tas, pr, gmt = data["tas"], data["pr"], data["gmt"]

    T = tas.shape[1]
    end = start + context_len + horizon
    if end > T:
        raise ValueError(
            f"start ({start}) + context_len ({context_len}) + horizon ({horizon}) = "
            f"{end} exceeds available time steps ({T})."
        )

    bundle = load_model(scaler_path=scaler_path, model_filename=model_filename,
                        n_regions=tas.shape[-1], device=device, **model_kwargs)
    model     = bundle["model"]
    y_scaler  = bundle["y_scaler"]
    u_scaler  = bundle["u_scaler"]
    pr_scaler = bundle["pr_scaler"]

    ctx_end = start + context_len

    # Split context / future
    y_ctx  = tas[:, start:ctx_end, :]
    pr_ctx = pr[:, start:ctx_end, :]
    u_ctx  = gmt[:, start:ctx_end, np.newaxis]       # [B, Tc, 1]
    u_fut  = gmt[:, ctx_end:end,   np.newaxis]       # [B, H,  1]

    # Normalise
    y_ctx_n  = torch.tensor(y_scaler.transform(y_ctx),   dtype=torch.float32, device=device)
    pr_ctx_n = torch.tensor(pr_scaler.transform(pr_ctx), dtype=torch.float32, device=device)
    u_ctx_n  = torch.tensor(u_scaler.transform(u_ctx),   dtype=torch.float32, device=device)
    u_fut_n  = torch.tensor(u_scaler.transform(u_fut),   dtype=torch.float32, device=device)

    # Forecast
    y_ssm, y_lin, pr_ssm = forecast(model, y_ctx_n, u_ctx_n, pr_ctx_n, u_fut_n,
                                     n_samples=n_samples)

    # Inverse-transform
    tas_pred = y_scaler.inverse_transform(y_ssm.cpu().numpy())
    pr_pred  = pr_scaler.inverse_transform(pr_ssm.cpu().numpy())
    tas_lin  = y_scaler.inverse_transform(y_lin.cpu().numpy())

    return {
        "tas_pred":    tas_pred,
        "pr_pred":     pr_pred,
        "tas_lin":     tas_lin,
        "tas_gt":      tas[:, ctx_end:end, :],
        "pr_gt":       pr[:, ctx_end:end, :],
        "gmt_ctx":     gmt[:, start:ctx_end],
        "gmt_fut":     gmt[:, ctx_end:end],
        "regions":     REGION_NAMES,
        "esm":         esm,
        "scenario":    scenario,
        "start":       start,
        "context_len": context_len,
        "horizon":     horizon,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Run a SCALES CNP forecast for a given ESM and scenario.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--esm", required=True, choices=SUPPORTED_ESMS)
    p.add_argument("--scenario", required=True,
                   help="e.g. ssp245, ssp585, ssp534-over")
    p.add_argument("--indicators", nargs="+", default=["tas", "pr"])
    p.add_argument("--scaler_path",
                   default="/home/kainverena/PythonProjects/outputs_ssm_scales/scales_cnp_20260615_150246/")
    p.add_argument("--model_filename", default="checkpoints/cnp_epoch1030.pt")
    p.add_argument("--output", required=True,
                   help="Path to write the forecast pickle (.pkl).")
    p.add_argument("--context_len", type=int, default=_DEFAULT_CONTEXT_LEN,
                   help="Number of months used as context.")
    p.add_argument("--horizon", type=int, default=_DEFAULT_HORIZON,
                   help="Number of months to forecast.")
    p.add_argument("--start", type=int, default=0,
                   help="First month index in the scenario timeseries where the context window begins.")
    p.add_argument("--n_samples", type=int, default=_DEFAULT_N_SAMPLES)
    p.add_argument("--device", default="cpu")
    p.add_argument("--base_data_path", default=BASE_DATA_PATH)
    p.add_argument("--zdim", type=int, default=_DEFAULT_ZDIM)
    p.add_argument("--rnn_hidden", type=int, default=_DEFAULT_RNN_HIDDEN)
    p.add_argument("--reservoir_dim", type=int, default=_DEFAULT_RESERVOIR_DIM)
    p.add_argument("--alpha_max", type=float, default=_DEFAULT_ALPHA_MAX)
    p.add_argument("--no_linear_model", action="store_true")
    p.add_argument("--no_emission_u", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    try:
        results = run_forecast(
            esm=args.esm,
            scenario=args.scenario,
            scaler_path=args.scaler_path,
            model_filename=args.model_filename,
            context_len=args.context_len,
            horizon=args.horizon,
            start=args.start,
            n_samples=args.n_samples,
            indicators=args.indicators,
            device=args.device,
            base_data_path=args.base_data_path,
            zdim=args.zdim,
            rnn_hidden=args.rnn_hidden,
            reservoir_dim=args.reservoir_dim,
            alpha_max=args.alpha_max,
            use_linear_model=not args.no_linear_model,
            emission_uses_u=not args.no_emission_u,
        )

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "wb") as f:
            pickle.dump(results, f)

        print(json.dumps({
            "status": "ok",
            "output_path": args.output,
            "esm": args.esm,
            "scenario": args.scenario,
            "start": args.start,
            "context_len": args.context_len,
            "horizon": args.horizon,
            "n_samples": args.n_samples,
            "tas_pred_shape": list(results["tas_pred"].shape),
            "pr_pred_shape":  list(results["pr_pred"].shape),
            "n_regions": len(results["regions"]),
        }))
        sys.exit(0)

    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
