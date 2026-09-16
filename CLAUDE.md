# SCALES Project

SCALES is a deep State Space Model (SSM) for climate emulation, projecting regional temperature (tas) and precipitation (pr) anomalies from global mean temperature (GMT) forcing.

## Available Commands

### `/scales-forecast`
Run a SCALES SSM forecast for a specific ESM and scenario using `load_esm_and_model.py`.
Use this when the user wants to generate a forecast for a particular ESM model (e.g. MIROC6, ACCESS-ESM1-5) and scenario (e.g. ssp245, ssp585), optionally specifying start month, context length, horizon, and number of samples.

### `/scales-run-fastmip`
Run SCALES CNP FastMIP projections on the IIASA cluster via SSH (`run_scales_unicc.py`).
Use this when the user wants to run the full FastMIP projection pipeline remotely, for a given scenario index into the FastMIP scenario CSV files.

## Key Files

- `proto_scales/ssm_model/scales_ssm_tas_pr_hyst_correlated.py` — core SSM model with low-rank covariance emission
- `proto_scales/train_scales_ssm_tas_pr_hyst_correlated.py` — training script
- `proto_scales/ssm_model/boostrap_ssm.py` — bootstrap correlated model from diagonal checkpoint
- `proto_scales/scales_tools/load_esm_and_model.py` — forecast wrapper used by `/scales-forecast`
- `proto_scales/scales_tools/_scales_cnp_fastmip.py` — FastMIP projection script (runs on IIASA cluster)
- `proto_scales/scales_tools/run_scales_unicc.py` — SSH launcher for IIASA cluster (used by `/scales-run-fastmip`)
- `proto_scales/data_prep/prepare_data.py` — data loading and preprocessing
- `proto_scales/scales-for-fastMIP.ipynb` — main analysis notebook

## Cluster

Remote execution is on `slurm-login.iiasa.ac.at` (port 30222) via paramiko SSH.
