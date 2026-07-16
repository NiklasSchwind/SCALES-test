---
description: Run a SCALES SSM forecast for a given ESM and scenario
allowed-tools: ["Bash"]
---

Run a SCALES forecast using `load_esm_and_model.py`. The user will specify
some or all of the parameters below; use the defaults for anything they omit.

## Parameters

| Parameter        | CLI flag          | Default                                                                 | Notes                                          |
|------------------|-------------------|-------------------------------------------------------------------------|------------------------------------------------|
| ESM              | `--esm`           | (required)                                                              | One of: MIROC6, MPI-ESM1-2-LR, ACCESS-ESM1-5, CanESM5 |
| Scenario         | `--scenario`      | (required)                                                              | e.g. ssp245, ssp585, ssp534-over               |
| Start month      | `--start`         | 0                                                                       | First month index in the scenario timeseries   |
| Context length   | `--context_len`   | 600                                                                     | Months of context fed to the model             |
| Horizon          | `--horizon`       | 2100                                                                    | Months to forecast                             |
| Samples          | `--n_samples`     | 1                                                                       | Stochastic samples (1 = deterministic mean)    |
| Output path      | `--output`        | `/tmp/scales_forecast_{esm}_{scenario}_start{start}.pkl`               | Where to write the forecast pickle             |
| Scaler path      | `--scaler_path`   | `/home/kainverena/PythonProjects/outputs_ssm_scales/scales_cnp_20260615_150246/` | Directory containing scalers and model weights |
| Model filename   | `--model_filename`| `checkpoints/cnp_epoch1030.pt`                                          | Relative to scaler_path, or absolute           |

## Steps

1. Parse all parameters from the user's message. For `--output`, if the user
   did not specify one, construct the default path substituting the actual
   `esm`, `scenario`, and `start` values.

2. Run the forecast via the wrapper script so the correct conda environment
   is active:

```bash
/home/kainverena/PythonProjects/SCALES-test/run_forecast.sh \
    --esm <ESM> \
    --scenario <SCENARIO> \
    --start <START> \
    --context_len <CONTEXT_LEN> \
    --horizon <HORIZON> \
    --n_samples <N_SAMPLES> \
    --scaler_path <SCALER_PATH> \
    --model_filename <MODEL_FILENAME> \
    --output <OUTPUT>
```

3. The script prints one JSON line to stdout. Parse it:
   - If `"status": "ok"` → report: ESM, scenario, start month, context length,
     horizon, output path, and prediction shapes in a short readable summary.
   - If `"status": "error"` → show the `"error"` value and suggest checking
     the ESM name, scenario availability, or whether start + context_len +
     horizon exceeds the timeseries length.

## Example invocations

```
/scales-forecast MIROC6 ssp245
/scales-forecast CanESM5 ssp585 start=120 horizon=600
/scales-forecast ACCESS-ESM1-5 ssp534-over context_len=300 start=60 horizon=1200 n_samples=50
```
