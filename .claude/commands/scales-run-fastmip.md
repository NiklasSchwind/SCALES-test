---
description: Run SCALES CNP FastMIP projections on the IIASA cluster for a given scenario index
allowed-tools: ["Bash"]
---

Submit a SCALES CNP FastMIP projection job to the IIASA cluster via
`run_scales_unicc.py`. The user specifies a scenario index; use the default if omitted.

## Parameters

| Parameter       | CLI flag            | Default | Notes                                                   |
|-----------------|---------------------|---------|---------------------------------------------------------|
| Scenario index  | `--scenario_index`  | 5       | Index into the sorted list of scenario CSV files in `/pdrive/projects/icigroup/projects/FastMIP/scenarios/` |

## Steps

1. Parse `scenario_index` from the user's message (use 5 if not given).

2. Run:

```bash
python /Users/vkain/vscodeProjects/SCALES-test/proto_scales/scales_tools/run_scales_unicc.py \
    --scenario_index <SCENARIO_INDEX>
```

3. The script connects to `slurm-login.iiasa.ac.at` and runs `_scales_cnp_fastmip.py`
   remotely. Wait for it to finish, then report:
   - EXIT CODE (0 = success)
   - Any errors from STDERR
   - The output path printed in STDOUT (looks like `Saved -> /pdrive/...`)

   If the exit code is non-zero, show the full STDERR and suggest checking:
   - Whether the scenario index is within range
   - Whether the model checkpoint and scaler files exist at the configured paths
   - Whether the SSH connection to the cluster succeeded

## Example invocations

```
/scales-run-fastmip
/scales-run-fastmip scenario_index=3
/scales-run-fastmip 0
```
