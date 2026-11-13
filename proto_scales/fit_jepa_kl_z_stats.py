"""
Fit the fixed z standardization stats of an already-trained annual JEPA-KL DiT
(runs trained before `ForcedAnnualJEPAKL.fit_z_stats` existed, or epoch
checkpoints, which are saved before the end-of-training fit).

    python -m proto_scales.fit_jepa_kl_z_stats \
        --run_dir outputs_ssm_scales/annual_jepa_kl_dit_20261105_120000 --cluster IIASA

Rebuilds exactly the training windows `run_train` used (same scenarios from
`train_scenarios.txt`, same seed-0 80/20 split, same scalers, context_len /
horizon / stride from `config.txt`), fits the stats with the checkpoint's own
encoder, and writes a copy of the checkpoint with them filled in
(default `<weights>_zstats`; the original is not touched). Everything in the
checkpoint other than the five z-stat buffers is copied unchanged, so the
result loads wherever the original did.
"""

import argparse
import os

import numpy as np
import torch

import proto_scales.ssm_model.scales_ssm as scales_ssm
import proto_scales.train_annual_jepa_kl_dit as train_script
from proto_scales.ssm_dit_model.annual_dit_training import LongHorizonWindowDataset
from proto_scales.ssm_dit_model.annual_jepa_kl_dit_training import z_stat_batches
from proto_scales.ssm_dit_model.annual_ssm_dit import ForcedAnnualLatentEncoder
from proto_scales.ssm_dit_model.eval_jepa_rollout import load_jepa, read_config

PREFIX = "ssm_encoder.ssm."


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--weights", default=None, help="default: <run_dir>/model_out")
    p.add_argument("--weights_key", default="ema", choices=["ema", "model"],
                   help="for epoch checkpoints {model, ema}; ignored for model_out")
    p.add_argument("--cluster", default="IIASA", choices=["IIASA", "ASC"])
    p.add_argument("--out", default=None, help="default: <weights>_zstats")
    p.add_argument("--seed", type=int, default=0, help="run_train's seed (split)")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    cfg = read_config(args.run_dir)
    if not cfg.get("normalize_z", True):
        raise SystemExit("run was trained with --no_normalize_z; nothing to fit")
    with open(os.path.join(args.run_dir, "train_scenarios.txt")) as f:
        scenarios = [s.strip() for s in f if s.strip()]

    model_path = (train_script.MODEL_PATH_ASC if args.cluster == "ASC"
                  else train_script.MODEL_PATH_IIASA)
    tas, pr, u = train_script.load_data(model_path, scenarios)

    # Same split as run_train: np seed, then permutation of the N chunks.
    np.random.seed(args.seed)
    idx = np.random.permutation(tas.shape[0])
    tr = idx[:max(1, int(0.8 * tas.shape[0]))]

    S = scales_ssm.StandardScaler
    tas_n = S.from_file(os.path.join(args.run_dir, "y_scaler.out")).transform(tas)
    pr_n = S.from_file(os.path.join(args.run_dir, "pr_scaler.out")).transform(pr)
    u_n = S.from_file(os.path.join(args.run_dir, "u_scaler.out")).transform(u)
    ds = LongHorizonWindowDataset(
        tas_n[tr], pr_n[tr], u_n[tr],
        context_len=cfg.get("context_len", 120), horizon=cfg.get("horizon", 1200),
        stride=cfg.get("stride", 7), data_start_month=cfg.get("data_start_month", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa = load_jepa(args.run_dir, args.weights, args.weights_key, cfg,
                     tas.shape[-1], u.shape[-1]).to(device)
    jepa.fit_z_stats(z_stat_batches(ds, ForcedAnnualLatentEncoder(jepa), device,
                                    args.batch_size))
    print(f"[z-stats] fitted on {len(ds)} training windows")
    print(f"  online mean {[round(v, 3) for v in jepa.z_mean.tolist()]}")
    print(f"  online std  {[round(v, 3) for v in jepa.z_std.tolist()]}")
    print(f"  target std  {[round(v, 3) for v in jepa.z_tgt_std.tolist()]}")

    src = args.weights or os.path.join(args.run_dir, "model_out")
    ckpt = torch.load(src, map_location="cpu")
    sds = ([ckpt[k] for k in ("model", "ema") if k in ckpt]
           if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt) else [ckpt])
    for sd in sds:  # fill both entries of an epoch checkpoint
        for name in jepa._Z_STAT_BUFFERS:
            sd[PREFIX + name] = getattr(jepa, name).detach().cpu().clone()
    out = args.out or src + "_zstats"
    torch.save(ckpt, out)
    print(f"[z-stats] wrote {out}")


if __name__ == "__main__":
    main()
