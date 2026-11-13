"""
Fit the fixed z standardization stats of an already-trained annual JEPA-KL DiT
(runs trained before `ForcedAnnualJEPAKL.fit_z_stats` existed, or epoch
checkpoints of a run that is still training — those are saved before the
end-of-training fit).

    # final model_out, or the latest epoch checkpoint if training isn't done
    python -m proto_scales.fit_jepa_kl_z_stats --run_dir outputs_ssm_scales/<run> --cluster IIASA
    # a specific epoch checkpoint
    python -m proto_scales.fit_jepa_kl_z_stats --run_dir outputs_ssm_scales/<run> --checkpoint 160

Rebuilds exactly the training windows `run_train` used (same scenarios from
`train_scenarios.txt`, same seed-0 80/20 split, same scalers, context_len /
horizon / stride from `config.txt`), fits the stats with the checkpoint's own
encoder, and writes a copy of the checkpoint with them filled in, next to it:
`model_out_zstats`, or `checkpoints/annual_jepa_kl_dit_epochNNNN_zstats.pt`.
The original is not touched. `eval_jepa_rollout` picks the fitted copy up
automatically. Everything other than the five z-stat buffers is copied
unchanged, so the result loads wherever the original did.

For an epoch checkpoint the stats are fitted with the `--weights_key` encoder
(default the sampling EMA) and written into both its `model` and `ema`
entries; the raw model's encoder differs from the EMA's only slightly.
"""

import argparse
import os

import numpy as np
import torch

PREFIX = "ssm_encoder.ssm."


def fit_on_training_set(jepa, run_dir, cfg, cluster, seed=0, batch_size=64):
    """
    Fit `jepa`'s fixed z stats in place on the run's training windows.
    Returns the number of windows used. `jepa` is returned to its original
    device.
    """
    import proto_scales.ssm_model.scales_ssm as scales_ssm
    import proto_scales.train_annual_jepa_kl_dit as train_script
    from proto_scales.ssm_dit_model.annual_dit_training import LongHorizonWindowDataset
    from proto_scales.ssm_dit_model.annual_jepa_kl_dit_training import z_stat_batches
    from proto_scales.ssm_dit_model.annual_ssm_dit import ForcedAnnualLatentEncoder

    if not jepa.normalize_z:
        raise SystemExit("run was trained with --no_normalize_z; nothing to fit")
    with open(os.path.join(run_dir, "train_scenarios.txt")) as f:
        scenarios = [s.strip() for s in f if s.strip()]

    model_path = (train_script.MODEL_PATH_ASC if cluster == "ASC"
                  else train_script.MODEL_PATH_IIASA)
    tas, pr, u = train_script.load_data(model_path, scenarios)

    # Same split as run_train: np seed, then permutation of the N chunks.
    np.random.seed(seed)
    idx = np.random.permutation(tas.shape[0])
    tr = idx[:max(1, int(0.8 * tas.shape[0]))]

    S = scales_ssm.StandardScaler
    tas_n = S.from_file(os.path.join(run_dir, "y_scaler.out")).transform(tas)
    pr_n = S.from_file(os.path.join(run_dir, "pr_scaler.out")).transform(pr)
    u_n = S.from_file(os.path.join(run_dir, "u_scaler.out")).transform(u)
    ds = LongHorizonWindowDataset(
        tas_n[tr], pr_n[tr], u_n[tr],
        context_len=cfg.get("context_len", 120), horizon=cfg.get("horizon", 1200),
        stride=cfg.get("stride", 7), data_start_month=cfg.get("data_start_month", 0))

    orig_device = next(jepa.parameters()).device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa.to(device)
    jepa.fit_z_stats(z_stat_batches(ds, ForcedAnnualLatentEncoder(jepa), device, batch_size))
    jepa.to(orig_device)

    print(f"[z-stats] fitted on {len(ds)} training windows")
    print(f"  online mean {[round(v, 3) for v in jepa.z_mean.tolist()]}")
    print(f"  online std  {[round(v, 3) for v in jepa.z_std.tolist()]}")
    print(f"  target std  {[round(v, 3) for v in jepa.z_tgt_std.tolist()]}")
    return len(ds)


def main():
    from proto_scales.ssm_dit_model.eval_jepa_rollout import (
        load_jepa, read_config, resolve_weights, zstats_path)

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run_dir", required=True)
    p.add_argument("--weights", default=None,
                   help="explicit weights file; overrides --checkpoint")
    p.add_argument("--checkpoint", default=None,
                   help="'latest' or an epoch number from <run_dir>/checkpoints. "
                        "Default: model_out, or the latest checkpoint if training hasn't finished")
    p.add_argument("--weights_key", default="ema", choices=["ema", "model"],
                   help="which encoder of an epoch checkpoint to fit with; ignored for model_out")
    p.add_argument("--cluster", default="IIASA", choices=["IIASA", "ASC"])
    p.add_argument("--out", default=None, help="default: <weights>_zstats[.pt] next to it")
    p.add_argument("--seed", type=int, default=0, help="run_train's seed (split)")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    src, _ = resolve_weights(args.run_dir, args.weights, args.checkpoint, prefer_zstats=False)
    cfg = read_config(args.run_dir)
    ckpt = torch.load(src, map_location="cpu")
    is_epoch = isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt)
    first = ckpt[args.weights_key] if is_epoch else ckpt
    # Field size from the checkpoint itself, so the data isn't loaded twice.
    obs_dim = first[PREFIX + "gru.weight_ih_l0"].shape[1] - first[PREFIX + "B.weight"].shape[1]
    u_dim = first[PREFIX + "B.weight"].shape[1]
    jepa = load_jepa(args.run_dir, src, args.weights_key, cfg, obs_dim // 2, u_dim)

    fit_on_training_set(jepa, args.run_dir, cfg, args.cluster, args.seed, args.batch_size)

    for sd in ([ckpt[k] for k in ("model", "ema") if k in ckpt] if is_epoch else [ckpt]):
        for name in jepa._Z_STAT_BUFFERS:
            sd[PREFIX + name] = getattr(jepa, name).detach().cpu().clone()
    out = args.out or zstats_path(src)
    torch.save(ckpt, out)
    print(f"[z-stats] wrote {out}")


if __name__ == "__main__":
    main()
