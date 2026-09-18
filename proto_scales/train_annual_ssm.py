"""
Train the simplified annual SSM (ELBO only).

Data loading is shared with train_ssm_dit.py so the two models see identical
arrays.

    torchrun --nproc_per_node=1 -m proto_scales.train_annual_ssm --cluster ASC

The resulting run directory holds `model_out` plus the monthly scalers, and is
what you pass to the DiT:

    torchrun --nproc_per_node=1 -m proto_scales.train_ssm_dit --cluster ASC \
        --annual_ssm_run_dir outputs_ssm_scales/annual_ssm_YYYYMMDD_HHMMSS

What to watch
-------------
The KL. It is printed every epoch and it is the diagnostic that matters: if it
sits at the free-bits floor, z is being ignored and the emission is too strong
(reduce --emit_hidden, or lower --kl_weight). A healthy run has KL comfortably
above the floor and still falling slowly.

At the end the learned oscillator periods and e-folding times are printed in
years. Those are directly interpretable — periods clustering in the 3-7 year
band mean the model has found ENSO-like variability.
"""

import argparse
import faulthandler
import os
import traceback
from datetime import datetime

import torch

import proto_scales.ssm_dit_model.annual_training as annual_training
from proto_scales.train_ssm_dit import (
    MODEL_PATH_ASC,
    MODEL_PATH_IIASA,
    TRAIN_SCENARIOS,
    load_data,
)

if __name__ == "__main__":
    faulthandler.enable()
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))

    p = argparse.ArgumentParser(description="Annual SSM for SCALES (ELBO only)")
    p.add_argument("--cluster", type=str, help="Cluster name ASC or IIASA")
    p.add_argument("--add_data", nargs="+", type=str, default=[])
    p.add_argument("--rm_data", nargs="+", type=str, default=[])

    # model — six knobs
    p.add_argument("--z_dim", type=int, default=16, help="Latent size (must be even)")
    p.add_argument("--rnn_hidden", type=int, default=64)
    p.add_argument("--emit_hidden", type=int, default=0,
                   help="0 = linear emission (recommended). A stronger emission invites posterior collapse")
    p.add_argument("--trans_hidden", type=int, default=0,
                   help="0 = purely linear-Gaussian transition (recommended)")
    p.add_argument("--osc_min_period", type=float, default=2.0,
                   help="Shortest oscillator period in YEARS (>= 2, the Nyquist limit for annual data)")
    p.add_argument("--osc_max_period", type=float, default=20.0)
    p.add_argument("--osc_damping_max", type=float, default=1.0)

    # loss — two knobs
    p.add_argument("--kl_weight", type=float, default=1.0,
                   help="1.0 is the true ELBO; >1 is a beta-VAE")
    p.add_argument("--kl_free_bits", type=float, default=0.05,
                   help="Per-latent-dimension KL floor; prevents dimensions of z from being permanently driven to the prior")

    # optimisation
    p.add_argument("--seq_years", type=int, default=100)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--weights_file", type=str, default=None)
    p.add_argument("--scaler_dir", type=str, default=None,
                   help="Reuse monthly scalers from an existing run instead of refitting")
    p.add_argument("--ckpt_every", type=int, default=20)

    args = p.parse_args()

    MODEL_PATH = MODEL_PATH_ASC if args.cluster == "ASC" else MODEL_PATH_IIASA
    scenarios = list(TRAIN_SCENARIOS) + args.add_data
    scenarios = [s for s in scenarios if s not in args.rm_data]
    print("MODEL_PATH", MODEL_PATH)

    tas, pr, u = load_data(MODEL_PATH, scenarios)
    print("tas", tas.shape, "pr", pr.shape, "GMT", u.shape)

    run_dir = os.path.join("outputs_ssm_scales",
                           "annual_ssm_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_scenarios.txt"), "w") as f:
        f.write("\n".join(scenarios))
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")

    try:
        model, tas_scaler, pr_scaler, u_scaler = annual_training.run_train(
            tas, pr, u,
            seq_years=args.seq_years, stride=args.stride,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
            z_dim=args.z_dim, rnn_hidden=args.rnn_hidden,
            emit_hidden=args.emit_hidden, trans_hidden=args.trans_hidden,
            osc_period_range=(args.osc_min_period, args.osc_max_period),
            osc_damping_max=args.osc_damping_max,
            kl_weight=args.kl_weight, kl_free_bits=args.kl_free_bits,
            ema_decay=args.ema_decay, run_dir=run_dir,
            weights_file=args.weights_file, scaler_dir=args.scaler_dir,
            ckpt_every=args.ckpt_every,
        )
    except Exception:
        print(f"[Rank {_local_rank}] run_train failed:", flush=True)
        traceback.print_exc()
        raise

    torch.save(model.state_dict(), os.path.join(run_dir, "model_out"))
    tas_scaler.save(os.path.join(run_dir, "y_scaler.out"))
    pr_scaler.save(os.path.join(run_dir, "pr_scaler.out"))
    u_scaler.save(os.path.join(run_dir, "u_scaler.out"))
    print("saved to", run_dir)
