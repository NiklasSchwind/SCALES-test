"""
Bootstrap a scales_ssm_cross_corr_osc_trans.DeepSSMPatternConditioned
(joint tas–pr emission + oscillatory transition) from a trained
scales_ssm_z2tasAndpr_hysteresis.DeepSSMPatternConditioned (diagonal,
independent tas/pr).

Differences vs. the source module the joint model has to accommodate:

  * `emit` has a new output layout of size 2·y_dim·(3 + cov_rank):
        [ μ_joint (2·D) | log_diag (2·D) | cov_factor (2·D·r) | ε_skew (D) | log δ (D) ]
    (There is no separate `emit_pr` in the joint model.)

  * `trans` is renamed to `trans_corr` (same shape). Together with the new
    linear oscillator (`log_damping`, `omega`, `B_osc`), it forms the
    transition prior; `trans_corr` acts as a correction on top of the
    oscillator.

Strategy
--------
Direct copies (shapes match, no reinterpretation):
    gru, q_head, u_gru, log_alpha, omega_lin, ctrl_lin
    trans.*                        → trans_corr.*
    emit.net.0.*, emit.net.2.*     (input/hidden layers of the emit MLP)

Constructed `emit.net.4` output rows (diagonal → joint reordering):
    joint  μ_tas       ← diagonal emit.mean_tas
    joint  μ_pr        ← diagonal emit_pr.μ_pr        (sliced to mlp_hidden)
    joint  log_diag_tas ← diagonal emit.log_diag_tas
    joint  log_diag_pr  ← diagonal emit_pr.log_σ_pr   (sliced to mlp_hidden)
    joint  cov_factor   ← small Gaussian noise (breaks zero-gradient trap)
    joint  ε_skew       ← diagonal emit_pr.ε_skew     (sliced to mlp_hidden)
    joint  log δ        ← diagonal emit_pr.log δ      (sliced to mlp_hidden)
Biases handled the same way (no column slicing needed).

Oscillator parameters (`log_damping`, `omega`, `B_osc`) are left at their
initialisation, with `B_osc.weight` zeroed so that the initial u→z
contribution goes through `trans_corr` alone (i.e. mimics the old
transition until training pulls the oscillator into use).
"""

import os
import math

import torch

import proto_scales.ssm_model.scales_ssm_z2tasAndpr_hysteresis as scales_ssm_z2pr_diagonal
import proto_scales.ssm_model.scales_ssm_cross_corr_osc_trans as scales_ssm_joint

# ── architecture params (must match the checkpoint being loaded) ──────────────
zdim             = 64
rnn_hidden       = 256
use_linear_model = True
emission_uses_u  = True
alpha_max        = 0.002
resevoir_dim     = 4
cov_rank         = 8   # joint MVN covers 2·y_dim; per-variable budget ~ cov_rank/2

model_filename = "/home/kainverena/PythonProjects/outputs_ssm_scales/scales_20260512_143221/model_out"
# ─────────────────────────────────────────────────────────────────────────────

diag_sd = torch.load(model_filename, map_location="cpu")

# Infer dims from the diagonal state dict
mlp_hidden   = diag_sd["emit.net.0.weight"].shape[0]
y_dim        = diag_sd["emit.net.4.weight"].shape[0] // 2
u_dim        = diag_sd["ctrl_lin.weight"].shape[1]
u_rnn_hidden = diag_sd["u_gru.weight_hh_l0"].shape[1]
pr_hidden    = diag_sd["emit_pr.net.4.weight"].shape[1]   # 230 in the current source

print(f"Inferred  y_dim={y_dim}  u_dim={u_dim}  mlp_hidden={mlp_hidden}  "
      f"u_rnn_hidden={u_rnn_hidden}  pr_emit_hidden={pr_hidden}")

joint_model = scales_ssm_joint.DeepSSMPatternConditioned(
    y_dim=y_dim,
    u_dim=u_dim,
    z_dim=zdim,
    rnn_hidden=rnn_hidden,
    mlp_hidden=mlp_hidden,
    u_rnn_hidden=u_rnn_hidden,
    emission_uses_u=emission_uses_u,
    use_linear_model=use_linear_model,
    reservoir_dim=resevoir_dim,
    alpha_max=alpha_max,
    cov_rank=cov_rank,
)

joint_sd = joint_model.state_dict()
copied, partial, skipped, init = [], [], [], []


def _slice_pr_rows(mat, rows, cols):
    """Return diag_sd matrix rows sliced to first `cols` cols (or padded with zeros)."""
    src = mat[rows]                                   # [len(rows), src_cols]
    src_cols = src.shape[1]
    if src_cols >= cols:
        return src[:, :cols].clone()
    out = torch.zeros(src.shape[0], cols, dtype=src.dtype)
    out[:, :src_cols] = src
    return out


D = y_dim
r = cov_rank
# Row offsets in joint emit.net.4 output vector
off_mu_joint   = 0                       # 2·D
off_log_diag   = off_mu_joint + 2 * D    # 2·D
off_cov_factor = off_log_diag + 2 * D    # 2·D·r
off_eps_skew   = off_cov_factor + 2 * D * r    # D
off_log_delta  = off_eps_skew + D              # D
total_emit_out = off_log_delta + D
assert total_emit_out == joint_sd["emit.net.4.weight"].shape[0]

# ─── Walk the joint state dict ──────────────────────────────────────────────
for name in joint_sd:

    # --- Special: trans_corr <- trans (same shape) ---
    if name.startswith("trans_corr."):
        src_name = name.replace("trans_corr.", "trans.", 1)
        if src_name in diag_sd and diag_sd[src_name].shape == joint_sd[name].shape:
            joint_sd[name] = diag_sd[src_name].clone()
            copied.append(f"  {name}  <- {src_name}  {tuple(joint_sd[name].shape)}")
            continue

    # --- Special: joint emit.net.4 output layer ---
    if name == "emit.net.4.weight":
        W = joint_sd[name].clone()
        cols = W.shape[1]  # == mlp_hidden
        # μ_tas rows (D)
        W[off_mu_joint:off_mu_joint + D] = diag_sd["emit.net.4.weight"][:D]
        # log_diag_tas rows (D)
        W[off_log_diag:off_log_diag + D] = diag_sd["emit.net.4.weight"][D:2 * D]
        # μ_pr rows (D)   ← emit_pr μ_pr
        W[off_mu_joint + D:off_mu_joint + 2 * D] = _slice_pr_rows(
            diag_sd["emit_pr.net.4.weight"], slice(0, D), cols)
        # log_diag_pr rows (D)   ← emit_pr log σ
        W[off_log_diag + D:off_log_diag + 2 * D] = _slice_pr_rows(
            diag_sd["emit_pr.net.4.weight"], slice(D, 2 * D), cols)
        # cov_factor rows (2·D·r) ← small Gaussian noise (break zero-gradient trap)
        W[off_cov_factor:off_cov_factor + 2 * D * r] = torch.randn(2 * D * r, cols) * 0.01
        # ε_skew rows (D)  ← emit_pr ε_skew
        W[off_eps_skew:off_eps_skew + D] = _slice_pr_rows(
            diag_sd["emit_pr.net.4.weight"], slice(2 * D, 3 * D), cols)
        # log δ rows (D)    ← emit_pr log δ
        W[off_log_delta:off_log_delta + D] = _slice_pr_rows(
            diag_sd["emit_pr.net.4.weight"], slice(3 * D, 4 * D), cols)
        joint_sd[name] = W
        partial.append(
            f"  {name}  rows assembled from diagonal emit + emit_pr "
            f"(pr rows sliced from {pr_hidden}→{cols} cols); cov_factor N(0,0.01)"
        )
        continue

    if name == "emit.net.4.bias":
        b = joint_sd[name].clone()
        b[off_mu_joint:off_mu_joint + D]             = diag_sd["emit.net.4.bias"][:D]
        b[off_log_diag:off_log_diag + D]             = diag_sd["emit.net.4.bias"][D:2 * D]
        b[off_mu_joint + D:off_mu_joint + 2 * D]     = diag_sd["emit_pr.net.4.bias"][:D]
        b[off_log_diag + D:off_log_diag + 2 * D]     = diag_sd["emit_pr.net.4.bias"][D:2 * D]
        b[off_cov_factor:off_cov_factor + 2 * D * r] = 0.0
        b[off_eps_skew:off_eps_skew + D]             = diag_sd["emit_pr.net.4.bias"][2 * D:3 * D]
        b[off_log_delta:off_log_delta + D]           = diag_sd["emit_pr.net.4.bias"][3 * D:4 * D]
        joint_sd[name] = b
        partial.append(f"  {name}  assembled from diagonal emit + emit_pr biases")
        continue

    # --- Zero out B_osc so initial u→z goes through trans_corr alone ---
    if name == "B_osc.weight":
        joint_sd[name] = torch.zeros_like(joint_sd[name])
        init.append(f"  {name}  zeroed (initial u→z via trans_corr only)")
        continue

    # --- Osc-specific params kept at init ---
    if name in ("log_damping", "omega"):
        init.append(f"  {name}  {tuple(joint_sd[name].shape)}  kept at model init")
        continue

    # --- Direct copy when shape matches ---
    if name in diag_sd and diag_sd[name].shape == joint_sd[name].shape:
        joint_sd[name] = diag_sd[name].clone()
        copied.append(f"  {name}  {tuple(joint_sd[name].shape)}")
    else:
        reason = ("not in diagonal model" if name not in diag_sd
                  else f"shape mismatch  diagonal={tuple(diag_sd[name].shape)}  "
                       f"joint={tuple(joint_sd[name].shape)}")
        skipped.append(f"  {name}  {reason}")

joint_model.load_state_dict(joint_sd)

print(f"\nCopied exactly ({len(copied)}):")
print("\n".join(copied))
print(f"\nPartially copied ({len(partial)}):")
print("\n".join(partial))
if init:
    print(f"\nLeft at model init ({len(init)}):")
    print("\n".join(init))
if skipped:
    print(f"\nLeft at random init ({len(skipped)}):")
    print("\n".join(skipped))

out_path = os.path.join(
    os.path.dirname(model_filename),
    os.path.basename(model_filename) + "_cross_corr_bootstrap",
)
torch.save(joint_model.state_dict(), out_path)
print(f"\nSaved to: {out_path}")
