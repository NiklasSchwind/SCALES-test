"""
Bootstrap a scales_ssm_tas_pr_hyst_correlated.DeepSSMPatternConditioned
(low-rank covariance) from a trained
scales_ssm_z2tasAndpr_hysteresis.DeepSSMPatternConditioned (diagonal).

The two models are architecturally identical except for emit.net.4:
  diagonal:   output size = 2 * y_dim
  correlated: output size = y_dim * (2 + cov_rank)

Strategy:
  - All parameters except emit.net.4 are copied directly (shapes match).
  - emit.net.4 weight/bias: first 2*y_dim rows copied, the remaining
    y_dim*cov_rank rows (the U-matrix / cov_factor) are zeroed.
"""

import os
import torch
import proto_scales.ssm_model.scales_ssm_z2tasAndpr_hysteresis as scales_ssm_z2pr_diagonal
import proto_scales.ssm_model.scales_ssm_tas_pr_hyst_correlated as scales_ssm_z2pr_correlated

# ── architecture params (must match the checkpoint being loaded) ──────────────
zdim           = 64
rnn_hidden     = 256
use_linear_model = True
emission_uses_u  = True
alpha_max        = 0.002
resevoir_dim     = 4
cov_rank         = 5

#scaler_path    = "/home/kainverena/PythonProjects/outputs_ssm_scales/scales_20260514_171221/"
model_filename = "/home/kainverena/PythonProjects/outputs_ssm_scales/scales_20260514_171221/model_out"
# ─────────────────────────────────────────────────────────────────────────────

diag_sd = torch.load(model_filename, map_location="cpu")

# Infer dims from the diagonal state dict
mlp_hidden  = diag_sd["emit.net.0.weight"].shape[0]
y_dim       = diag_sd["emit.net.4.weight"].shape[0] // 2
u_dim       = diag_sd["ctrl_lin.weight"].shape[1]
u_rnn_hidden = diag_sd["u_gru.weight_hh_l0"].shape[1]

print(f"Inferred  y_dim={y_dim}  u_dim={u_dim}  mlp_hidden={mlp_hidden}  "
      f"u_rnn_hidden={u_rnn_hidden}")

corr_model = scales_ssm_z2pr_correlated.DeepSSMPatternConditioned(
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

corr_sd = corr_model.state_dict()
copied, partial, skipped = [], [], []

for name in corr_sd:
    if name == "emit.net.4.weight":
        corr_sd[name] = torch.zeros_like(corr_sd[name])
        corr_sd[name][: 2 * y_dim] = diag_sd["emit.net.4.weight"]
        partial.append(
            f"  {name}  rows copied={2*y_dim}/{corr_sd[name].shape[0]}  rest=zero"
        )
    elif name == "emit.net.4.bias":
        corr_sd[name] = torch.zeros_like(corr_sd[name])
        corr_sd[name][: 2 * y_dim] = diag_sd["emit.net.4.bias"]
        partial.append(
            f"  {name}  entries copied={2*y_dim}/{corr_sd[name].shape[0]}  rest=zero"
        )
    elif name in diag_sd and diag_sd[name].shape == corr_sd[name].shape:
        corr_sd[name] = diag_sd[name].clone()
        copied.append(f"  {name}  {tuple(corr_sd[name].shape)}")
    else:
        reason = ("not in diagonal model" if name not in diag_sd
                  else f"shape mismatch  diagonal={tuple(diag_sd[name].shape)}  "
                       f"correlated={tuple(corr_sd[name].shape)}")
        skipped.append(f"  {name}  {reason}")

corr_model.load_state_dict(corr_sd)

print(f"\nCopied exactly ({len(copied)}):")
print("\n".join(copied))
print(f"\nPartially copied ({len(partial)}):")
print("\n".join(partial))
if skipped:
    print(f"\nLeft at random init ({len(skipped)}):")
    print("\n".join(skipped))

out_path = os.path.join(
    os.path.dirname(model_filename),
    os.path.basename(model_filename) + "_correlated_bootstrap",
)
torch.save(corr_model.state_dict(), out_path)
print(f"\nSaved to: {out_path}")
