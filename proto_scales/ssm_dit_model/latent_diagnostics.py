"""
latent_diagnostics
===================

Context-length diagnostics for any of the annual latent processes
(`ForcedAnnualSSM`, `ForcedAnnualJEPA`, `ForcedAnnualJEPAKL`) via the outer DiT
models that wrap them (`AnnualSSMOutpaintingDiT` / `AnnualJEPAOutpaintingDiT` /
`AnnualJEPAKLOutpaintingDiT`), so the three can be compared on equal footing.

Motivation: a latent process whose context-conditioned state (`z_last`, what
`roll_forward` is seeded from) depends strongly on how long a context it is
given is extrapolating outside whatever the DiT was trained to condition on,
and predictions degrade as a result. `efolding_summary` reports a static,
context-independent property of the transition; `z_norm_by_context_length`
reports how `z_last`'s norm actually behaves as context length grows. Used
together: if the decay structure (e-folding times) looks stable but z_last's
norm still grows with context length, the growth is coming from the encoder,
not from the transition's own dynamics.

A second, separate question `transition_weight_summary` is for: whether a
slow dimension the decay structure *has capacity for* actually carries
anything. The transition is a bank of independent real-pole exponential
filters of the forcing (`decay() * z_prev + B(u_t)`), which can in principle
combine into a non-monotonic response (e.g. an overshoot scenario's delayed,
multi-decadal re-warming) via mixed-sign `B` weights across dimensions with
different e-folding times — but only if training actually populated the slow
dimension(s) rather than starving them. A long e-folding time paired with a
near-zero `B`-row norm for that dimension means the capacity is there but
unused ("dead"), which is a training-signal problem (the ELBO's per-dimension
KL free-bits floor forces every dimension to carry something at every step;
a plain scalar MSE, as in the deterministic JEPA loss, does not), not an
architectural ceiling.
"""

import torch
import torch.nn as nn

from proto_scales.ssm_dit_model.annual_ssm import to_annual


@torch.no_grad()
def efolding_summary(model):
    """
    Per-dimension e-folding times (years) of the transition's decay. A static
    model property — independent of context length — reported alongside
    `z_norm_by_context_length` as a reference.
    """
    return model.ssm_encoder.ssm.efolding_years()


@torch.no_grad()
def z_norm_by_context_length(model, tas_ctx, pr_ctx, u_ctx, context_lengths):
    """
    Run the latent process's posterior at several different context lengths,
    truncating `tas_ctx`/`pr_ctx`/`u_ctx` to the trailing `n` months for each
    `n` in `context_lengths`, and report `||z_last||` statistics (mean/std/max
    across the batch).

    A model whose `z_last` norm stays roughly flat across context lengths is
    stable to how much history it is given; one whose norm grows (or blows
    up) with context length is drifting out of the regime the DiT was trained
    to condition on — exactly the failure mode this function exists to make
    visible.

    `tas_ctx`/`pr_ctx`/`u_ctx` should be at least as long as `max(context_lengths)`
    months; shorter requested lengths use the trailing `n` months of the data
    given. Returns a list of dicts, one per entry in `context_lengths`:
    `{"context_len": n, "mean": ..., "std": ..., "max": ...}`.
    """
    ssm = model.ssm_encoder.ssm
    results = []
    for n in context_lengths:
        n = int(n)
        tas_n = tas_ctx[:, -n:]
        pr_n = pr_ctx[:, -n:]
        u_n = u_ctx[:, -n:]

        y_ann = to_annual(torch.cat([tas_n, pr_n], dim=-1))
        u_ann = to_annual(u_n)
        _, z_last = ssm.posterior(y_ann, u_ann)

        norms = z_last.norm(dim=-1)
        results.append({
            "context_len": n,
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "max": float(norms.max()),
        })
    return results


def print_context_length_report(model, tas_ctx, pr_ctx, u_ctx, context_lengths, label=""):
    """Pretty-print `efolding_summary` + `z_norm_by_context_length` for one model."""
    efold = efolding_summary(model)
    print(f"{label} e-folding times (yr): {[round(v, 1) for v in efold.tolist()]}")
    for row in z_norm_by_context_length(model, tas_ctx, pr_ctx, u_ctx, context_lengths):
        print(f"  context_len={row['context_len']:>5d} months | "
              f"||z_last|| mean={row['mean']:.3f} std={row['std']:.3f} max={row['max']:.3f}")


@torch.no_grad()
def transition_weight_summary(model):
    """
    Per z-dimension: e-folding time (years), the norm of that dimension's row
    of `B` (how strongly forcing drives it directly), and — for latent
    processes that have one (`ForcedAnnualSSM`) — the norm of its column in
    `emit`'s first layer (how strongly it's read back out into y).

    A dimension with a long e-folding time but a near-zero `B`-row norm has
    capacity the architecture offers but training never used: see the module
    docstring for why that's a training-signal gap (no per-dimension floor on
    the plain-MSE JEPA loss) rather than evidence the linear transition can't
    represent a delayed, multi-decadal response.

    For `emit_hidden > 0` the emission column norm is only a rough proxy —
    a hidden layer can still amplify or null a dimension further downstream —
    but with the default `emit_hidden=0` it is exact.

    Returns a list of dicts, one per z-dimension, sorted by e-folding time
    (ascending, so the slow dimensions this is usually about are at the end):
    `{"dim": i, "efold_years": ..., "b_row_norm": ...}`, plus
    `"emit_col_norm"` when the model has an emission layer.
    """
    ssm = model.ssm_encoder.ssm
    efold = ssm.efolding_years()
    b_norm = ssm.B.weight.norm(dim=1)  # [z_dim]: one norm per output z-dim row

    emit_norm = None
    if hasattr(ssm, "emit"):
        first_linear = ssm.emit if isinstance(ssm.emit, nn.Linear) else ssm.emit[0]
        emit_norm = first_linear.weight[:, :ssm.z_dim].norm(dim=0)  # [z_dim]

    rows = []
    for i in range(ssm.z_dim):
        row = {"dim": i, "efold_years": float(efold[i]), "b_row_norm": float(b_norm[i])}
        if emit_norm is not None:
            row["emit_col_norm"] = float(emit_norm[i])
        rows.append(row)
    return sorted(rows, key=lambda r: r["efold_years"])


def print_transition_weight_report(model, label=""):
    """Pretty-print `transition_weight_summary` for one model, sorted by e-folding time."""
    rows = transition_weight_summary(model)
    has_emit = "emit_col_norm" in rows[0]
    print(f"{label} per-dimension transition weights (sorted by e-folding time):")
    for row in rows:
        line = (f"  dim={row['dim']:>2d} efold={row['efold_years']:>5.1f} yr | "
                f"||B_row||={row['b_row_norm']:.4f}")
        if has_emit:
            line += f" | ||emit_col||={row['emit_col_norm']:.4f}"
        print(line)
