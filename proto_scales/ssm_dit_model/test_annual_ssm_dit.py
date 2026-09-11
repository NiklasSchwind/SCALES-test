"""
Tests for the annual-SSM-conditioned outpainting DiT.

The state tests are the important ones. This architecture exists because
`ssm_dit.sample_blocks` re-derived z from the trailing context window at every
block, so no latent state crossed a block boundary and nothing could carry
information further back than `context_len` months. `test_z_carries_state_across_
blocks` is the regression test for exactly that: it perturbs only the *first*
year of forcing and requires the perturbation to still be visible in z a century
later. Nothing in the training loss would flag a regression here — the model
would simply stop generalising across pathway shapes, which is only visible
after a full retrain.

The causality tests matter for the same reason as in `test_ssm_dit`: every
conditioning signal must be a causal function of u_{<=t} and of the observed
context, and an off-by-one in the annual indexing would quietly leak the target.

Run: pytest proto_scales/ssm_dit_model/test_annual_ssm_dit.py
"""

import math

import numpy as np
import pytest
import torch

from proto_scales.ssm_dit_model.annual_dit_training import LongHorizonWindowDataset
from proto_scales.ssm_dit_model.annual_ssm_dit import build_model

B, Y_DIM, U_DIM = 2, 6, 1
TC, BLK, HOR = 24, 24, 120           # 2 yr context, 2 yr block, 10 yr future


def make_model(**kw):
    torch.manual_seed(0)
    base = dict(
        y_dim=Y_DIM, u_dim=U_DIM, z_dim=8, rnn_hidden=16,
        context_len=TC, block_len=BLK, cond_dim=64, hidden=64, depth=2, heads=4,
        n_diffusion_steps=100, field_memory_rank=8, timescales_years=(1.0, 5.0),
    )
    base.update(kw)
    return build_model(**base)


@pytest.fixture(scope="module")
def model():
    return make_model()


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return dict(
        tas_ctx=torch.randn(B, TC, Y_DIM), pr_ctx=torch.randn(B, TC, Y_DIM),
        u_ctx=torch.randn(B, TC, U_DIM),
        tas_fut=torch.randn(B, HOR, Y_DIM), pr_fut=torch.randn(B, HOR, Y_DIM),
        u_fut=torch.randn(B, HOR, U_DIM),
    )


# ─────────────────────────────────────────────────────────────────────────────
# latent state
# ─────────────────────────────────────────────────────────────────────────────
def test_z_carries_state_across_blocks(model):
    """
    A perturbation confined to the first year of forcing must still be visible
    in z ten years (five blocks) later.

    This is the property `ssm_dit` did not have: there, z was recomputed from the
    trailing `context_len` window per block, so anything older than the window
    was unreachable.
    """
    tas_ctx = torch.randn(B, TC, Y_DIM)
    pr_ctx = torch.randn(B, TC, Y_DIM)
    u_ctx = torch.randn(B, TC, U_DIM)

    u_a = torch.zeros(B, HOR, U_DIM)
    u_b = u_a.clone()
    u_b[:, :12] += 1.0                      # first year only

    with torch.no_grad():
        z_a = model.latents(tas_ctx, pr_ctx, u_ctx, u_a)
        z_b = model.latents(tas_ctx, pr_ctx, u_ctx, u_b)

    assert (z_a[:, -1] - z_b[:, -1]).abs().max() > 1e-5
    # and the context half, which precedes the perturbation, must be untouched
    assert torch.allclose(z_a[:, :TC], z_b[:, :TC], atol=1e-6)


def test_z_is_causal_in_forcing(model):
    """Perturbing forcing at year 5 must not move z before year 5."""
    tas_ctx, pr_ctx = torch.randn(B, TC, Y_DIM), torch.randn(B, TC, Y_DIM)
    u_ctx = torch.randn(B, TC, U_DIM)
    u_a = torch.zeros(B, HOR, U_DIM)
    u_b = u_a.clone()
    u_b[:, 60:72] += 5.0                    # year 5 of the future

    with torch.no_grad():
        z_a = model.latents(tas_ctx, pr_ctx, u_ctx, u_a)
        z_b = model.latents(tas_ctx, pr_ctx, u_ctx, u_b)

    assert torch.allclose(z_a[:, : TC + 60], z_b[:, : TC + 60], atol=1e-6)
    assert (z_a[:, TC + 60:] - z_b[:, TC + 60:]).abs().max() > 1e-5


def test_transition_is_a_contraction(model):
    """Real eigenvalues strictly inside the unit circle, so rollouts are stable."""
    d = model.ssm_encoder.ssm.decay()
    assert (d > 0).all() and (d < 1).all()


def test_long_rollout_stays_bounded(model):
    """A 500-year rollout must not diverge — the point of the sigmoid decay."""
    ssm = model.ssm_encoder.ssm
    with torch.no_grad():
        z = ssm.roll_forward(torch.randn(B, ssm.z_dim), torch.randn(B, 500, U_DIM))
    assert torch.isfinite(z).all()
    assert z.abs().max() < 1e3


def test_mean_propagation_matches_manual_iteration(model):
    """`roll_forward(stochastic=False)` is exactly the conditional mean."""
    ssm = model.ssm_encoder.ssm
    u = torch.randn(B, 7, U_DIM)
    z0 = torch.randn(B, ssm.z_dim)
    with torch.no_grad():
        rolled = ssm.roll_forward(z0, u, stochastic=False)
        z = z0
        manual = []
        for k in range(u.shape[1]):
            z = ssm.decay() * z + ssm.B(u[:, k])
            manual.append(z)
    assert torch.allclose(rolled, torch.stack(manual, dim=1), atol=1e-6)


def test_conditioning_z_is_deterministic_by_default(model):
    """Two calls must agree: the default conditioning rollout has no noise."""
    args = (torch.randn(B, TC, Y_DIM), torch.randn(B, TC, Y_DIM),
            torch.randn(B, TC, U_DIM), torch.randn(B, HOR, U_DIM))
    with torch.no_grad():
        assert torch.allclose(model.latents(*args), model.latents(*args), atol=1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# blocks
# ─────────────────────────────────────────────────────────────────────────────
def test_block_bounds_tile_the_future(model):
    n = model.n_blocks(HOR)
    assert n == HOR // BLK
    covered = []
    for bi in range(n):
        w0, bs, nb = model.block_bounds(bi, HOR)
        assert bs - w0 == TC                     # always a full context window
        covered.append((bs, bs + nb))
    # contiguous, no gaps, no overlap, starting right after the context
    assert covered[0][0] == TC
    assert covered[-1][1] == TC + HOR
    for (_, end), (start, _) in zip(covered, covered[1:]):
        assert end == start


def test_block_idx_past_end_raises(model):
    with pytest.raises(IndexError):
        model.block_bounds(model.n_blocks(HOR), HOR)


# ─────────────────────────────────────────────────────────────────────────────
# losses
# ─────────────────────────────────────────────────────────────────────────────
def test_loss_finite_and_all_params_get_gradient(batch):
    m = make_model()
    loss, parts = m.loss(**batch, block_idx=[0, 4], start_month=3, return_parts=True)
    assert torch.isfinite(loss)
    for k in ("diffusion", "ssm_nll", "ssm_kl", "ssm_rollout"):
        assert k in parts
    # per-block reporting, so drift with lead time is visible from one call
    assert "diff_blk0" in parts and "diff_blk4" in parts

    loss.backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, missing


def test_rollout_loss_trains_the_transition():
    """
    The multi-step term must reach the transition parameters — it is the only
    thing that scores the operator the sampler iterates.
    """
    m = make_model(ssm_elbo_weight=0.0, ssm_rollout_weight=1.0)
    ssm = m.ssm_encoder.ssm
    y_ctx = torch.randn(B, 2, 2 * Y_DIM)
    u_ctx = torch.randn(B, 2, U_DIM)
    y_fut = torch.randn(B, 10, 2 * Y_DIM)
    u_fut = torch.randn(B, 10, U_DIM)

    ssm.rollout_loss(y_ctx, u_ctx, y_fut, u_fut).backward()
    assert ssm.logit_decay.grad is not None
    assert ssm.logit_decay.grad.abs().sum() > 0
    assert ssm.B.weight.grad.abs().sum() > 0


def test_block_choice_changes_the_diffusion_term(batch):
    """
    Different blocks see different z, so they must not give identical losses —
    if they did, z would not be carrying anything across the horizon.
    """
    m = make_model()
    t_fixed = torch.zeros(B, dtype=torch.long) + 50
    with torch.no_grad():
        _, p0 = m.loss(**batch, block_idx=[0], t_diff=t_fixed, return_parts=True)
        _, p4 = m.loss(**batch, block_idx=[4], t_diff=t_fixed, return_parts=True)
    assert abs(float(p0["diffusion"]) - float(p4["diffusion"])) > 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# sampling
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("horizon", [BLK, HOR, HOR + 10])
def test_sample_blocks_shapes(model, horizon):
    """Includes a horizon that is not a whole number of blocks."""
    with torch.no_grad():
        tas, pr = model.sample_blocks(
            torch.randn(B, TC, Y_DIM), torch.randn(B, TC, Y_DIM),
            torch.randn(B, TC, U_DIM), torch.randn(B, horizon, U_DIM),
            start_month=3, n_steps=4)
    assert tas.shape == (B, horizon, Y_DIM)
    assert pr.shape == (B, horizon, Y_DIM)
    assert torch.isfinite(tas).all() and torch.isfinite(pr).all()


def test_sample_blocks_is_reproducible(model):
    args = (torch.randn(B, TC, Y_DIM), torch.randn(B, TC, Y_DIM),
            torch.randn(B, TC, U_DIM), torch.randn(B, HOR, U_DIM))
    with torch.no_grad():
        a = model.sample_blocks(*args, n_steps=4,
                                generator=torch.Generator().manual_seed(7))
        b = model.sample_blocks(*args, n_steps=4,
                                generator=torch.Generator().manual_seed(7))
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


# ─────────────────────────────────────────────────────────────────────────────
# guards
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kw", [{"context_len": 25}, {"block_len": 30},
                                {"context_len": 12}])
def test_window_lengths_must_be_whole_years(kw):
    with pytest.raises(ValueError):
        make_model(**kw)


def test_bad_parameterization_rejected():
    with pytest.raises(ValueError):
        make_model(parameterization="x0")


# ─────────────────────────────────────────────────────────────────────────────
# dataset
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    T = 400
    return (rng.standard_normal((5, T, Y_DIM)), rng.standard_normal((5, T, Y_DIM)),
            rng.standard_normal((5, T, U_DIM)))


def test_dataset_shapes(arrays):
    ds = LongHorizonWindowDataset(*arrays, context_len=TC, horizon=HOR, stride=7)
    tas_c, pr_c, u_c, tas_f, pr_f, u_f, sm = ds[0]
    assert tas_c.shape == (TC, Y_DIM) and pr_c.shape == (TC, Y_DIM)
    assert u_c.shape == (TC, U_DIM)
    assert tas_f.shape == (HOR, Y_DIM) and u_f.shape == (HOR, U_DIM)
    assert 0 <= int(sm) < 12


def test_dataset_covers_every_calendar_phase(arrays):
    """A stride coprime with 12 must visit all twelve month-embedding rows."""
    ds = LongHorizonWindowDataset(*arrays, context_len=TC, horizon=HOR, stride=7)
    assert {int(ds[i][6]) for i in range(len(ds))} == set(range(12))


@pytest.mark.parametrize("stride", [12, 24, 6, 4, 3, 2])
def test_dataset_rejects_stride_sharing_a_factor_with_12(arrays, stride):
    """
    Not cosmetic: start_month = (st + data_start_month) % 12, so a stride of 12
    pins the phase to one value and the month embedding trains one of twelve
    rows. Smaller common factors restrict it to a subset.
    """
    assert math.gcd(stride, 12) != 1
    with pytest.raises(ValueError):
        LongHorizonWindowDataset(*arrays, context_len=TC, horizon=HOR, stride=stride)


def test_dataset_rejects_horizon_longer_than_trajectory(arrays):
    with pytest.raises(ValueError):
        LongHorizonWindowDataset(*arrays, context_len=TC, horizon=400)
