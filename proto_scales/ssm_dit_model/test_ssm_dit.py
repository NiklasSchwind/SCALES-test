"""
Smoke tests for the SSM-conditioned outpainting DiT.

The causality tests are the important ones: the whole reason this architecture
is expected to generalise across GSAT trajectory shapes is that every
conditioning signal is a causal function of u_{<=t}. That property is easy to
break silently with an off-by-one in the annual indexing, and nothing in the
training loss would flag it — it would just quietly leak the target and look
like unusually good validation performance.

Run: pytest proto_scales/ssm_dit_model/test_ssm_dit.py
"""

import pytest
import torch

from proto_scales.ssm_dit_model import MultiTimescaleMemory, annual_means, build_model

B, Y_DIM, U_DIM, TC, H = 2, 6, 1, 24, 24


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(
        y_dim=Y_DIM, u_dim=U_DIM, z_dim=8, rnn_hidden=16, cov_rank=3,
        context_len=TC, horizon=H, cond_dim=64, hidden=64, depth=2, heads=4,
        n_diffusion_steps=100, field_memory_rank=8,
    )


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return dict(
        tas_ctx=torch.randn(B, TC, Y_DIM), pr_ctx=torch.randn(B, TC, Y_DIM),
        u_ctx=torch.randn(B, TC, U_DIM),
        tas_fut=torch.randn(B, H, Y_DIM), pr_fut=torch.randn(B, H, Y_DIM),
        u_fut=torch.randn(B, H, U_DIM),
    )


# ---------------------------------------------------------------- memory kernel
def test_memory_kernel_shape():
    mk = MultiTimescaleMemory((1.0, 5.0, 20.0, 100.0))
    x = torch.randn(B, 12 * 8, 3)
    assert mk(x).shape == (B, 12 * 8, 3 * 4)


def test_memory_kernel_is_causal():
    """Perturbing the future must not change any past feature."""
    mk = MultiTimescaleMemory((1.0, 5.0, 20.0, 100.0))
    x = torch.randn(B, 12 * 8, 3)
    t = 12 * 4 + 5
    x2 = x.clone()
    x2[:, t:] += 100.0
    assert torch.allclose(mk(x)[:, : t + 1], mk(x2)[:, : t + 1], atol=1e-5)


def test_memory_kernel_no_same_year_leakage():
    """A month must not see its own year's annual mean — that is the target."""
    mk = MultiTimescaleMemory((1.0, 5.0, 20.0, 100.0))
    x = torch.randn(B, 12 * 8, 3)
    yr = 4
    x2 = x.clone()
    x2[:, yr * 12:(yr + 1) * 12] += 100.0
    f, f2 = mk(x), mk(x2)
    assert torch.allclose(f[:, yr * 12:(yr + 1) * 12], f2[:, yr * 12:(yr + 1) * 12], atol=1e-5)
    # but the perturbation must register in later years, or the kernel is inert
    assert not torch.allclose(f[:, (yr + 1) * 12:], f2[:, (yr + 1) * 12:], atol=1e-3)


def test_memory_kernel_first_year_is_zero():
    mk = MultiTimescaleMemory((1.0, 5.0))
    f = mk(torch.randn(B, 12 * 3, 3))
    assert torch.allclose(f[:, :12], torch.zeros_like(f[:, :12]))


def test_annual_means_drops_partial_year():
    assert annual_means(torch.randn(B, 12 * 3 + 7, 3)).shape[1] == 3


# ------------------------------------------------------------- parameterisation
def test_default_is_v_parameterization(model):
    assert model.parameterization == "v"


@pytest.fixture(scope="module")
def model_1000():
    """Full 1000-step schedule, so the extreme-noise tail can be exercised."""
    torch.manual_seed(0)
    return build_model(
        y_dim=Y_DIM, u_dim=U_DIM, z_dim=8, rnn_hidden=16, cov_rank=3,
        context_len=TC, horizon=H, cond_dim=32, hidden=32, depth=2, heads=4,
        n_diffusion_steps=1000, field_memory_rank=8)


@pytest.mark.parametrize("t", [0, 1, 100, 500, 900, 999])
def test_v_target_inverts_exactly(model_1000, t):
    """
    Feeding the analytically correct v back through `_to_x0_eps` must recover
    x0 and eps. If this drifts, sampling is biased in a way that is very hard
    to see in the loss curve.
    """
    torch.manual_seed(0)
    x0 = torch.randn(3, 24, 2 * Y_DIM)
    eps = torch.randn_like(x0)
    tt = torch.tensor(t)
    a, s = model_1000.sqrt_ab[tt], model_1000.sqrt_1mab[tt]
    x_t = a * x0 + s * eps
    x0_hat, eps_hat = model_1000._to_x0_eps(a * eps - s * x0, x_t, tt)
    assert torch.allclose(x0_hat, x0, atol=1e-4)
    assert torch.allclose(eps_hat, eps, atol=1e-3)


def test_v_is_better_conditioned_than_eps_at_high_noise():
    """
    The reason for the v default: the eps route computes x0 = (x_t - s*eps)/a
    and a -> 0 at high noise, so output error is amplified without bound.
    """
    torch.manual_seed(0)
    kw = dict(y_dim=Y_DIM, u_dim=U_DIM, z_dim=8, rnn_hidden=16, cov_rank=3,
              context_len=TC, horizon=H, cond_dim=32, hidden=32, depth=2,
              heads=4, n_diffusion_steps=1000, field_memory_rank=8)
    mv = build_model(parameterization="v", **kw)
    me = build_model(parameterization="eps", **kw)

    x0 = torch.randn(2, 24, 2 * Y_DIM)
    eps = torch.randn_like(x0)
    tt = torch.tensor(999)
    a, s = mv.sqrt_ab[tt], mv.sqrt_1mab[tt]
    x_t = a * x0 + s * eps
    perturb = 0.01 * torch.randn_like(x0)

    err_v = (mv._to_x0_eps(a * eps - s * x0 + perturb, x_t, tt)[0] - x0).abs().max()
    err_e = (me._to_x0_eps(eps + perturb, x_t, tt)[0] - x0).abs().max()
    assert err_v < 0.5
    assert err_e > 10 * err_v


@pytest.mark.parametrize("param", ["v", "eps"])
def test_both_parameterizations_train_and_sample(param, batch):
    torch.manual_seed(0)
    m = build_model(
        y_dim=Y_DIM, u_dim=U_DIM, z_dim=8, rnn_hidden=16, cov_rank=3,
        context_len=TC, horizon=H, cond_dim=64, hidden=64, depth=2, heads=4,
        n_diffusion_steps=100, field_memory_rank=8, parameterization=param)
    loss = m.loss(**batch, start_month=0)
    assert torch.isfinite(loss)
    loss.backward()
    m.eval()
    tas, pr = m.sample(batch["tas_ctx"], batch["pr_ctx"], batch["u_ctx"],
                       batch["u_fut"], start_month=0, n_steps=10)
    assert torch.isfinite(tas).all() and torch.isfinite(pr).all()


# ---------------------------------------------------------------------- model
def test_loss_backward_and_ssm_stays_frozen(model, batch):
    """
    adaLN-Zero means final.linear starts at exactly zero, so no gradient reaches
    the trunk on step 1 by design. Gradient flow to the conditioning path is
    therefore only meaningful after the first optimiser step.
    """
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        loss = model.loss(**batch, start_month=3)
        loss.backward()
        opt.step()

    gsum = lambda m: sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
    assert gsum(model.dit) > 0
    assert gsum(model.cond_builder) > 0, "conditioning path is receiving no gradient"
    tau_grad = model.cond_builder.mem_u.log_tau.grad
    assert tau_grad is not None and float(tau_grad.abs().sum()) > 0
    assert not [p for p in model.ssm_encoder.ssm.parameters() if p.grad is not None], \
        "frozen SSM received gradients"


def test_sample_shapes_and_stochasticity(model, batch):
    model.eval()
    kw = dict(tas_ctx=batch["tas_ctx"], pr_ctx=batch["pr_ctx"],
              u_ctx=batch["u_ctx"], u_fut=batch["u_fut"], start_month=3, n_steps=10)
    tas, pr = model.sample(**kw)
    assert tas.shape == (B, H, Y_DIM) and pr.shape == (B, H, Y_DIM)
    assert torch.isfinite(tas).all() and torch.isfinite(pr).all()
    # it is a generative model; two draws must not coincide
    assert not torch.allclose(model.sample(**kw)[0], model.sample(**kw)[0])


def test_sample_blocks_handles_ragged_horizon(model, batch):
    """Horizon deliberately not a multiple of the block size."""
    model.eval()
    h_long = 3 * H + 7
    u_long = torch.randn(B, h_long, U_DIM)
    tas, pr = model.sample_blocks(
        batch["tas_ctx"], batch["pr_ctx"], batch["u_ctx"], u_long,
        start_month=3, n_steps=5)
    assert tas.shape == (B, h_long, Y_DIM) and pr.shape == (B, h_long, Y_DIM)
    assert torch.isfinite(tas).all() and torch.isfinite(pr).all()


def test_short_context_is_rejected():
    """Below two years the annual-mean field memory is identically zero."""
    with pytest.raises(ValueError, match="context_len"):
        build_model(y_dim=Y_DIM, u_dim=U_DIM, z_dim=8, rnn_hidden=16, cov_rank=3,
                    context_len=12, horizon=12, cond_dim=32, hidden=32, depth=2,
                    heads=4, n_diffusion_steps=100, field_memory_rank=8)


@pytest.mark.parametrize("freeze", [True, False])
def test_no_trainable_parameter_is_left_unreduced(freeze):
    """
    Every parameter that requires grad must actually receive one, or DDP raises
    on unreduced parameters. In joint mode this means the SSM's emission head
    and reservoir - which the DiT replaces - must stay frozen.
    """
    torch.manual_seed(0)
    m = build_model(y_dim=Y_DIM, u_dim=U_DIM, z_dim=8, rnn_hidden=16, cov_rank=3,
                    context_len=TC, horizon=H, cond_dim=32, hidden=32, depth=2,
                    heads=4, n_diffusion_steps=100, field_memory_rank=8,
                    freeze_ssm=freeze)
    args = (torch.randn(B, TC, Y_DIM), torch.randn(B, TC, Y_DIM), torch.randn(B, TC, U_DIM),
            torch.randn(B, H, Y_DIM), torch.randn(B, H, Y_DIM), torch.randn(B, H, U_DIM))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    for i in range(4):
        opt.zero_grad()
        m.loss(*args).backward()
        if i < 3:
            opt.step()
    bad = [n for n, p in m.named_parameters()
           if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)]
    assert not bad, f"trainable but unreduced: {bad}"
    if freeze:
        assert not m.ssm_encoder.trainable_ssm_parameters()
    else:
        assert m.ssm_encoder.trainable_ssm_parameters()
