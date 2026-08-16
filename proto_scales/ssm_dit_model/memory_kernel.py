"""
memory_kernel
=============

Causal multi-timescale memory features on *annual* means.

Motivation
----------
In the SSM the slow reservoir (`_reservoir_step`) existed because the latent
`z` could not simultaneously carry (a) the 12-month seasonal cycle and (b) the
multi-decadal memory of the GSAT trajectory. Those are two different jobs and
they interfere: a state with a time constant short enough to resolve the
seasonal cycle cannot also integrate over a century.

This module removes job (b) from `z` entirely. It builds an explicit, causal,
multi-timescale summary of the forcing and of the regional fields, evaluated on
*annual means* so that the seasonal cycle is averaged out before the memory
kernel ever sees it. Seasonality is then handled separately and explicitly by a
month-of-year embedding (see `conditioning.py`), so `z` is left to do the one
thing it is good at: internal variability with oscillatory structure.

Form
----
For each timescale tau (in years) the feature is a discrete exponential moving
average over annual means

    ema_k[y] = ema_k[y-1] + lambda_k * (a[y] - ema_k[y-1]),
    lambda_k = 1 - exp(-1 / tau_k)

which is the discrete Green's function of a first-order system with relaxation
time tau_k. A bank of these over tau in {1, 5, 20, 100} yr approximates the
impulse response of the climate system to forcing, in the same spirit as
simple-climate-model (FaIR-like) response functions.

Why this generalises across GSAT trajectory shapes
--------------------------------------------------
The feature at month t depends on the forcing only through u_{<=t}, and only
through fixed exponential kernels. It therefore cannot key on the *global
shape* of the trajectory (monotonic ramp vs. overshoot vs. stabilisation) the
way a model conditioned on the whole trajectory at once can. Trajectory-shape
generalisation becomes structural rather than something that has to be learned
from the scenario set.
"""

import torch
import torch.nn as nn

MONTHS_PER_YEAR = 12


def annual_means(x, months_per_year=MONTHS_PER_YEAR):
    """
    Monthly series -> annual means.

    Parameters
    ----------
    x : [B, T, D] monthly values.

    Returns
    -------
    [B, Y, D] with Y = T // months_per_year. A trailing partial year is dropped
    (it is not a complete annual mean and would alias the seasonal cycle in).
    """
    B, T, D = x.shape
    Y = T // months_per_year
    if Y == 0:
        return x.new_zeros(B, 0, D)
    return x[:, : Y * months_per_year].reshape(B, Y, months_per_year, D).mean(dim=2)


class MultiTimescaleMemory(nn.Module):
    """
    Bank of causal exponential memory kernels over annual means.

    Parameters
    ----------
    timescales_years : relaxation times tau_k, in years.
    learnable        : if True the tau_k are trained (in log space, so they stay
                       positive); if False the bank is a fixed basis.

    Shapes
    ------
    forward(x): [B, T, D] monthly -> [B, T, D * n_tau] monthly.

    Causality
    ---------
    A month in calendar year `yr` sees the EMA through year `yr - 1`, i.e. only
    *completed* years strictly in the past. Months in year 0 see zeros. This
    matters: using the current year's mean would leak the value being predicted.
    """

    def __init__(self, timescales_years=(1.0, 5.0, 20.0, 100.0), learnable=True):
        super().__init__()
        tau = torch.tensor(tuple(timescales_years), dtype=torch.float32)
        if (tau <= 0).any():
            raise ValueError("timescales_years must be positive")
        self.log_tau = nn.Parameter(torch.log(tau), requires_grad=bool(learnable))
        self.n_tau = len(timescales_years)

    def extra_repr(self):
        with torch.no_grad():
            tau = torch.exp(self.log_tau).tolist()
        return f"tau_years={[round(v, 2) for v in tau]}, learnable={self.log_tau.requires_grad}"

    def forward(self, x, months_per_year=MONTHS_PER_YEAR):
        B, T, D = x.shape
        a = annual_means(x, months_per_year)                       # [B, Y, D]
        Y = a.shape[1]
        if Y == 0:
            return x.new_zeros(B, T, D * self.n_tau)

        tau = torch.exp(self.log_tau).clamp(min=1e-2)              # [K]
        lam = 1.0 - torch.exp(-1.0 / tau)                          # [K]

        ema = x.new_zeros(B, self.n_tau, D)
        hist = []
        for y in range(Y):
            # ema after this update summarises annual means 0..y inclusive
            ema = ema + lam.view(1, -1, 1) * (a[:, y].unsqueeze(1) - ema)
            hist.append(ema)
        E = torch.stack(hist, dim=1)                               # [B, Y, K, D]

        # Prepend a zero state for "year -1" so that indexing by the calendar
        # year of each month yields the EMA through the *previous* year.
        E = torch.cat([E.new_zeros(B, 1, self.n_tau, D), E], dim=1)  # [B, Y+1, K, D]

        yr = torch.arange(T, device=x.device) // months_per_year
        yr = yr.clamp(max=Y)                                        # months past the
        # last complete year reuse the most recent completed-year state
        feats = E[:, yr]                                            # [B, T, K, D]
        return feats.reshape(B, T, self.n_tau * D)

    @property
    def out_multiplier(self):
        """Output feature count per input channel."""
        return self.n_tau
