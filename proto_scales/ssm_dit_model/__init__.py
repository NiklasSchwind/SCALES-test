"""SSM-conditioned outpainting diffusion transformer for SCALES."""

from proto_scales.ssm_dit_model.memory_kernel import MultiTimescaleMemory, annual_means
from proto_scales.ssm_dit_model.conditioning import SSMLatentEncoder, ConditioningBuilder
from proto_scales.ssm_dit_model.dit import DiT1D
from proto_scales.ssm_dit_model.annual_ssm import (
    AnnualSSM,
    AnnualSSMLatentEncoder,
    broadcast_to_monthly,
    to_annual,
)
from proto_scales.ssm_dit_model.ssm_dit import (
    SSMConditionedOutpaintingDiT,
    build_model,
    cosine_beta_schedule,
)
# `annual_ssm_dit` defines its own `build_model` and `cosine_beta_schedule`;
# they are deliberately not re-exported here so they cannot be confused with the
# ones above. Import them from the module directly.
from proto_scales.ssm_dit_model.annual_ssm_dit import (
    AnnualSSMOutpaintingDiT,
    ForcedAnnualLatentEncoder,
    ForcedAnnualSSM,
)

__all__ = [
    "MultiTimescaleMemory",
    "annual_means",
    "SSMLatentEncoder",
    "ConditioningBuilder",
    "DiT1D",
    "AnnualSSM",
    "AnnualSSMLatentEncoder",
    "to_annual",
    "broadcast_to_monthly",
    "SSMConditionedOutpaintingDiT",
    "build_model",
    "cosine_beta_schedule",
    "AnnualSSMOutpaintingDiT",
    "ForcedAnnualSSM",
    "ForcedAnnualLatentEncoder",
]
