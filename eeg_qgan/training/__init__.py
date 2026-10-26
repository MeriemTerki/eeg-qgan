"""Training loops and the shared latent coordinate system."""

from eeg_qgan.training.latent_space import (
    LatentTransform,
    fit_latent_transform,
    support_diagnostics,
)

__all__ = ["LatentTransform", "fit_latent_transform", "support_diagnostics"]
