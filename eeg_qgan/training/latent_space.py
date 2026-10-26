"""The latent coordinate system.

Two audit findings live here.

E2, the range mismatch. Every generator ends in a bounded nonlinearity: the
classical arms in ``tanh``, the quantum arm in a Pauli-Z expectation. Their
reachable set is therefore the cube ``(-1, 1)^d``. The encoder bottleneck is also
``tanh``, so with no transform at all the two ranges match exactly. Standardising
the target to zero mean and unit variance breaks that match: a standardised target
places roughly a third of its mass per coordinate outside ``[-1, 1]``, where no arm
can place any mass, and the reported distance inherits an irreducible floor. Worse,
the severity varies by fold, because a coordinate whose standard deviation collapses
to 0.018 has its support stretched to ``±55`` while the generator still reaches
``±1``, giving two percent coverage.

F1, the coordinate system. The mixture arm was fitted and scored on raw codes while
the adversarial arms were scored on standardised ones, so a single leaderboard mixed
two coordinate systems. The inverse transform was also applied to every arm on the
sampling path, including the arm that was never transformed in the first place, so
its samples were rescaled once too often.

The fix is to make the coordinate system an explicit object that every arm shares.
``LatentTransform`` is fitted once per fold on the generator's training codes and
then handed to every arm, every metric and every sampling call. ``mode`` records
which convention is in force so it appears in the saved record.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

TransformMode = Literal["identity", "whiten", "whiten_scaled"]

ArrayLike = np.ndarray | torch.Tensor


@dataclass
class LatentTransform:
    """A fitted, invertible map between encoder codes and generator space.

    Attributes:
        mode: which convention is in force.
        mean: per-coordinate mean of the training codes.
        std: per-coordinate standard deviation, floored to stay invertible.
        output_scale: multiplier a generator should apply to its bounded output so
            that it can cover the transformed target. 1.0 when no scaling is needed.
    """

    mode: TransformMode
    mean: np.ndarray
    std: np.ndarray
    output_scale: float = 1.0
    coverage_before_scale: float = 1.0

    # ---------------------------------------------------------------- forward
    def to_generator_space(self, z: ArrayLike) -> ArrayLike:
        """Map encoder codes into the space the critic and the metrics use."""
        if self.mode == "identity":
            return z
        mean, std = self._params_like(z)
        return (z - mean) / std

    def to_code_space(self, z: ArrayLike) -> ArrayLike:
        """Map generator output back to encoder-code space for decoding."""
        if self.mode == "identity":
            return z
        mean, std = self._params_like(z)
        return z * std + mean

    def _params_like(self, z: ArrayLike):
        if isinstance(z, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=z.dtype, device=z.device)
            std = torch.as_tensor(self.std, dtype=z.dtype, device=z.device)
        else:
            mean, std = self.mean, self.std
        return mean, std

    # ------------------------------------------------------------- reporting
    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mean": [float(v) for v in np.asarray(self.mean).ravel()],
            "std": [float(v) for v in np.asarray(self.std).ravel()],
            "output_scale": float(self.output_scale),
            "coverage_before_scale": float(self.coverage_before_scale),
        }

    @property
    def is_identity(self) -> bool:
        return self.mode == "identity"


def fit_latent_transform(
    z_train: np.ndarray,
    mode: TransformMode = "identity",
    std_floor: float = 1e-2,
    robust: bool = True,
) -> LatentTransform:
    """Fit the coordinate system on the generator's training codes only.

    Args:
        z_train: encoder codes the generator will be trained on. Never pass the
            selection or reporting split here.
        mode: ``identity`` keeps the encoder's own bounded range, which already
            matches the generators. ``whiten_scaled`` standardises and tells the
            generator how far to scale its bounded output so the target stays
            reachable. ``whiten`` is standardisation with no compensation and is
            rejected by config validation.
        std_floor: lower bound on the per-coordinate scale. The previous floor of
            1e-6 let a standard deviation of 0.018 through, which stretched that
            coordinate's support to roughly ``±55`` while the generator still
            reached ``±1``.
        robust: derive the scale from the interquartile range rather than the
            standard deviation, so one saturated coordinate cannot collapse it.
    """
    z = np.asarray(z_train, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected codes of shape (n, d), got {z.shape}")

    mean = z.mean(axis=0).astype(np.float32)
    if robust:
        q75, q25 = np.percentile(z, [75, 25], axis=0)
        # 1.349 converts the IQR of a normal distribution to its standard deviation.
        scale = ((q75 - q25) / 1.349).astype(np.float32)
        fallback = z.std(axis=0).astype(np.float32)
        scale = np.where(scale > std_floor, scale, fallback)
    else:
        scale = z.std(axis=0).astype(np.float32)
    std = np.maximum(scale, np.float32(std_floor))

    if mode == "identity":
        return LatentTransform(mode="identity", mean=np.zeros_like(mean), std=np.ones_like(std))

    # Fraction of the standardised target a generator bounded to +/-1 could reach,
    # recorded so the cost of standardising is visible in the saved record.
    standardised = (z - mean) / std
    reach = float(np.mean(np.abs(standardised) <= 1.0))

    output_scale = 1.0
    if mode == "whiten_scaled":
        # Cover the observed support with a small margin so the tails stay reachable.
        output_scale = float(np.abs(standardised).max() * 1.05)
        output_scale = max(output_scale, 1.0)

    return LatentTransform(
        mode=mode,
        mean=mean,
        std=std,
        output_scale=output_scale,
        coverage_before_scale=reach,
    )


def support_diagnostics(z_reference: np.ndarray, transform: LatentTransform) -> dict:
    """Quantify how much of the target a bounded generator can reach.

    Reported per fold so that the range mismatch, if anyone reintroduces it, shows
    up as a number in the record rather than as an unexplained distance floor.
    """
    z = np.asarray(z_reference, dtype=np.float32)
    target = np.asarray(transform.to_generator_space(z), dtype=np.float32)
    reachable = float(transform.output_scale)

    within = np.abs(target) <= reachable
    per_coord = within.mean(axis=0)
    return {
        "reachable_bound": reachable,
        "fraction_within_reach": float(within.mean()),
        "worst_coordinate_coverage": float(per_coord.min()),
        "per_coordinate_coverage": [float(v) for v in per_coord],
        "target_abs_max": float(np.abs(target).max()),
        "target_std_mean": float(target.std(axis=0).mean()),
    }
