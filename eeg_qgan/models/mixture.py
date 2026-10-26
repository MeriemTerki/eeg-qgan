"""Gaussian mixture reference, fitted in the shared coordinate system.

Audit item F1. The mixture arm used to be fitted and scored on raw encoder codes while
the adversarial arms were fitted and scored on standardised ones, and the two were then
listed in one table as though the numbers were comparable. They were not: they were
distances in different coordinate systems. On top of that the sampling path applied the
inverse standardisation to every arm's output, including this one, so the mixture's
samples were rescaled by a transform that had never been applied to them in the first
place, pushing them outside the encoder's valid range before decoding.

The fix is that this arm receives the same :class:`LatentTransform` as every other arm
and applies it in the same places. Being non-adversarial it never sees the critic, but
it is fitted, sampled and scored in exactly one space.

This arm matters more than its simplicity suggests: it is the reference that says
whether adversarial training buys anything at all. In the audited runs it won on the
distance metric, which is a result worth reporting correctly.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from eeg_qgan.training.latent_space import LatentTransform


class GaussianMixtureGenerator:
    """Full-covariance Gaussian mixture over latent codes.

    Not an ``nn.Module``: it has no gradient-based training and no noise input of the
    kind the adversarial arms use, so pretending otherwise would obscure the
    difference. It exposes the same sampling contract the runner needs.
    """

    def __init__(
        self,
        latent_dim: int,
        n_components: int = 8,
        covariance_type: str = "full",
        reg_covar: float = 1e-4,
        seed: int = 42,
        clip_to_range: bool = True,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.n_components = int(n_components)
        self.covariance_type = str(covariance_type)
        self.clip_to_range = bool(clip_to_range)
        self.seed = int(seed)
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=float(reg_covar),
            random_state=self.seed,
            max_iter=500,
            n_init=3,
        )
        self._fitted = False
        self._fit_space = "unfitted"

    def fit(self, z_train: np.ndarray, transform: LatentTransform | None = None) -> dict:
        """Fit in the shared generator space.

        The transform is applied here and its inverse in :meth:`sample`, so this arm
        occupies the same coordinate system as every other arm.
        """
        z = np.asarray(z_train, dtype=np.float64)
        if transform is not None:
            z = np.asarray(transform.to_generator_space(z), dtype=np.float64)
        n_components = min(self.n_components, max(1, len(z) // 10))
        if n_components != self.n_components:
            self._gmm.set_params(n_components=n_components)
            self.n_components = n_components

        self._gmm.fit(z)
        self._fitted = True
        self._fit_space = transform.mode if transform is not None else "identity"
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "converged": bool(self._gmm.converged_),
            "n_iter": int(self._gmm.n_iter_),
            "log_likelihood": float(self._gmm.score(z)),
            "fit_space": self._fit_space,
            "n_train": int(len(z)),
        }

    def sample(
        self, n: int, transform: LatentTransform | None = None, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Draw ``n`` codes and return them in encoder-code space.

        The inverse transform is applied exactly once, and only if this arm was fitted
        in a transformed space.
        """
        if not self._fitted:
            raise RuntimeError("GaussianMixtureGenerator.sample called before fit")
        if rng is not None:
            self._gmm.random_state = int(rng.integers(0, 2**31 - 1))

        z, _ = self._gmm.sample(int(n))
        z = np.asarray(z, dtype=np.float32)

        if transform is not None:
            if transform.mode != self._fit_space:
                raise ValueError(
                    f"Mixture was fitted in '{self._fit_space}' space but sampling was "
                    f"asked for '{transform.mode}'. Mixing the two is exactly the bug "
                    "that made this arm's numbers incomparable with the others."
                )
            z = np.asarray(transform.to_code_space(z), dtype=np.float32)

        if self.clip_to_range:
            # The encoder bottleneck is a tanh, so codes outside [-1, 1] cannot be
            # produced by the encoder and the decoder has never seen them. Clipping is
            # recorded rather than silent because it changes the distribution.
            z = np.clip(z, -1.0, 1.0)
        return z.astype(np.float32)

    def count_params(self) -> int:
        """Effective parameter count, for the capacity table.

        A full-covariance mixture carries means, covariances and weights, and this
        typically dwarfs both the circuit and its twin. That is worth stating whenever
        this arm wins.
        """
        d, k = self.latent_dim, self.n_components
        if self.covariance_type == "full":
            per_component_cov = d * (d + 1) // 2
        elif self.covariance_type == "diag":
            per_component_cov = d
        else:
            per_component_cov = 1
        return int(k * (d + per_component_cov) + (k - 1))

    def describe(self) -> dict:
        return {
            "kind": "gaussian_mixture",
            "latent_dim": self.latent_dim,
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "effective_params": self.count_params(),
            "fit_space": self._fit_space,
            "clips_to_encoder_range": self.clip_to_range,
            "is_adversarial": False,
        }


def build_mixture(cfg: dict, seed: int = 42) -> GaussianMixtureGenerator:
    mix_cfg = cfg.get("mixture", {}) or {}
    return GaussianMixtureGenerator(
        latent_dim=int(cfg["latent_dim"]),
        n_components=int(mix_cfg.get("n_components", 8)),
        covariance_type=str(mix_cfg.get("covariance_type", "full")),
        reg_covar=float(mix_cfg.get("reg_covar", 1e-4)),
        seed=int(seed),
        clip_to_range=bool(mix_cfg.get("clip_to_range", True)),
    )
