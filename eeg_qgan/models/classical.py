"""Classical latent generators: the controls the quantum arm is compared against.

The capacity-matched twin is the one that decides the paper's verdict, so its
construction is deliberately mechanical: the target parameter count comes from the
circuit, and the hidden widths are searched to land as close to it as an integer
architecture allows. In the headline configuration that gives 127 against 128.

Audit findings addressed here:

F6, noise parity. Every arm now takes ``noise_dim`` explicitly rather than assuming
the latent width, so a comparison cannot silently give one arm a wider input than the
other.

F12, the diverged control. The large network's selection distance sat at 0.49 to 0.60
across all ten runs against 0.02 to 0.05 for the small arms, because matched learning
rates were applied to a 5256-parameter LeakyReLU stack that simply diverges under
WGAN-GP at that setting. Any conclusion that extra capacity does not help was
therefore an untuned-optimiser result, not a capacity result. The architecture now
uses a bounded activation and optional spectral normalisation, and the runner records
a convergence flag so a diverged arm is excluded from capacity claims rather than
quietly reported.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from eeg_qgan.models.base import LatentGenerator, choose_hidden_for_target, count_mlp_params


def _mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: str = "tanh",
    spectral_norm: bool = False,
) -> nn.Sequential:
    act = {"tanh": nn.Tanh, "leaky_relu": lambda: nn.LeakyReLU(0.2), "gelu": nn.GELU}[activation]
    layers: list[nn.Module] = []
    prev = int(in_dim)
    for h in hidden:
        linear = nn.Linear(prev, int(h))
        layers.append(nn_utils.spectral_norm(linear) if spectral_norm else linear)
        layers.append(act())
        prev = int(h)
    layers.append(nn.Linear(prev, int(out_dim)))
    layers.append(nn.Tanh())  # bounded output, matching the encoder's own range
    return nn.Sequential(*layers)


class MLPGenerator(LatentGenerator):
    """Feed-forward generator with an explicit hidden layout."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden: Sequence[int] = (10,),
        noise_dim: int | None = None,
        activation: str = "tanh",
        spectral_norm: bool = False,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            noise_dim=int(noise_dim) if noise_dim else int(latent_dim),
            output_scale=output_scale,
        )
        self.hidden = tuple(int(h) for h in hidden)
        self.activation = activation
        self.net = _mlp(self.noise_dim, self.hidden, self.latent_dim, activation, spectral_norm)

    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        # Noise arrives on [-pi, pi] to match the quantum encoding; rescale to [-1, 1]
        # so the first linear layer sees a comparable input magnitude.
        return self.net(noise / math.pi)

    def describe(self) -> dict:
        base = super().describe()
        base.update({"hidden": list(self.hidden), "activation": self.activation})
        return base


class CapacityMatchedGenerator(MLPGenerator):
    """Classical twin sized to a target parameter count.

    This is the control that matters. Its hidden widths are chosen so the total
    parameter count, biases included, sits as close to the circuit's as possible.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        target_params: int = 128,
        noise_dim: int | None = None,
        hidden: Sequence[int] | None = None,
        max_width: int = 256,
        output_scale: float = 1.0,
    ) -> None:
        n_in = int(noise_dim) if noise_dim else int(latent_dim)
        if hidden is None:
            hidden = choose_hidden_for_target(n_in, int(latent_dim), int(target_params), max_width)
        super().__init__(
            latent_dim=latent_dim,
            hidden=hidden,
            noise_dim=n_in,
            activation="tanh",
            output_scale=output_scale,
        )
        self.target_params = int(target_params)

    def describe(self) -> dict:
        base = super().describe()
        actual = self.count_params()
        base.update(
            {
                "target_params": self.target_params,
                "actual_params": actual,
                "match_error": actual - self.target_params,
                "match_error_fraction": abs(actual - self.target_params)
                / max(1, self.target_params),
            }
        )
        return base


class LargeMLPGenerator(MLPGenerator):
    """Deliberately over-parameterised control, the capacity upper bound.

    Uses a bounded activation and optional spectral normalisation because the
    unbounded LeakyReLU version diverged at the matched learning rate in every run,
    which made it useless as evidence about capacity.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        hidden: Sequence[int] = (64, 64),
        noise_dim: int | None = None,
        spectral_norm: bool = True,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            hidden=hidden,
            noise_dim=noise_dim,
            activation="tanh",
            spectral_norm=spectral_norm,
            output_scale=output_scale,
        )
        self.spectral_norm = spectral_norm


class LastStyleGenerator(MLPGenerator):
    """The LaSt-QGAN reference layout, retained so earlier runs stay reproducible."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden: Sequence[int] = (50, 30),
        noise_dim: int | None = None,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            hidden=hidden,
            noise_dim=noise_dim,
            activation="leaky_relu",
            output_scale=output_scale,
        )


class FourierGenerator(LatentGenerator):
    """Truncated Fourier feature map, the classical surrogate for the circuit.

    A re-uploading circuit is a bounded-degree trigonometric function of its inputs,
    so a Fourier basis of matching degree is the natural classical surrogate. Fitting
    one to the trained circuit is how the product topology was shown to be
    reproducible to a normalised error of about 5e-4 by roughly a hundred classical
    parameters.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        n_frequencies: int = 4,
        noise_dim: int | None = None,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            noise_dim=int(noise_dim) if noise_dim else int(latent_dim),
            output_scale=output_scale,
        )
        self.n_frequencies = int(n_frequencies)
        # sin and cos at each integer frequency for each input coordinate.
        n_features = 2 * self.n_frequencies * self.noise_dim
        self.readout = nn.Linear(n_features, self.latent_dim)
        freqs = torch.arange(1, self.n_frequencies + 1, dtype=torch.float32)
        self.register_buffer("frequencies", freqs, persistent=True)

    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        # noise: (B, noise_dim) -> features: (B, 2 * n_freq * noise_dim)
        scaled = noise.unsqueeze(-1) * self.frequencies.to(noise.device)
        features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return torch.tanh(self.readout(features.flatten(start_dim=1)))

    def describe(self) -> dict:
        base = super().describe()
        base["n_frequencies"] = self.n_frequencies
        return base


def expected_matched_params(latent_dim: int, noise_dim: int, hidden: Sequence[int]) -> int:
    """Convenience wrapper used by the capacity audit."""
    return count_mlp_params(noise_dim, hidden, latent_dim)
