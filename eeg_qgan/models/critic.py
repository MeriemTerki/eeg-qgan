"""Discriminators for latent adversarial training.

Audit item F14. The critic width was read from ``wgan_critic_hidden`` with a default
of 128, giving two hidden layers of 128 units, so the critic carried roughly 33000
parameters against the generator's 128. A discriminator that outweighs the generator
by two orders of magnitude is not automatically wrong under WGAN-GP, because the
critic is meant to approximate a supremum, but it is a design choice that was never
stated and never varied, and it is a plausible cause of a result in which no generator
can win. The width is now explicit, recorded in the report, and swept by a dedicated
study so the choice is evidence rather than an assumption.

The critic is shared across arms within a fold, which is what makes the comparison a
comparison: the same measuring instrument is applied to every generator.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class WassersteinCritic(nn.Module):
    """MLP critic for WGAN-GP. Outputs an unbounded score."""

    def __init__(self, latent_dim: int, hidden: Sequence[int] | int = (128, 128)) -> None:
        super().__init__()
        widths = [int(hidden), int(hidden)] if isinstance(hidden, int) else [int(h) for h in hidden]
        self.hidden = tuple(widths)
        layers: list[nn.Module] = []
        in_dim = int(latent_dim)
        for h in widths:
            layers.extend([nn.Linear(in_dim, h), nn.LeakyReLU(0.2)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict:
        return {
            "kind": "wgan_gp_critic",
            "hidden": list(self.hidden),
            "n_params": self.count_params(),
        }


class SpectralDiscriminator(nn.Module):
    """Spectrally normalised discriminator for the hinge-loss variant."""

    def __init__(self, latent_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(int(latent_dim), self.hidden)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(self.hidden, self.hidden)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(self.hidden, 1)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> dict:
        return {
            "kind": "spectral_discriminator",
            "hidden": [self.hidden, self.hidden],
            "n_params": self.count_params(),
        }


def gradient_penalty(
    critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, lambda_gp: float
) -> torch.Tensor:
    """Two-sided gradient penalty on the line between real and fake samples."""
    batch = real.size(0)
    alpha = torch.rand(batch, 1, device=real.device)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = critic(interp)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return lambda_gp * ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def build_critic(latent_dim: int, cfg: dict, loss_mode: str = "wgan_gp") -> nn.Module:
    critic_cfg = cfg.get("critic", {}) or {}
    if loss_mode == "wgan_gp":
        hidden = critic_cfg.get("hidden", cfg.get("wgan_critic_hidden", (128, 128)))
        return WassersteinCritic(latent_dim, hidden=hidden)
    return SpectralDiscriminator(latent_dim, hidden=int(critic_cfg.get("hidden", 64)))
