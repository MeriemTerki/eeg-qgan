"""The generator contract every arm implements.

Two audit findings shaped this interface.

F6, noise dimension parity. The quantum generator hard-wired its noise width to the
qubit count while the classical arms used the latent width. In the wider-latent runs
that gave the quantum arm eight input dimensions against the twin's thirty-two, so
its output was supported on a manifold of at most eight dimensions inside a
thirty-two dimensional space and could never match a full-rank target. That confound
worked against the quantum arm and was never stated. ``noise_dim`` is now explicit
on every generator and the runner checks that compared arms agree.

E2, output range. Every arm is bounded by construction, so a generator must be able
to report the range it can reach and to accept a scale that widens it. ``output_scale``
carries that, and it is the mechanism behind the ``whiten_scaled`` coordinate system.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class LatentGenerator(nn.Module, ABC):
    """A model that maps noise to latent codes in the shared generator space.

    Subclasses must set ``latent_dim`` and ``noise_dim`` and implement
    :meth:`generate`. ``forward`` handles noise creation and output scaling so that
    every arm behaves identically from the training loop's point of view.
    """

    latent_dim: int
    noise_dim: int

    def __init__(self, latent_dim: int, noise_dim: int, output_scale: float = 1.0) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.noise_dim = int(noise_dim)
        if self.noise_dim < 1:
            raise ValueError(f"noise_dim must be positive, got {self.noise_dim}")
        # Registered as a buffer so it travels with the checkpoint and cannot drift
        # between training and evaluation.
        self.register_buffer("output_scale", torch.tensor(float(output_scale)), persistent=True)

    # ----------------------------------------------------------------- noise
    def sample_noise(self, batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Uniform noise on ``[-pi, pi]^noise_dim``.

        The angular range is what the quantum encoding expects; classical arms
        rescale it internally so all arms see the same input distribution.
        """
        return (torch.rand(int(batch), self.noise_dim, device=device) * 2.0 - 1.0) * math.pi

    # ------------------------------------------------------------- interface
    @abstractmethod
    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        """Map noise to bounded latent codes in ``(-1, 1)^latent_dim``."""

    def forward(self, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            device = next(self.parameters()).device if any(True for _ in self.parameters()) else "cpu"
            noise = self.sample_noise(1, device)
        if noise.dim() == 1:
            noise = noise.unsqueeze(0)
        if noise.shape[-1] != self.noise_dim:
            raise ValueError(
                f"{type(self).__name__} expects noise of width {self.noise_dim}, "
                f"got {noise.shape[-1]}"
            )
        out = self.generate(noise)
        scale = self.output_scale
        return out * scale if float(scale) != 1.0 else out

    # ------------------------------------------------------------ reporting
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> dict:
        """Parameters split by component.

        Audit F5: in the wider-latent configuration 288 of the quantum arm's 416
        parameters lived in a dense classical layer, so the arm labelled quantum was
        69 percent classical. Reporting the split makes that impossible to miss.
        """
        return {"total": self.count_params(), "quantum": 0, "classical": self.count_params()}

    def describe(self) -> dict:
        return {
            "kind": type(self).__name__,
            "latent_dim": self.latent_dim,
            "noise_dim": self.noise_dim,
            "output_scale": float(self.output_scale),
            "params": self.parameter_breakdown(),
        }


def count_mlp_params(in_dim: int, hidden, out_dim: int) -> int:
    """Parameter count for a Linear stack including biases."""
    widths = [int(in_dim), *[int(h) for h in hidden], int(out_dim)]
    return sum(a * b + b for a, b in zip(widths[:-1], widths[1:]))


def choose_hidden_for_target(
    in_dim: int,
    out_dim: int,
    target_params: int,
    max_width: int = 256,
) -> tuple[int, ...]:
    """Pick hidden widths whose parameter count is closest to ``target_params``.

    Searches one- and two-hidden-layer layouts. Ties break toward not exceeding the
    target, then toward fewer layers. This is the mechanism behind the
    capacity-matched twin, and it reaches 127 against the circuit's 128 in the
    headline configuration.
    """
    target = max(1, int(target_params))
    best: tuple[int, ...] | None = None
    best_err = float("inf")
    best_over = 0

    candidates: list[tuple[int, ...]] = []
    for h in range(1, int(max_width) + 1):
        candidates.append((h,))
        candidates.append((h, h))
        if h > 2:
            candidates.append((h, max(1, h // 2)))
            candidates.append((max(1, h // 2), h))

    for hid in candidates:
        n = count_mlp_params(in_dim, hid, out_dim)
        err = abs(n - target)
        over = n - target
        if best is None:
            best, best_err, best_over = hid, err, over
            continue
        better = err < best_err
        if err == best_err:
            better = (over <= 0 and best_over > 0) or (
                (over <= 0) == (best_over <= 0) and len(hid) < len(best)
            )
        if better:
            best, best_err, best_over = hid, err, over

    assert best is not None
    return best
