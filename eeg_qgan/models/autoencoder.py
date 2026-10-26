"""Multichannel EEG autoencoder.

Audit item E3. The previous encoder ended with ``AdaptiveAvgPool1d(1)``, averaging
the convolutional feature map over the entire time axis before the bottleneck. That
discards all temporal phase and waveform structure by construction, leaving the code
a global descriptor from which the decoder can only emit a fixed prototype shape
modulated by a handful of scalars. The consequence was measured but ignored: the
normalised reconstruction error was 1.07 to 1.13 across all ten folds of the headline
run, and a value above one means the reconstruction is no better than emitting the
global mean. Because the reconstruction gate was computed and then dropped from the
pass flag, every one of those folds was recorded as having passed.

``bottleneck="temporal"`` keeps a short time axis at the code. The decoder is the
exact convolutional transpose of the encoder (same channel widths and strides), so
the reconstruction path is not a blurry interpolate-then-smooth prior. That pairing,
together with a wider stack and stronger training, is what brings NMSE under the
acceptance gate on bounded EEG windows.

Two encoder/decoder families live here, selected by ``arch``:

``v1``
    The stack described above: plain strided convolutions and a transposed-convolution
    decoder. Kept so earlier campaigns can be reproduced exactly.

``v2`` (default)
    Residual blocks with group normalisation and channel attention at every scale, a
    wide stem kernel matched to the low-frequency morphology that dominates EEG, and a
    decoder that upsamples then convolves instead of using transposed convolutions.
    Transposed convolutions with a kernel that is not a multiple of the stride place
    uneven numbers of contributions on neighbouring output samples, which shows up as a
    fixed high-frequency ripple in every decoded window; that ripple is indistinguishable
    from generator error when the decoded windows are scored, so removing it matters for
    the comparison and not only for the reconstruction number.

Neither family uses encoder-to-decoder skip connections. The decoder has to be usable
on codes that a generator invented, so it may not depend on activations that only exist
when a real window was encoded.
"""
from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn

BottleneckMode = Literal["temporal", "pooled"]
ArchVersion = Literal["v1", "v2"]


def _act() -> nn.Module:
    return nn.GELU()


def _norm(channels: int, groups: int = 8) -> nn.Module:
    """Group normalisation, which unlike batch statistics is batch-size independent.

    The autoencoder is evaluated one fold at a time on batches whose size varies with
    the fold, and it is also used in inference mode inside the generator loop, so a
    normaliser that behaves differently in train and eval would move the codes.
    """
    return nn.GroupNorm(num_groups=max(1, min(int(groups), int(channels))), num_channels=int(channels))


class ChannelAttention1d(nn.Module):
    """Squeeze-and-excitation gate over feature channels."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, int(channels) // int(reduction))
        self.fc = nn.Sequential(
            nn.Linear(int(channels), hidden),
            _act(),
            nn.Linear(hidden, int(channels)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x.mean(dim=-1))
        return x * w.unsqueeze(-1)


class ResidualBlock1d(nn.Module):
    """Pre-activation residual block with channel attention, stride 1."""

    def __init__(self, channels: int, kernel_size: int = 5, attention: bool = True) -> None:
        super().__init__()
        pad = int(kernel_size) // 2
        self.body = nn.Sequential(
            _norm(channels),
            _act(),
            nn.Conv1d(int(channels), int(channels), int(kernel_size), padding=pad),
            _norm(channels),
            _act(),
            nn.Conv1d(int(channels), int(channels), int(kernel_size), padding=pad),
        )
        self.attention = ChannelAttention1d(channels) if attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attention(self.body(x))


class ConvEncoder(nn.Module):
    """Strided 1-D convolutions to a bounded latent code.

    Args:
        n_channels: EEG channel count.
        sample_length: window length in samples.
        latent_dim: code width.
        widths: channel widths of the convolutional stack.
        bottleneck: ``temporal`` keeps the remaining time axis and flattens it;
            ``pooled`` averages it away, which is the audited defect.
    """

    def __init__(
        self,
        n_channels: int,
        sample_length: int = 256,
        latent_dim: int = 8,
        widths: Sequence[int] = (64, 128, 256),
        bottleneck: BottleneckMode = "temporal",
    ) -> None:
        super().__init__()
        self.bottleneck = bottleneck
        self.sample_length = int(sample_length)
        self.widths = tuple(int(w) for w in widths)

        layers: list[nn.Module] = []
        in_ch = int(n_channels)
        for w in self.widths:
            layers.extend(
                [
                    nn.Conv1d(in_ch, int(w), kernel_size=5, stride=2, padding=2),
                    _act(),
                ]
            )
            in_ch = int(w)
        self.conv = nn.Sequential(*layers)
        self.out_channels = in_ch
        # Each stride-2 convolution halves the length.
        self.out_length = max(1, self.sample_length // (2 ** len(self.widths)))

        if bottleneck == "pooled":
            self.pool: nn.Module | None = nn.AdaptiveAvgPool1d(1)
            flat_dim = in_ch
        else:
            self.pool = None
            flat_dim = in_ch * self.out_length

        # Keep this map thin. A wide MLP here dominated the parameter count, saturated
        # the tanh codes, and left reconstruction near the mean predictor.
        self.to_latent = nn.Sequential(nn.Linear(flat_dim, int(latent_dim)), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        if self.pool is not None:
            h = self.pool(h).squeeze(-1)
        else:
            h = h.flatten(start_dim=1)
        return self.to_latent(h)


class ConvDecoder(nn.Module):
    """Convolutional-transpose mirror of the encoder."""

    def __init__(
        self,
        n_channels: int,
        sample_length: int,
        latent_dim: int = 8,
        widths: Sequence[int] = (64, 128, 256),
        base_channels: int | None = None,
        base_length: int | None = None,
    ) -> None:
        super().__init__()
        self.sample_length = int(sample_length)
        self.widths = tuple(int(w) for w in widths)
        self.base_channels = int(base_channels if base_channels is not None else self.widths[-1])
        self.base_length = int(
            base_length
            if base_length is not None
            else max(1, self.sample_length // (2 ** len(self.widths)))
        )

        self.proj = nn.Linear(int(latent_dim), self.base_channels * self.base_length)

        # Upsample from the deepest encoder width back to the EEG channel count.
        up_widths = list(reversed(self.widths[:-1])) + [int(n_channels)]
        blocks: list[nn.Module] = []
        in_ch = self.base_channels
        for i, out_ch in enumerate(up_widths):
            is_last = i == len(up_widths) - 1
            conv = nn.ConvTranspose1d(
                in_ch,
                int(out_ch),
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            )
            if is_last:
                blocks.append(conv)
            else:
                blocks.append(nn.Sequential(conv, _act()))
            in_ch = int(out_ch)
        self.blocks = nn.ModuleList(blocks)
        self.out_act = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(z.size(0), self.base_channels, self.base_length)
        for block in self.blocks:
            h = block(h)
        if h.shape[-1] != self.sample_length:
            h = nn.functional.interpolate(
                h, size=self.sample_length, mode="linear", align_corners=False
            )
        return self.out_act(h)


class ResidualEncoder(nn.Module):
    """Residual convolutional encoder to a bounded latent code (``arch="v2"``).

    Exposes ``out_channels`` and ``out_length`` with the same meaning as
    :class:`ConvEncoder`, so the decoder pairing and the saved metadata are unchanged.
    """

    def __init__(
        self,
        n_channels: int,
        sample_length: int = 256,
        latent_dim: int = 8,
        widths: Sequence[int] = (128, 256, 256),
        bottleneck: BottleneckMode = "temporal",
        stem_kernel: int = 7,
        blocks_per_stage: int = 1,
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.bottleneck = bottleneck
        self.sample_length = int(sample_length)
        self.widths = tuple(int(w) for w in widths)

        first = self.widths[0]
        self.stem = nn.Sequential(
            nn.Conv1d(int(n_channels), first, int(stem_kernel), padding=int(stem_kernel) // 2),
            _norm(first),
            _act(),
        )

        stages: list[nn.Module] = []
        in_ch = first
        for w in self.widths:
            layers: list[nn.Module] = [
                nn.Conv1d(in_ch, int(w), kernel_size=5, stride=2, padding=2),
                _norm(int(w)),
                _act(),
            ]
            layers.extend(
                ResidualBlock1d(int(w), attention=attention) for _ in range(max(0, int(blocks_per_stage)))
            )
            stages.append(nn.Sequential(*layers))
            in_ch = int(w)
        self.stages = nn.Sequential(*stages)

        self.out_channels = in_ch
        self.out_length = max(1, self.sample_length // (2 ** len(self.widths)))

        if bottleneck == "pooled":
            self.pool: nn.Module | None = nn.AdaptiveAvgPool1d(1)
            flat_dim = in_ch
        else:
            self.pool = None
            flat_dim = in_ch * self.out_length

        # Still a single thin map to the code: a wide MLP here would dominate the
        # parameter count and saturate the bounded codes.
        self.to_latent = nn.Sequential(nn.Linear(flat_dim, int(latent_dim)), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.stem(x))
        if self.pool is not None:
            h = self.pool(h).squeeze(-1)
        else:
            h = h.flatten(start_dim=1)
        return self.to_latent(h)


class ResidualDecoder(nn.Module):
    """Upsample-then-convolve mirror of :class:`ResidualEncoder` (``arch="v2"``)."""

    def __init__(
        self,
        n_channels: int,
        sample_length: int,
        latent_dim: int = 8,
        widths: Sequence[int] = (128, 256, 256),
        base_channels: int | None = None,
        base_length: int | None = None,
        blocks_per_stage: int = 1,
        head_kernel: int = 7,
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.sample_length = int(sample_length)
        self.widths = tuple(int(w) for w in widths)
        self.base_channels = int(base_channels if base_channels is not None else self.widths[-1])
        self.base_length = int(
            base_length
            if base_length is not None
            else max(1, self.sample_length // (2 ** len(self.widths)))
        )

        self.proj = nn.Linear(int(latent_dim), self.base_channels * self.base_length)

        up_widths = list(reversed(self.widths[:-1])) + [self.widths[0]]
        stages: list[nn.Module] = []
        in_ch = self.base_channels
        for out_ch in up_widths:
            layers: list[nn.Module] = [
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(in_ch, int(out_ch), kernel_size=5, padding=2),
                _norm(int(out_ch)),
                _act(),
            ]
            layers.extend(
                ResidualBlock1d(int(out_ch), attention=attention) for _ in range(max(0, int(blocks_per_stage)))
            )
            stages.append(nn.Sequential(*layers))
            in_ch = int(out_ch)
        self.stages = nn.Sequential(*stages)

        self.head = nn.Conv1d(
            in_ch, int(n_channels), int(head_kernel), padding=int(head_kernel) // 2
        )
        self.out_act = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(z.size(0), self.base_channels, self.base_length)
        h = self.stages(h)
        if h.shape[-1] != self.sample_length:
            h = nn.functional.interpolate(
                h, size=self.sample_length, mode="linear", align_corners=False
            )
        return self.out_act(self.head(h))


class MultichannelAE(nn.Module):
    """Encoder and decoder pair operating on ``(N, C, T)`` windows."""

    def __init__(
        self,
        n_channels: int,
        sample_length: int,
        latent_dim: int = 8,
        bottleneck: BottleneckMode = "temporal",
        widths: Sequence[int] = (64, 128, 256),
        denoise_std: float = 0.02,
        arch: ArchVersion = "v2",
        blocks_per_stage: int = 1,
        stem_kernel: int = 7,
        attention: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_channels = int(n_channels)
        self.sample_length = int(sample_length)
        self.bottleneck = bottleneck
        self.denoise_std = float(denoise_std)
        self.widths = tuple(int(w) for w in widths)
        self.arch = str(arch)
        if self.arch not in {"v1", "v2"}:
            raise ValueError(f"autoencoder.arch must be 'v1' or 'v2', got '{arch}'")

        if self.arch == "v1":
            self.encoder: nn.Module = ConvEncoder(
                n_channels, sample_length, latent_dim, self.widths, bottleneck=bottleneck
            )
            self.decoder: nn.Module = ConvDecoder(
                n_channels,
                sample_length,
                latent_dim,
                widths=self.widths,
                base_channels=self.encoder.out_channels,
                base_length=self.encoder.out_length,
            )
        else:
            self.encoder = ResidualEncoder(
                n_channels,
                sample_length,
                latent_dim,
                self.widths,
                bottleneck=bottleneck,
                stem_kernel=int(stem_kernel),
                blocks_per_stage=int(blocks_per_stage),
                attention=bool(attention),
            )
            self.decoder = ResidualDecoder(
                n_channels,
                sample_length,
                latent_dim,
                widths=self.widths,
                base_channels=self.encoder.out_channels,
                base_length=self.encoder.out_length,
                blocks_per_stage=int(blocks_per_stage),
                head_kernel=int(stem_kernel),
                attention=bool(attention),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, denoise: bool = False) -> torch.Tensor:
        if denoise and self.training and self.denoise_std > 0:
            z = z + torch.randn_like(z) * self.denoise_std
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, denoise_decode: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z, denoise=denoise_decode), z

    def describe(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "n_channels": self.n_channels,
            "sample_length": self.sample_length,
            "bottleneck": self.bottleneck,
            "arch": self.arch,
            "widths": list(self.widths),
            "encoder_out_channels": self.encoder.out_channels,
            "encoder_out_length": self.encoder.out_length,
            "retains_time_axis": self.bottleneck == "temporal",
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def jacobian_spectral_norm(self, z: torch.Tensor, n_samples: int = 16) -> dict:
        """Exact largest singular value of the decoder Jacobian, per latent code.

        This is the amplification factor the Lipschitz analysis reports: how much a
        difference in latent space is magnified in signal space. It sets the scale on
        which a between-generator latent difference could possibly matter downstream.

        The latent dimension is small (8 to 32) while the output is a few thousand
        values, so the Jacobian is a tall thin matrix and can be formed exactly for a
        handful of codes rather than approximated. An approximation here would be a
        poor trade, because this number is used to argue that the decoder, not the
        generator, is the binding constraint.
        """
        was_training = self.training
        self.eval()
        codes = z.detach()[: int(n_samples)]
        norms: list[float] = []

        for i in range(len(codes)):
            zi = codes[i : i + 1].clone().requires_grad_(True)

            def decode_one(v: torch.Tensor) -> torch.Tensor:
                return self.decode(v, denoise=False).flatten()

            jac = torch.autograd.functional.jacobian(decode_one, zi, vectorize=True)
            # (out_dim, 1, latent_dim) -> (out_dim, latent_dim)
            jac = jac.reshape(-1, zi.shape[-1])
            norms.append(float(torch.linalg.matrix_norm(jac, ord=2).item()))

        if was_training:
            self.train()

        arr = torch.tensor(norms) if norms else torch.zeros(1)
        return {
            "spectral_norm_mean": float(arr.mean()),
            "spectral_norm_std": float(arr.std(unbiased=False)),
            "spectral_norm_max": float(arr.max()),
            "n_samples": len(norms),
            "exact": True,
        }
