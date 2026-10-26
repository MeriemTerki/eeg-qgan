"""Multi-resolution STFT loss, used alongside MSE for autoencoder training.

Matching spectra at several resolutions discourages the smooth, low-frequency
reconstructions that a pure time-domain loss encourages, which matters for EEG where
band structure carries the clinically relevant information.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _stft_magnitude(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """Magnitude spectrogram of a ``(B, C, T)`` batch, computed per channel."""
    b, c, t = x.shape
    flat = x.reshape(b * c, t)
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True,
    )
    mag = spec.abs()
    return mag.reshape(b, c, mag.shape[-2], mag.shape[-1])


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    resolutions: tuple[tuple[int, int], ...] = ((64, 16), (128, 32), (256, 64)),
) -> torch.Tensor:
    """Mean L1 distance between magnitude spectra across several window sizes.

    A Hann window is applied, which the earlier implementation omitted. Without it
    ``torch.stft`` uses a rectangular window whose spectral leakage puts energy in
    bands that carry none, so the loss was partly fitting leakage artefacts.
    """
    total = torch.zeros((), device=pred.device, dtype=pred.dtype)
    usable = 0
    for n_fft, hop in resolutions:
        if pred.shape[-1] < n_fft:
            continue
        total = total + F.l1_loss(
            _stft_magnitude(pred, n_fft, hop), _stft_magnitude(target, n_fft, hop)
        )
        usable += 1
    return total / max(1, usable)
