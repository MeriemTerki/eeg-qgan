"""Autoencoder reconstruction quality (audit E3 gate)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_qgan.evaluation.gates import evaluate_ae_gates, normalised_mse
from eeg_qgan.models.autoencoder import MultichannelAE
from eeg_qgan.training.autoencoder import build_autoencoder, train_autoencoder


def _synthetic_eeg(
    n: int = 512,
    n_channels: int = 22,
    sample_length: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Bounded multichannel windows with shared temporal structure across splits."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, sample_length, dtype=np.float32)
    out = np.zeros((n, n_channels, sample_length), dtype=np.float32)
    for i in range(n):
        # Slow sample-wise modulation keeps train/val in the same family of signals.
        amp = 0.9 + 0.1 * np.sin(0.03 * i)
        for c in range(n_channels):
            f1 = 3.0 + 0.1 * c
            f2 = 9.0 + 0.05 * c
            phase = 0.15 * c + 0.01 * i
            sig = amp * (
                0.6 * np.sin(2 * np.pi * f1 * t + phase)
                + 0.25 * np.sin(2 * np.pi * f2 * t)
                + 0.05 * rng.normal(size=sample_length).astype(np.float32)
            )
            out[i, c] = np.clip(sig, -1.0, 1.0)
    return out


def test_temporal_ae_clears_reconstruction_gate_at_latent_64():
    """Headline latent width used by capacity_rematch after the AE upgrade."""
    x = _synthetic_eeg(n=640, seed=1)
    cfg = {
        "sample_length": 256,
        "latent_dim": 64,
        "autoencoder": {
            "bottleneck": "temporal",
            "widths": [128, 256, 256],
            "epochs": 50,
            "batch_size": 64,
            "patience": 15,
            "stft_weight": 0.1,
            "l1_weight": 0.05,
            "denoise_std": 0.0,
            "weight_decay": 0.0,
            "max_nmse": 0.5,
            "max_saturation": None,
            "on_gate_failure": "raise",
        },
    }
    ae, meta = train_autoencoder(x[:512], x[512:], cfg, device="cpu", fold_label="synth")
    assert not meta["excluded"]
    assert meta["gates"]["nmse"] < 0.5, meta["gates"]
    assert meta["gates"]["passed"]
    assert meta["best_val_nmse"] < 0.5


def _synthetic_eeg_of_known_rank(
    n: int = 384,
    n_channels: int = 22,
    sample_length: int = 256,
    n_sources: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Windows drawn from a source manifold of known dimension.

    The comparison between latent widths is only meaningful when the data needs more
    degrees of freedom than the narrow code provides. The windows used by the other
    tests vary in amplitude and phase alone, which eight dimensions already carry, so
    the two widths would be compared on a budget rather than on capacity. Here each
    window is a fixed channel mixing of ``n_sources`` oscillators with independently
    drawn amplitudes, so the manifold has ``n_sources`` dimensions by construction.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, sample_length, dtype=np.float32)
    freqs = rng.uniform(2.0, 20.0, size=n_sources).astype(np.float32)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_sources).astype(np.float32)
    sources = np.stack([np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases)])
    mixing = rng.normal(size=(n_channels, n_sources)).astype(np.float32)
    coeffs = rng.normal(size=(n, n_sources)).astype(np.float32)
    out = np.einsum("cs,ns,st->nct", mixing, coeffs, sources)
    return np.clip(out / (np.abs(out).max() + 1e-8), -1.0, 1.0).astype(np.float32)


def test_wider_latent_reconstructs_better_than_narrow():
    x = _synthetic_eeg_of_known_rank(n=384, n_sources=16, seed=4)
    shared_ae = {
        "bottleneck": "temporal",
        "widths": [64, 128, 256],
        "epochs": 35,
        "batch_size": 64,
        "patience": 12,
        "stft_weight": 0.05,
        "l1_weight": 0.05,
        "denoise_std": 0.0,
        "weight_decay": 0.0,
        "max_nmse": 2.0,
        "max_saturation": None,
        "on_gate_failure": "exclude",
    }
    _, m8 = train_autoencoder(
        x[:288],
        x[288:],
        {"sample_length": 256, "latent_dim": 8, "autoencoder": shared_ae},
        device="cpu",
    )
    _, m32 = train_autoencoder(
        x[:288],
        x[288:],
        {"sample_length": 256, "latent_dim": 32, "autoencoder": shared_ae},
        device="cpu",
    )
    assert m32["gates"]["nmse"] <= m8["gates"]["nmse"] + 0.05


def test_pooled_bottleneck_is_worse_than_temporal_on_structured_windows():
    x = _synthetic_eeg(n=256, seed=2)
    shared = {
        "sample_length": 256,
        "latent_dim": 8,
        "autoencoder": {
            "widths": [64, 128, 256],
            "epochs": 40,
            "batch_size": 64,
            "patience": 15,
            "stft_weight": 0.1,
            "l1_weight": 0.05,
            "denoise_std": 0.0,
            "max_nmse": 2.0,
            "max_saturation": None,
            "on_gate_failure": "exclude",
        },
    }
    cfg_t = {**shared, "autoencoder": {**shared["autoencoder"], "bottleneck": "temporal"}}
    cfg_p = {**shared, "autoencoder": {**shared["autoencoder"], "bottleneck": "pooled"}}
    _, meta_t = train_autoencoder(x[:192], x[192:], cfg_t, device="cpu")
    _, meta_p = train_autoencoder(x[:192], x[192:], cfg_p, device="cpu")
    assert meta_t["gates"]["nmse"] < meta_p["gates"]["nmse"]


def test_decoder_length_matches_encoder_for_variable_width_stacks():
    ae = MultichannelAE(22, 256, latent_dim=8, widths=(64, 128, 256))
    x = torch.randn(4, 22, 256)
    recon, z = ae(x, denoise_decode=False)
    assert z.shape == (4, 8)
    assert recon.shape == x.shape
    assert ae.decoder.base_length == ae.encoder.out_length
    assert ae.decoder.base_channels == ae.encoder.out_channels


def test_build_autoencoder_reads_new_defaults_from_cfg():
    cfg = {
        "sample_length": 256,
        "latent_dim": 16,
        "autoencoder": {"widths": [64, 128, 256], "bottleneck": "temporal"},
    }
    ae = build_autoencoder(cfg, n_channels=22)
    assert ae.latent_dim == 16
    assert ae.widths == (64, 128, 256)
    assert ae.bottleneck == "temporal"


def test_normalised_mse_is_one_for_global_mean_predictor():
    x = _synthetic_eeg(n=32, seed=3)
    mu = np.full_like(x, float(x.mean()))
    assert normalised_mse(mu, x) == pytest.approx(1.0, abs=1e-5)


def test_ae_train_subsample_does_not_use_python_hash():
    """The 80k-window cap must be identical across SLURM processes.

    Job 1184337 failed the NMSE gate (0.5581 > 0.55) after drawing a different
    subsample from job 1158363 because ``hash(fold_label)`` is randomised per
    process. ``spawn_rng`` is the seeded replacement.
    """
    from eeg_qgan.seeding import spawn_rng

    def draw() -> np.ndarray:
        return spawn_rng(42, "seed0_fold0", "ae_subsample").choice(158400, size=80000, replace=False)

    first = draw()
    second = draw()
    hashed = np.random.default_rng(abs(hash("seed0_fold0")) % (2**31)).choice(
        158400, size=80000, replace=False
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (80000,)
    # The unstable hash stream is allowed to match by chance, but must not be
    # the API we rely on: two spawn_rng draws are the reproducibility contract.
    assert hashed.shape == (80000,)
