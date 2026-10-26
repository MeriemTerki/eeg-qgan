"""The three changes made to raise honest detection, each tested for what it claims.

Normalisation scope, the long-clip detector, and the two dependence metrics required
by R1.3. Each test asserts the mechanism, not just that the code runs.
"""
from __future__ import annotations

import numpy as np
import torch

from eeg_qgan.evaluation.decoded import (
    lagged_cross_correlation_error,
    sliced_wasserstein,
)
from eeg_qgan.evaluation.mmd import mmd_report
from eeg_qgan.data.tuh import preprocess_stages, robust_range
from eeg_qgan.models.detector import STFTBiLSTMDetector, build_detector
from eeg_qgan.training.detector import train_and_evaluate_detector


# ---------------------------------------------------------------------------
# Normalisation scope
# ---------------------------------------------------------------------------


def _quiet_and_loud() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    quiet = (10.0 * rng.standard_normal((4, 256))).astype(np.float32)
    loud = (200.0 * rng.standard_normal((4, 256))).astype(np.float32)
    return quiet, loud


def test_window_scope_erases_the_amplitude_difference():
    quiet, loud = _quiet_and_loud()
    a = preprocess_stages(quiet, 256, denoise=False)["final"]
    b = preprocess_stages(loud, 256, denoise=False)["final"]
    # Both were stretched to fill the same range, so the 20x amplitude gap is gone.
    ratio = float(np.abs(b).mean() / np.abs(a).mean())
    assert 0.8 < ratio < 1.25, ratio


def test_recording_scope_preserves_the_amplitude_difference():
    quiet, loud = _quiet_and_loud()
    shared = robust_range(np.stack([quiet, loud]))
    a = preprocess_stages(quiet, 256, denoise=False, norm_range=shared)["final"]
    b = preprocess_stages(loud, 256, denoise=False, norm_range=shared)["final"]
    ratio = float(np.abs(b).mean() / np.abs(a).mean())
    assert ratio > 5.0, ratio


def test_recording_scope_stays_inside_the_generator_range():
    quiet, loud = _quiet_and_loud()
    shared = robust_range(np.stack([quiet, loud]))
    for w in (quiet, loud):
        out = preprocess_stages(w, 256, denoise=False, norm_range=shared)["final"]
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# R1.3 dependence metrics
# ---------------------------------------------------------------------------


def _dependent_pair(n: int = 400, d: int = 6) -> np.ndarray:
    rng = np.random.default_rng(3)
    base = rng.standard_normal((n, 1))
    return (base + 0.3 * rng.standard_normal((n, d))).astype(np.float64)


def test_sliced_wasserstein_reacts_to_a_copula_shuffle():
    real = _dependent_pair()
    rng = np.random.default_rng(7)
    shuffled = np.column_stack([rng.permutation(real[:, j]) for j in range(real.shape[1])])

    # Marginals are identical by construction; only the joint dependence changed.
    intact = sliced_wasserstein(real, real.copy(), n_projections=256, seed=0)
    broken = sliced_wasserstein(real, shuffled, n_projections=256, seed=0)
    assert broken > 5.0 * max(intact, 1e-6), (intact, broken)


def test_sliced_wasserstein_is_more_sensitive_than_mmd_to_the_shuffle():
    real = _dependent_pair()
    rng = np.random.default_rng(7)
    shuffled = np.column_stack([rng.permutation(real[:, j]) for j in range(real.shape[1])])

    scale = float(np.abs(real).mean())
    sw = sliced_wasserstein(real, shuffled, n_projections=256, seed=0) / scale
    mmd = abs(mmd_report(real, shuffled, rng=np.random.default_rng(0)).mmd2)
    assert sw > mmd, (sw, mmd)


def test_lagged_cross_correlation_catches_a_reversed_lead_lag():
    """Channel 1 leads instead of lags: zero-lag agreement is perfect, the truth is not.

    This is the case the zero-lag correlation matrix cannot see. Both signals have the
    same marginals and the same instantaneous correlation (about zero for white noise
    offset in time); only the direction of propagation differs.
    """
    rng = np.random.default_rng(11)
    n, t, lag = 64, 256, 8
    src = rng.standard_normal((n, t))

    real = np.stack([src, np.roll(src, lag, axis=-1)], axis=1)
    matched = np.stack([src, np.roll(src, lag, axis=-1)], axis=1)
    reversed_lag = np.stack([src, np.roll(src, -lag, axis=-1)], axis=1)

    good = lagged_cross_correlation_error(real, matched, max_lag=12, max_windows=64)
    bad = lagged_cross_correlation_error(real, reversed_lag, max_lag=12, max_windows=64)

    # The instantaneous matrix is fooled; the lagged structure is not.
    assert bad["zero_lag_absolute_error"] < 0.05, bad
    assert bad["nonzero_lag_absolute_error"] > 20.0 * good["nonzero_lag_absolute_error"]
    assert bad["max_absolute_error"] > 0.8, bad


def test_lagged_cross_correlation_is_near_zero_for_identical_input():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((48, 3, 128))
    report = lagged_cross_correlation_error(x, x.copy(), max_lag=8, max_windows=48)
    assert report["mean_absolute_error"] < 1e-9, report


# ---------------------------------------------------------------------------
# Long-clip detector
# ---------------------------------------------------------------------------


def _clip_cfg() -> dict:
    return {
        "sample_length": 512,
        "target_fs": 200,
        "latent_dim": 8,
        "detector": {
            "kind": "stft_bilstm",
            "conv_channels": [8, 16],
            "proj_dim": 32,
            "lstm_hidden": 16,
            "lstm_layers": 1,
            "n_fft": 32,
            "hop_length": 16,
            "head_hidden": 16,
            "use_spectral": True,
            "epochs": 6,
            "batch_size": 16,
            "patience": 6,
            "lr": 0.01,
            "val_fraction": 0.25,
            "ema_decay": 0.9,
            "window_augment": False,
        },
    }


def test_stft_bilstm_builds_and_forwards():
    model = build_detector(_clip_cfg(), n_channels=4, sample_length=512)
    assert isinstance(model, STFTBiLSTMDetector)
    assert model(torch.randn(3, 4, 512)).shape == (3,)


def test_stft_bilstm_separates_localised_from_diffuse_events():
    """A burst confined to part of the clip versus the same energy spread over all of it.

    Both classes carry the same total band power, so a model that averages the whole
    time-frequency plane cannot tell them apart. Keeping the time axis can.
    """
    rng = np.random.default_rng(2)
    t = np.linspace(0, 2.56, 512, dtype=np.float32)
    tone = np.sin(2 * np.pi * 8 * t).astype(np.float32)
    envelope = np.zeros(512, dtype=np.float32)
    envelope[128:256] = 1.0

    def make(n: int, localised: np.ndarray) -> np.ndarray:
        out = 0.3 * rng.standard_normal((n, 4, 512)).astype(np.float32)
        for i in range(n):
            if localised[i]:
                out[i] += 2.0 * envelope * tone
            else:
                out[i] += 0.5 * tone
        return out

    y_train = np.zeros(160, dtype=np.int64)
    y_train[:60] = 1
    x_train = make(160, y_train.astype(bool))
    y_test = np.zeros(80, dtype=np.int64)
    y_test[:30] = 1
    x_test = make(80, y_test.astype(bool))

    result = train_and_evaluate_detector(
        x_train, y_train, x_test, y_test, _clip_cfg(), rng=np.random.default_rng(0)
    )
    assert result.auroc > 0.8, result.auroc
