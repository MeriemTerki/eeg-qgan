"""Detector architecture and imbalance handling.

The clinical numbers are the part of the study a reader will scrutinise hardest, so the
things that were wrong in ae_v5b are pinned here: the rebalancing mechanisms must not
stack, the validation split must stay real-only, and the band-power branch must be a
function of the window rather than of the batch it arrived in.
"""
from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from eeg_qgan.experiments.runner import _build_overrides
from eeg_qgan.models.detector import CNNBiLSTMDetectorV2, SpectralFeatures, build_detector
from eeg_qgan.training.detector import train_and_evaluate_detector

N_CHANNELS = 4
SAMPLE_LENGTH = 64


def _cfg(**detector_overrides) -> dict:
    detector = {
        "kind": "cnn_bilstm_v2",
        "conv_widths": [16, 32],
        "lstm_hidden": 16,
        "head_hidden": 16,
        "epochs": 6,
        "batch_size": 32,
        "patience": 6,
        "lr": 0.003,
        "val_fraction": 0.25,
        "ema_decay": 0.9,
    }
    detector.update(detector_overrides)
    return {
        "sample_length": SAMPLE_LENGTH,
        "target_fs": 200,
        "latent_dim": 8,
        "seed": 0,
        "detector": detector,
    }


def _labelled_windows(
    n: int, positive_rate: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Positives carry a rhythmic burst; negatives are broadband noise."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < positive_rate).astype(np.int64)
    t = np.linspace(0.0, 1.0, SAMPLE_LENGTH, dtype=np.float32)
    x = 0.3 * rng.standard_normal((n, N_CHANNELS, SAMPLE_LENGTH)).astype(np.float32)
    for i in np.flatnonzero(y):
        burst = np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi)).astype(np.float32)
        x[i] += 0.9 * burst
    return np.clip(x, -1.0, 1.0), y


def test_v2_detector_is_the_default_and_reports_a_shape_per_window():
    model = build_detector(_cfg(), n_channels=N_CHANNELS)
    assert isinstance(model, CNNBiLSTMDetectorV2)
    logits = model(torch.randn(5, N_CHANNELS, SAMPLE_LENGTH))
    assert logits.shape == (5,)


def test_unknown_detector_kind_is_refused_rather_than_silently_substituted():
    with pytest.raises(ValueError, match="Unknown detector kind"):
        build_detector(_cfg(kind="transformer"), n_channels=N_CHANNELS)


def test_band_power_features_depend_on_the_window_not_on_its_neighbours():
    features = SpectralFeatures(N_CHANNELS, SAMPLE_LENGTH, target_fs=200.0)
    x = torch.randn(8, N_CHANNELS, SAMPLE_LENGTH)
    alone = features(x[:1])
    in_batch = features(x)[:1]
    assert torch.allclose(alone, in_batch, atol=1e-6)
    # Relative band powers sum to one per channel, so the branch cannot smuggle in
    # amplitude twice; the log total is the one absolute term.
    n_bands = len(SpectralFeatures.BANDS)
    per_channel = features(x).view(len(x), N_CHANNELS, n_bands + 1)
    assert torch.allclose(
        per_channel[..., :n_bands].sum(dim=-1), torch.ones(len(x), N_CHANNELS), atol=1e-5
    )


def test_detector_separates_a_rhythmic_burst_from_noise():
    x_train, y_train = _labelled_windows(400, positive_rate=0.2, seed=1)
    x_test, y_test = _labelled_windows(200, positive_rate=0.2, seed=2)
    result = train_and_evaluate_detector(
        x_train, y_train, x_test, y_test, _cfg(), rng=np.random.default_rng(0)
    )
    assert result.auroc > 0.8, result.auroc
    assert result.n_synthetic_added == 0


def test_rebalancing_mechanisms_do_not_stack_by_default():
    """Balanced sampling alone, so the same correction is not applied three times."""
    x_train, y_train = _labelled_windows(300, positive_rate=0.1, seed=3)
    x_test, y_test = _labelled_windows(120, positive_rate=0.1, seed=4)
    cfg = _cfg(balanced_batches=True, oversample_positives=True, use_pos_weight=True)
    result = train_and_evaluate_detector(
        x_train, y_train, x_test, y_test, cfg, rng=np.random.default_rng(0)
    )
    strategy = result.augmentation["imbalance_strategy"]
    assert strategy["balanced_batches"] is True
    assert strategy["oversample_requested"] is True
    assert strategy["oversample_applied"] is False
    assert strategy["pos_weight"] is None


def test_stacked_rebalancing_is_available_only_when_asked_for_explicitly():
    x_train, y_train = _labelled_windows(300, positive_rate=0.1, seed=5)
    x_test, y_test = _labelled_windows(120, positive_rate=0.1, seed=6)
    cfg = _cfg(
        balanced_batches=True,
        oversample_positives=True,
        use_pos_weight=True,
        allow_stacked_rebalance=True,
        max_oversample_factor=3.0,
    )
    result = train_and_evaluate_detector(
        x_train, y_train, x_test, y_test, cfg, rng=np.random.default_rng(0)
    )
    strategy = result.augmentation["imbalance_strategy"]
    assert strategy["oversample_applied"] is True
    # The duplication cap keeps a handful of positives from being repeated until the fit
    # is measuring memorisation.
    assert strategy["oversample_duplication_factor"] <= 3.0 + 1e-6
    assert strategy["pos_weight"] is not None


def test_synthetic_windows_never_reach_the_validation_split():
    x_train, y_train = _labelled_windows(300, positive_rate=0.15, seed=7)
    x_test, y_test = _labelled_windows(120, positive_rate=0.15, seed=8)
    synthetic, _ = _labelled_windows(64, positive_rate=1.0, seed=9)
    result = train_and_evaluate_detector(
        x_train,
        y_train,
        x_test,
        y_test,
        _cfg(),
        synthetic=synthetic,
        rng=np.random.default_rng(0),
    )
    assert result.augmentation["validation_split_is_real_only"] is True
    assert result.n_synthetic_added == 64
    assert result.augmentation["positive_rate_after"] > result.augmentation["positive_rate_before"]


def test_window_jitter_is_off_by_default_and_recorded_when_switched_on():
    x_train, y_train = _labelled_windows(300, positive_rate=0.15, seed=10)
    x_test, y_test = _labelled_windows(120, positive_rate=0.15, seed=11)
    plain = train_and_evaluate_detector(
        x_train, y_train, x_test, y_test, _cfg(), rng=np.random.default_rng(0)
    )
    assert plain.augmentation["imbalance_strategy"]["window_augment"] is False

    jittered = train_and_evaluate_detector(
        x_train,
        y_train,
        x_test,
        y_test,
        _cfg(window_augment=True, augment_time_shift=8, augment_noise_std=0.02),
        rng=np.random.default_rng(0),
    )
    strategy = jittered.augmentation["imbalance_strategy"]
    assert strategy["window_augment"] is True
    assert strategy["window_augment_settings"]["augment_time_shift"] == 8
    # A rhythmic burst survives a shift and a little noise, so the task is still learnable.
    assert jittered.auroc > 0.7, jittered.auroc


def _args(overrides: list[str] | None) -> argparse.Namespace:
    return argparse.Namespace(
        arms=None,
        data_root=None,
        output_root=None,
        skip_downstream=False,
        overrides=overrides,
    )


def test_set_overrides_reach_nested_config_keys():
    overrides = _build_overrides(
        _args(["augmentation.dose_count=2048", "latent.train_fraction=0.25", "n_synthetic=2048"])
    )
    assert overrides["augmentation"]["dose_count"] == 2048
    assert overrides["latent"]["train_fraction"] == pytest.approx(0.25)
    assert overrides["n_synthetic"] == 2048


def test_set_override_without_a_value_is_refused():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _build_overrides(_args(["augmentation.dose_count"]))
