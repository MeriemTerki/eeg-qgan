"""Long-clip clinical protocol helpers and STFT detector."""
from __future__ import annotations

import numpy as np
import torch

from eeg_qgan.data.clips import build_clips_from_indices, pack_synthetic_clips
from eeg_qgan.data.tuh import EEGWindows
from eeg_qgan.models.detector import STFTCNNDetector, build_detector
from eeg_qgan.training.detector import train_and_evaluate_detector


def _fake_windows(n_per_group: int = 20, n_groups: int = 3) -> EEGWindows:
    rng = np.random.default_rng(0)
    xs, ys, gs, parts = [], [], [], []
    for g in range(n_groups):
        for i in range(n_per_group):
            xs.append(rng.standard_normal((4, 32)).astype(np.float32))
            ys.append(1 if i % 5 == 0 else 0)
            gs.append(f"train:file{g}")
            parts.append("train")
    x = np.stack(xs)
    return EEGWindows(
        x=x,
        y=np.asarray(ys, dtype=np.int64),
        groups=np.asarray(gs, dtype=object),
        group_key="file",
        group_source="test",
        n_channels=4,
        partitions=np.asarray(parts, dtype=object),
        provenance={},
    )


def test_pack_synthetic_clips_concatenates_along_time():
    syn = np.zeros((16, 3, 8), dtype=np.float32)
    packed = pack_synthetic_clips(syn, n_windows=4)
    assert packed is not None
    assert packed.shape == (4, 3, 32)


def test_crossfade_packing_keeps_the_clip_length():
    rng = np.random.default_rng(4)
    syn = rng.standard_normal((64, 3, 32)).astype(np.float32)
    packed = pack_synthetic_clips(syn, n_windows=4, crossfade=8)
    assert packed is not None
    assert packed.shape[1:] == (3, 128)


def test_crossfade_packing_removes_the_seam_step():
    """Every window is the same rising ramp, so the only discontinuity is the seam."""
    n_windows, t = 4, 32
    ramp = np.linspace(-1.0, 1.0, t, dtype=np.float32)
    syn = np.tile(ramp, (40, 3, 1)).astype(np.float32)

    plain = pack_synthetic_clips(syn, n_windows=n_windows, crossfade=0)
    faded = pack_synthetic_clips(syn, n_windows=n_windows, crossfade=8)
    assert plain is not None and faded is not None

    plain_step = float(np.abs(np.diff(plain, axis=-1)).max())
    faded_step = float(np.abs(np.diff(faded, axis=-1)).max())
    assert plain_step > 1.5, plain_step
    assert faded_step < 0.5 * plain_step, (plain_step, faded_step)


def test_build_clips_requires_unit_steps_when_contiguous():
    windows = _fake_windows()
    # Take every other index inside a group → no unit-step blocks of length 4.
    idxs = np.arange(0, len(windows.x), 2)
    try:
        build_clips_from_indices(
            windows, idxs, n_windows=4, stride=1, require_contiguous_file_order=True
        )
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_build_clips_from_contiguous_indices():
    windows = _fake_windows(n_per_group=16, n_groups=2)
    idxs = np.arange(len(windows.x))
    clips = build_clips_from_indices(
        windows, idxs, n_windows=4, stride=4, label_mode="any", require_contiguous_file_order=True
    )
    assert clips.x.ndim == 3
    assert clips.x.shape[-1] == 128
    assert len(clips.y) == len(clips.x)


def test_stft_cnn_forward_and_trains_on_toy_clips():
    cfg = {
        "sample_length": 256,
        "target_fs": 200,
        "latent_dim": 8,
        "detector": {
            "kind": "stft_cnn",
            "conv_channels": [8, 16],
            "n_fft": 32,
            "hop_length": 16,
            "head_hidden": 16,
            "epochs": 4,
            "batch_size": 16,
            "patience": 4,
            "lr": 0.01,
            "val_fraction": 0.25,
            "ema_decay": 0.9,
            "window_augment": False,
        },
    }
    model = build_detector(cfg, n_channels=4, sample_length=256)
    assert isinstance(model, STFTCNNDetector)
    logits = model(torch.randn(3, 4, 256))
    assert logits.shape == (3,)

    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 256, dtype=np.float32)
    x_train = 0.2 * rng.standard_normal((120, 4, 256)).astype(np.float32)
    y_train = np.zeros(120, dtype=np.int64)
    y_train[:40] = 1
    for i in np.flatnonzero(y_train):
        x_train[i] += 0.8 * np.sin(2 * np.pi * 8 * t)
    x_test = 0.2 * rng.standard_normal((60, 4, 256)).astype(np.float32)
    y_test = np.zeros(60, dtype=np.int64)
    y_test[:20] = 1
    for i in np.flatnonzero(y_test):
        x_test[i] += 0.8 * np.sin(2 * np.pi * 8 * t)

    result = train_and_evaluate_detector(
        x_train, y_train, x_test, y_test, cfg, rng=np.random.default_rng(0)
    )
    assert result.auroc > 0.7, result.auroc
