"""Regression tests for the preprocessing chain, especially wavelet denoising."""
from __future__ import annotations

import numpy as np
import pytest

from eeg_qgan.data.tuh import preprocess_stages, wavelet_denoise


def _noisy_sine(n: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    clean = np.sin(3.0 * t)
    noise = 0.35 * rng.standard_normal(n)
    return (clean + noise).astype(np.float32)


class TestWaveletDenoise:
    def test_reduces_high_frequency_energy(self):
        pytest.importorskip("pywt")
        noisy = _noisy_sine()
        cleanish = wavelet_denoise(noisy)
        # Soft thresholding should shrink the finest detail energy.
        assert cleanish.shape == noisy.shape
        assert float(np.std(np.diff(cleanish))) < float(np.std(np.diff(noisy)))

    def test_preserves_length(self):
        pytest.importorskip("pywt")
        for n in (64, 200, 256, 513):
            x = _noisy_sine(n=n)
            y = wavelet_denoise(x)
            assert y.shape == x.shape

    def test_missing_pywt_raises_clear_error(self, monkeypatch):
        import eeg_qgan.data.tuh as tuh

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "pywt":
                raise ImportError("forced")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        with pytest.raises(ImportError, match="PyWavelets"):
            tuh.wavelet_denoise(_noisy_sine())


class TestPreprocessStages:
    def test_final_shape_and_bounds(self):
        pytest.importorskip("pywt")
        window = np.stack([_noisy_sine(n=300, seed=i) for i in range(4)], axis=0)
        stages = preprocess_stages(window, sample_length=256, denoise=True, normalize=True)
        assert set(stages) >= {"raw", "resampled", "denoised", "normalized", "final"}
        assert stages["raw"].shape == (4, 300)
        assert stages["resampled"].shape == (4, 256)
        assert stages["denoised"].shape == (4, 256)
        assert stages["normalized"].shape == (4, 256)
        assert stages["final"].shape == (4, 256)
        assert float(stages["normalized"].min()) >= -1.0 - 1e-5
        assert float(stages["normalized"].max()) <= 1.0 + 1e-5

    def test_denoise_false_skips_wavelet(self):
        # Must work even without pywt when denoise is off.
        window = np.stack([_noisy_sine(n=256, seed=i) for i in range(2)], axis=0)
        stages = preprocess_stages(window, sample_length=256, denoise=False, normalize=True)
        np.testing.assert_allclose(stages["denoised"], stages["resampled"])

    def test_normalize_false_keeps_amplitude_scale(self):
        pytest.importorskip("pywt")
        window = np.stack([_noisy_sine(n=256, seed=i) * 5.0 for i in range(2)], axis=0)
        stages = preprocess_stages(window, sample_length=256, denoise=True, normalize=False)
        # Without normalisation the values are not forced into [-1, 1].
        assert float(np.max(np.abs(stages["final"]))) > 1.0
