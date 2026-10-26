"""Metrics on decoded windows: spectra, autocorrelation and cross-channel structure.

Audit item F13 concerns the dependence measures, which are the ones the entanglement
hypothesis actually rests on.

The cross-channel error averaged a signed difference of correlation matrices, so a
generator that got half its channel pairs too positive and the other half too negative
by the same amount scored a perfect zero. Any measure of agreement has to use the
absolute or squared difference. It also included the unit diagonal, which is identical
for every generator and therefore dilutes the part that varies.

The lagged autocorrelation comparison collapsed the whole curve to a mean before
comparing, so two signals with different shapes but equal mean autocorrelation were
scored as identical.

Both are fixed here, and the off-diagonal cross-channel error is reported separately
from the full-matrix version so the informative part is not diluted. This matters
because the entanglement study reported no relationship between entanglement and
cross-channel fidelity, and a measure that can return zero for a wrong answer cannot
support that conclusion.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats


def _flatten_channels(x: np.ndarray) -> np.ndarray:
    n, c, t = x.shape
    return x.reshape(n * c, t)


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def autocorrelation_curve(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Mean autocorrelation as a function of lag, shape ``(max_lag + 1,)``."""
    flat = _flatten_channels(np.asarray(x, dtype=np.float64))
    flat = flat - flat.mean(axis=1, keepdims=True)
    denom = (flat**2).sum(axis=1) + 1e-12
    out = np.ones(int(max_lag) + 1, dtype=np.float64)
    for lag in range(1, int(max_lag) + 1):
        out[lag] = float(np.mean((flat[:, :-lag] * flat[:, lag:]).sum(axis=1) / denom))
    return out


def autocorrelation_distance(real: np.ndarray, fake: np.ndarray, max_lag: int = 50) -> float:
    """Mean absolute difference between the two autocorrelation curves.

    Compares the curves lag by lag. Averaging each curve to a scalar first, as the
    previous version did, discards exactly the shape information the comparison is
    meant to capture.
    """
    a = autocorrelation_curve(real, max_lag)
    b = autocorrelation_curve(fake, max_lag)
    return float(np.mean(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Cross-channel structure
# ---------------------------------------------------------------------------


def mean_channel_correlation(x: np.ndarray) -> np.ndarray:
    """Average per-window channel-by-channel correlation matrix."""
    x = np.asarray(x, dtype=np.float64)
    n, c, _ = x.shape
    acc = np.zeros((c, c), dtype=np.float64)
    valid = 0
    for i in range(n):
        w = x[i]
        sd = w.std(axis=1)
        if np.any(sd < 1e-10):
            # A flat channel has undefined correlation; skipping it is preferable to
            # letting a nan propagate into the mean.
            continue
        acc += np.corrcoef(w)
        valid += 1
    return acc / max(1, valid)


def channel_correlation_error(
    real: np.ndarray, fake: np.ndarray, off_diagonal_only: bool = True
) -> float:
    """Mean absolute difference between mean channel-correlation matrices.

    Absolute rather than signed, so errors of opposite sign cannot cancel to a
    spurious zero. Off-diagonal by default, because the unit diagonal is identical for
    every generator and only dilutes the comparison.
    """
    a = mean_channel_correlation(real)
    b = mean_channel_correlation(fake)
    diff = np.abs(a - b)
    if off_diagonal_only:
        mask = ~np.eye(diff.shape[0], dtype=bool)
        return float(diff[mask].mean())
    return float(diff.mean())


def channel_correlation_report(real: np.ndarray, fake: np.ndarray) -> dict:
    """Full account of cross-channel agreement, including the sign-cancellation check."""
    a = mean_channel_correlation(real)
    b = mean_channel_correlation(fake)
    mask = ~np.eye(a.shape[0], dtype=bool)
    signed = float((a - b)[mask].mean())
    absolute = float(np.abs(a - b)[mask].mean())
    return {
        "offdiagonal_absolute_error": absolute,
        "offdiagonal_signed_error": signed,
        "full_matrix_absolute_error": float(np.abs(a - b).mean()),
        "max_pairwise_error": float(np.abs(a - b)[mask].max()),
        "real_mean_offdiagonal": float(a[mask].mean()),
        "fake_mean_offdiagonal": float(b[mask].mean()),
        # When the absolute error greatly exceeds the signed error, opposite-sign
        # errors were cancelling: the previous signed-only measure would have
        # understated the disagreement.
        "sign_cancellation_ratio": float(absolute / abs(signed)) if abs(signed) > 1e-9 else float("inf"),
    }


def lagged_cross_correlation_curve(
    x: np.ndarray, max_lag: int = 20, max_windows: int = 512, seed: int = 0
) -> np.ndarray:
    """Mean cross-channel correlation at each non-negative lag, shape ``(C, C, L + 1)``.

    Entry ``[i, j, l]`` is the correlation between channel ``i`` and channel ``j``
    advanced by ``l`` samples, averaged over windows. Negative lags are covered by the
    transposed pair, so the full lagged dependence structure is represented.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) > int(max_windows):
        rng = np.random.default_rng(int(seed))
        x = x[rng.choice(len(x), int(max_windows), replace=False)]
    x = x - x.mean(axis=2, keepdims=True)
    norm = np.sqrt((x**2).sum(axis=2, keepdims=True)) + 1e-12
    x = x / norm

    c = x.shape[1]
    out = np.zeros((c, c, int(max_lag) + 1), dtype=np.float64)
    for lag in range(int(max_lag) + 1):
        a = x[:, :, : x.shape[2] - lag] if lag else x
        b = x[:, :, lag:]
        out[:, :, lag] = np.einsum("nct,ndt->cd", a, b) / len(x)
    return out


def lagged_cross_correlation_error(
    real: np.ndarray, fake: np.ndarray, max_lag: int = 20, max_windows: int = 512
) -> dict:
    """Absolute error in the lagged cross-channel correlation structure (audit F13).

    The zero-lag matrix says whether two channels move together; it says nothing about
    whether one leads the other. Seizure propagation is exactly a lead-lag relationship,
    so a generator can match the instantaneous correlation and still produce channels
    with no temporal dependence between them. Errors are taken in absolute value: a
    signed average lets a lead in one pair cancel a lag in another and report zero
    disagreement for a wrong answer.
    """
    a = lagged_cross_correlation_curve(real, max_lag, max_windows)
    b = lagged_cross_correlation_curve(fake, max_lag, max_windows)
    diff = np.abs(a - b)
    mask = ~np.eye(diff.shape[0], dtype=bool)
    cross = diff[mask]
    return {
        "mean_absolute_error": float(cross.mean()),
        "zero_lag_absolute_error": float(diff[:, :, 0][mask].mean()),
        "nonzero_lag_absolute_error": (
            float(diff[:, :, 1:][mask].mean()) if max_lag > 0 else 0.0
        ),
        "max_absolute_error": float(cross.max()),
        "max_lag": int(max_lag),
    }


def sliced_wasserstein(
    a: np.ndarray,
    b: np.ndarray,
    n_projections: int = 256,
    seed: int = 0,
    max_samples: int = 4096,
) -> float:
    """Mean 1-D Wasserstein-1 distance over random unit projections of the joint.

    MMD with an RBF kernel on a median bandwidth is weakly sensitive to cross-dimensional
    dependence: shuffling each coordinate independently leaves the marginals intact and
    moves the statistic very little. Projecting the *joint* vector onto random directions
    and comparing the resulting 1-D distributions does react to that shuffle, because a
    direction that mixes coordinates sees the dependence directly. This is the companion
    measure for claim C5.
    """
    a = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Dimension mismatch: {a.shape[1]} vs {b.shape[1]}")

    rng = np.random.default_rng(int(seed))
    # Equal sample sizes on both sides; an unequal comparison confounds the distance
    # with the estimator's sample-size bias (audit R1.1).
    n = min(len(a), len(b), int(max_samples))
    if n < 2:
        return float("nan")
    a = a[rng.choice(len(a), n, replace=False)]
    b = b[rng.choice(len(b), n, replace=False)]

    directions = rng.normal(size=(a.shape[1], int(n_projections)))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12
    pa = np.sort(a @ directions, axis=0)
    pb = np.sort(b @ directions, axis=0)
    return float(np.abs(pa - pb).mean())


# ---------------------------------------------------------------------------
# Spectral
# ---------------------------------------------------------------------------

EEG_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def band_powers(x: np.ndarray, fs: float = 200.0) -> dict[str, float]:
    """Mean relative power in each canonical EEG band."""
    flat = _flatten_channels(np.asarray(x, dtype=np.float64))
    nperseg = min(256, flat.shape[1])
    freqs, psd = sp_signal.welch(flat, fs=fs, nperseg=nperseg, axis=1)
    total = psd.sum(axis=1) + 1e-12
    out: dict[str, float] = {}
    for name, (lo, hi) in EEG_BANDS.items():
        sel = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.mean(psd[:, sel].sum(axis=1) / total)) if sel.any() else 0.0
    return out


def spectral_error(real: np.ndarray, fake: np.ndarray, fs: float = 200.0) -> dict:
    """Per-band relative power differences, plus a summary."""
    a = band_powers(real, fs)
    b = band_powers(fake, fs)
    per_band = {name: float(abs(a[name] - b[name])) for name in a}
    return {
        "per_band_absolute_error": per_band,
        "mean_band_error": float(np.mean(list(per_band.values()))),
        "real_band_powers": a,
        "fake_band_powers": b,
    }


def ks_statistic(real: np.ndarray, fake: np.ndarray, max_samples: int = 20000) -> dict:
    """Kolmogorov-Smirnov distance between pooled amplitude distributions."""
    a = _flatten_channels(np.asarray(real, dtype=np.float64)).ravel()
    b = _flatten_channels(np.asarray(fake, dtype=np.float64)).ravel()
    rng = np.random.default_rng(0)
    if len(a) > max_samples:
        a = a[rng.choice(len(a), max_samples, replace=False)]
    if len(b) > max_samples:
        b = b[rng.choice(len(b), max_samples, replace=False)]
    stat, p = sp_stats.ks_2samp(a, b)
    return {"ks_statistic": float(stat), "ks_p_value": float(p)}


def evaluate_decoded(
    real: np.ndarray, fake: np.ndarray, fs: float = 200.0, max_lag: int = 50
) -> dict:
    """Every decoded-space metric for one arm.

    These numbers are only meaningful if the autoencoder passed its reconstruction
    gate. When the decoder explains none of the signal variance, this compares one
    noise process against another, which is why the gate is enforced upstream.
    """
    return {
        "autocorrelation_distance": autocorrelation_distance(real, fake, max_lag),
        "channel_correlation": channel_correlation_report(real, fake),
        "lagged_cross_correlation": lagged_cross_correlation_error(real, fake),
        "sliced_wasserstein": sliced_wasserstein(real, fake),
        "spectral": spectral_error(real, fake, fs),
        **ks_statistic(real, fake),
    }
