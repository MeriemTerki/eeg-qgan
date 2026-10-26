"""The biased estimator's 1/n bias, and why it inverted the shuffle control.

The old reporting path averaged full kernel matrices including the self-kernel
diagonal, which adds roughly (1 - E k_xx)/n + (1 - E k_yy)/m to every value. The
real-against-real floor was computed on two halves of the sample while the arms were
scored on the full sample, so the floor carried about twice the bias of the values it
was supposed to bound. That is sufficient to produce the reported inversion in which
shuffled data appeared closer to real data than real data was to itself.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_qgan.evaluation.mmd import mmd2_unbiased, mmd_rbf_multi_biased


def _pair(n: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n, 8)).astype(np.float32),
        rng.normal(size=(n, 8)).astype(np.float32),
    )


def test_unbiased_is_near_zero_for_identical_distributions():
    a, b = _pair(400)
    value = mmd2_unbiased(a, b)
    assert abs(value) < 0.01, f"expected near zero for identical distributions, got {value}"


def test_unbiased_detects_a_real_shift():
    a, _ = _pair(400)
    rng = np.random.default_rng(7)
    shifted = (rng.normal(size=(400, 8)) + 1.0).astype(np.float32)
    assert mmd2_unbiased(a, shifted) > 0.05


def test_biased_estimator_bias_shrinks_with_n_but_unbiased_does_not_drift():
    """The confound: the biased value depends on n even when the distributions match."""
    biased_values, unbiased_values = [], []
    for n in (125, 250, 500, 1000):
        a, b = _pair(n)
        biased_values.append(
            float(mmd_rbf_multi_biased(torch.tensor(a), torch.tensor(b)).item())
        )
        unbiased_values.append(mmd2_unbiased(a, b))

    # The biased estimator falls monotonically toward zero as n grows: that decay is
    # the bias term, not a change in the distributions being compared.
    assert biased_values[0] > biased_values[-1]
    assert biased_values[0] > 4 * max(abs(v) for v in unbiased_values)

    # The unbiased estimator shows no such trend.
    assert max(abs(v) for v in unbiased_values) < 0.01


def test_half_sample_floor_carries_more_bias_than_full_sample_comparison():
    """Exactly the asymmetry that inverted the shuffle control."""
    a_half, b_half = _pair(500)
    a_full, b_full = _pair(1000)
    floor_biased = float(
        mmd_rbf_multi_biased(torch.tensor(a_half), torch.tensor(b_half)).item()
    )
    arm_biased = float(mmd_rbf_multi_biased(torch.tensor(a_full), torch.tensor(b_full)).item())

    # The floor, measured on fewer samples, is inflated relative to the arm value even
    # though both compare samples from the same distribution.
    assert floor_biased > arm_biased

    # With the diagonal removed, neither is meaningfully above zero.
    assert abs(mmd2_unbiased(a_half, b_half)) < 0.01
    assert abs(mmd2_unbiased(a_full, b_full)) < 0.01


def test_unbiased_requires_two_samples_per_side():
    a, _ = _pair(4)
    with pytest.raises(ValueError):
        mmd2_unbiased(a[:1], a[:1])
