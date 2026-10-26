"""Maximum mean discrepancy, unbiased.

Audit item F3 (and the cause of the "shuffled data beats the real-against-real
floor" artefact). The previous estimator averaged the full kernel matrices,
including the self-kernel diagonal where ``k(x, x) = 1``. That leaves a positive
bias of roughly ``(1 - E k_xx) / n + (1 - E k_yy) / m``, which:

  * does not vanish for identical distributions, so a "floor" measured this way is
    not a floor at all; and
  * scales as ``1 / n``, so any comparison between two sample sizes is confounded.

In the shuffle control the floor was computed on two halves while the shuffled
comparison used the full sample, giving the floor roughly twice the bias and
producing an inversion that was read as a property of the metric.

Here the diagonal is removed and each term is normalised by ``n(n - 1)``, giving
the standard unbiased estimator of Gretton et al. Sample sizes are equalised
explicitly, and ``mmd_report`` records the ``n`` actually used so a mismatch can
never again pass unnoticed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_BANDWIDTH_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def _as_tensor(x: np.ndarray | torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def _sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xx = (x * x).sum(1, keepdim=True)
    yy = (y * y).sum(1, keepdim=True)
    d = xx + yy.t() - 2.0 * (x @ y.t())
    return d.clamp_min(0.0)


def median_heuristic(x: torch.Tensor) -> float:
    """Median pairwise distance of the reference sample.

    Computed from the reference (real) sample only. Because the real holdout is
    identical across arms within a fold, every arm is then scored at the same
    bandwidth, so the leaderboard cannot be confounded by per-arm bandwidth drift.
    """
    with torch.no_grad():
        d = _sq_dists(x, x)
        off = d[~torch.eye(len(x), dtype=torch.bool, device=x.device)]
        positive = off[off > 0]
        if positive.numel() == 0:
            return 1.0
        med = torch.median(positive)
        return float(torch.sqrt(med / 2.0).clamp_min(1e-4).item())


def _mmd2_unbiased_single(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Unbiased MMD^2 for one bandwidth."""
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        raise ValueError(f"Unbiased MMD needs at least 2 samples per side, got n={n}, m={m}")

    gamma = 1.0 / (2.0 * sigma**2)
    k_xx = torch.exp(-gamma * _sq_dists(x, x))
    k_yy = torch.exp(-gamma * _sq_dists(y, y))
    k_xy = torch.exp(-gamma * _sq_dists(x, y))

    # Remove the self-kernel diagonal: this is the whole point of the fix.
    sum_xx = k_xx.sum() - k_xx.diagonal().sum()
    sum_yy = k_yy.sum() - k_yy.diagonal().sum()

    term_xx = sum_xx / (n * (n - 1))
    term_yy = sum_yy / (m * (m - 1))
    term_xy = k_xy.mean()
    return term_xx + term_yy - 2.0 * term_xy


def mmd2_unbiased(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    bandwidth_multipliers: tuple[float, ...] = DEFAULT_BANDWIDTH_MULTIPLIERS,
    device: str = "cpu",
) -> float:
    """Multi-bandwidth unbiased MMD^2 between two samples.

    The value may be slightly negative for identical distributions, which is
    expected of an unbiased estimator and is informative: it tells you the sample
    is inside the noise band rather than hiding that fact behind a positive bias.
    """
    xt, yt = _as_tensor(x, device), _as_tensor(y, device)
    base = median_heuristic(xt)
    total = 0.0
    for mult in bandwidth_multipliers:
        total += float(_mmd2_unbiased_single(xt, yt, base * mult).item())
    return total / len(bandwidth_multipliers)


def mmd_rbf_multi_biased(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth_multipliers: tuple[float, ...] = DEFAULT_BANDWIDTH_MULTIPLIERS,
) -> torch.Tensor:
    """Biased, differentiable MMD for use as a *training* loss only.

    The diagonal bias is a constant with respect to the generator parameters when
    the batch size is fixed, so it does not affect gradients. It must never be
    used for reporting; use :func:`mmd2_unbiased` there.
    """
    base = median_heuristic(x)
    out = torch.zeros((), device=x.device)
    for mult in bandwidth_multipliers:
        sigma = base * mult
        gamma = 1.0 / (2.0 * sigma**2)
        out = out + (
            torch.exp(-gamma * _sq_dists(x, x)).mean()
            + torch.exp(-gamma * _sq_dists(y, y)).mean()
            - 2.0 * torch.exp(-gamma * _sq_dists(x, y)).mean()
        )
    return out / len(bandwidth_multipliers)


@dataclass
class MMDReport:
    """An MMD value together with the conditions under which it was measured."""

    mmd2: float
    n_reference: int
    n_candidate: int
    bandwidth_base: float
    bandwidth_multipliers: tuple[float, ...]
    estimator: str = "unbiased"
    subsampled: bool = False

    def as_dict(self) -> dict:
        return {
            "mmd2": self.mmd2,
            "n_reference": self.n_reference,
            "n_candidate": self.n_candidate,
            "bandwidth_base": self.bandwidth_base,
            "bandwidth_multipliers": list(self.bandwidth_multipliers),
            "estimator": self.estimator,
            "subsampled": self.subsampled,
        }


def mmd_report(
    reference: np.ndarray,
    candidate: np.ndarray,
    n_max: int | None = 1024,
    rng: np.random.Generator | None = None,
    bandwidth_multipliers: tuple[float, ...] = DEFAULT_BANDWIDTH_MULTIPLIERS,
    match_n: bool = True,
) -> MMDReport:
    """Measure MMD with the sample sizes made explicit and, by default, equal.

    Args:
        reference: real sample; also sets the kernel bandwidth.
        candidate: generated or perturbed sample.
        n_max: cap on the number of samples per side.
        rng: source of randomness for subsampling. Subsampling is random rather
            than a prefix slice, because taking a prefix made different dosage
            cells reuse the same synthetic samples.
        match_n: equalise both sides. Leave this on unless you specifically want
            to study the sample-size dependence, because unequal ``n`` was the
            confound behind the shuffle-control inversion.
    """
    ref = np.asarray(reference, dtype=np.float32).reshape(len(reference), -1)
    cand = np.asarray(candidate, dtype=np.float32).reshape(len(candidate), -1)
    rng = rng if rng is not None else np.random.default_rng(0)

    n = min(len(ref), len(cand)) if match_n else max(len(ref), len(cand))
    if n_max is not None:
        n = min(n, int(n_max))
    if n < 2:
        raise ValueError(f"Need at least 2 samples per side, got {len(ref)} and {len(cand)}")

    subsampled = False
    if len(ref) > n:
        ref = ref[rng.choice(len(ref), size=n, replace=False)]
        subsampled = True
    if len(cand) > n:
        cand = cand[rng.choice(len(cand), size=n, replace=False)]
        subsampled = True

    xt = _as_tensor(ref)
    base = median_heuristic(xt)
    value = mmd2_unbiased(ref, cand, bandwidth_multipliers=bandwidth_multipliers)
    return MMDReport(
        mmd2=value,
        n_reference=len(ref),
        n_candidate=len(cand),
        bandwidth_base=base,
        bandwidth_multipliers=tuple(bandwidth_multipliers),
        subsampled=subsampled,
    )


def mmd_noise_floor(
    real: np.ndarray,
    n_repeats: int = 50,
    n_max: int | None = 1024,
    rng: np.random.Generator | None = None,
) -> dict:
    """Distribution of MMD between two disjoint halves of the real sample.

    This is the reference point every arm should be read against. Both halves are
    drawn at the same size used for the arms, so the floor and the arm values are
    directly comparable, which was not true before.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    data = np.asarray(real, dtype=np.float32).reshape(len(real), -1)
    half = len(data) // 2
    if half < 2:
        raise ValueError("Need at least 4 real samples to estimate a noise floor")

    per_side = min(half, n_max) if n_max is not None else half
    values = []
    for _ in range(int(n_repeats)):
        perm = rng.permutation(len(data))
        a = data[perm[:per_side]]
        b = data[perm[half : half + per_side]]
        values.append(mmd2_unbiased(a, b))

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "q025": float(np.quantile(arr, 0.025)),
        "q975": float(np.quantile(arr, 0.975)),
        "n_repeats": int(n_repeats),
        "n_per_side": int(per_side),
        "estimator": "unbiased",
        "values": [float(v) for v in arr],
    }
