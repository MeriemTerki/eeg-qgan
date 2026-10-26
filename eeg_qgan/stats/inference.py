"""Statistical inference that respects the dependence structure of the runs.

Audit item E6. The outer splitter was constructed once, before the seed loop, so every
seed iterated the same partition and produced byte-identical folds. The saved records
confirm it: for the capacity-matched runs each fold held exactly the same number of
test windows under both seeds. The two seeds therefore differ only in weight
initialisation and batch order, not in data, so the ten seed-by-fold rows are five
partitions measured twice rather than ten independent observations.

Treating them as independent inflates precision. A paired test over ten rows with five
effective clusters has roughly four degrees of freedom, not nine, so any reported
p-value is optimistic and a claimed equivalence interval is narrower than the data
support.

Two remedies live here. The splitter is now rebuilt per seed, which is a change in the
runner. And inference is done with a cluster bootstrap that resamples whole folds, so
the correlation between rows sharing a partition is respected regardless.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class EstimateWithInterval:
    """A point estimate with an interval and the sample structure behind it."""

    estimate: float
    ci_low: float
    ci_high: float
    n_observations: int
    n_clusters: int
    method: str
    p_value: float | None = None
    extras: dict = field(default_factory=dict)

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_observations": self.n_observations,
            "n_clusters": self.n_clusters,
            "method": self.method,
            "p_value": self.p_value,
            "excludes_zero": self.excludes_zero,
            **self.extras,
        }


def effective_sample_size(clusters: np.ndarray) -> dict:
    """Report how many independent units the rows actually represent.

    When rows are pseudo-replicates the cluster count, not the row count, is what
    governs the degrees of freedom.
    """
    clusters = np.asarray(clusters)
    unique, counts = np.unique(clusters, return_counts=True)
    return {
        "n_rows": int(len(clusters)),
        "n_clusters": int(len(unique)),
        "rows_per_cluster_mean": float(counts.mean()),
        "is_pseudo_replicated": bool(len(unique) < len(clusters)),
        "inflation_factor": float(len(clusters)) / max(1, len(unique)),
    }


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> EstimateWithInterval:
    """Bootstrap the mean by resampling whole clusters.

    Resampling clusters rather than rows preserves the within-cluster correlation, so
    the interval widens appropriately when rows are pseudo-replicates instead of
    pretending the extra rows carry new information.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)

    mask = np.isfinite(values)
    values, clusters = values[mask], clusters[mask]
    if len(values) == 0:
        raise ValueError("No finite observations to bootstrap")

    unique = np.unique(clusters)
    by_cluster = [values[clusters == c] for c in unique]

    draws = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        picks = rng.integers(0, len(by_cluster), size=len(by_cluster))
        pooled = np.concatenate([by_cluster[i] for i in picks])
        draws[b] = pooled.mean()

    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    # Two-sided bootstrap p-value for the null that the mean is zero.
    p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())
    return EstimateWithInterval(
        estimate=float(values.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        n_observations=int(len(values)),
        n_clusters=int(len(unique)),
        method="cluster_bootstrap",
        p_value=float(min(1.0, p)),
        extras=effective_sample_size(clusters),
    )


def paired_difference(
    a: np.ndarray,
    b: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> EstimateWithInterval:
    """Compare two arms measured on the same folds, respecting the clustering.

    The pairing is what makes the comparison sensitive, since both arms see the same
    partition and the same autoencoder. The clustering is what keeps it honest.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired inputs must align: {a.shape} vs {b.shape}")

    result = cluster_bootstrap_mean(a - b, clusters, n_boot=n_boot, alpha=alpha, rng=rng)
    result.method = "paired_cluster_bootstrap"

    finite = np.isfinite(a - b)
    if finite.sum() > 1:
        # Reported alongside the bootstrap for reference, with the caveat that it
        # assumes independent rows and so is anti-conservative here.
        t_stat, t_p = stats.ttest_rel(a[finite], b[finite])
        result.extras["naive_paired_t_p"] = float(t_p)
        result.extras["naive_paired_t_statistic"] = float(t_stat)
        result.extras["naive_t_note"] = (
            "assumes independent rows, so it is anti-conservative when folds repeat "
            "across seeds; prefer the cluster bootstrap interval"
        )
    return result


def equivalence_test(
    a: np.ndarray,
    b: np.ndarray,
    clusters: np.ndarray,
    margin: float,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Two one-sided test for practical equivalence within ``margin``.

    Equivalence is concluded only when the whole interval falls inside the margin.
    Stating the margin in advance is what separates a claim of equivalence from a
    failure to detect a difference, and the two must not be conflated: an interval
    that contains zero but extends beyond the margin is simply inconclusive.
    """
    result = paired_difference(a, b, clusters, n_boot=n_boot, alpha=alpha, rng=rng)
    inside = bool(result.ci_low > -abs(margin) and result.ci_high < abs(margin))
    return {
        **result.as_dict(),
        "margin": float(abs(margin)),
        "equivalent": inside,
        "conclusion": (
            "equivalent within the stated margin"
            if inside
            else (
                "inconclusive: the interval extends beyond the margin, so this is a "
                "failure to detect a difference rather than evidence of equivalence"
            )
        ),
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Holm-Bonferroni correction across a family of comparisons.

    The leaderboard compares several arms against one reference, so the family-wise
    error rate needs controlling. Reporting uncorrected p-values across six arms and
    four metrics would make an incidental result almost certain.
    """
    items = sorted(
        ((k, v) for k, v in p_values.items() if v is not None and np.isfinite(v)),
        key=lambda kv: kv[1],
    )
    m = len(items)
    out: dict[str, dict] = {}
    previous = 0.0
    for rank, (key, p) in enumerate(items):
        adjusted = min(1.0, max(previous, (m - rank) * p))
        previous = adjusted
        out[key] = {
            "p_raw": float(p),
            "p_adjusted": float(adjusted),
            "significant": bool(adjusted < alpha),
            "rank": rank + 1,
        }
    return {"n_comparisons": m, "alpha": alpha, "results": out}


def minimum_detectable_effect(
    values: np.ndarray,
    clusters: np.ndarray,
    power: float = 0.8,
    alpha: float = 0.05,
) -> dict:
    """Smallest paired difference the design could detect, given the real clustering.

    A null result is only informative next to this number. With five independent
    partitions the detectable effect is considerably larger than with ten independent
    rows, which is precisely why the distinction matters.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    clusters = np.asarray(clusters)[: len(values)]
    n_clusters = int(len(np.unique(clusters)))

    if n_clusters < 2:
        return {"error": "need at least 2 clusters", "n_clusters": n_clusters}

    sd = float(values.std(ddof=1))
    df = n_clusters - 1
    t_alpha = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    t_beta = float(stats.t.ppf(power, df))
    mde = (t_alpha + t_beta) * sd / np.sqrt(n_clusters)

    return {
        "minimum_detectable_effect": float(mde),
        "observed_sd": sd,
        "n_clusters": n_clusters,
        "n_rows": int(len(values)),
        "degrees_of_freedom": df,
        "power": power,
        "alpha": alpha,
        "note": (
            f"With {n_clusters} independent partitions the design can detect a paired "
            f"difference of about {mde:.4f} at {power:.0%} power. A null result below "
            "this magnitude is uninformative rather than evidence of equivalence."
        ),
    }
