"""Statistical inference that accounts for repeated folds across seeds."""

from eeg_qgan.stats.inference import (
    EstimateWithInterval,
    cluster_bootstrap_mean,
    effective_sample_size,
    equivalence_test,
    holm_bonferroni,
    minimum_detectable_effect,
    paired_difference,
)

__all__ = [
    "EstimateWithInterval",
    "cluster_bootstrap_mean",
    "effective_sample_size",
    "equivalence_test",
    "holm_bonferroni",
    "minimum_detectable_effect",
    "paired_difference",
]
