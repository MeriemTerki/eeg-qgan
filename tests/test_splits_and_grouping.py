"""Regression tests for the splitting and grouping fixes.

These guard the corrections that change reported numbers the most. The previous code split
each batch file into about a hundred equal chunks of the array index and treated those as
groups, which put windows from one recording into every fold and turned a cross-patient
claim into a within-recording one. It also rebuilt folds once outside the seed loop, so
extra seeds produced duplicate rows rather than replicates. Both were invisible in the
output, which is why they are pinned here.
"""
from __future__ import annotations

import numpy as np
import pytest

from eeg_qgan.data.splits import (
    GroupingError,
    assert_grouping_is_plausible,
    make_outer_splitter,
    nested_latent_splits,
)


def make_grouped_data(n_patients: int = 12, per_patient: int = 30, seed: int = 0):
    """Grouped data with unequal recording lengths, as the corpus has."""
    rng = np.random.default_rng(seed)
    groups, labels = [], []
    for p in range(n_patients):
        n = int(per_patient + rng.integers(-per_patient // 3, per_patient // 3 + 1))
        groups.extend([f"p{p:03d}"] * n)
        # Seizure prevalence varies by patient, as it does in the corpus.
        rate = 0.2 + 0.5 * (p % 3) / 3.0
        labels.extend((rng.random(n) < rate).astype(int).tolist())
    x = rng.standard_normal((len(groups), 4, 32)).astype(np.float32)
    return x, np.asarray(labels), np.asarray(groups)


class TestNoPatientCrossesFolds:
    def test_train_and_test_never_share_a_patient(self):
        x, y, groups = make_grouped_data()
        splitter = make_outer_splitter(n_splits=5, seed=0)
        for train_idx, test_idx in splitter.split(x, y, groups):
            overlap = set(groups[train_idx]) & set(groups[test_idx])
            assert not overlap, f"patients in both sides: {overlap}"

    def test_every_patient_is_tested_exactly_once(self):
        x, y, groups = make_grouped_data()
        splitter = make_outer_splitter(n_splits=5, seed=0)
        seen: list[str] = []
        for _, test_idx in splitter.split(x, y, groups):
            seen.extend(sorted(set(groups[test_idx])))
        assert sorted(seen) == sorted(set(groups))
        assert len(seen) == len(set(seen)), "a patient was tested in more than one fold"


class TestIndexBasedGroupingIsRejected:
    """The specific bug: grouping by ``index // 20``.

    That key looks like a patient id and satisfies a naive "more than one group" check,
    so nothing complained. The plausibility check exists to catch it.
    """

    def test_index_slice_grouping_is_flagged(self):
        # 625 groups of exactly 20 windows: what the old fallback produced.
        n = 12_500
        fake = np.asarray([f"g{i // 20}" for i in range(n)])
        labels = np.tile([0, 1], n // 2)
        with pytest.raises(GroupingError):
            assert_grouping_is_plausible(fake, labels, n_splits=5, group_key="patient")

    def test_real_patient_grouping_passes(self):
        _, y, groups = make_grouped_data(n_patients=40, per_patient=60)
        diagnostics = assert_grouping_is_plausible(
            groups, y, n_splits=5, group_key="patient", min_group_size=25
        )
        assert diagnostics["n_groups"] == 40
        # Real recordings differ in length; uniform sizes are the tell-tale of chunking.
        assert diagnostics["distinct_group_sizes"] > 1

    def test_one_group_is_rejected(self):
        groups = np.asarray(["p000"] * 100)
        labels = np.tile([0, 1], 50)
        with pytest.raises(GroupingError):
            assert_grouping_is_plausible(groups, labels, n_splits=5, group_key="patient")

    def test_uniform_group_sizes_are_flagged(self):
        # Uniform sizes plus many groups is the chunking signature, even at 20 groups.
        groups = np.asarray([f"g{i // 20}" for i in range(2000)])
        labels = np.tile([0, 1], 1000)
        with pytest.raises(GroupingError):
            assert_grouping_is_plausible(
                groups, labels, n_splits=5, group_key="patient", min_group_size=25
            )


class TestSeedsProduceDifferentPartitions:
    """The E6 fix: a splitter built per seed, not once for all seeds."""

    def test_different_seeds_give_different_folds(self):
        x, y, groups = make_grouped_data()
        partitions = []
        for seed in (11, 22, 33):
            splitter = make_outer_splitter(n_splits=5, seed=seed)
            partitions.append(
                [tuple(sorted(set(groups[t]))) for _, t in splitter.split(x, y, groups)]
            )
        assert partitions[0] != partitions[1], "seeds produced identical folds"
        assert partitions[1] != partitions[2]

    def test_same_seed_is_reproducible(self):
        x, y, groups = make_grouped_data()
        a = [tuple(t) for _, t in make_outer_splitter(5, seed=7).split(x, y, groups)]
        b = [tuple(t) for _, t in make_outer_splitter(5, seed=7).split(x, y, groups)]
        assert a == b


class TestThreeWaySplitsAreDisjoint:
    """The E4 fix: checkpoint selection and reporting must not share windows.

    Selecting the best step on the same windows the distance is then reported on
    optimises the reported number directly, which flatters whichever arm is noisiest.
    """

    def test_train_select_report_share_no_window(self):
        _, _, groups = make_grouped_data()
        idx = np.arange(len(groups))
        splits = nested_latent_splits(idx, groups, select_size=0.2, report_size=0.2, seed=0)
        assert not set(splits.train) & set(splits.select)
        assert not set(splits.train) & set(splits.report)
        assert not set(splits.select) & set(splits.report)

    def test_train_select_report_share_no_patient(self):
        _, _, groups = make_grouped_data()
        idx = np.arange(len(groups))
        splits = nested_latent_splits(idx, groups, select_size=0.2, report_size=0.2, seed=0)
        g_train = set(groups[splits.train])
        g_select = set(groups[splits.select])
        g_report = set(groups[splits.report])
        assert not g_train & g_select
        assert not g_train & g_report
        assert not g_select & g_report

    def test_all_windows_are_accounted_for(self):
        _, _, groups = make_grouped_data()
        idx = np.arange(len(groups))
        splits = nested_latent_splits(idx, groups, select_size=0.2, report_size=0.2, seed=0)
        total = len(splits.train) + len(splits.select) + len(splits.report)
        assert total == len(idx)

    def test_report_split_is_large_enough_for_an_unbiased_estimate(self):
        _, _, groups = make_grouped_data()
        idx = np.arange(len(groups))
        splits = nested_latent_splits(idx, groups, select_size=0.15, report_size=0.15, seed=0)
        # The unbiased estimator needs at least two samples per side and is very noisy
        # near that limit, which is what the noise floor exists to expose.
        assert len(splits.report) >= 8

    def test_too_few_groups_raises_rather_than_returning_an_empty_split(self):
        # Two patients cannot yield three disjoint patient-respecting splits.
        groups = np.asarray(["p0"] * 20 + ["p1"] * 20)
        idx = np.arange(40)
        with pytest.raises(GroupingError, match="at least 3 groups"):
            nested_latent_splits(idx, groups, select_size=0.3, report_size=0.3, seed=0)

    def test_impossible_holdout_fractions_raise(self):
        _, _, groups = make_grouped_data()
        idx = np.arange(len(groups))
        with pytest.raises(GroupingError, match=r"\(0, 1\)"):
            nested_latent_splits(idx, groups, select_size=0.6, report_size=0.5, seed=0)


class TestPredefinedFolds:
    """Supervisor protocol: train vs holdout folders, not invented patient chunks."""

    def test_train_vs_each_holdout_and_no_index_overlap(self):
        from eeg_qgan.data.tuh import EEGWindows, make_predefined_folds

        n_train, n_dev, n_eval = 40, 12, 10
        parts = (
            ["train"] * n_train + ["dev"] * n_dev + ["eval"] * n_eval
        )
        n = len(parts)
        windows = EEGWindows(
            x=np.zeros((n, 2, 8), dtype=np.float32),
            y=np.zeros(n, dtype=np.int64),
            groups=np.asarray([f"f{i // 5}" for i in range(n)], dtype=object),
            partitions=np.asarray(parts, dtype=object),
        )
        folds = make_predefined_folds(windows, holdouts=("dev", "eval"))
        assert [name for _, _, name in folds] == ["dev", "eval"]
        train_idx, dev_idx, _ = folds[0]
        assert set(train_idx) == set(range(n_train))
        assert set(dev_idx) == set(range(n_train, n_train + n_dev))
        assert not set(train_idx) & set(dev_idx)
        _, eval_idx, _ = folds[1]
        assert set(eval_idx) == set(range(n_train + n_dev, n))
        # Same train set for every holdout fold.
        assert set(folds[0][0]) == set(folds[1][0])
