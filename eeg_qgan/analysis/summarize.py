"""Pool fold rows into the tables that go into the manuscript.

Previously every study had its own ad hoc summarising script, and the numbers in the
manuscript were transcribed by hand from console output. Several disagreed with the
artefacts by the time they were checked. This module is the single path from raw fold
rows to reported numbers, so a table can always be regenerated from the artefacts.

Three things happen here that the old scripts did not do.

* Excluded folds are dropped and counted, so a study that lost folds to the autoencoder
  gate cannot silently report a mean over the survivors as though nothing happened.
* Intervals come from a cluster bootstrap over folds rather than a standard error over
  rows, because rows from different seeds share fold assignments only when the splitter
  is reused, and treating them as independent understates the interval.
* Comparisons against the reference arm are corrected for multiplicity, and a null result
  is reported next to the smallest effect the design could have detected.

Usage::

    eeg-qgan-analyze results/capacity_rematch
    eeg-qgan-analyze results/capacity_rematch --reference classical_equiv --csv tables/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eeg_qgan.stats.inference import (
    cluster_bootstrap_mean,
    effective_sample_size,
    equivalence_test,
    holm_bonferroni,
    minimum_detectable_effect,
    paired_difference,
)
from eeg_qgan.utils.io import save_json

# Metrics pulled from each arm, with the path into the arm record and whether smaller is
# better. Keeping the direction next to the metric stops the leaderboard from ranking a
# distance the wrong way round.
LATENT_METRICS: dict[str, tuple[tuple[str, ...], bool]] = {
    "mmd2": (("latent_distance", "mmd2"), True),
    # Reported next to mmd2, never blended with it: this one reacts to cross-coordinate
    # dependence that the RBF kernel is largely blind to.
    "latent_sliced_w1": (("latent_sliced_wasserstein",), True),
}
DECODED_METRICS: dict[str, tuple[tuple[str, ...], bool]] = {
    "autocorrelation_distance": (("decoded", "autocorrelation_distance"), True),
    # The absolute error, not the signed one: signed errors cancel across channel pairs
    # and made a generator with no channel structure look correlated.
    "channel_abs_error": (("decoded", "channel_correlation", "offdiagonal_absolute_error"), True),
    "lagged_xcorr_error": (
        ("decoded", "lagged_cross_correlation", "nonzero_lag_absolute_error"),
        True,
    ),
    "decoded_sliced_w1": (("decoded", "sliced_wasserstein"), True),
    "spectral_band_error": (("decoded", "spectral", "mean_band_error"), True),
    "ks_statistic": (("decoded", "ks_statistic"), True),
}


def dig(record: dict, path: tuple[str, ...]) -> float:
    node: Any = record
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return float("nan")
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(path: Path) -> tuple[list[dict], dict]:
    """Collect rows from one results file or a directory of split-job results.

    Splitting a study by seed produces one file per job. Pooling them here is what makes
    ``--split-by seed`` transparent to the analysis.
    """
    files = sorted(path.rglob("results*.json")) if path.is_dir() else [path]
    files = [f for f in files if f.name != "summary.json"]
    if not files:
        raise SystemExit(f"No results*.json found under {path}")

    rows: list[dict] = []
    meta: dict = {}
    seen: set[tuple] = set()
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  skipping unreadable {f.name}: {exc}")
            continue
        meta.setdefault("study", payload.get("study"))
        meta.setdefault("config", payload.get("config", {}))
        meta.setdefault("grouping", payload.get("grouping", {}))
        meta.setdefault("capacity_audit", payload.get("capacity_audit", {}))
        for row in payload.get("rows", []):
            # A requeued job rewrites folds it already finished; keep one copy.
            key = (row.get("seed"), row.get("fold"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    meta["n_result_files"] = len(files)
    meta["source_files"] = [f.name for f in files]
    return rows, meta


def cluster_ids(rows: list[dict]) -> np.ndarray:
    """Cluster label per row: the fold partition.

    Rows sharing a fold index across seeds are correlated whenever the partition is
    reused, so the fold is the resampling unit.
    """
    return np.asarray([f"fold{r.get('fold')}" for r in rows])


def summarize_arm_metrics(
    rows: list[dict],
    arm: str,
    metrics: dict[str, tuple[tuple[str, ...], bool]],
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    out: dict[str, dict] = {}
    for name, (path, _) in metrics.items():
        values, clusters = [], []
        for row in rows:
            arm_record = row.get("arms", {}).get(arm)
            if not arm_record:
                continue
            v = dig(arm_record, path)
            if np.isfinite(v):
                values.append(v)
                clusters.append(f"fold{row.get('fold')}")
        if not values:
            continue
        est = cluster_bootstrap_mean(
            np.asarray(values), np.asarray(clusters), n_boot=n_boot, rng=rng
        )
        out[name] = est.as_dict()
        out[name]["std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out


def paired_vs_reference(
    rows: list[dict],
    arm: str,
    reference: str,
    metrics: dict[str, tuple[tuple[str, ...], bool]],
    n_boot: int,
    equivalence_margins: dict[str, float],
    rng: np.random.Generator,
) -> dict:
    """Compare one arm to the reference on folds where both are present."""
    out: dict[str, dict] = {}
    for name, (path, lower_is_better) in metrics.items():
        a_vals, b_vals, clusters = [], [], []
        for row in rows:
            arms = row.get("arms", {})
            if arm not in arms or reference not in arms:
                continue
            a, b = dig(arms[arm], path), dig(arms[reference], path)
            if np.isfinite(a) and np.isfinite(b):
                a_vals.append(a)
                b_vals.append(b)
                clusters.append(f"fold{row.get('fold')}")
        if len(a_vals) < 2:
            continue

        a_arr, b_arr, c_arr = map(np.asarray, (a_vals, b_vals, clusters))
        diff = paired_difference(a_arr, b_arr, c_arr, n_boot=n_boot, rng=rng)
        entry = diff.as_dict()
        # "Better" depends on the metric direction, so state it rather than leaving the
        # reader to infer it from the sign.
        entry["arm_is_better"] = bool(
            (diff.estimate < 0) if lower_is_better else (diff.estimate > 0)
        )
        entry["significant"] = bool(diff.excludes_zero)
        entry["mde"] = minimum_detectable_effect(a_arr - b_arr, c_arr)

        margin = equivalence_margins.get(name)
        if margin is None:
            # Default margin: a tenth of the reference's spread. Any margin is a choice,
            # so it is recorded in the output.
            spread = float(np.std(b_arr, ddof=1)) if len(b_arr) > 1 else 0.0
            margin = max(0.1 * spread, 1e-6)
            entry["margin_source"] = "0.1 x reference std"
        else:
            entry["margin_source"] = "configured"
        entry["equivalence"] = equivalence_test(
            a_arr, b_arr, c_arr, margin=margin, n_boot=n_boot, rng=rng
        )
        out[name] = entry
    return out


def annotate_against_floor(comparisons: dict, floor: float | None) -> None:
    """Mark differences that are significant but smaller than the noise floor.

    This is the failure mode the audit named a manufactured advantage. A paired test over
    folds is sensitive enough to resolve a gap that the distance estimator itself cannot
    resolve between two samples of real data, because pairing removes the fold-to-fold
    variance that the floor measures. Such a gap is statistically real and scientifically
    empty: it is smaller than the estimator's own resolution, so it cannot support a claim
    that one generator matches the data better than another. Flagging it here is what
    stops a table of stars from being read as a quantum advantage.
    """
    if floor is None or not np.isfinite(floor) or floor <= 0:
        return
    for block in comparisons.values():
        for entry in block.get("latent", {}).values():
            if not entry.get("significant"):
                continue
            magnitude = abs(entry["estimate"])
            if magnitude < abs(floor):
                entry["below_noise_floor"] = True
                entry["floor_ratio"] = float(magnitude / abs(floor))
                entry["interpretation"] = (
                    f"The difference ({magnitude:.5f}) is {abs(floor) / max(magnitude, 1e-12):.0f} "
                    f"times smaller than the real-against-real floor ({abs(floor):.5f}). "
                    "The paired test resolves it only because pairing removes fold "
                    "variance; it is below the resolution of the distance itself and "
                    "must not be reported as one arm fitting the data better."
                )
            else:
                entry["below_noise_floor"] = False
                entry["floor_ratio"] = float(magnitude / abs(floor))


_CLINICAL_METRICS = (
    "auroc",
    "auprc",
    "mcc",
    "f1",
    "sensitivity",
    "specificity",
    "precision",
    "ece",
    "brier",
)


def summarize_detector(rows: list[dict], n_boot: int, rng: np.random.Generator) -> dict:
    """Downstream utility per augmentation source, against the real-only baseline.

    Under the natural TUH imbalance, ranking metrics (AUROC, AUPRC) and MCC are the
    primary clinical scores; F1 / precision / sensitivity are reported too but are
    threshold-dependent and harsh when positives are rare.
    """
    per_source: dict[str, dict[str, list]] = {}
    for row in rows:
        det = row.get("detector") or {}
        baseline = det.get("baseline")
        if not baseline:
            continue
        fold = f"fold{row.get('fold')}"
        for source, result in det.items():
            if source == "baseline" or not isinstance(result, dict):
                continue
            bucket = per_source.setdefault(
                source,
                {
                    **{m: [] for m in _CLINICAL_METRICS},
                    **{f"delta_{m}": [] for m in _CLINICAL_METRICS},
                    "dose": [],
                    "clusters": [],
                },
            )
            for metric in _CLINICAL_METRICS:
                val = float(result.get(metric, float("nan")))
                base = float(baseline.get(metric, float("nan")))
                bucket[metric].append(val)
                bucket[f"delta_{metric}"].append(val - base)
            bucket["dose"].append(float(result.get("n_synthetic_added", 0)))
            bucket["clusters"].append(fold)

    baseline_block: dict[str, Any] = {}
    for metric in _CLINICAL_METRICS:
        vals = [
            float((r.get("detector") or {}).get("baseline", {}).get(metric, float("nan")))
            for r in rows
            if (r.get("detector") or {}).get("baseline")
        ]
        finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
        if len(finite) < 2:
            continue
        clusters = np.asarray(
            [
                f"fold{r.get('fold')}"
                for r in rows
                if (r.get("detector") or {}).get("baseline")
            ]
        )
        idx = [i for i, _ in finite]
        baseline_block[metric] = cluster_bootstrap_mean(
            np.asarray([v for _, v in finite], dtype=np.float64),
            clusters[idx],
            n_boot=n_boot,
            rng=rng,
        ).as_dict()

    out: dict[str, Any] = {}
    if baseline_block:
        # Keep legacy key "baseline" as the F1 summary for older readers, plus "metrics".
        out["baseline"] = baseline_block.get("f1", {})
        out["baseline_metrics"] = baseline_block

    sources: dict[str, dict] = {}
    for source, bucket in per_source.items():
        clusters = np.asarray(bucket["clusters"])
        entry: dict[str, Any] = {}
        for key in list(_CLINICAL_METRICS) + [f"delta_{m}" for m in _CLINICAL_METRICS]:
            vals = np.asarray(bucket[key], dtype=np.float64)
            mask = np.isfinite(vals)
            if mask.sum() < 2:
                continue
            entry[key] = cluster_bootstrap_mean(
                vals[mask], clusters[mask], n_boot=n_boot, rng=rng
            ).as_dict()
        doses = np.asarray(bucket["dose"], dtype=np.float64)
        entry["dose"] = {
            "mean": float(np.nanmean(doses)) if len(doses) else 0.0,
            "min": float(np.nanmin(doses)) if len(doses) else 0.0,
            "max": float(np.nanmax(doses)) if len(doses) else 0.0,
        }
        # Collect shortfall provenance when present on fold metrics.
        shortfalls, requesteds = [], []
        for row in rows:
            det = row.get("detector") or {}
            result = det.get(source)
            if not isinstance(result, dict):
                continue
            aug_meta = result.get("augmentation") or {}
            if "shortfall" in aug_meta and "requested" in aug_meta:
                shortfalls.append(float(aug_meta["shortfall"]))
                requesteds.append(float(aug_meta["requested"]))
        if requesteds:
            entry["dose"]["mean_requested"] = float(np.mean(requesteds))
            entry["dose"]["mean_shortfall"] = float(np.mean(shortfalls))
            mean_req = float(np.mean(requesteds))
            mean_short = float(np.mean(shortfalls))
            if mean_req > 0 and mean_short / mean_req > 0.05:
                entry["warning"] = (
                    f"Augmentation shortfall {mean_short:.0f}/{mean_req:.0f} "
                    f"({mean_short / mean_req:.0%}): do not claim the configured dose "
                    "was delivered. Re-run with a deliverable fixed dose_count."
                )
        # A dose that collapses to nothing explains a null result, and saying so here
        # prevents "augmentation did not help" being read as a claim about the generator.
        if entry["dose"]["mean"] < 1.0:
            entry["warning"] = (
                "Dose collapsed to zero synthetic windows: this arm adds nothing, so the "
                "null result reflects the quality gate, not the generator."
            )
        sources[source] = entry
    out["sources"] = sources
    return out


def rank_arms(arm_summaries: dict[str, dict], metric: str, lower_is_better: bool) -> list[dict]:
    ranked = []
    for arm, summary in arm_summaries.items():
        entry = summary.get("latent", {}).get(metric) or summary.get("decoded", {}).get(metric)
        if entry:
            ranked.append({"arm": arm, "estimate": entry["estimate"],
                           "ci_low": entry["ci_low"], "ci_high": entry["ci_high"]})
    return sorted(ranked, key=lambda e: e["estimate"], reverse=not lower_is_better)


def build_summary(rows: list[dict], meta: dict, reference: str, n_boot: int) -> dict:
    rng = np.random.default_rng(20260804)
    cfg = meta.get("config", {})

    excluded = [r for r in rows if r.get("excluded")]
    kept = [r for r in rows if not r.get("excluded")]

    summary: dict[str, Any] = {
        "study": meta.get("study"),
        "n_result_files": meta.get("n_result_files"),
        "folds_total": len(rows),
        "folds_used": len(kept),
        "folds_excluded": len(excluded),
        # Naming the reason keeps a shrinking denominator visible.
        "exclusion_reasons": sorted({str(r.get("exclusion_reason")) for r in excluded}),
        "grouping": meta.get("grouping", {}),
        "capacity_audit": meta.get("capacity_audit", {}),
    }
    if not kept:
        summary["error"] = "Every fold was excluded; there is nothing to summarise."
        return summary

    summary["effective_sample_size"] = effective_sample_size(cluster_ids(kept))

    floors = [dig(r, ("mmd_noise_floor", "mean")) for r in kept]
    floors = [f for f in floors if np.isfinite(f)]
    if floors:
        # Every distance must be read against this; a value near it means the estimator
        # cannot separate the arm from real data at this sample size.
        summary["mmd_noise_floor"] = {
            "mean": float(np.mean(floors)),
            "std": float(np.std(floors, ddof=1)) if len(floors) > 1 else 0.0,
        }

    reach = [dig(r, ("latent_support", "fraction_within_reach")) for r in kept]
    reach = [v for v in reach if np.isfinite(v)]
    if reach:
        summary["latent_support"] = {"fraction_within_reach_mean": float(np.mean(reach))}
        if np.mean(reach) < 0.9:
            summary["latent_support"]["warning"] = (
                f"Only {np.mean(reach):.1%} of real latent codes lie inside the bounded "
                "generator range, so part of the reported distance is an unreachable "
                "target rather than a modelling failure."
            )

    arms = sorted({a for r in kept for a in (r.get("arms") or {})})
    margins = dict((cfg.get("equivalence_margins") or {}))

    arm_summaries: dict[str, dict] = {}
    for arm in arms:
        arm_summaries[arm] = {
            "latent": summarize_arm_metrics(kept, arm, LATENT_METRICS, n_boot, rng),
            "decoded": summarize_arm_metrics(kept, arm, DECODED_METRICS, n_boot, rng),
            "n_folds": sum(1 for r in kept if arm in (r.get("arms") or {})),
        }
        params = [
            dig(r["arms"][arm], ("n_params",)) for r in kept if arm in (r.get("arms") or {})
        ]
        params = [p for p in params if np.isfinite(p)]
        if params:
            arm_summaries[arm]["parameters"] = int(params[0])

        # A diverged arm carries no information about capacity, so the count travels with
        # the summary rather than being left in the console log of a finished job.
        flags = [
            bool(r["arms"][arm].get("converged", True))
            for r in kept
            if arm in (r.get("arms") or {})
        ]
        if flags and not all(flags):
            arm_summaries[arm]["n_diverged"] = int(sum(1 for f in flags if not f))
            arm_summaries[arm]["divergence_warning"] = (
                f"{sum(1 for f in flags if not f)} of {len(flags)} folds failed to "
                "converge; those folds measure an optimisation failure, not capacity."
            )
    summary["arms"] = arm_summaries

    if reference in arms:
        comparisons: dict[str, dict] = {}
        p_values: dict[str, float] = {}
        for arm in arms:
            if arm == reference:
                continue
            latent = paired_vs_reference(
                kept, arm, reference, LATENT_METRICS, n_boot, margins, rng
            )
            decoded = paired_vs_reference(
                kept, arm, reference, DECODED_METRICS, n_boot, margins, rng
            )
            comparisons[arm] = {"latent": latent, "decoded": decoded}
            for space, block in (("latent", latent), ("decoded", decoded)):
                for name, entry in block.items():
                    if entry.get("p_value") is not None:
                        p_values[f"{arm}.{space}.{name}"] = entry["p_value"]
        summary["reference_arm"] = reference
        annotate_against_floor(comparisons, (summary.get("mmd_noise_floor") or {}).get("mean"))
        summary["comparisons"] = comparisons
        if p_values:
            # Six arms times eight metrics makes an uncorrected result nearly certain.
            summary["multiplicity_correction"] = holm_bonferroni(p_values)

    summary["leaderboard"] = {
        "mmd2": rank_arms(arm_summaries, "mmd2", lower_is_better=True),
        "autocorrelation_distance": rank_arms(
            arm_summaries, "autocorrelation_distance", lower_is_better=True
        ),
    }

    detector = summarize_detector(kept, n_boot, rng)
    if detector:
        summary["detector"] = detector

    summary["notes"] = build_notes(summary)
    return summary


def build_notes(summary: dict) -> list[str]:
    """Caveats a reader needs in order to interpret the numbers correctly."""
    notes: list[str] = []
    ess = summary.get("effective_sample_size", {})
    if ess.get("is_pseudo_replicated"):
        notes.append(
            f"{ess['n_rows']} rows represent {ess['n_clusters']} independent partitions; "
            "intervals come from a cluster bootstrap over folds, not from a standard "
            "error over rows."
        )
    if summary.get("folds_excluded"):
        notes.append(
            f"{summary['folds_excluded']} of {summary['folds_total']} folds were excluded "
            f"by the autoencoder gate ({'; '.join(summary['exclusion_reasons'])}). Means "
            "are over the surviving folds."
        )
    floor = (summary.get("mmd_noise_floor") or {}).get("mean")
    if floor is not None:
        notes.append(
            f"The real-against-real MMD floor is {floor:+.5f}. Any arm within roughly one "
            "floor width of it is indistinguishable from real data at this sample size, "
            "so ranking such arms against one another is not meaningful."
        )
    if (summary.get("latent_support") or {}).get("warning"):
        notes.append(summary["latent_support"]["warning"])
    for arm, s in (summary.get("arms") or {}).items():
        if s.get("divergence_warning"):
            notes.append(f"{arm}: {s['divergence_warning']}")
    below = [
        arm
        for arm, block in (summary.get("comparisons") or {}).items()
        if (block.get("latent", {}).get("mmd2") or {}).get("below_noise_floor")
    ]
    if below:
        notes.append(
            f"For {', '.join(below)} the difference against the reference is significant "
            "but smaller than the noise floor. Pairing removes the fold variance that the "
            "floor measures, which is why a test can resolve a gap the distance cannot. "
            "These differences are not evidence that one arm fits the data better."
        )
    for source, entry in (summary.get("detector") or {}).get("sources", {}).items():
        if entry.get("warning"):
            notes.append(f"{source}: {entry['warning']}")
    audit = summary.get("capacity_audit") or {}
    if audit and not audit.get("noise_dim_consistent", True):
        notes.append(
            "Arms do not share a noise dimension, so a capacity comparison between them "
            "confounds capacity with input width."
        )
    return notes


def write_csv_tables(summary: dict, out_dir: Path) -> None:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "arms.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "n_folds", "parameters", "metric", "space",
                    "mean", "ci_low", "ci_high", "std"])
        for arm, s in summary.get("arms", {}).items():
            for space in ("latent", "decoded"):
                for name, entry in s.get(space, {}).items():
                    w.writerow([arm, s.get("n_folds"), s.get("parameters", ""), name, space,
                                f"{entry['estimate']:.6f}", f"{entry['ci_low']:.6f}",
                                f"{entry['ci_high']:.6f}", f"{entry.get('std', 0.0):.6f}"])

    if "comparisons" in summary:
        with (out_dir / "comparisons.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["arm", "reference", "space", "metric", "difference", "ci_low",
                        "ci_high", "p_value", "significant", "arm_is_better", "equivalent"])
            for arm, block in summary["comparisons"].items():
                for space in ("latent", "decoded"):
                    for name, entry in block.get(space, {}).items():
                        w.writerow([
                            arm, summary["reference_arm"], space, name,
                            f"{entry['estimate']:+.6f}", f"{entry['ci_low']:+.6f}",
                            f"{entry['ci_high']:+.6f}",
                            "" if entry.get("p_value") is None else f"{entry['p_value']:.4f}",
                            entry.get("significant"), entry.get("arm_is_better"),
                            (entry.get("equivalence") or {}).get("equivalent"),
                        ])

    if "detector" in summary:
        with (out_dir / "detector.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([
                "source", "metric", "mean", "ci_low", "ci_high",
                "delta_mean", "delta_ci_low", "delta_ci_high", "delta_significant",
                "mean_dose",
            ])
            # Real-only baseline row first (absolute metrics only).
            base_metrics = summary["detector"].get("baseline_metrics") or {}
            for metric in _CLINICAL_METRICS:
                b = base_metrics.get(metric) or (
                    summary["detector"].get("baseline") if metric == "f1" else None
                )
                if not b:
                    continue
                w.writerow([
                    "baseline",
                    metric,
                    f"{b.get('estimate', float('nan')):.4f}",
                    f"{b.get('ci_low', float('nan')):.4f}",
                    f"{b.get('ci_high', float('nan')):.4f}",
                    "",
                    "",
                    "",
                    "",
                    "0",
                ])
            for source, entry in summary["detector"].get("sources", {}).items():
                dose = f"{entry.get('dose', {}).get('mean', 0.0):.0f}"
                for metric in _CLINICAL_METRICS:
                    abs_m = entry.get(metric, {})
                    d = entry.get(f"delta_{metric}", {})
                    if not abs_m and not d:
                        continue
                    w.writerow([
                        source,
                        metric,
                        f"{abs_m.get('estimate', float('nan')):.4f}" if abs_m else "",
                        f"{abs_m.get('ci_low', float('nan')):.4f}" if abs_m else "",
                        f"{abs_m.get('ci_high', float('nan')):.4f}" if abs_m else "",
                        f"{d.get('estimate', float('nan')):+.4f}" if d else "",
                        f"{d.get('ci_low', float('nan')):+.4f}" if d else "",
                        f"{d.get('ci_high', float('nan')):+.4f}" if d else "",
                        d.get("excludes_zero") if d else "",
                        dose,
                    ])
    print(f"CSV tables written to {out_dir}")


def print_report(summary: dict) -> None:
    print("=" * 74)
    print(f"Study: {summary.get('study')}")
    print("=" * 74)
    print(
        f"Folds: {summary['folds_used']} used, {summary['folds_excluded']} excluded "
        f"of {summary['folds_total']}"
    )
    if summary.get("error"):
        print(f"\nERROR: {summary['error']}")
        return

    ess = summary.get("effective_sample_size", {})
    print(f"Independent partitions: {ess.get('n_clusters')} (rows: {ess.get('n_rows')})")
    floor = (summary.get("mmd_noise_floor") or {}).get("mean")
    if floor is not None:
        print(f"MMD noise floor: {floor:+.5f}")

    print("\nLatent MMD^2 by arm (mean [95% CI])")
    print("-" * 74)
    for entry in summary.get("leaderboard", {}).get("mmd2", []):
        arm_summary = summary["arms"].get(entry["arm"], {})
        params = arm_summary.get("parameters", "")
        near = ""
        if floor is not None and abs(entry["estimate"] - floor) < abs(floor) * 0.5:
            near = "  (at the floor)"
        print(
            f"  {entry['arm']:<22} {entry['estimate']:+.5f} "
            f"[{entry['ci_low']:+.5f}, {entry['ci_high']:+.5f}]  "
            f"params={params}{near}"
        )

    if "comparisons" in summary:
        print(f"\nPaired differences against {summary['reference_arm']} (negative favours arm)")
        print("-" * 74)
        for arm, block in summary["comparisons"].items():
            entry = block.get("latent", {}).get("mmd2")
            if not entry:
                continue
            verdict = "differs" if entry.get("significant") else "no difference detected"
            if (entry.get("equivalence") or {}).get("equivalent"):
                verdict = "equivalent within margin"
            if entry.get("below_noise_floor"):
                verdict = "differs, but below the noise floor"
            print(
                f"  {arm:<22} {entry['estimate']:+.5f} "
                f"[{entry['ci_low']:+.5f}, {entry['ci_high']:+.5f}]  {verdict}"
            )
            if entry.get("below_noise_floor"):
                print(f"    {'':<20} {entry['floor_ratio']:.3f} x the floor: not interpretable")
            if not entry.get("significant"):
                mde = entry.get("mde", {}).get("mde")
                if mde is not None and np.isfinite(mde):
                    print(f"    {'':<20} smallest detectable difference: {mde:.5f}")

    if "detector" in summary:
        det = summary["detector"]
        base_metrics = det.get("baseline_metrics") or {}
        print("\nDownstream detection (real-only baseline; primary = AUROC / AUPRC / MCC / ECE)")
        print("-" * 74)
        if base_metrics:
            parts = []
            for metric in ("auroc", "auprc", "mcc", "ece", "f1"):
                b = base_metrics.get(metric)
                if b:
                    parts.append(f"{metric.upper()}={b['estimate']:.4f}")
            if parts:
                print("  baseline  " + "  ".join(parts))
        for metric in ("auroc", "auprc", "mcc", "ece", "f1"):
            print(f"\n  Δ{metric.upper()} vs baseline (dose marked; * = CI excludes 0)")
            for source, entry in det.get("sources", {}).items():
                d = entry.get(f"delta_{metric}", {})
                abs_m = entry.get(metric, {})
                if not d:
                    continue
                mark = "*" if d.get("excludes_zero") else " "
                abs_s = f"{abs_m['estimate']:.4f}" if abs_m else "n/a"
                print(
                    f" {mark} {source:<28} {abs_s}  delta {d['estimate']:+.4f} "
                    f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
                    f"dose={entry.get('dose', {}).get('mean', 0):.0f}"
                )

    if summary.get("notes"):
        print("\nHow to read these numbers")
        print("-" * 74)
        for note in summary["notes"]:
            print(f"  - {note}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="results.json file or a directory of them")
    p.add_argument("--reference", default="classical_equiv",
                   help="arm every other arm is compared against")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", default=None, help="where to write summary.json")
    p.add_argument("--csv", default=None, help="directory for CSV tables")
    args = p.parse_args(argv)

    path = Path(args.results)
    rows, meta = load_rows(path)
    print(f"Loaded {len(rows)} fold rows from {meta['n_result_files']} file(s)\n")

    summary = build_summary(rows, meta, reference=args.reference, n_boot=args.n_boot)
    print_report(summary)

    out = Path(args.out) if args.out else (path if path.is_dir() else path.parent) / "summary.json"
    save_json(summary, out)
    print(f"\nSummary written to {out}")

    if args.csv:
        write_csv_tables(summary, Path(args.csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
