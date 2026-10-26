#!/usr/bin/env python3
"""Dependence-destruction (copula-shuffle) test — brief item E-D, claim C5.

Permute each latent coordinate independently across samples. The marginals are then
exactly preserved and the joint dependence is destroyed. A distance that claims to
measure distributional agreement should react strongly. If the headline MMD does not,
the leaderboard it produces cannot be read as evidence about dependence structure, which
is the whole substance of the entanglement argument.

Three quantities are computed per fold, all with the unbiased estimator and matched
sample sizes:

  floor        MMD^2 between two disjoint halves of the real sample, repeated many times.
               This is the reference: it is what "no difference at all" looks like given
               finite samples.
  shuffle      MMD^2 between the real sample and its coordinate-shuffled copy.
  sliced W1    the same comparison read through random 1-D projections of the joint.

Claim C5 holds when the shuffle MMD sits inside or barely above the floor distribution
while the sliced distance is far outside it. If the shuffle MMD is clearly separated from
the floor, C5 is withdrawn — the metric is not blind and the paper says so.

No retraining and no GPU: this reads the saved ``z_report.npy`` artefacts.

    python scripts/copula_shuffle_test.py $SCRATCH/eeg_qgan_results_ae_v5c/capacity_rematch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eeg_qgan.evaluation.decoded import sliced_wasserstein  # noqa: E402
from eeg_qgan.evaluation.mmd import mmd_noise_floor, mmd_report  # noqa: E402


def shuffle_coordinates(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute every coordinate independently: marginals kept, joint destroyed."""
    out = np.empty_like(z)
    for j in range(z.shape[1]):
        out[:, j] = z[rng.permutation(len(z)), j]
    return out


def analyse_fold(z: np.ndarray, n_shuffles: int, n_partitions: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    floor = mmd_noise_floor(z, n_repeats=n_partitions, rng=np.random.default_rng(seed + 1))

    mmds: list[float] = []
    slices: list[float] = []
    for k in range(n_shuffles):
        shuffled = shuffle_coordinates(z, rng)
        mmds.append(mmd_report(z, shuffled, rng=np.random.default_rng(seed + 100 + k)).mmd2)
        slices.append(sliced_wasserstein(z, shuffled, seed=seed + 200 + k))

    # The same projection distance between two disjoint halves of the real data, so the
    # sliced numbers have a floor of their own rather than being read against zero.
    half = len(z) // 2
    floor_rng = np.random.default_rng(seed + 2)
    sliced_floor: list[float] = []
    for k in range(n_partitions):
        perm = floor_rng.permutation(len(z))
        sliced_floor.append(
            sliced_wasserstein(z[perm[:half]], z[perm[half : 2 * half]], seed=seed + 300 + k)
        )

    mmds_arr = np.asarray(mmds)
    slices_arr = np.asarray(slices)
    sliced_floor_arr = np.asarray(sliced_floor)
    return {
        "n_samples": int(len(z)),
        "n_dims": int(z.shape[1]),
        "floor_mmd2": {
            "mean": float(floor["mean"]),
            "q975": float(floor["q975"]),
            "n_repeats": int(floor["n_repeats"]),
        },
        "shuffle_mmd2": {
            "mean": float(mmds_arr.mean()),
            "std": float(mmds_arr.std(ddof=1)) if len(mmds_arr) > 1 else 0.0,
        },
        "floor_sliced_w1": {"mean": float(sliced_floor_arr.mean())},
        "shuffle_sliced_w1": {
            "mean": float(slices_arr.mean()),
            "std": float(slices_arr.std(ddof=1)) if len(slices_arr) > 1 else 0.0,
        },
        # MMD^2 is quadratic in the discrepancy and W1 is linear, so the two ratios
        # below are not on a common scale and must not be compared to each other. The
        # verdict uses each metric against its own floor.
        "mmd2_ratio_to_floor": float(mmds_arr.mean() / max(float(floor["mean"]), 1e-12)),
        "sliced_ratio_to_floor": float(
            slices_arr.mean() / max(float(sliced_floor_arr.mean()), 1e-12)
        ),
        "mmd2_separates_from_floor": bool(mmds_arr.mean() > float(floor["q975"])),
        "sliced_separates_from_floor": bool(
            slices_arr.mean() > float(np.quantile(sliced_floor_arr, 0.975))
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_root", help="study results directory containing artifacts/")
    p.add_argument("--n-shuffles", type=int, default=50)
    p.add_argument("--n-partitions", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="where to write the JSON (default: results_root)")
    args = p.parse_args()

    root = Path(args.results_root)
    paths = sorted(root.glob("artifacts/*/z_report.npy"))
    if not paths:
        print(f"No z_report.npy under {root}/artifacts. Was save.latents enabled?")
        return 2

    per_fold: dict[str, dict] = {}
    for path in paths:
        z = np.asarray(np.load(path), dtype=np.float64)
        label = path.parent.name
        per_fold[label] = analyse_fold(z, args.n_shuffles, args.n_partitions, args.seed)
        r = per_fold[label]
        print(
            f"{label:<20} n={r['n_samples']:<6} d={r['n_dims']:<4} "
            f"MMD2 {r['mmd2_ratio_to_floor']:>8.2f}x floor "
            f"({'separates' if r['mmd2_separates_from_floor'] else 'INSIDE floor'})   "
            f"slicedW1 {r['sliced_ratio_to_floor']:>6.2f}x floor "
            f"({'separates' if r['sliced_separates_from_floor'] else 'INSIDE floor'})"
        )

    mmd_ratios = [r["mmd2_ratio_to_floor"] for r in per_fold.values()]
    sliced_ratios = [r["sliced_ratio_to_floor"] for r in per_fold.values()]
    mmd_blind = not any(r["mmd2_separates_from_floor"] for r in per_fold.values())
    sliced_sees = all(r["sliced_separates_from_floor"] for r in per_fold.values())

    if mmd_blind and sliced_sees:
        verdict = (
            "C5 supported: the headline MMD cannot separate a dependence-destroyed sample "
            "from the real-vs-real floor, while the sliced distance can."
        )
    elif mmd_blind:
        verdict = (
            "Partial: MMD is blind to the shuffle, but the sliced distance does not "
            "cleanly separate it either. C5 needs a different sensitive companion metric "
            "before it can be claimed."
        )
    else:
        verdict = (
            "C5 NOT supported by this run: the MMD does separate the shuffle from the "
            "floor. The metric-blindness claim must be withdrawn or restated as a "
            "difference of degree, quoting both ratios above."
        )
    print()
    print(verdict)

    payload = {
        "results_root": str(root),
        "n_shuffles": args.n_shuffles,
        "n_partitions": args.n_partitions,
        "per_fold": per_fold,
        "mean_mmd2_ratio_to_floor": float(np.mean(mmd_ratios)),
        "mean_sliced_ratio_to_floor": float(np.mean(sliced_ratios)),
        "verdict": verdict,
    }
    out = Path(args.out) if args.out else root / "copula_shuffle_test.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
