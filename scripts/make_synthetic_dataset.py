#!/usr/bin/env python3
"""Create a small synthetic dataset in the on-disk layout the loader expects.

Default layout matches BinaryClassification_Data under ``split_mode: predefined``:
train / dev / eval folders with distinct patients, so outer evaluation is
patient-disjoint without per-window patient sidecars.

Pass ``--layout grouped_cv`` to write a flat seizure/non_seizure tree with
``*_patients.npy`` sidecars for the alternative GroupKFold path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Names the predefined loader looks for (see eeg_qgan.data.tuh.PREDEFINED_PARTITIONS).
PARTITION_DIRS = {
    "train": ("train-seizure_converted", "train-NS_converted"),
    "dev": ("dev-seizure_converted", "dev-non-seizure_converted"),
    "eval": ("eval-seizure_converted", "eval-non-seizure_converted"),
}


def make_windows(
    n: int, n_channels: int, sample_length: int, seizure: bool, rng: np.random.Generator
) -> np.ndarray:
    """Generate ``(n, n_channels, sample_length)`` windows."""
    t = np.linspace(0, 4 * np.pi, sample_length)
    out = np.zeros((n, n_channels, sample_length), dtype=np.float32)

    for i in range(n):
        shared_weight = 0.6 if seizure else 0.2
        freq = rng.uniform(3.0, 6.0) if seizure else rng.uniform(0.5, 2.0)
        amplitude = rng.uniform(0.8, 1.2) if seizure else rng.uniform(0.3, 0.6)
        shared = amplitude * np.sin(freq * t + rng.uniform(0, 2 * np.pi))

        for c in range(n_channels):
            private = 0.3 * rng.standard_normal(sample_length)
            private += 0.2 * np.sin(rng.uniform(8, 20) * t + rng.uniform(0, 2 * np.pi))
            signal = shared_weight * shared + (1.0 - shared_weight) * private
            peak = np.abs(signal).max()
            out[i, c] = (signal / peak if peak > 0 else signal).astype(np.float32)
    return out


def _write_batch(
    directory: Path,
    stem: str,
    data: np.ndarray,
    patients: list[str] | None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{stem}.npy", data)
    if patients is not None:
        np.save(directory / f"{stem}_patients.npy", np.asarray(patients, dtype=object))


def write_predefined(root: Path, args: argparse.Namespace, rng: np.random.Generator) -> None:
    """train/dev/eval folders; patients do not cross partitions."""
    n_per = max(2, args.n_patients // 3)
    leftover = args.n_patients - 2 * n_per
    sizes = {"train": n_per + leftover, "dev": n_per, "eval": n_per}

    for partition, (seiz_name, ns_name) in PARTITION_DIRS.items():
        n_patients = sizes[partition]
        for label, dirname, per_patient, stem_prefix in (
            (1, seiz_name, args.seizure_per_patient, "seizure_batch"),
            (0, ns_name, args.non_seizure_per_patient, "non_seizure_batch"),
        ):
            batches: list[np.ndarray] = []
            for _k in range(n_patients):
                n = int(per_patient * rng.uniform(0.6, 1.4))
                batches.append(
                    make_windows(n, args.n_channels, args.sample_length, label == 1, rng)
                )
            data = np.concatenate(batches, axis=0)
            # File-level groups: the nested generator splits need train, select and
            # report to be disjoint, so two files were one short of what the splitter
            # requires and the offline smoke run could never start a fold.
            n_files = max(3, int(args.n_batch_files))
            directory = root / dirname
            for idx, chunk in enumerate(np.array_split(data, n_files)):
                _write_batch(directory, f"{stem_prefix}_{idx:04d}", chunk, patients=None)
            print(
                f"{partition}/{dirname}: {data.shape} in {n_files} batch files "
                f"({n_patients} patients, disjoint from other partitions) -> {directory}"
            )

    print(f"\nWrote predefined-layout synthetic dataset to {root.resolve()}")
    print("Run smoke with:")
    print(f"  python -m eeg_qgan.experiments.runner smoke --data-root {root}")


def write_grouped_cv(root: Path, args: argparse.Namespace, rng: np.random.Generator) -> None:
    """Flat seizure/non_seizure with per-window patient sidecars."""
    for label, dirname, per_patient in (
        (1, "seizure", args.seizure_per_patient),
        (0, "non_seizure", args.non_seizure_per_patient),
    ):
        batches: list[np.ndarray] = []
        patients: list[str] = []
        for k in range(args.n_patients):
            n = int(per_patient * rng.uniform(0.6, 1.4))
            patient_id = f"{10000000 + k * 137:08d}"
            batches.append(
                make_windows(n, args.n_channels, args.sample_length, label == 1, rng)
            )
            patients.extend([patient_id] * n)

        data = np.concatenate(batches, axis=0)
        stem = "seizure_batch_0000" if label == 1 else "non_seizure_batch_0000"
        _write_batch(root / dirname, stem, data, patients)
        print(
            f"{dirname}: {data.shape} across {len(set(patients))} patients "
            f"-> {root / dirname}"
        )

    print(f"\nWrote grouped_cv synthetic dataset to {root.resolve()}")
    print(
        "Run with: python -m eeg_qgan.experiments.runner smoke --data-root "
        f"{root} --override data.split_mode=grouped_cv"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic", help="output root")
    p.add_argument(
        "--layout",
        choices=("predefined", "grouped_cv"),
        default="predefined",
        help="predefined = train/dev/eval folders; grouped_cv = flat + sidecars",
    )
    p.add_argument("--n-patients", type=int, default=12)
    p.add_argument(
        "--n-batch-files",
        type=int,
        default=4,
        help="batch files per class per partition; at least 3 for the nested splits",
    )
    p.add_argument("--n-channels", type=int, default=22)
    p.add_argument("--sample-length", type=int, default=256)
    p.add_argument("--seizure-per-patient", type=int, default=40)
    p.add_argument("--non-seizure-per-patient", type=int, default=90)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    if args.layout == "predefined":
        write_predefined(root, args, rng)
    else:
        write_grouped_cv(root, args, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
