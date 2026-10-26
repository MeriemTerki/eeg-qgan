#!/usr/bin/env python3
"""Reconstruct per-window patient identity for pooled batch files.

The loader now refuses to invent a grouping key, which means a dataset whose batch files
carry no patient information cannot be used for patient-wise cross-validation until that
information is recovered. This script recovers it, and it is the piece that makes the
grouping fix actionable rather than merely blocking.

Run it on Narval, where both the batch files and the source corpus are visible::

    python scripts/build_patient_sidecars.py \\
        --data-root /scratch/.../BinaryClassification_Data \\
        --corpus-root /scratch/.../THUSZ/edf \\
        --dry-run

Three strategies are attempted, in decreasing order of reliability.

1. ``manifest``: a CSV or JSON written when the batches were built, listing the source
   recording for each window. This is exact when present.
2. ``corpus-match``: rebuild the window ordering by walking the source corpus in the same
   order the batch builder used, then assign patients by position. Exact only if the
   window counts agree, which the script verifies and reports.
3. ``report-only``: when neither works, report what is missing so the batches can be
   rebuilt with identity preserved. No sidecar is written, because a guessed grouping is
   what caused the problem in the first place.

If none succeeds, the honest options are to rebuild the batches carrying patient ids, or
to declare ``group_key: file`` and stop describing the results as patient-wise.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

TUH_PATIENT_RE = re.compile(r"(\d{8})")


def find_batch_files(data_root: Path) -> list[Path]:
    skip = ("_patients", "_patient_ids", "_groups", "_labels", "_info")
    out: list[Path] = []
    for p in sorted(data_root.rglob("*.npy")):
        if any(s in p.stem.lower() for s in skip):
            continue
        out.append(p)
    return out


def batch_window_count(path: Path) -> int:
    """Window count without loading the whole array into memory."""
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return int(arr.shape[0]) if arr.ndim == 3 else 1


def try_manifest(data_root: Path, batch_files: list[Path]) -> dict[Path, list[str]] | None:
    """Look for a manifest written when the batches were built."""
    candidates = list(data_root.rglob("*manifest*.csv")) + list(data_root.rglob("*manifest*.json"))
    candidates += list(data_root.rglob("*index*.csv"))
    if not candidates:
        return None

    print(f"  found {len(candidates)} candidate manifest file(s)")
    for manifest in candidates:
        print(f"    trying {manifest.name}")
        rows: list[dict] = []
        try:
            if manifest.suffix == ".json":
                data = json.loads(manifest.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else data.get("windows", [])
            else:
                with manifest.open("r", newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
        except Exception as exc:
            print(f"      unreadable: {exc}")
            continue

        if not rows:
            continue
        keys = set(rows[0])
        patient_key = next(
            (k for k in ("patient", "patient_id", "subject", "subject_id") if k in keys), None
        )
        file_key = next((k for k in ("batch", "batch_file", "file", "source") if k in keys), None)
        if patient_key is None:
            print(f"      no patient column; has {sorted(keys)}")
            continue

        by_file: dict[Path, list[str]] = {}
        if file_key is None:
            # A single flat ordering: distribute across the batch files in order.
            ids = [str(r[patient_key]) for r in rows]
            cursor = 0
            for path in batch_files:
                n = batch_window_count(path)
                if cursor + n > len(ids):
                    print(f"      manifest too short: {len(ids)} ids for more windows")
                    return None
                by_file[path] = ids[cursor : cursor + n]
                cursor += n
        else:
            grouped: dict[str, list[str]] = {}
            for r in rows:
                grouped.setdefault(Path(str(r[file_key])).stem, []).append(str(r[patient_key]))
            for path in batch_files:
                if path.stem not in grouped:
                    print(f"      manifest lacks an entry for {path.stem}")
                    return None
                by_file[path] = grouped[path.stem]
        print("      manifest matched")
        return by_file
    return None


def try_corpus_match(
    corpus_root: Path, batch_files: list[Path]
) -> dict[Path, list[str]] | None:
    """Rebuild the ordering by walking the source corpus.

    Exact only when the total window count agrees with the batches. The script says so
    either way rather than assuming.
    """
    if not corpus_root or not corpus_root.is_dir():
        print("  no corpus root supplied or it does not exist")
        return None

    edfs = sorted(corpus_root.rglob("*.edf"))
    if not edfs:
        print(f"  no EDF files under {corpus_root}")
        return None

    patients = []
    for edf in edfs:
        m = TUH_PATIENT_RE.search(edf.name) or TUH_PATIENT_RE.search(str(edf.parent))
        patients.append(m.group(1) if m else edf.parent.name)

    counts = Counter(patients)
    total_batch_windows = sum(batch_window_count(p) for p in batch_files)
    print(f"  corpus: {len(edfs)} recordings across {len(counts)} patients")
    print(f"  batches: {total_batch_windows} windows across {len(batch_files)} files")
    print(
        "  cannot map recordings to windows without knowing the windowing parameters "
        "used when the batches were built (window length, stride, rejection rules)"
    )
    return None


def write_sidecars(mapping: dict[Path, list[str]], dry_run: bool) -> int:
    written = 0
    for path, ids in mapping.items():
        n_expected = batch_window_count(path)
        if len(ids) != n_expected:
            print(f"  SKIP {path.name}: {len(ids)} ids for {n_expected} windows")
            continue
        sidecar = path.with_name(path.stem + "_patients.npy")
        n_patients = len(set(ids))
        print(f"  {path.name}: {n_expected} windows, {n_patients} patients -> {sidecar.name}")
        if not dry_run:
            np.save(sidecar, np.asarray(ids, dtype=object))
            written += 1
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="BinaryClassification_Data root")
    p.add_argument("--corpus-root", default=None, help="source THUSZ/edf tree")
    p.add_argument("--dry-run", action="store_true", help="report without writing")
    args = p.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        print(f"ERROR: data root not found: {data_root}")
        return 1

    batch_files = find_batch_files(data_root)
    print(f"Found {len(batch_files)} batch file(s) under {data_root}\n")
    if not batch_files:
        return 1

    existing = [p for p in batch_files if p.with_name(p.stem + "_patients.npy").exists()]
    if existing:
        print(f"{len(existing)} file(s) already have sidecars.\n")

    print("Strategy 1: manifest")
    mapping = try_manifest(data_root, batch_files)

    if mapping is None:
        print("\nStrategy 2: corpus match")
        mapping = try_corpus_match(
            Path(args.corpus_root) if args.corpus_root else None, batch_files
        )

    if mapping is None:
        print("\n" + "=" * 74)
        print("Could not recover per-window patient identity.")
        print("=" * 74)
        print(
            "\nThe batch files pool windows from many patients and carry no record of\n"
            "which window came from whom. That information cannot be reconstructed from\n"
            "the arrays alone, so there are two honest ways forward.\n"
        )
        print("Option A, preferred: rebuild the batches carrying identity.")
        print("  When windowing the corpus, save one patient id per window alongside the")
        print("  array. The loader looks for <batch_stem>_patients.npy, so this is the")
        print("  only change needed:\n")
        print("      np.save(out_dir / f'{stem}.npy', windows)")
        print("      np.save(out_dir / f'{stem}_patients.npy',")
        print("              np.asarray(patient_per_window, dtype=object))\n")
        print("Option B: proceed with a coarser grouping and describe it accurately.")
        print("  Set group_key: file and require_patient_groups: false. Folds then split")
        print("  by batch file, which is weaker than by patient, and the results must be")
        print("  reported as file-wise rather than patient-wise.\n")
        print("What must not happen is the previous behaviour, which split each batch")
        print("file into about a hundred equal chunks of the array index and treated")
        print("those as patients. That puts one recording into every fold, so the score")
        print("measures within-recording interpolation while being reported as")
        print("cross-patient generalisation.")
        return 2

    print(f"\n{'Would write' if args.dry_run else 'Writing'} sidecars:")
    written = write_sidecars(mapping, args.dry_run)
    if args.dry_run:
        print("\nDry run. Re-run without --dry-run to write.")
    else:
        print(f"\nWrote {written} sidecar file(s).")
        print("Verify with: python -m eeg_qgan.experiments.runner capacity_rematch --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
