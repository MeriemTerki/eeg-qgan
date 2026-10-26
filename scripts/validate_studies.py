#!/usr/bin/env python3
"""Load and validate every study config.

Worth running before submitting a batch: an inconsistent study fails here in seconds
rather than after its jobs have queued and started. Several of the old repositories had
job scripts pointing at configs that no longer existed, which only surfaced when the job
died on the cluster.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg_qgan.config import list_studies, load_config  # noqa: E402


def main() -> int:
    studies = list_studies()
    print(f"Validating {len(studies)} studies\n")
    print(f"  {'study':<26} {'seeds x folds':<14} {'arms':<44} downstream")
    print("  " + "-" * 92)

    failures: list[tuple[str, str]] = []
    for name in studies:
        try:
            cfg = load_config(name)
        except Exception as exc:
            print(f"  {name:<26} FAILED  {type(exc).__name__}: {exc}")
            failures.append((name, str(exc)))
            continue
        arms = ", ".join(cfg["generator_arms"])
        if len(arms) > 42:
            arms = arms[:39] + "..."
        shape = f"{cfg['n_seeds']} x {cfg['n_folds']}"
        downstream = "no" if cfg.get("skip_downstream") else "yes"
        print(f"  {name:<26} {shape:<14} {arms:<44} {downstream}")

    print()
    if failures:
        print(f"{len(failures)} study/studies failed validation:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    print(f"All {len(studies)} studies valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
