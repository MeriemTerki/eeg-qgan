#!/usr/bin/env python3
"""Verify that a study output directory contains the expected artefacts.

Run after a job (or after a local smoke) to confirm checkpoints and metrics exist
before you delete logs or submit the next study::

    python scripts/verify_outputs.py results/smoke
    python scripts/verify_outputs.py $OUTPUT_ROOT/capacity_rematch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_RESULT_KEYS = (
    "study",
    "config",
    "rows",
    "completed",
    "grouping",
    "capacity_audit",
)

REQUIRED_ROW_METRIC_PATHS = (
    ("arms",),
    ("mmd_noise_floor",),
    ("autoencoder",),
)


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _bad(msg: str) -> None:
    print(f"  BAD {msg}")


def verify(study_dir: Path) -> int:
    print(f"Checking {study_dir}")
    if not study_dir.is_dir():
        _bad(f"not a directory: {study_dir}")
        return 1

    failures = 0
    results = sorted(study_dir.glob("results_*.json"))
    if not results:
        _bad("no results_*.json found")
        failures += 1
    else:
        for path in results:
            payload = json.loads(path.read_text(encoding="utf-8"))
            missing = [k for k in REQUIRED_RESULT_KEYS if k not in payload]
            if missing:
                _bad(f"{path.name}: missing keys {missing}")
                failures += 1
            else:
                _ok(f"{path.name}: {payload['completed']} fold row(s)")

            for i, row in enumerate(payload.get("rows", [])):
                for keys in REQUIRED_ROW_METRIC_PATHS:
                    node = row
                    for k in keys:
                        if not isinstance(node, dict) or k not in node:
                            _bad(f"{path.name} row {i}: missing {'/'.join(keys)}")
                            failures += 1
                            break
                        node = node[k]

                arts = row.get("artifacts") or {}
                if not arts.get("enabled", False):
                    _bad(f"{path.name} row {i}: artifacts disabled or missing")
                    failures += 1
                    continue

                fold_dir = study_dir / arts.get("dir", "")
                if not fold_dir.is_dir():
                    _bad(f"{path.name} row {i}: artifact dir missing ({arts.get('dir')})")
                    failures += 1
                    continue
                _ok(f"row {i}: artifacts at {arts['dir']}")

                for rel in (
                    arts.get("autoencoder"),
                    arts.get("latent_transform"),
                    arts.get("z_report"),
                ):
                    if rel and not (study_dir / rel).exists():
                        _bad(f"missing file {rel}")
                        failures += 1

                for arm, rel in (arts.get("generators") or {}).items():
                    if not (study_dir / rel).exists():
                        _bad(f"missing generator checkpoint for {arm}: {rel}")
                        failures += 1
                    else:
                        _ok(f"generator checkpoint {arm}")

                for arm, rel in (arts.get("fake_latents") or {}).items():
                    if not (study_dir / rel).exists():
                        _bad(f"missing fake latents for {arm}: {rel}")
                        failures += 1

    if (study_dir / "config_resolved.json").exists():
        _ok("config_resolved.json")
    else:
        _bad("config_resolved.json missing")
        failures += 1

    if (study_dir / "capacity_audit.json").exists():
        _ok("capacity_audit.json")
    else:
        _bad("capacity_audit.json missing")
        failures += 1

    print()
    if failures:
        print(f"{failures} problem(s) found.")
        return 1
    print("All expected checkpoints and metrics are present.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study_dir", help="results/<study> directory")
    args = p.parse_args()
    return verify(Path(args.study_dir))


if __name__ == "__main__":
    raise SystemExit(main())
