#!/usr/bin/env python3
"""Report whether the clinical inference fold is ready for the demo."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.paths import discover_studies  # noqa: E402


def main() -> int:
    rows = discover_studies()
    print("Clinical inference fold (needs AE + generators + detectors):\n")
    ready = 0
    for s in rows:
        mark = "READY" if s["available"] else "MISSING"
        if s["available"]:
            ready += 1
        print(f"[{mark}] {s['id']}")
        print(f"         {s['label']}")
        print(f"         path: {s['path']}")
        print(
            f"         AE={s['has_autoencoder']}  gens={s['generators_on_disk']}  "
            f"gmm={s['has_gmm']}  detectors={s['n_detectors']}"
        )
        if s["missing_arms"]:
            print(f"         still missing arms: {s['missing_arms']}")
        print()
    print(
        "Mechanism studies (entanglement / Fourier / rematch extras) are generation-only "
        "and are not required for this clinical inference demo."
    )
    print(f"{ready}/{len(rows)} clinical studies ready.")
    if not ready:
        print("Sync clinical fold:  .\\demo\\sync_from_narval.ps1")
    print(json.dumps({"ready": ready, "total": len(rows)}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
