"""Discover saved checkpoints on this machine."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from demo.paths import ARM_LABELS, THESIS_ROOT


def inventory(root: Path | None = None) -> dict[str, Any]:
    root = Path(root or THESIS_ROOT)
    folds: list[dict[str, Any]] = []
    seen: set[str] = set()
    for meta in root.rglob("meta.json"):
        fold = meta.parent
        key = str(fold)
        if key in seen:
            continue
        seen.add(key)
        gens = sorted((fold / "generators").glob("*.pt")) if (fold / "generators").is_dir() else []
        gmms = sorted((fold / "gmm").glob("*.json")) if (fold / "gmm").is_dir() else []
        dets = sorted((fold / "detectors").glob("*.pt")) if (fold / "detectors").is_dir() else []
        ae = fold / "autoencoder.pt"
        real = fold / "decoded" / "real_figure.npy"
        if not gens and not ae.is_file():
            continue
        arms = [p.stem for p in gens]
        if gmms:
            arms.extend(p.stem for p in gmms)
        folds.append(
            {
                "path": str(fold),
                "study": fold.parent.parent.name if fold.parent.name == "artifacts" else fold.parent.name,
                "fold": fold.name,
                "autoencoder": ae.is_file(),
                "autoencoder_bytes": ae.stat().st_size if ae.is_file() else 0,
                "generators": [
                    {"arm": p.stem, "label": ARM_LABELS.get(p.stem, p.stem), "bytes": p.stat().st_size}
                    for p in gens
                ],
                "mixtures": [p.stem for p in gmms],
                "detectors": [p.stem for p in dets],
                "has_real_tuh_windows": real.is_file(),
                "live": (fold / "autoencoder.pt").is_file() and bool(gens),
            }
        )
    legacy: list[dict[str, Any]] = []
    for gen in root.rglob("generator.pt"):
        rel = str(gen.relative_to(root)).replace("\\", "/")
        if "paper_phase" not in rel and "phase1" not in rel:
            continue
        legacy.append(
            {
                "path": str(gen.parent),
                "arm": gen.parent.name,
                "label": ARM_LABELS.get(gen.parent.name.replace("_adv", "").replace("_mmd", ""), gen.parent.name),
                "bytes": gen.stat().st_size,
                "note": "Earlier campaign. Different autoencoder; not loaded in this session.",
            }
        )
    return {"n_folds": len(folds), "folds": folds, "n_legacy": len(legacy), "legacy": legacy[:40]}
