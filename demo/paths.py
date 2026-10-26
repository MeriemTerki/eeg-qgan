"""Filesystem layout for the clinical inference demo."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
THESIS_ROOT = REPO_ROOT.parent
SAMPLES_DIR = DEMO_DIR / "samples"
ARTIFACTS_DIR = DEMO_DIR / "artifacts"
STATIC_DIR = DEMO_DIR / "static"
CATALOG_PATH = SAMPLES_DIR / "catalog.json"

DEFAULT_CAMPAIGN = (
    THESIS_ROOT
    / "eeg_qgan_results_thesis_figures"
    / "thesis_figure_capture"
    / "artifacts"
    / "seed0_fold0"
)

# Only studies that support the clinical inference demo (AE + generators + detectors).
# Mechanism folds (entanglement / Fourier / rematch extras) are generation-only on
# Narval and are intentionally excluded from this page.
KNOWN_STUDIES: tuple[dict[str, Any], ...] = (
    {
        "id": "thesis_figure_capture",
        "label": "Clinical inference (hybrid / twin / GMM + detectors)",
        "rel": "eeg_qgan_results_thesis_figures/thesis_figure_capture/artifacts/seed0_fold0",
        "arms": ("quantum", "classical_equiv", "gmm"),
        "has_detectors_expected": True,
        "role": "clinical",
    },
)

_env = os.environ.get("EEG_QGAN_DEMO_ARTIFACTS", "").strip()
CAMPAIGN_DIR = Path(_env).expanduser() if _env else (
    DEFAULT_CAMPAIGN if (DEFAULT_CAMPAIGN / "autoencoder.pt").is_file() else None
)

ARM_LABELS = {
    "quantum": "Hybrid Style-qGAN",
    "classical_equiv": "Capacity-matched twin",
    "gmm": "Gaussian mixture",
    "fourier": "Fourier analogue",
    "classical_large": "Larger MLP",
    "classical_matched": "Matched MLP",
    "quantum_none": "Product circuit",
    "quantum_linear": "Linear entanglement",
    "quantum_ring": "Ring entanglement",
    "quantum_full": "Full entanglement",
}

TCP_CHANNELS = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


def study_path(rel: str) -> Path:
    return THESIS_ROOT / Path(rel)


def discover_studies() -> list[dict[str, Any]]:
    """Local availability of clinical inference folds."""
    rows: list[dict[str, Any]] = []
    for spec in KNOWN_STUDIES:
        fold = study_path(spec["rel"])
        gens_dir = fold / "generators"
        gen_files = sorted(p.stem for p in gens_dir.glob("*.pt")) if gens_dir.is_dir() else []
        has_ae = (fold / "autoencoder.pt").is_file()
        has_gmm = (fold / "gmm" / "gmm.json").is_file()
        n_det = (
            len(list((fold / "detectors").glob("*.pt")))
            if (fold / "detectors").is_dir()
            else 0
        )
        present_arms = list(gen_files)
        if has_gmm and "gmm" not in present_arms:
            present_arms.append("gmm")
        missing = [a for a in spec["arms"] if a not in present_arms]
        inference_ready = has_ae and bool(present_arms) and n_det > 0
        rows.append(
            {
                **spec,
                "path": str(fold),
                "available": inference_ready,
                "has_autoencoder": has_ae,
                "generators_on_disk": gen_files,
                "has_gmm": has_gmm,
                "n_detectors": n_det,
                "present_arms": present_arms,
                "missing_arms": missing,
            }
        )
    return rows
