#!/usr/bin/env python3
"""Prepare sample windows and optional demo checkpoints.

Usage, from the eeg_qgan repository root:

    python demo/prepare.py
    python demo/prepare.py --data-root PATH\\TO\\BinaryClassification_Data
    python demo/prepare.py --skip-train
    python demo/prepare.py --from-artifacts PATH\\TO\\artifacts\\seed0_fold0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.paths import ARTIFACTS_DIR, CATALOG_PATH, SAMPLES_DIR  # noqa: E402
from eeg_qgan.utils.artifacts import save_latent_transform, save_model_checkpoint  # noqa: E402
from scripts.make_synthetic_dataset import make_windows  # noqa: E402


def _write_window(path: Path, window: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(window, dtype=np.float32))


def _pick_from_array(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[None]
    n = min(int(n), len(arr))
    idx = rng.choice(len(arr), size=n, replace=False)
    return arr[idx]


def collect_tuh_windows(data_root: Path, n_each: int, rng: np.random.Generator) -> list[dict]:
    from eeg_qgan.config import load_config
    from eeg_qgan.data.tuh import load_windows

    cfg = load_config("demo")
    cfg["data"]["root"] = str(data_root)
    cfg["data"]["max_batch_files"] = 2
    cfg["data"]["max_segments_per_batch"] = 80
    windows = load_windows(cfg)
    samples: list[dict] = []
    for label, name in ((1, "seizure"), (0, "non_seizure")):
        mask = np.asarray(windows.y) == label
        if not np.any(mask):
            continue
        chosen = _pick_from_array(windows.x[mask], n_each, rng)
        groups = np.asarray(windows.groups)[mask]
        for i, window in enumerate(chosen):
            sample_id = f"{'seiz' if label else 'ns'}_{i + 1:03d}"
            file_name = f"{sample_id}.npy"
            _write_window(SAMPLES_DIR / file_name, window)
            src = str(groups[min(i, len(groups) - 1)])
            samples.append(
                {
                    "id": sample_id,
                    "label": name,
                    "title": f"{name.replace('_', ' ').title()} window {i + 1}",
                    "file": file_name,
                    "provenance": "tuh",
                    "source_file": src,
                    "note": "Preprocessed TUH-SZ window: 22 x 256 at 200 Hz.",
                }
            )
    return samples


def collect_synthetic(n_each: int, rng: np.random.Generator) -> list[dict]:
    samples: list[dict] = []
    for label, name in ((1, "seizure"), (0, "non_seizure")):
        batch = make_windows(n_each, 22, 256, seizure=label == 1, rng=rng)
        for i, window in enumerate(batch):
            sample_id = f"{'seiz' if label else 'ns'}_{i + 1:03d}"
            file_name = f"{sample_id}.npy"
            _write_window(SAMPLES_DIR / file_name, window)
            samples.append(
                {
                    "id": sample_id,
                    "label": name,
                    "title": f"{name.replace('_', ' ').title()} stand-in {i + 1}",
                    "file": file_name,
                    "provenance": "synthetic",
                    "source_file": None,
                    "note": (
                        "Synthetic window matching the TUH-SZ contract "
                        "(22 x 256 at 200 Hz). Replace by running with --data-root."
                    ),
                }
            )
    return samples


def copy_campaign(src: Path) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("autoencoder.pt", "latent_transform.npz"):
        source = src / name
        if source.is_file():
            shutil.copy2(source, ARTIFACTS_DIR / name)
    gen_dir = src / "generators"
    if gen_dir.is_dir():
        dest = ARTIFACTS_DIR / "generators"
        dest.mkdir(parents=True, exist_ok=True)
        for arm in ("quantum.pt", "classical_equiv.pt"):
            source = gen_dir / arm
            if source.is_file():
                shutil.copy2(source, dest / arm)
    cfg = src / "config_resolved.json"
    if not cfg.is_file():
        cfg = src.parent.parent / "config_resolved.json"
    if cfg.is_file():
        shutil.copy2(cfg, ARTIFACTS_DIR / "config_resolved.json")
    print(f"Copied campaign artifacts from {src} -> {ARTIFACTS_DIR}")


def train_demo_weights(samples: list[dict]) -> None:
    import torch

    from eeg_qgan.config import load_config
    from eeg_qgan.models.registry import build_generator
    from eeg_qgan.training.autoencoder import build_autoencoder, encode_windows, train_autoencoder
    from eeg_qgan.training.latent_space import fit_latent_transform

    cfg = load_config("demo")
    cfg["use_gpu"] = False
    windows = np.stack(
        [np.load(SAMPLES_DIR / item["file"]).astype(np.float32) for item in samples],
        axis=0,
    )
    if windows.ndim == 4:
        windows = windows[:, 0]
    n_val = max(2, len(windows) // 4)
    x_val = windows[:n_val]
    x_train = windows[n_val:] if len(windows) > n_val else windows

    print(f"Training demo autoencoder on {len(x_train)} windows...", flush=True)
    ae, meta = train_autoencoder(x_train, x_val, cfg, device="cpu", fold_label="demo")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "generators").mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(ae, ARTIFACTS_DIR / "autoencoder.pt", kind="autoencoder", extra={"meta": meta})

    z = encode_windows(ae, x_train, device="cpu")
    transform = fit_latent_transform(z, mode="identity")
    save_latent_transform(transform, ARTIFACTS_DIR / "latent_transform.npz")

    scale = float(transform.output_scale)
    for arm in ("quantum", "classical_equiv"):
        gen = build_generator(arm, cfg, output_scale=scale)
        gen.eval()
        save_model_checkpoint(
            gen,
            ARTIFACTS_DIR / "generators" / f"{arm}.pt",
            kind=f"generator:{arm}",
            extra={"note": "Thesis architecture instantiated for the local demo."},
        )
        print(f"  saved {arm}: {gen.count_params()} parameters", flush=True)

    (ARTIFACTS_DIR / "config_resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("Demo weights written to", ARTIFACTS_DIR)
    _ = torch  # keep import used if checkpoints need it later


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the Style-qGAN deployment demo.")
    parser.add_argument("--data-root", type=Path, default=None, help="TUH-SZ BinaryClassification_Data root")
    parser.add_argument("--n-each", type=int, default=6, help="Windows per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true", help="Write samples only")
    parser.add_argument(
        "--from-artifacts",
        type=Path,
        default=None,
        help="Copy a campaign fold (autoencoder.pt and generators/*.pt)",
    )
    args = parser.parse_args()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.data_root is not None:
        samples = collect_tuh_windows(args.data_root, args.n_each, rng)
        print(f"Wrote {len(samples)} TUH-SZ windows to {SAMPLES_DIR}")
    else:
        samples = collect_synthetic(args.n_each, rng)
        print(f"Wrote {len(samples)} synthetic stand-in windows to {SAMPLES_DIR}")

    catalog = {
        "window": {"n_channels": 22, "sample_length": 256, "fs_hz": 200, "duration_s": 1.28},
        "samples": samples,
    }
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print("Catalog:", CATALOG_PATH)

    if args.from_artifacts is not None:
        copy_campaign(args.from_artifacts)
        return 0

    if not args.skip_train:
        try:
            train_demo_weights(samples)
        except Exception as exc:
            print(
                f"Demo weight training failed ({exc}). Samples are ready; "
                "the API can still instantiate the architecture."
            )
    else:
        print("Skipped training. The API will instantiate the architecture without demo AE weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
