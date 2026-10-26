"""Persist checkpoints and arrays so a finished fold can be reused later.

The metrics JSON alone is not enough for Phase-1 style re-analysis (Fourier surrogate,
shot-noise ablation, Jacobian panel): those need the selected generator weights, the
autoencoder, the latent transform, and a held-out latent sample. This module writes
those artefacts under a predictable layout and records their paths in the fold row so
the results file is a complete index.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from eeg_qgan.training.latent_space import LatentTransform
from eeg_qgan.utils.io import ensure_dir, save_json


def fold_artifact_dir(out_dir: Path, seed_index: int, fold: int) -> Path:
    return ensure_dir(out_dir / "artifacts" / f"seed{seed_index}_fold{fold}")


def save_model_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    kind: str,
    extra: dict | None = None,
) -> str:
    """Save a torch checkpoint with enough metadata to reload it later."""
    ensure_dir(path.parent)
    payload: dict[str, Any] = {
        "kind": kind,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "class_name": type(model).__name__,
    }
    if hasattr(model, "describe"):
        try:
            payload["description"] = model.describe()
        except Exception:
            pass
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return str(path)


def save_latent_transform(transform: LatentTransform, path: Path) -> str:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        mode=np.asarray(transform.mode),
        mean=np.asarray(transform.mean, dtype=np.float64),
        std=np.asarray(transform.std, dtype=np.float64),
        output_scale=np.asarray(transform.output_scale, dtype=np.float64),
        coverage_before_scale=np.asarray(transform.coverage_before_scale, dtype=np.float64),
    )
    return str(path)


def load_latent_transform(path: Path | str) -> LatentTransform:
    data = np.load(path, allow_pickle=True)
    mode_raw = data["mode"]
    mode = str(mode_raw.item() if getattr(mode_raw, "shape", None) == () else mode_raw)
    return LatentTransform(
        mode=mode,  # type: ignore[arg-type]
        mean=np.asarray(data["mean"], dtype=np.float64),
        std=np.asarray(data["std"], dtype=np.float64),
        output_scale=float(np.asarray(data["output_scale"]).reshape(())),
        coverage_before_scale=float(
            np.asarray(data["coverage_before_scale"]).reshape(())
            if "coverage_before_scale" in data.files
            else 1.0
        ),
    )


def save_array(array: np.ndarray, path: Path) -> str:
    ensure_dir(path.parent)
    np.save(path, np.asarray(array))
    return str(path)


def save_fold_artifacts(
    *,
    out_dir: Path,
    seed_index: int,
    fold: int,
    cfg: dict,
    ae: nn.Module | None,
    ae_meta: dict,
    transform: LatentTransform | None,
    z_report: np.ndarray | None,
    arm_models: dict[str, dict[str, Any]],
    fold_record: dict,
    real_figure_windows: np.ndarray | None = None,
) -> dict:
    """Write everything needed to reuse this fold after the job ends.

    Layout::

        artifacts/seed{i}_fold{f}/
          meta.json
          autoencoder.pt
          latent_transform.npz
          z_report.npy
          generators/<arm>.pt
          critics/<arm>.pt          (if present)
          latents/<arm>_fake.npy    (generator-space codes)
          decoded/<arm>.npy         (optional, can be large)

    Returns a dict of relative paths that is merged into the fold JSON row.
    """
    save_cfg = cfg.get("save", {}) or {}
    if not bool(save_cfg.get("artifacts", True)):
        return {"enabled": False}

    fold_dir = fold_artifact_dir(out_dir, seed_index, fold)
    paths: dict[str, Any] = {
        "enabled": True,
        "dir": fold_dir.relative_to(out_dir).as_posix(),
    }

    if ae is not None and bool(save_cfg.get("autoencoder", True)):
        abs_path = save_model_checkpoint(
            ae,
            fold_dir / "autoencoder.pt",
            kind="autoencoder",
            extra={"meta": ae_meta},
        )
        paths["autoencoder"] = Path(abs_path).relative_to(out_dir).as_posix()

    if transform is not None and bool(save_cfg.get("latent_transform", True)):
        abs_path = save_latent_transform(transform, fold_dir / "latent_transform.npz")
        paths["latent_transform"] = Path(abs_path).relative_to(out_dir).as_posix()

    if z_report is not None and bool(save_cfg.get("latents", True)):
        abs_path = save_array(z_report, fold_dir / "z_report.npy")
        paths["z_report"] = Path(abs_path).relative_to(out_dir).as_posix()

    gen_paths: dict[str, str] = {}
    critic_paths: dict[str, str] = {}
    fake_paths: dict[str, str] = {}
    decoded_paths: dict[str, str] = {}
    figure_paths: dict[str, str] = {}

    figure_n = int(save_cfg.get("decoded_figure_windows", 0) or 0)

    for arm, bundle in arm_models.items():
        generator = bundle.get("generator")
        critic = bundle.get("critic")
        fake_codes = bundle.get("codes")
        decoded = bundle.get("decoded")

        if generator is not None and bool(save_cfg.get("generators", True)):
            abs_path = save_model_checkpoint(
                generator,
                fold_dir / "generators" / f"{arm}.pt",
                kind=f"generator:{arm}",
                extra={
                    "arm": arm,
                    "selection_step": bundle.get("selection_step"),
                    "selection_distance": bundle.get("selection_distance"),
                },
            )
            gen_paths[arm] = Path(abs_path).relative_to(out_dir).as_posix()
        if critic is not None and bool(save_cfg.get("critics", False)):
            abs_path = save_model_checkpoint(
                critic,
                fold_dir / "critics" / f"{arm}.pt",
                kind=f"critic:{arm}",
                extra={"arm": arm},
            )
            critic_paths[arm] = Path(abs_path).relative_to(out_dir).as_posix()
        if fake_codes is not None and bool(save_cfg.get("latents", True)):
            abs_path = save_array(
                np.asarray(fake_codes, dtype=np.float32),
                fold_dir / "latents" / f"{arm}_fake.npy",
            )
            fake_paths[arm] = Path(abs_path).relative_to(out_dir).as_posix()
        if decoded is not None and bool(save_cfg.get("decoded", False)):
            # Off by default: n_synthetic x 22 x 256 float32 is large on disk.
            abs_path = save_array(
                np.asarray(decoded, dtype=np.float32),
                fold_dir / "decoded" / f"{arm}.npy",
            )
            decoded_paths[arm] = Path(abs_path).relative_to(out_dir).as_posix()
        if decoded is not None and figure_n > 0:
            subset = np.asarray(decoded[:figure_n], dtype=np.float32)
            abs_path = save_array(subset, fold_dir / "decoded" / f"{arm}_figure.npy")
            figure_paths[arm] = Path(abs_path).relative_to(out_dir).as_posix()

    if gen_paths:
        paths["generators"] = gen_paths
    if critic_paths:
        paths["critics"] = critic_paths
    if fake_paths:
        paths["fake_latents"] = fake_paths
    if decoded_paths:
        paths["decoded"] = decoded_paths
    if figure_paths:
        paths["decoded_figure"] = figure_paths
    if real_figure_windows is not None and len(real_figure_windows) > 0:
        abs_path = save_array(
            np.asarray(real_figure_windows, dtype=np.float32),
            fold_dir / "decoded" / "real_figure.npy",
        )
        paths["real_figure"] = Path(abs_path).relative_to(out_dir).as_posix()

    meta = {
        "seed_index": seed_index,
        "fold": fold,
        "study": cfg.get("study"),
        "paths": {k: v for k, v in paths.items() if k != "enabled"},
        "autoencoder_meta": ae_meta,
        "latent_transform": transform.as_dict() if transform is not None else None,
        "arms": sorted(arm_models),
        "excluded": bool(fold_record.get("excluded", False)),
        "exclusion_reason": fold_record.get("exclusion_reason"),
    }
    save_json(meta, fold_dir / "meta.json")
    paths["meta"] = (fold_dir / "meta.json").relative_to(out_dir).as_posix()
    return paths
