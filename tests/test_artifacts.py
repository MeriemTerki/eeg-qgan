"""Tests for checkpoint and metrics persistence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eeg_qgan.training.latent_space import LatentTransform
from eeg_qgan.utils.artifacts import (
    load_latent_transform,
    save_fold_artifacts,
    save_model_checkpoint,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

    def describe(self):
        return {"kind": "tiny", "n_params": sum(p.numel() for p in self.parameters())}


def test_model_checkpoint_roundtrip(tmp_path: Path):
    model = Tiny()
    path = tmp_path / "model.pt"
    save_model_checkpoint(model, path, kind="tiny", extra={"note": "ok"})
    loaded = torch.load(path, map_location="cpu")
    assert loaded["kind"] == "tiny"
    assert loaded["extra"]["note"] == "ok"
    assert "fc.weight" in loaded["state_dict"]
    clone = Tiny()
    clone.load_state_dict(loaded["state_dict"])
    x = torch.randn(3, 4)
    assert torch.allclose(model(x), clone(x))


def test_latent_transform_roundtrip(tmp_path: Path):
    transform = LatentTransform(
        mode="whiten",
        mean=np.array([0.1, -0.2, 0.3]),
        std=np.array([1.0, 0.5, 2.0]),
        output_scale=1.2,
        coverage_before_scale=0.8,
    )
    from eeg_qgan.utils.artifacts import save_latent_transform

    path = tmp_path / "lt.npz"
    save_latent_transform(transform, path)
    loaded = load_latent_transform(path)
    assert loaded.mode == "whiten"
    np.testing.assert_allclose(loaded.mean, transform.mean)
    np.testing.assert_allclose(loaded.std, transform.std)
    assert abs(loaded.output_scale - 1.2) < 1e-9


def test_fold_artifacts_layout(tmp_path: Path):
    ae = Tiny()
    gen = Tiny()
    transform = LatentTransform(
        mode="identity",
        mean=np.zeros(2),
        std=np.ones(2),
        output_scale=1.0,
    )
    z_report = np.random.randn(16, 2).astype(np.float32)
    codes = np.random.randn(8, 2).astype(np.float32)
    cfg = {
        "study": "unit",
        "save": {
            "artifacts": True,
            "autoencoder": True,
            "generators": True,
            "critics": False,
            "latents": True,
            "latent_transform": True,
            "decoded": False,
            "detectors": False,
        },
    }
    paths = save_fold_artifacts(
        out_dir=tmp_path,
        seed_index=0,
        fold=1,
        cfg=cfg,
        ae=ae,
        ae_meta={"excluded": False, "nmse": 0.1},
        transform=transform,
        z_report=z_report,
        arm_models={
            "quantum": {
                "generator": gen,
                "critic": None,
                "codes": codes,
                "decoded": None,
                "selection_step": 100,
                "selection_distance": 0.2,
            }
        },
        fold_record={"excluded": False},
    )
    assert paths["enabled"] is True
    assert (tmp_path / "artifacts" / "seed0_fold1" / "autoencoder.pt").exists()
    assert (tmp_path / "artifacts" / "seed0_fold1" / "generators" / "quantum.pt").exists()
    assert (tmp_path / "artifacts" / "seed0_fold1" / "latents" / "quantum_fake.npy").exists()
    assert (tmp_path / "artifacts" / "seed0_fold1" / "z_report.npy").exists()
    assert (tmp_path / "artifacts" / "seed0_fold1" / "latent_transform.npz").exists()
    assert (tmp_path / "artifacts" / "seed0_fold1" / "meta.json").exists()
    # Paths recorded in the fold row are relative to the study output directory.
    assert not Path(paths["autoencoder"]).is_absolute()
    assert paths["generators"]["quantum"].startswith("artifacts/")
