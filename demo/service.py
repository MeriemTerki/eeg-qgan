"""Load campaign generators and serve inference for the research workstation."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from demo.inventory import inventory
from demo.paths import (
    ARM_LABELS,
    ARTIFACTS_DIR,
    CAMPAIGN_DIR,
    CATALOG_PATH,
    SAMPLES_DIR,
    TCP_CHANNELS,
    discover_studies,
    study_path,
)
from demo.pipeline import (
    ADDED_VALUE,
    CLIP_WINDOWS,
    LEVELS,
    LIVE_ARMS,
    MANUSCRIPT_ONLY_ARMS,
    STAGES,
    SYNTHETIC_CROSSFADE,
    THESIS_HEADLINE,
    CampaignRecord,
    DetectorBank,
    clip_seconds,
    latent_noise_floor,
    level1_latent,
    level2_decoded,
    pack_one_synthetic_clip,
    pack_real_clip,
    windows_needed_for_synthetic_clip,
)
from eeg_qgan.config import load_config
from eeg_qgan.models.mixture import build_mixture
from eeg_qgan.models.quantum import entangler_pairs
from eeg_qgan.models.registry import ALL_ARMS, build_generator
from eeg_qgan.training.adversarial import decode_codes, sample_latent_codes
from eeg_qgan.training.autoencoder import build_autoencoder, encode_windows, reconstruct
from eeg_qgan.training.latent_space import LatentTransform
from eeg_qgan.utils.artifacts import load_latent_transform


def _identity_transform(dim: int) -> LatentTransform:
    zeros = np.zeros(int(dim), dtype=np.float64)
    ones = np.ones(int(dim), dtype=np.float64)
    return LatentTransform(mode="identity", mean=zeros, std=ones, output_scale=1.0)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def window_payload(window: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected (C, T) window, got {arr.shape}")
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "n_channels": int(arr.shape[0]),
        "sample_length": int(arr.shape[1]),
        "fs_hz": 200,
        "duration_s": float(arr.shape[1] / 200.0),
        "montage": "TCP bipolar",
        "channels": TCP_CHANNELS[: arr.shape[0]],
        "values": arr.tolist(),
        "peak": float(np.max(np.abs(arr))),
    }


def _load_gmm(path: Path, cfg: dict):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mix = build_mixture(cfg, seed=42)
    weights = np.asarray(payload["weights"], dtype=np.float64)
    means = np.asarray(payload["means"], dtype=np.float64)
    cov = np.asarray(payload["covariances"], dtype=np.float64)
    gmm = mix._gmm
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = cov
    try:
        from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

        gmm.precisions_cholesky_ = _compute_precision_cholesky(cov, str(gmm.covariance_type))
    except Exception:
        inv = np.linalg.inv(cov)
        gmm.precisions_cholesky_ = np.linalg.cholesky(inv)
    mix._fitted = True
    mix._fit_space = str((payload.get("fit") or {}).get("fit_space") or "identity")
    mix.n_components = int(len(weights))
    mix.latent_dim = int(means.shape[1])
    return mix


class DemoEngine:
    """Campaign autoencoder plus every generator found in the selected fold."""

    def __init__(self, fold_dir: Path | None = None) -> None:
        self.lock = threading.Lock()
        self.device = "cpu"
        self.cfg = load_config("demo")
        self.cfg["use_gpu"] = False
        self.weight_source = "architecture-only"
        self.fold_dir: Path | None = None
        self.study_id: str | None = None
        self.data_source = "synthetic"
        self.notes: list[str] = []
        self.transform = _identity_transform(int(self.cfg["latent_dim"]))
        self.ae = build_autoencoder(self.cfg, int(self.cfg["n_channels"]))
        self.generators: dict[str, Any] = {}
        self.mixtures: dict[str, Any] = {}
        self.loaded: dict[str, dict[str, Any]] = {}
        self._preferred_fold = Path(fold_dir) if fold_dir is not None else None
        self._load_weights()
        self.ae.eval()
        for gen in self.generators.values():
            gen.eval()
        self.catalog = self._build_catalog()
        # Stage 4 and the three evidence levels.
        self.detectors = DetectorBank()
        self.campaign = CampaignRecord(None)
        self.z_report: np.ndarray | None = None
        self.real_windows: np.ndarray | None = None
        self.fake_latents: dict[str, np.ndarray] = {}
        self.decoded_figure: dict[str, np.ndarray] = {}
        self._noise_floor: dict[str, Any] | None = None
        self._ictal_recon: dict[str, Any] | None = None
        self._load_pipeline_assets()
        self._resolve_study_id()

    def _resolve_study_id(self) -> None:
        if self.fold_dir is None:
            self.study_id = None
            return
        fold = self.fold_dir.resolve()
        for spec in discover_studies():
            if Path(spec["path"]).resolve() == fold:
                self.study_id = spec["id"]
                return
        self.study_id = self.fold_dir.parent.parent.name

    def arm_status(self) -> dict[str, Any]:
        """Only clinical inference arms that are actually loaded (generate + detect)."""
        clinical = ("quantum", "classical_equiv", "gmm")
        why = {
            "quantum": "Model under test under matched capacity.",
            "classical_equiv": "Equal-budget classical rematch peer.",
            "gmm": "Density reference in the same latent space (not equal-budget).",
        }
        role_of = {
            "quantum": "rematch",
            "classical_equiv": "rematch_peer",
            "gmm": "density_reference",
        }
        live_rows = []
        for arm in clinical:
            info = self.loaded.get(arm)
            if not info:
                continue
            live_rows.append(
                {
                    "arm": arm,
                    "label": ARM_LABELS.get(arm, arm),
                    "role": role_of[arm],
                    "live": True,
                    "params": info.get("params"),
                    "why": why[arm],
                    "supports": "generation+detection",
                }
            )
        return {
            "study_id": self.study_id,
            "fold": str(self.fold_dir) if self.fold_dir else None,
            "live_arms": live_rows,
            "missing_arms": [],
            "studies": discover_studies(),
            "added_value": ADDED_VALUE,
        }

    def _load_pipeline_assets(self) -> None:
        """Detectors, the real holdout codes, and the cached per-arm outputs."""
        if self.fold_dir is None:
            self.notes.append("No campaign fold: Stage 4 and Levels 1 to 3 are unavailable.")
            return

        self.detectors = DetectorBank.load(
            self.fold_dir,
            self.cfg,
            n_channels=int(self.cfg.get("n_channels", 22)),
            n_windows=CLIP_WINDOWS,
            device=self.device,
        )
        self.notes.extend(self.detectors.notes)

        # The study directory holds the fold record: two levels up from seed0_fold0.
        study_dir = self.fold_dir.parent.parent
        self.campaign = CampaignRecord(study_dir)
        if self.campaign.available:
            self.notes.append(
                f"Campaign record: study '{self.campaign.study}', "
                f"holdout '{self.campaign.fold_name}'"
            )
        else:
            self.notes.append("No results JSON beside this fold. Recorded metrics unavailable.")

        z_path = self.fold_dir / "z_report.npy"
        if z_path.is_file():
            self.z_report = np.load(z_path).astype(np.float32)
            self.notes.append(
                f"Real holdout codes for Level 1: {self.z_report.shape[0]} x {self.z_report.shape[1]}"
            )

        real_path = self.fold_dir / "decoded" / "real_figure.npy"
        if real_path.is_file():
            self.real_windows = np.load(real_path).astype(np.float32)

        latent_dir = self.fold_dir / "latents"
        if latent_dir.is_dir():
            for path in sorted(latent_dir.glob("*_fake.npy")):
                arm = path.stem.replace("_fake", "")
                self.fake_latents[arm] = np.load(path).astype(np.float32)

        decoded_dir = self.fold_dir / "decoded"
        if decoded_dir.is_dir():
            for path in sorted(decoded_dir.glob("*_figure.npy")):
                arm = path.stem.replace("_figure", "")
                if arm == "real":
                    continue
                self.decoded_figure[arm] = np.load(path).astype(np.float32)

    def noise_floor(self) -> dict[str, Any] | None:
        """Real against real. Cached because it bootstraps 20 times."""
        if self.z_report is None:
            return None
        if self._noise_floor is None:
            self._noise_floor = latent_noise_floor(self.z_report, n_repeats=20)
        return self._noise_floor

    def ictal_reconstruction(self) -> dict[str, Any] | None:
        """Reconstruction error over the cached ictal windows.

        The campaign gate was measured on the fold's mixed-class validation split.
        The cached windows here are all ictal, and the ictal class carries more
        amplitude, so it reconstructs worse than the split the gate was read on.
        Both numbers are reported so neither is mistaken for the other.
        """
        if self.real_windows is None:
            return None
        if self._ictal_recon is None:
            with self.lock:
                recon = reconstruct(self.ae, self.real_windows, device=self.device)
            x = self.real_windows
            per = np.mean((recon - x) ** 2, axis=(1, 2)) / np.maximum(
                1e-8, np.mean(x**2, axis=(1, 2))
            )
            self._ictal_recon = {
                "n_windows": int(len(x)),
                "class": "ictal",
                "mean_nmse": float(per.mean()),
                "median_nmse": float(np.median(per)),
                "min_nmse": float(per.min()),
                "max_nmse": float(per.max()),
            }
        return self._ictal_recon

    def _candidate_dirs(self) -> list[Path]:
        dirs: list[Path] = []
        if self._preferred_fold is not None and self._preferred_fold.is_dir():
            dirs.append(self._preferred_fold)
        if CAMPAIGN_DIR is not None and CAMPAIGN_DIR.is_dir():
            dirs.append(CAMPAIGN_DIR)
        if (ARTIFACTS_DIR / "autoencoder.pt").is_file():
            dirs.append(ARTIFACTS_DIR)
        for spec in discover_studies():
            p = Path(spec["path"])
            if spec.get("available") and p.is_dir():
                dirs.append(p)
        seen: set[str] = set()
        out: list[Path] = []
        for d in dirs:
            key = str(d.resolve())
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    def _try_load_cfg(self, fold_dir: Path) -> None:
        for candidate in (
            fold_dir / "config_resolved.json",
            fold_dir.parent.parent / "config_resolved.json",
            fold_dir.parent / "config_resolved.json",
        ):
            if candidate.is_file():
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and "latent_dim" in loaded:
                    self.cfg = loaded
                    self.notes.append(f"Loaded study config from {candidate.name}")
                    return

    def _load_autoencoder(self, fold_dir: Path) -> None:
        ae_path = fold_dir / "autoencoder.pt"
        if not ae_path.is_file():
            return
        ckpt = torch.load(ae_path, map_location="cpu", weights_only=False)
        arch = (ckpt.get("description") or (ckpt.get("extra") or {}).get("meta") or {}).get(
            "architecture"
        )
        if isinstance(arch, dict):
            self.ae = self.ae.__class__(
                n_channels=int(arch.get("n_channels", self.cfg.get("n_channels", 22))),
                sample_length=int(arch.get("sample_length", self.cfg.get("sample_length", 256))),
                latent_dim=int(arch.get("latent_dim", self.cfg["latent_dim"])),
                bottleneck=str(arch.get("bottleneck", "temporal")),
                widths=tuple(arch.get("widths", (128, 256, 256))),
                arch=str(arch.get("arch", "v2")),
            )
        self.ae.load_state_dict(ckpt["state_dict"])
        nmse = None
        extra = ckpt.get("extra") or {}
        gates = ((extra.get("meta") or {}).get("gates") or {})
        nmse = gates.get("nmse")
        note = f"Loaded frozen autoencoder ({ae_path.stat().st_size / 1e6:.1f} MB)"
        if nmse is not None:
            note += f", campaign NMSE {float(nmse):.3f}"
        self.notes.append(note)

    def _load_weights(self) -> None:
        fold_dir = next((p for p in self._candidate_dirs()), None)
        if fold_dir is None:
            self.generators["quantum"] = build_generator("quantum", self.cfg)
            self.generators["classical_equiv"] = build_generator("classical_equiv", self.cfg)
            self.notes.append("No campaign fold found. Architecture only.")
            return

        self.fold_dir = fold_dir
        self._try_load_cfg(fold_dir)
        self.ae = build_autoencoder(self.cfg, int(self.cfg.get("n_channels", 22)))
        self._load_autoencoder(fold_dir)

        transform_path = fold_dir / "latent_transform.npz"
        if transform_path.is_file():
            self.transform = load_latent_transform(transform_path)
        else:
            self.transform = _identity_transform(int(self.cfg["latent_dim"]))

        scale = float(self.transform.output_scale)
        gen_dir = fold_dir / "generators"
        arms = []
        if gen_dir.is_dir():
            arms = [p.stem for p in gen_dir.glob("*.pt")]
        if not arms:
            arms = ["quantum", "classical_equiv"]

        for arm in arms:
            if arm not in ALL_ARMS and not arm.startswith("quantum"):
                self.notes.append(f"Skipped unknown generator file {arm}.pt")
                continue
            try:
                gen = build_generator(arm, self.cfg, output_scale=scale)
            except Exception as exc:
                self.notes.append(f"Could not construct {arm}: {exc}")
                continue
            path = gen_dir / f"{arm}.pt"
            loaded = False
            if path.is_file():
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                gen.load_state_dict(ckpt["state_dict"])
                loaded = True
            self.generators[arm] = gen
            self.loaded[arm] = {
                "kind": "generator",
                "label": ARM_LABELS.get(arm, arm),
                "path": str(path) if path.is_file() else None,
                "params": gen.count_params(),
                "checkpoint": loaded,
            }

        gmm_path = fold_dir / "gmm" / "gmm.json"
        if gmm_path.is_file():
            mix = _load_gmm(gmm_path, self.cfg)
            self.mixtures["gmm"] = mix
            self.loaded["gmm"] = {
                "kind": "mixture",
                "label": ARM_LABELS["gmm"],
                "path": str(gmm_path),
                "params": mix.count_params(),
                "checkpoint": True,
            }
            self.notes.append("Loaded Gaussian mixture from campaign gmm.json")

        if any(v.get("checkpoint") for v in self.loaded.values()) and (
            CAMPAIGN_DIR is not None and fold_dir == CAMPAIGN_DIR
        ):
            self.weight_source = "campaign"
            self.notes.append(f"Live fold: {fold_dir}")
        elif any(v.get("checkpoint") for v in self.loaded.values()):
            self.weight_source = "demo-trained"

    def _build_catalog(self) -> list[dict[str, Any]]:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        real_path = None
        if self.fold_dir is not None:
            candidate = self.fold_dir / "decoded" / "real_figure.npy"
            if candidate.is_file():
                real_path = candidate
        if real_path is None:
            if CATALOG_PATH.is_file():
                data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
                self.data_source = "synthetic"
                self.notes.append("No TUH holdout cache. Samples are synthetic stand-ins.")
                return list(data.get("samples", []))
            return []

        windows = np.load(real_path).astype(np.float32)
        samples: list[dict[str, Any]] = []
        n = min(12, len(windows))
        for i in range(n):
            sample_id = f"tuh_ictal_{i + 1:02d}"
            np.save(SAMPLES_DIR / f"{sample_id}.npy", windows[i])
            samples.append(
                {
                    "id": sample_id,
                    "label": "seizure",
                    "title": f"Holdout ictal {i + 1:02d}",
                    "file": f"{sample_id}.npy",
                    "provenance": "tuh-sz-holdout",
                    "source_file": "TUH-SZ official holdout (decoded/real_figure.npy)",
                    "note": (
                        "Real TUH EEG Seizure Corpus window from the campaign holdout, "
                        "22 bipolar channels, 1.28 s at 200 Hz, already preprocessed."
                    ),
                }
            )
        self.data_source = "tuh-sz-holdout"
        self.notes.append(
            f"Loaded {n} real TUH-SZ holdout ictal windows from {real_path.name} "
            f"({len(windows)} cached in the fold)."
        )
        CATALOG_PATH.write_text(
            json.dumps({"window": {"n_channels": 22, "sample_length": 256, "fs_hz": 200}, "samples": samples}, indent=2),
            encoding="utf-8",
        )
        return samples

    def meta(self) -> dict[str, Any]:
        models = []
        for arm, info in self.loaded.items():
            extra = dict(info)
            extra["arm"] = arm
            if arm in self.generators:
                extra["describe"] = _jsonable(self.generators[arm].describe())
            elif arm in self.mixtures:
                extra["describe"] = _jsonable(self.mixtures[arm].describe())
            models.append(extra)
        ae = self.ae.describe()
        return {
            "title": "TUH-SZ seizure EEG research workstation",
            "school": "ESI 8 Mai 1945, Sidi Bel Abbes",
            "weight_source": self.weight_source,
            "data_source": self.data_source,
            "fold": str(self.fold_dir) if self.fold_dir else None,
            "device": self.device,
            "notes": self.notes,
            "window": {
                "n_channels": int(self.cfg.get("n_channels", 22)),
                "sample_length": int(self.cfg.get("sample_length", 256)),
                "fs_hz": int(self.cfg.get("target_fs", 200)),
                "duration_s": 1.28,
                "montage": "TCP bipolar 10-20",
            },
            "latent_dim": int(self.cfg["latent_dim"]),
            "models": models,
            "autoencoder": _jsonable(ae),
            "n_samples": len(self.catalog),
            "disclaimer": (
                "Research prototype. Not a medical device. Not for clinical diagnosis."
            ),
        }

    def architecture(self) -> dict[str, Any]:
        n_qubits = int(self.cfg.get("n_qubits", 8))
        entanglement = str((self.cfg.get("quantum") or {}).get("entanglement", "ring"))
        n_layers = int((self.cfg.get("quantum") or {}).get("n_layers", 4))
        q = self.generators.get("quantum")
        breakdown = q.parameter_breakdown() if q is not None else {}
        return {
            "corpus": "Temple University Hospital EEG Seizure Corpus (TUH-SZ)",
            "protocol": [
                "Official train / dev / eval folders, patient-disjoint by construction",
                "22-channel TCP bipolar montage, 200 Hz, 1.28 s generative windows",
                "Daubechies-6 denoise, recording-scope scaling to [-1, 1]",
                "Frozen autoencoder host; only the generator body changes",
                "Synthetic data never enters the reported holdout",
            ],
            "hybrid": {
                "noise": 64,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "entanglement": entanglement,
                "pairs": entangler_pairs(n_qubits, entanglement),
                "per_layer": "data-dependent RY, then RY and RZ, then ring CZ",
                "readout": "analytic Pauli-Z expectations",
                "expand": "Linear 8 to 64 + tanh",
                "params": _jsonable(breakdown),
            },
            "models": self.meta()["models"],
        }

    def list_samples(self) -> list[dict[str, Any]]:
        return [
            {
                "id": item["id"],
                "label": item["label"],
                "title": item.get("title", item["id"]),
                "provenance": item.get("provenance", "synthetic"),
                "source_file": item.get("source_file"),
                "note": item.get("note", ""),
            }
            for item in self.catalog
        ]

    def load_sample(self, sample_id: str) -> tuple[dict[str, Any], np.ndarray]:
        item = next((s for s in self.catalog if s["id"] == sample_id), None)
        if item is None:
            raise KeyError(sample_id)
        path = SAMPLES_DIR / item["file"]
        window = np.load(path).astype(np.float32)
        if window.ndim == 3:
            window = window[0]
        return item, window

    def _sample_index(self, sample_id: str) -> int:
        for i, item in enumerate(self.catalog):
            if item["id"] == sample_id:
                return i
        raise KeyError(sample_id)

    @staticmethod
    def _call_from_score(score: dict[str, Any] | None) -> dict[str, Any]:
        """Map one detector score to a seizure / no-seizure call for the UI."""
        if not score:
            return {
                "call": "unavailable",
                "label": "Unavailable",
                "probability": None,
                "threshold": None,
                "fired": None,
                "note": "Detector not loaded.",
            }
        probs = score.get("probabilities") or []
        prob = float(probs[0]) if probs else score.get("mean_probability")
        thr = float(score.get("threshold", 0.5))
        fired = bool(score.get("fired", [False])[0]) if score.get("fired") else (
            prob is not None and prob >= thr
        )
        if prob is None:
            call, label = "unavailable", "Unavailable"
        elif fired:
            call, label = "seizure", "SEIZURE"
        else:
            call, label = "not_seizure", "NOT SEIZURE"
        return {
            "call": call,
            "label": label,
            "probability": prob,
            "threshold": thr,
            "fired": fired,
            "detector": score.get("detector"),
            "detector_label": score.get("label"),
            "note": (
                f"Operating point from the saved checkpoint (threshold {thr:.3f}). "
                "This is a research detector call, not a clinical diagnosis."
            ),
        }

    def detect_holdout(self, sample_id: str) -> dict[str, Any]:
        """Score a contiguous 10.24 s clip around the selected holdout window.

        Detectors expect eight packed windows, not a single 1.28 s window. The
        selected catalog index is the start (clamped so the clip fits).
        """
        item, window = self.load_sample(sample_id)
        if self.real_windows is None or not self.detectors.detectors:
            return {
                "sample": item,
                "window": window_payload(window),
                "truth": {
                    "label": item.get("label", "seizure"),
                    "class": "ictal",
                    "source": "TUH-SZ holdout catalog (all listed windows are ictal)",
                },
                "clip": None,
                "baseline": self._call_from_score(None),
                "error": "Real windows or detectors unavailable.",
            }

        idx = self._sample_index(sample_id)
        max_start = max(0, int(len(self.real_windows) - CLIP_WINDOWS))
        start = min(idx, max_start)
        clip = pack_real_clip(self.real_windows, start=start, n_windows=CLIP_WINDOWS)
        scores = self.detectors.score_all(clip[None], device=self.device)
        baseline = scores.get("baseline")
        return {
            "sample": item,
            "window": window_payload(window),
            "truth": {
                "label": item.get("label", "seizure"),
                "class": "ictal",
                "source": "TUH-SZ holdout catalog (all listed windows are ictal)",
            },
            "clip": {
                "start_window": int(start),
                "n_windows": CLIP_WINDOWS,
                "seconds": clip_seconds(CLIP_WINDOWS),
                "payload": window_payload(clip),
            },
            "baseline": self._call_from_score(baseline),
            "all_detectors": {
                name: self._call_from_score(sc) for name, sc in scores.items()
            },
            "raw_scores": scores,
        }

    def generate(self, arm: str, seed: int | None = None, n: int = 1) -> dict[str, Any]:
        n = max(1, min(int(n), 4))
        t0 = time.perf_counter()
        with self.lock:
            if seed is not None:
                torch.manual_seed(int(seed))
                np.random.seed(int(seed))
            if arm in self.mixtures:
                rng = np.random.default_rng(seed if seed is not None else None)
                codes = self.mixtures[arm].sample(n, transform=self.transform, rng=rng)
                describe = self.mixtures[arm].describe()
            elif arm in self.generators:
                codes = sample_latent_codes(
                    self.generators[arm],
                    n=n,
                    device=self.device,
                    transform=self.transform,
                    batch_size=n,
                )
                describe = self.generators[arm].describe()
            else:
                raise KeyError(arm)
            windows = decode_codes(self.ae, codes, device=self.device, batch_size=n)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "arm": arm,
            "label": ARM_LABELS.get(arm, arm),
            "seed": seed,
            "latency_ms": round(elapsed, 1),
            "weight_source": self.weight_source,
            "latent_codes": codes.tolist(),
            "windows": [window_payload(w) for w in windows],
            "describe": _jsonable(describe),
        }

    def reconstruct_sample(self, sample_id: str) -> dict[str, Any]:
        item, window = self.load_sample(sample_id)
        t0 = time.perf_counter()
        with self.lock:
            recon = reconstruct(self.ae, window[None], device=self.device)[0]
            code = encode_windows(self.ae, window[None], device=self.device)[0]
        elapsed = (time.perf_counter() - t0) * 1000.0
        err = float(np.mean((recon - window) ** 2) / max(1e-8, np.mean(window**2)))
        return {
            "sample": item,
            "latency_ms": round(elapsed, 1),
            "nmse": err,
            "latent_code": code.tolist(),
            "real": window_payload(window),
            "reconstruction": window_payload(recon),
        }

    def compare(self, sample_id: str, seed: int = 42) -> dict[str, Any]:
        item, window = self.load_sample(sample_id)
        recon = self.reconstruct_sample(sample_id)
        generated = {}
        for arm in list(self.generators) + list(self.mixtures):
            generated[arm] = self.generate(arm, seed=seed, n=1)
        return {
            "sample": item,
            "real": window_payload(window),
            "reconstruction": recon["reconstruction"],
            "reconstruction_nmse": recon["nmse"],
            "generated": generated,
            "weight_source": self.weight_source,
            "data_source": self.data_source,
        }


    # ---- Stage 4 and the three evidence levels -------------------------------

    def pipeline_overview(self) -> dict[str, Any]:
        """What the pipeline is, what is live, and what is quoted."""
        n_real_clips = 0
        if self.real_windows is not None:
            n_real_clips = int(len(self.real_windows) // CLIP_WINDOWS)
        return {
            "stages": STAGES,
            "levels": LEVELS,
            "live_arms": self.arm_status()["live_arms"],
            "added_value": ADDED_VALUE,
            "active_study": self.study_id,
            "clip": {
                "windows": CLIP_WINDOWS,
                "samples": CLIP_WINDOWS * int(self.cfg.get("sample_length", 256)),
                "seconds": clip_seconds(CLIP_WINDOWS),
                "synthetic_crossfade": SYNTHETIC_CROSSFADE,
                "windows_per_synthetic_clip": windows_needed_for_synthetic_clip(),
                "label_rule": "a clip is positive if any of its windows is ictal",
            },
            "detectors": self.detectors.describe(),
            "campaign": self.campaign.context(),
            "thesis": THESIS_HEADLINE,
            "noise_floor": self.noise_floor(),
            "ictal_reconstruction": self.ictal_reconstruction(),
            "generators": [
                {"arm": arm, "label": ARM_LABELS.get(arm, arm)}
                for arm in list(self.generators) + list(self.mixtures)
            ],
            "available": {
                "live_generation": bool(self.generators or self.mixtures),
                "live_level1": self.z_report is not None,
                "live_level2": self.real_windows is not None,
                "live_detection": bool(self.detectors.detectors),
                "recorded_level3": self.campaign.available,
                "n_real_holdout_windows": int(len(self.real_windows))
                if self.real_windows is not None
                else 0,
                "n_real_holdout_clips": n_real_clips,
            },
            "honesty": {
                "live": (
                    "Reconstruction, generation, latent MMD^2, decoded distances, and "
                    "detector probabilities are computed on this machine now."
                ),
                "quoted": (
                    "AUROC, AUPRC and MCC over the whole holdout are read from the "
                    "campaign record, because that needs the corpus, which is on the cluster."
                ),
                "real_windows": (
                    "The cached holdout windows are all ictal, so a live AUROC cannot be "
                    "formed from them. Live detection reports probability against each "
                    "detector's own operating point."
                ),
            },
        }

    def real_clips(self, n_clips: int | None = None) -> np.ndarray:
        """Contiguous real holdout ictal clips from the cached windows."""
        if self.real_windows is None:
            raise RuntimeError("No real holdout windows cached in this fold")
        total = int(len(self.real_windows) // CLIP_WINDOWS)
        take = total if n_clips is None else max(1, min(int(n_clips), total))
        return np.stack(
            [pack_real_clip(self.real_windows, start=i * CLIP_WINDOWS) for i in range(take)]
        )

    def run_pipeline(
        self,
        arm: str,
        sample_id: str | None = None,
        seed: int = 42,
        n_codes: int = 256,
    ) -> dict[str, Any]:
        """One arm, end to end, with a result at every stage."""
        if arm not in self.generators and arm not in self.mixtures:
            raise KeyError(arm)

        out: dict[str, Any] = {
            "arm": arm,
            "label": ARM_LABELS.get(arm, arm),
            "seed": seed,
            "weight_source": self.weight_source,
            "data_source": self.data_source,
            "fold": str(self.fold_dir) if self.fold_dir else None,
        }

        # ---- Stage 1. Data -------------------------------------------------
        if sample_id is None and self.catalog:
            sample_id = self.catalog[0]["id"]
        item, window = self.load_sample(sample_id) if sample_id else (None, None)
        real_clip = self.real_clips() if self.real_windows is not None else None
        out["stage1_data"] = {
            "stage": 1,
            "title": "Data and preprocessing",
            "mode": "live",
            "sample": item,
            "window": window_payload(window) if window is not None else None,
            "n_channels": int(self.cfg.get("n_channels", 22)),
            "sample_length": int(self.cfg.get("sample_length", 256)),
            "fs_hz": int(self.cfg.get("target_fs", 200)),
            "window_seconds": 1.28,
            "n_real_clips": int(len(real_clip)) if real_clip is not None else 0,
            "clip_seconds": clip_seconds(CLIP_WINDOWS),
            "note": (
                "Official TUH-SZ holdout windows, already resampled, denoised with "
                "Daubechies-6, and scaled on a recording-wide range."
            ),
        }

        # ---- Stage 2. Representation --------------------------------------
        if window is not None:
            recon = self.reconstruct_sample(sample_id)
            out["stage2_representation"] = {
                "stage": 2,
                "title": "Representation",
                "mode": "live",
                "latent_dim": int(self.cfg["latent_dim"]),
                "nmse": recon["nmse"],
                "window_class": (item or {}).get("label"),
                "ictal_subset": self.ictal_reconstruction(),
                "gate": {
                    "threshold": 0.55,
                    "measured_on": "fold validation split, mixed classes",
                    "campaign_nmse": (self.campaign.context() or {}).get("ae_nmse"),
                    "passed": (self.campaign.context() or {}).get("ae_gate_passed"),
                },
                "latency_ms": recon["latency_ms"],
                "real": recon["real"],
                "reconstruction": recon["reconstruction"],
                "note": (
                    "The host is frozen, so this error is a floor shared by every arm. "
                    "The gate is a fold-level criterion on the mixed validation split, "
                    "not a per-window test. Ictal windows carry more amplitude and "
                    "reconstruct worse than that average, which is itself part of why a "
                    "latent win need not survive into detection."
                ),
            }

        # ---- Stage 3. Generation ------------------------------------------
        need = windows_needed_for_synthetic_clip()
        n_gen = max(int(need), int(n_codes))
        t0 = time.perf_counter()
        with self.lock:
            if seed is not None:
                torch.manual_seed(int(seed))
                np.random.seed(int(seed))
            if arm in self.mixtures:
                rng = np.random.default_rng(seed)
                codes = self.mixtures[arm].sample(n_gen, transform=self.transform, rng=rng)
                describe = self.mixtures[arm].describe()
            else:
                codes = sample_latent_codes(
                    self.generators[arm],
                    n=n_gen,
                    device=self.device,
                    transform=self.transform,
                    batch_size=min(128, n_gen),
                )
                describe = self.generators[arm].describe()
            gen_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            decoded = decode_codes(self.ae, codes, device=self.device, batch_size=64)
            dec_ms = (time.perf_counter() - t1) * 1000.0

        out["stage3_generation"] = {
            "stage": 3,
            "title": "Generation",
            "mode": "live",
            "n_codes": int(len(codes)),
            "generator_latency_ms": round(gen_ms, 1),
            "decoder_latency_ms": round(dec_ms, 1),
            "ms_per_window": round(gen_ms / max(1, len(codes)), 2),
            "describe": _jsonable(describe),
            "windows": [window_payload(w) for w in decoded[:2]],
            "note": (
                "Noise to code through the generator body, then the frozen decoder. "
                "The hybrid arm runs a PennyLane circuit on CPU, so it is the slowest."
            ),
        }

        # ---- Level 1. Latent fidelity -------------------------------------
        if self.z_report is not None:
            live = level1_latent(self.z_report, codes, rng=np.random.default_rng(seed))
            recorded = (self.campaign.generative().get(arm) or {}).get("mmd2")
            out["level1_latent"] = {
                "level": 1,
                "title": "Latent fidelity",
                "mode": "live",
                "mmd2": live["mmd2"],
                "n_reference": live["n_reference"],
                "n_candidate": live["n_candidate"],
                "estimator": live["estimator"],
                "bandwidth_base": live["bandwidth_base"],
                "noise_floor": self.noise_floor(),
                "recorded_mmd2": recorded,
                "note": (
                    "MMD^2 against the real holdout codes of this fold. The live value "
                    "uses the codes generated a moment ago, so it moves with n and the seed; "
                    "the recorded value used 16384 codes."
                ),
            }

        # ---- Level 2. Decoded signal --------------------------------------
        if self.real_windows is not None:
            n_cmp = min(len(self.real_windows), len(decoded))
            live2 = level2_decoded(self.real_windows[:n_cmp], decoded[:n_cmp])
            recorded2 = (self.campaign.generative().get(arm) or {}).get("decoded") or {}
            out["level2_decoded"] = {
                "level": 2,
                "title": "Decoded signal",
                "mode": "live",
                "n_windows": int(n_cmp),
                "live": {k: v for k, v in live2.items() if k != "detail"},
                "recorded": recorded2,
                "note": (
                    "Distances between decoded and real windows. Lower is closer. "
                    "The hybrid tends to win the distributional distances and lose "
                    "autocorrelation and spectral band error."
                ),
            }

        # ---- Stage 4 and Level 3. Detection --------------------------------
        detection: dict[str, Any] = {
            "stage": 4,
            "level": 3,
            "title": "Detection",
            "mode": "live probabilities, recorded holdout metrics",
            "clip": {
                "windows": CLIP_WINDOWS,
                "seconds": clip_seconds(CLIP_WINDOWS),
                "crossfade": SYNTHETIC_CROSSFADE,
            },
        }
        if self.detectors.detectors and real_clip is not None:
            detection["real_clips"] = {
                "n_clips": int(len(real_clip)),
                "class": "ictal (all positive)",
                "scores": self.detectors.score_all(real_clip, device=self.device),
                "note": (
                    "Real holdout ictal clips. A detector should fire on these. "
                    "Because every cached clip is positive, this is a sensitivity "
                    "reading, not an AUROC."
                ),
            }
            try:
                syn_clip = pack_one_synthetic_clip(decoded)
                detection["synthetic_clip"] = {
                    "arm": arm,
                    "n_clips": 1,
                    "scores": self.detectors.score_all(syn_clip[None], device=self.device),
                    "clip": window_payload(syn_clip),
                    "note": (
                        f"One {clip_seconds(CLIP_WINDOWS):.2f} s clip packed from "
                        f"{windows_needed_for_synthetic_clip()} generated windows with a "
                        f"{SYNTHETIC_CROSSFADE}-sample raised-cosine seam."
                    ),
                }
            except Exception as exc:
                detection["synthetic_clip"] = {"error": str(exc)}

        if self.campaign.available:
            rec = self.campaign.detection()
            det_name = f"augmented_{arm}"
            detection["recorded"] = {
                "baseline": rec.get("baseline"),
                "this_arm": rec.get(det_name),
                "holdout": self.campaign.fold_name,
                "note": (
                    "Recorded on the whole real holdout during the campaign. "
                    "Synthetic clips entered detector training only."
                ),
            }
        out["stage4_detection"] = detection
        return out

    def _generate_batch(
        self, arm: str, n: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        """Generate ``n`` codes and decode them. Returns codes, windows, gen_ms, describe."""
        t0 = time.perf_counter()
        with self.lock:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            if arm in self.mixtures:
                rng = np.random.default_rng(seed)
                codes = self.mixtures[arm].sample(n, transform=self.transform, rng=rng)
                describe = self.mixtures[arm].describe()
            else:
                codes = sample_latent_codes(
                    self.generators[arm],
                    n=n,
                    device=self.device,
                    transform=self.transform,
                    batch_size=min(128, n),
                )
                describe = self.generators[arm].describe()
            gen_ms = (time.perf_counter() - t0) * 1000.0
            decoded = decode_codes(self.ae, codes, device=self.device, batch_size=64)
        return codes, decoded, gen_ms, describe

    def live_compare(
        self,
        sample_id: str | None = None,
        seed: int = 42,
        n_codes: int = 256,
    ) -> dict[str, Any]:
        """Run every loaded generator live, then score generation and detection.

        Everything in the returned rows is computed now from the saved checkpoints.
        AUROC over the full holdout is omitted here on purpose: that needs the corpus.
        Detection impact is reported as live probabilities on real ictal clips and on
        a freshly packed synthetic clip from each generator.
        """
        arms = [a for a in ("quantum", "classical_equiv", "gmm") if a in self.generators or a in self.mixtures]
        if not arms:
            raise RuntimeError("No generators loaded")

        if sample_id is None and self.catalog:
            sample_id = self.catalog[0]["id"]
        item, window = self.load_sample(sample_id) if sample_id else (None, None)
        recon = self.reconstruct_sample(sample_id) if sample_id else None
        real_clip = self.real_clips() if self.real_windows is not None else None

        real_scores = None
        if real_clip is not None and self.detectors.detectors:
            real_scores = self.detectors.score_all(real_clip, device=self.device)

        need = windows_needed_for_synthetic_clip()
        n_gen = max(int(need), int(n_codes))
        rows: list[dict[str, Any]] = []
        waveforms: dict[str, Any] = {}
        if window is not None:
            waveforms["real"] = window_payload(window)
        if recon is not None:
            waveforms["reconstruction"] = recon["reconstruction"]
            waveforms["reconstruction_nmse"] = recon["nmse"]

        for i, arm in enumerate(arms):
            codes, decoded, gen_ms, describe = self._generate_batch(arm, n_gen, seed + i)
            row: dict[str, Any] = {
                "arm": arm,
                "label": ARM_LABELS.get(arm, arm),
                "mode": "live",
                "n_params": int((self.loaded.get(arm) or {}).get("params") or 0),
                "generation_ms": round(gen_ms, 1),
                "n_codes": int(len(codes)),
                "window": window_payload(decoded[0]),
            }

            if self.z_report is not None:
                live1 = level1_latent(self.z_report, codes, rng=np.random.default_rng(seed + i))
                row["generation"] = {
                    "mmd2": live1["mmd2"],
                    "n_reference": live1["n_reference"],
                    "n_candidate": live1["n_candidate"],
                }

            if self.real_windows is not None:
                n_cmp = min(len(self.real_windows), len(decoded))
                live2 = level2_decoded(self.real_windows[:n_cmp], decoded[:n_cmp])
                row["decoded"] = {k: v for k, v in live2.items() if k != "detail"}
                row["decoded"]["n_windows"] = int(n_cmp)

            if self.detectors.detectors:
                det_block: dict[str, Any] = {}
                if real_scores is not None:
                    own = real_scores.get(f"augmented_{arm}") or real_scores.get("baseline")
                    base = real_scores.get("baseline")
                    det_block["real_ictal"] = {
                        "own": own,
                        "baseline": base,
                        "n_clips": int(len(real_clip)),
                    }
                try:
                    syn_clip = pack_one_synthetic_clip(decoded)
                    syn_scores = self.detectors.score_all(syn_clip[None], device=self.device)
                    base_syn = syn_scores.get("baseline")
                    own_syn = syn_scores.get(f"augmented_{arm}")
                    det_block["synthetic_clip"] = {
                        "own": own_syn,
                        "baseline": base_syn,
                        "all": syn_scores,
                        "clip": window_payload(syn_clip),
                        "baseline_call": self._call_from_score(base_syn),
                        "own_call": self._call_from_score(own_syn),
                    }
                except Exception as exc:
                    det_block["synthetic_clip"] = {"error": str(exc)}
                row["detection"] = det_block

            rows.append(row)
            waveforms[arm] = window_payload(decoded[0])

        selected = self.detect_holdout(sample_id) if sample_id else None

        # Rank by live generation closeness (lower MMD better).
        mmds = [(r["arm"], r.get("generation", {}).get("mmd2")) for r in rows]
        mmds = [(a, m) for a, m in mmds if m is not None]
        best_gen = min(mmds, key=lambda x: x[1])[0] if mmds else None

        return {
            "mode": "live",
            "seed": seed,
            "n_codes": n_gen,
            "sample": item,
            "selected_detection": selected,
            "weight_source": self.weight_source,
            "data_source": self.data_source,
            "noise_floor": self.noise_floor(),
            "real_detection": {
                "n_clips": int(len(real_clip)) if real_clip is not None else 0,
                "class": "ictal",
                "scores": real_scores,
                "note": (
                    "Live probabilities on real holdout ictal clips. "
                    "All clips are positive, so this is a sensitivity reading."
                ),
            },
            "rows": rows,
            "waveforms": waveforms,
            "best_generation_arm": best_gen,
            "reading": {
                "generation": "Lower MMD² and lower decoded distances mean closer to real seizure EEG.",
                "detection": (
                    "Selected holdout uses the real-only detector call (SEIZURE / NOT SEIZURE) "
                    "at the saved threshold. Synthetic rows use the same detector on a packed clip."
                ),
                "impact": (
                    "Compare generation closeness with detection behaviour. A model can win "
                    "latent MMD and still not look like a seizure to the real-only detector."
                ),
            },
        }

    def comparison(self) -> dict[str, Any]:
        """Fast side-by-side using cached campaign codes, then live detector scores.

        Prefer :meth:`live_compare` for the supervisor demo: that regenerates from the
        saved weights. This method is kept for the quick table and for verify.py.
        """
        rows: list[dict[str, Any]] = []
        recorded_gen = self.campaign.generative()
        recorded_det = self.campaign.detection()
        baseline = recorded_det.get("baseline") or {}

        real_clip = self.real_clips() if self.real_windows is not None else None
        real_scores = None
        if real_clip is not None and self.detectors.detectors:
            real_scores = self.detectors.score_all(real_clip, device=self.device)

        arms = [a for a in ("quantum", "classical_equiv", "gmm") if a in self.fake_latents or a in self.generators or a in self.mixtures]
        for arm in arms:
            row: dict[str, Any] = {
                "arm": arm,
                "label": ARM_LABELS.get(arm, arm),
                "n_params": (recorded_gen.get(arm) or {}).get("n_params"),
                "converged": (recorded_gen.get(arm) or {}).get("converged"),
            }

            codes = self.fake_latents.get(arm)
            if codes is not None and self.z_report is not None:
                live1 = level1_latent(self.z_report, codes, rng=np.random.default_rng(0))
                row["level1"] = {
                    "mmd2": live1["mmd2"],
                    "n_candidate": int(len(codes)),
                    "recorded_mmd2": (recorded_gen.get(arm) or {}).get("mmd2"),
                    "mode": "live",
                }

            fake = self.decoded_figure.get(arm)
            if fake is not None and self.real_windows is not None:
                n_cmp = min(len(self.real_windows), len(fake))
                live2 = level2_decoded(self.real_windows[:n_cmp], fake[:n_cmp])
                row["level2"] = {
                    **{k: v for k, v in live2.items() if k != "detail"},
                    "n_windows": int(n_cmp),
                    "mode": "live",
                }

            if real_scores is not None:
                own = real_scores.get(f"augmented_{arm}")
                base = real_scores.get("baseline")
                row["detection_live"] = {
                    "mode": "live",
                    "own_mean_p": (own or {}).get("mean_probability"),
                    "own_fired": (own or {}).get("n_fired"),
                    "baseline_mean_p": (base or {}).get("mean_probability"),
                    "baseline_fired": (base or {}).get("n_fired"),
                    "n_clips": (base or {}).get("n_clips"),
                }

            det = recorded_det.get(f"augmented_{arm}")
            if det is not None:
                row["level3"] = {**det, "mode": "recorded"}
            rows.append(row)

        return {
            "baseline": {**baseline, "mode": "recorded"} if baseline else None,
            "rows": rows,
            "noise_floor": self.noise_floor(),
            "campaign": self.campaign.context(),
            "thesis": THESIS_HEADLINE,
            "scope": {
                "served": (
                    f"one fold: seed {self.campaign.context().get('seed')}, "
                    f"holdout '{self.campaign.fold_name}'"
                ),
                "manuscript": THESIS_HEADLINE["scope"],
                "warning": THESIS_HEADLINE["warning"],
            },
            "reading": {
                "level1": "lower MMD^2 is closer to the real holdout codes",
                "level2": "lower is closer for every decoded distance",
                "level3": "higher AUROC is better; read delta against the real-only baseline",
                "headline": (
                    "The levels are allowed to disagree, and that disagreement is the "
                    "finding. Level 1 ranks the hybrid first among the adversarial arms; "
                    "Level 2 shows it buying distributional closeness at the cost of "
                    "autocorrelation; and pooled over four records Level 3 gives it no "
                    "detection benefit at full dose."
                ),
            },
        }


_ENGINE: DemoEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> DemoEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = DemoEngine()
        return _ENGINE


def reset_engine(fold_dir: Path | None = None) -> DemoEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        _ENGINE = DemoEngine(fold_dir=fold_dir)
        return _ENGINE


def select_study(study_id: str) -> DemoEngine:
    """Reload the demo on a synced study fold (own AE + generators together)."""
    match = next((s for s in discover_studies() if s["id"] == study_id), None)
    if match is None:
        raise KeyError(study_id)
    fold = Path(match["path"])
    if not (fold / "autoencoder.pt").is_file():
        raise FileNotFoundError(
            f"Study '{study_id}' has no autoencoder.pt yet. "
            "Run demo/sync_from_narval.ps1 -Optional first."
        )
    return reset_engine(fold_dir=fold)
