"""Full-pipeline inference: four stages, three levels of evidence.

The demo runs the same code path as the campaign. Clip packing is the campaign's
own :func:`pack_synthetic_clips`, the metrics are the campaign's own estimators,
and the detectors are the campaign's own checkpoints.

What is computed live, and what is quoted, is kept strictly apart:

Live on this machine
    Stage 2 reconstruction NMSE, Stage 3 generation, Level 1 latent MMD^2 against
    the cached real holdout codes, Level 2 decoded distances, and Level 3 detector
    probabilities on cached real holdout ictal clips and on freshly generated clips.

Quoted from the campaign record
    Level 3 AUROC, AUPRC and MCC over the whole holdout. Those need the corpus,
    which lives on the cluster, so they are read from ``results_all.json`` and
    always labelled as a recorded result rather than a live one.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg_qgan.data.clips import pack_synthetic_clips
from eeg_qgan.evaluation.decoded import evaluate_decoded
from eeg_qgan.evaluation.mmd import mmd_noise_floor, mmd_report
from eeg_qgan.models.detector import build_detector

# The campaign's clinical settings for this fold. Kept as named constants so the
# website can show them next to the numbers they produced.
CLIP_WINDOWS = 8
SYNTHETIC_CROSSFADE = 32
WINDOW_SAMPLES = 256
TARGET_FS = 200

DETECTOR_LABELS = {
    "baseline": "Real only (baseline)",
    "augmented_quantum": "Real + hybrid Style-qGAN",
    "augmented_classical_equiv": "Real + capacity-matched twin",
    "augmented_gmm": "Real + Gaussian mixture",
}

# Which generator arm feeds which augmented detector.
DETECTOR_ARM = {
    "augmented_quantum": "quantum",
    "augmented_classical_equiv": "classical_equiv",
    "augmented_gmm": "gmm",
}

STAGES = [
    {
        "id": "data",
        "n": 1,
        "title": "Data and preprocessing",
        "detail": "Official TUH-SZ holdout, 22 bipolar channels, 200 Hz, Daubechies-6, recording-scope scaling",
    },
    {
        "id": "representation",
        "n": 2,
        "title": "Representation",
        "detail": "Frozen autoencoder, 22 x 256 window to a 64-d code and back",
    },
    {
        "id": "generation",
        "n": 3,
        "title": "Generation",
        "detail": "Noise to code through the generator body, then the frozen decoder",
    },
    {
        "id": "detection",
        "n": 4,
        "title": "Detection",
        "detail": "Eight windows packed to a 10.24 s clip, then the frozen STFT-BiLSTM",
    },
]

# The manuscript headline, pooled over two seeds by two official holdouts with
# cluster-bootstrap intervals. The demo serves ONE fold, so its detection numbers
# are not the headline and must never be quoted as if they were. This constant
# exists so the website can show the two side by side and say which is which.
THESIS_HEADLINE = {
    "scope": "2 seeds x 2 official holdouts = 4 seed-by-fold records, cluster-bootstrap intervals",
    "baseline_auroc": 0.821,
    "arms": {
        "quantum": {
            "delta_auroc": 0.0004,
            "interval_excludes_zero": False,
            "reading": "no detection benefit at full dose; the interval contains zero",
        },
        "classical_equiv": {
            "delta_auroc": 0.005,
            "interval_excludes_zero": True,
            "reading": "small but interval-supported lift",
        },
        "gmm": {
            "delta_auroc": 0.010,
            "interval_excludes_zero": True,
            "reading": "largest interval-supported lift, and it uses no critic",
        },
    },
    "latent_mmd2": {"quantum": 0.016, "classical_equiv": 0.088, "gmm": 0.0008},
    "warning": (
        "One fold is not the campaign. Fold-level detection deltas swing far wider "
        "than the pooled mean, which is exactly why the manuscript reports intervals "
        "over four records instead of a single run."
    ),
}

LEVELS = [
    {
        "id": "level1",
        "n": 1,
        "title": "Latent fidelity",
        "metric": "MMD^2 between generated and real holdout codes",
    },
    {
        "id": "level2",
        "n": 2,
        "title": "Decoded signal",
        "metric": "Sliced W1, autocorrelation, spectral band error, KS",
    },
    {
        "id": "level3",
        "n": 3,
        "title": "Clinical detection",
        "metric": "AUROC, AUPRC, MCC on real holdout clips",
    },
]

# What the live deployment page can run versus what the manuscript discusses.
# Extra classical / mechanism arms are real thesis controls, but their checkpoints
# are not in this fold folder, so the website must not pretend they are live.
LIVE_ARMS = (
    {
        "arm": "quantum",
        "label": "Hybrid Style-qGAN",
        "role": "rematch",
        "live": True,
        "params": 704,
        "why": "Model under test under matched capacity.",
    },
    {
        "arm": "classical_equiv",
        "label": "Capacity-matched twin",
        "role": "rematch_peer",
        "live": True,
        "params": 709,
        "why": "Only fair classical twin for the generative rematch.",
    },
    {
        "arm": "gmm",
        "label": "Gaussian mixture",
        "role": "density_reference",
        "live": True,
        "params": 17159,
        "why": "Non-adversarial density fit in the same latent space.",
    },
)

# Mechanism / ablation arms (Fourier, entanglement, large MLP, …) are thesis
# generation controls. They are not part of this clinical inference demo.
MANUSCRIPT_ONLY_ARMS: tuple[dict, ...] = ()

ADDED_VALUE = {
    "yes": [
        "Under matched capacity (~704 vs 709), hybrid is closer than the twin on latent MMD².",
        "Same frozen AE host, same critic family, same TUH-SZ holdout protocol as the campaign.",
        "Live generation and detector calls from saved clinical weights.",
    ],
    "no": [
        "No quantum advantage claim.",
        "No medical-device / hospital-ready claim.",
        "Hybrid ΔAUROC at full dose is not distinguishable from zero in the pooled study.",
        "GMM is denser and not equal-budget; it is not a rematch peer.",
    ],
}


def clip_seconds(n_windows: int = CLIP_WINDOWS) -> float:
    return float(n_windows * WINDOW_SAMPLES) / float(TARGET_FS)


def pack_real_clip(windows: np.ndarray, start: int = 0, n_windows: int = CLIP_WINDOWS) -> np.ndarray:
    """A real clip is a contiguous stretch of one recording, concatenated in time.

    This mirrors ``build_clips_from_indices``, which concatenates a contiguous
    block of windows along the last axis. No crossfade is applied, because a real
    clip has no seams to hide.
    """
    arr = np.asarray(windows, dtype=np.float32)
    if len(arr) < start + n_windows:
        raise ValueError(f"Need {n_windows} windows from index {start}, have {len(arr)}")
    block = arr[start : start + n_windows]
    return np.concatenate(list(block), axis=-1).astype(np.float32)


def windows_needed_for_synthetic_clip(
    n_windows: int = CLIP_WINDOWS, crossfade: int = SYNTHETIC_CROSSFADE
) -> int:
    """How many generated windows one crossfaded clip consumes.

    With a raised-cosine overlap the seams eat samples, so a 2048-sample clip
    needs more than eight windows. The campaign accounts for this in
    ``pack_synthetic_clips``; the demo asks it the same question up front so the
    generator is called exactly the right number of times.
    """
    t = WINDOW_SAMPLES
    target = n_windows * t
    fade = max(0, min(int(crossfade), t // 2))
    if fade == 0:
        return int(n_windows)
    return int(np.ceil((target - fade) / (t - fade)))


def pack_one_synthetic_clip(
    windows: np.ndarray,
    n_windows: int = CLIP_WINDOWS,
    crossfade: int = SYNTHETIC_CROSSFADE,
) -> np.ndarray:
    """One synthetic clip through the campaign's own packing function."""
    clips = pack_synthetic_clips(windows, n_windows, crossfade=crossfade)
    if clips is None or len(clips) == 0:
        raise ValueError("Not enough generated windows to pack a clip")
    return np.asarray(clips[0], dtype=np.float32)


@torch.no_grad()
def predict_clips(
    model: torch.nn.Module, clips: np.ndarray, device: str = "cpu", batch_size: int = 32
) -> np.ndarray:
    """Seizure probability per clip. Mirrors the private ``_predict`` in training."""
    model.eval()
    arr = np.asarray(clips, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    out: list[np.ndarray] = []
    for start in range(0, len(arr), batch_size):
        chunk = torch.as_tensor(arr[start : start + batch_size], dtype=torch.float32, device=device)
        out.append(torch.sigmoid(model(chunk)).cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


@dataclass
class LoadedDetector:
    """One frozen detector plus the operating point chosen on real validation."""

    name: str
    label: str
    model: torch.nn.Module
    threshold: float
    n_params: int
    arm: str | None = None
    path: str | None = None

    def score(self, clips: np.ndarray, device: str = "cpu") -> dict[str, Any]:
        t0 = time.perf_counter()
        probs = predict_clips(self.model, clips, device=device)
        elapsed = (time.perf_counter() - t0) * 1000.0
        fires = probs >= self.threshold
        return {
            "detector": self.name,
            "label": self.label,
            "arm": self.arm,
            "threshold": self.threshold,
            "probabilities": [float(p) for p in probs],
            "mean_probability": float(probs.mean()) if len(probs) else None,
            "n_clips": int(len(probs)),
            "n_fired": int(fires.sum()),
            "fired": [bool(f) for f in fires],
            "latency_ms": round(elapsed, 1),
        }


@dataclass
class DetectorBank:
    """The four detectors of one fold, rebuilt at the campaign's architecture."""

    detectors: dict[str, LoadedDetector] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        fold_dir: Path,
        cfg: dict,
        n_channels: int = 22,
        n_windows: int = CLIP_WINDOWS,
        device: str = "cpu",
    ) -> DetectorBank:
        bank = cls()
        det_dir = Path(fold_dir) / "detectors"
        if not det_dir.is_dir():
            bank.notes.append("No detectors directory in this fold. Level 3 is unavailable live.")
            return bank

        clip_len = int(n_windows * int(cfg.get("sample_length", WINDOW_SAMPLES)))
        for path in sorted(det_dir.glob("*.pt")):
            name = path.stem
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                model = build_detector(cfg, n_channels, sample_length=clip_len)
                missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
                if missing or unexpected:
                    bank.notes.append(
                        f"{name}: {len(missing)} missing and {len(unexpected)} unexpected tensors"
                    )
                model.eval().to(device)
                threshold = float((ckpt.get("extra") or {}).get("threshold", 0.5))
                bank.detectors[name] = LoadedDetector(
                    name=name,
                    label=DETECTOR_LABELS.get(name, name),
                    model=model,
                    threshold=threshold,
                    n_params=int(sum(p.numel() for p in model.parameters())),
                    arm=DETECTOR_ARM.get(name),
                    path=str(path),
                )
            except Exception as exc:  # a broken checkpoint must not take the site down
                bank.notes.append(f"Could not load detector {name}: {exc}")
        if bank.detectors:
            bank.notes.append(
                f"Loaded {len(bank.detectors)} frozen detectors at clip length {clip_len} "
                f"({clip_seconds(n_windows):.2f} s)"
            )
        return bank

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": d.name,
                "label": d.label,
                "arm": d.arm,
                "threshold": d.threshold,
                "n_params": d.n_params,
                "kind": str(type(d.model).__name__),
            }
            for d in self.detectors.values()
        ]

    def score_all(self, clips: np.ndarray, device: str = "cpu") -> dict[str, dict[str, Any]]:
        return {name: det.score(clips, device=device) for name, det in self.detectors.items()}


def level1_latent(
    z_report: np.ndarray,
    codes: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """MMD^2 between generated codes and the real holdout codes of this fold."""
    report = mmd_report(
        np.asarray(z_report, dtype=np.float32),
        np.asarray(codes, dtype=np.float32),
        rng=rng if rng is not None else np.random.default_rng(0),
    )
    return report.as_dict()


def latent_noise_floor(z_report: np.ndarray, n_repeats: int = 20) -> dict[str, Any]:
    """Real against real, the same estimator and the same n.

    Any arm whose MMD^2 sits inside this band is not distinguishable from the
    real holdout at this sample size, so the floor is reported beside every arm.
    """
    out = mmd_noise_floor(
        np.asarray(z_report, dtype=np.float32),
        n_repeats=int(n_repeats),
        rng=np.random.default_rng(0),
    )
    out.pop("values", None)
    return out


def level2_decoded(real_windows: np.ndarray, fake_windows: np.ndarray) -> dict[str, Any]:
    """Decoded distances, flattened to the four the deck reports."""
    full = evaluate_decoded(
        np.asarray(real_windows, dtype=np.float32),
        np.asarray(fake_windows, dtype=np.float32),
        fs=float(TARGET_FS),
    )
    channel = full.get("channel_correlation") or {}
    spectral = full.get("spectral") or {}
    return {
        "sliced_wasserstein": float(full.get("sliced_wasserstein", float("nan"))),
        "autocorrelation_distance": float(full.get("autocorrelation_distance", float("nan"))),
        "spectral_band_error": float(spectral.get("mean_band_error", float("nan"))),
        "ks_statistic": float(full.get("ks_statistic", float("nan"))),
        "channel_l1": float(
            channel.get("offdiagonal_absolute_error", float("nan"))
            if isinstance(channel, dict)
            else float("nan")
        ),
        "detail": full,
    }


class CampaignRecord:
    """The recorded fold result, read only, used for the numbers that need the corpus."""

    def __init__(self, study_dir: Path | None) -> None:
        self.study_dir = Path(study_dir) if study_dir is not None else None
        self.row: dict[str, Any] = {}
        self.study: str | None = None
        self.available = False
        self._load()

    def _load(self) -> None:
        if self.study_dir is None:
            return
        for name in ("results_all.json", "results_s0_f0.json"):
            path = self.study_dir / name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows = payload.get("rows") or []
            if not rows:
                continue
            self.row = rows[0]
            self.study = payload.get("study")
            self.available = True
            return

    @property
    def fold_name(self) -> str | None:
        return self.row.get("fold_name")

    def detection(self) -> dict[str, dict[str, Any]]:
        """Recorded holdout detection metrics per detector, history stripped."""
        det = self.row.get("detector") or {}
        out: dict[str, dict[str, Any]] = {}
        for name, rec in det.items():
            if not isinstance(rec, dict):
                continue
            keep = {
                k: rec.get(k)
                for k in (
                    "auroc",
                    "auprc",
                    "mcc",
                    "f1",
                    "sensitivity",
                    "specificity",
                    "precision",
                    "threshold",
                    "brier",
                    "ece",
                    "n_test",
                    "n_synthetic_added",
                )
            }
            keep["label"] = DETECTOR_LABELS.get(name, name)
            keep["arm"] = DETECTOR_ARM.get(name)
            keep["confusion"] = rec.get("confusion")
            out[name] = keep
        baseline = out.get("baseline") or {}
        base_auroc = baseline.get("auroc")
        for name, rec in out.items():
            if name == "baseline" or base_auroc is None or rec.get("auroc") is None:
                rec["delta_auroc"] = None
            else:
                rec["delta_auroc"] = float(rec["auroc"]) - float(base_auroc)
        return out

    def generative(self) -> dict[str, dict[str, Any]]:
        """Recorded Level 1 and Level 2 per arm, so live values can be checked against them."""
        arms = self.row.get("arms") or {}
        out: dict[str, dict[str, Any]] = {}
        for arm, rec in arms.items():
            if not isinstance(rec, dict):
                continue
            latent = rec.get("latent_distance") or {}
            decoded = rec.get("decoded") or {}
            spectral = decoded.get("spectral") or {}
            channel = decoded.get("channel_correlation") or {}
            out[arm] = {
                "n_params": rec.get("n_params"),
                "converged": rec.get("converged"),
                "convergence_note": rec.get("convergence_note"),
                "selection_step": rec.get("selection_step"),
                "mmd2": latent.get("mmd2"),
                "sliced_wasserstein_latent": rec.get("latent_sliced_wasserstein"),
                "mean_entanglement": rec.get("mean_entanglement"),
                "n_synthetic": rec.get("n_synthetic"),
                "decoded": {
                    "sliced_wasserstein": decoded.get("sliced_wasserstein"),
                    "autocorrelation_distance": decoded.get("autocorrelation_distance"),
                    "spectral_band_error": spectral.get("mean_band_error"),
                    "ks_statistic": decoded.get("ks_statistic"),
                    "channel_l1": (
                        channel.get("offdiagonal_absolute_error")
                        if isinstance(channel, dict)
                        else None
                    ),
                },
            }
        return out

    def context(self) -> dict[str, Any]:
        """Everything needed to say what these numbers are a result *of*."""
        ae = self.row.get("autoencoder") or {}
        gates = (ae.get("gates") or {}) if isinstance(ae, dict) else {}
        floor = self.row.get("mmd_noise_floor") or {}
        clinical = self.row.get("clinical") or {}
        return {
            "available": self.available,
            "study": self.study,
            "fold_name": self.fold_name,
            "fold": self.row.get("fold"),
            "seed": self.row.get("seed"),
            "n_train": self.row.get("n_train"),
            "n_test": self.row.get("n_test"),
            "n_train_positive": self.row.get("n_train_positive"),
            "n_test_positive": self.row.get("n_test_positive"),
            "ae_nmse": gates.get("nmse"),
            "ae_gate_passed": gates.get("passed"),
            "ae_variance_explained": gates.get("variance_explained"),
            "mmd_noise_floor": {
                "mean": floor.get("mean"),
                "q025": floor.get("q025"),
                "q975": floor.get("q975"),
            },
            "clinical": {
                "clip_windows": (clinical.get("test") or {}).get("n_windows", CLIP_WINDOWS),
                "clip_length": (clinical.get("test") or {}).get("clip_length"),
                "duration_seconds": clinical.get("duration_seconds", clip_seconds()),
                "label_mode": (clinical.get("test") or {}).get("label_mode"),
                "n_clips_train": (clinical.get("train") or {}).get("n_clips"),
                "n_clips_test": (clinical.get("test") or {}).get("n_clips"),
            },
        }
