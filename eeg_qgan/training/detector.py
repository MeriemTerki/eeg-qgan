"""Training and scoring the downstream detector.

Audit items F8 and A16.

F8: the early-stopping validation split is carved out of the real training data before
any synthetic window is added. Adding synthetic data first, then splitting, puts
synthetic samples into the validation set, so the stopping criterion partly measures
how well the detector fits the generator. That inflates the augmented arms in a way the
baseline never enjoys.

A16: the decision threshold is chosen on that validation split rather than left at 0.5.
With a positive rate near four percent a fixed threshold of 0.5 conflates the quality of
the ranking with the calibration of the probabilities, and it is the ranking that the
augmentation is supposed to improve. The threshold is tuned on validation data and then
applied unchanged to the test fold, so the test data plays no part in choosing it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from eeg_qgan.models.detector import FocalLoss, build_detector


@dataclass
class DetectorResult:
    """Test-fold performance plus the provenance of the decision threshold."""

    f1: float
    auroc: float
    auprc: float
    mcc: float
    sensitivity: float
    specificity: float
    precision: float
    threshold: float
    brier: float
    ece: float
    n_train: int
    n_val: int
    n_test: int
    n_synthetic_added: int
    epochs_run: int
    wall_seconds: float
    augmentation: dict = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    # Kept for checkpointing; never serialised into the metrics JSON.
    model: Any = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict:
        out = {
            "f1": self.f1,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "mcc": self.mcc,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "threshold": self.threshold,
            "brier": self.brier,
            "ece": self.ece,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_synthetic_added": self.n_synthetic_added,
            "epochs_run": self.epochs_run,
            "wall_seconds": self.wall_seconds,
            "augmentation": self.augmentation,
            "confusion": self.confusion,
        }
        if self.history:
            out["history"] = self.history
        return out


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Binned gap between confidence and accuracy."""
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (probs > lo) & (probs <= hi)
        if not sel.any():
            continue
        total += (sel.mean()) * abs(float(y_true[sel].mean()) - float(probs[sel].mean()))
    return float(total)


def _augment_windows(
    xb: torch.Tensor, det_cfg: dict, rng: np.random.Generator
) -> torch.Tensor:
    """Label-preserving jitter on a training batch.

    Off by default. It is the control a reader will ask for, because a gain from
    synthetic windows only means something if simple signal jitter does not deliver the
    same gain. When it is enabled it must be enabled for every arm including the
    real-only baseline, otherwise the comparison measures the jitter instead.

    Each transform preserves whether a seizure is present: a circular shift moves the
    event within the window, a gain change rescales it, dropping a channel mimics a bad
    electrode, and the added noise stays well below the signal amplitude.
    """
    shift_max = int(det_cfg.get("augment_time_shift", 0))
    if shift_max > 0:
        shift = int(rng.integers(-shift_max, shift_max + 1))
        if shift:
            xb = torch.roll(xb, shifts=shift, dims=-1)

    scale_range = float(det_cfg.get("augment_scale", 0.0))
    if scale_range > 0:
        gains = torch.as_tensor(
            rng.uniform(1.0 - scale_range, 1.0 + scale_range, size=(len(xb), 1, 1)),
            dtype=xb.dtype,
            device=xb.device,
        )
        xb = xb * gains

    drop_p = float(det_cfg.get("augment_channel_dropout", 0.0))
    if drop_p > 0:
        keep = torch.as_tensor(
            (rng.random((len(xb), xb.shape[1], 1)) >= drop_p).astype(np.float32),
            dtype=xb.dtype,
            device=xb.device,
        )
        xb = xb * keep

    noise_std = float(det_cfg.get("augment_noise_std", 0.0))
    if noise_std > 0:
        xb = xb + torch.randn_like(xb) * noise_std

    return xb.clamp(-1.0, 1.0)


class _WeightEMA:
    """Exponential moving average of the detector weights.

    Only floating-point entries of the state dict are averaged, which covers the
    parameters and the normalisation running statistics while leaving integer counters
    alone. The averaged weights are swapped into the live model for scoring and swapped
    back out afterwards, so training continues from the raw iterate.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.n_updates = 0
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model: nn.Module) -> None:
        self.n_updates += 1
        # Ramp the horizon in, so the average is not anchored to the random start.
        decay = min(self.decay, (1.0 + self.n_updates) / (10.0 + self.n_updates))
        with torch.no_grad():
            state = model.state_dict()
            for key, shadow in self.shadow.items():
                shadow.mul_(decay).add_(state[key].detach().float(), alpha=1.0 - decay)

    def swap_in(self, model: nn.Module) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            state = model.state_dict()
            backup = {k: state[k].detach().clone() for k in self.shadow}
            for key, shadow in self.shadow.items():
                state[key].copy_(shadow.to(state[key].dtype))
        return backup

    def swap_out(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            state = model.state_dict()
            for key, value in backup.items():
                state[key].copy_(value)


def choose_threshold(
    y_val: np.ndarray,
    probs_val: np.ndarray,
    metric: str = "mcc",
) -> float:
    """Pick an operating point on the real-only validation split.

    ``metric``:
      - ``mcc`` (default): Matthews correlation — preferred under class imbalance.
      - ``f1``: harmonic mean of precision and recall (harsh when positives are rare).
      - ``youden``: sensitivity + specificity − 1.

    Ranking metrics (AUROC / AUPRC) and calibration (ECE / Brier) do not use this
    threshold.
    """
    if len(np.unique(y_val)) < 2:
        return 0.5
    metric = str(metric).lower()
    candidates = np.unique(np.round(probs_val, 4))
    if len(candidates) > 200:
        candidates = np.quantile(probs_val, np.linspace(0.01, 0.99, 200))
    best_t, best_score = 0.5, -np.inf
    for t in candidates:
        preds = (probs_val >= t).astype(int)
        if metric in {"f1", "f1_score"}:
            score = float(f1_score(y_val, preds, zero_division=0))
        elif metric in {"youden", "balanced", "j"}:
            tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
            sens = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            score = float(sens + spec - 1.0)
        else:
            score = float(matthews_corrcoef(y_val, preds))
            if not np.isfinite(score):
                score = -1.0
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


@torch.no_grad()
def _predict(model: nn.Module, x: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(x), int(batch_size)):
        chunk = torch.as_tensor(
            np.asarray(x[start : start + int(batch_size)], dtype=np.float32), device=device
        )
        out.append(torch.sigmoid(model(chunk)).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty(0, dtype=np.float32)


def train_and_evaluate_detector(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    device: str = "cpu",
    synthetic: np.ndarray | None = None,
    augmentation_meta: dict | None = None,
    groups_train: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> DetectorResult:
    """Train the detector on one fold and score it on the held-out patients.

    Args:
        x_train: real training windows.
        y_train: labels for ``x_train``.
        x_test: held-out windows from patients absent from training.
        y_test: labels for ``x_test``.
        cfg: resolved configuration.
        device: torch device.
        synthetic: synthetic positive windows to inject, already selected.
        augmentation_meta: record from the selection step, carried into the result.
        groups_train: groups for the training windows, so the early-stopping split can
            respect patient boundaries too.
        rng: randomness for the split and batching.
    """
    t0 = time.perf_counter()
    det_cfg = cfg.get("detector", {}) or {}
    rng = rng if rng is not None else np.random.default_rng(int(cfg.get("seed", 42)))

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)

    # ---- validation split, carved from REAL data only, before augmentation ----
    val_fraction = float(det_cfg.get("val_fraction", 0.15))
    if groups_train is not None:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=int(cfg.get("seed", 42))
        )
        tr_idx, va_idx = next(splitter.split(x_train, y_train, groups=groups_train))
    else:
        from sklearn.model_selection import train_test_split

        tr_idx, va_idx = train_test_split(
            np.arange(len(x_train)),
            test_size=val_fraction,
            random_state=int(cfg.get("seed", 42)),
            stratify=y_train if len(np.unique(y_train)) > 1 else None,
        )

    x_fit, y_fit = x_train[tr_idx], y_train[tr_idx]
    x_val, y_val = x_train[va_idx], y_train[va_idx]

    # ---- how the class imbalance is handled, decided once and recorded ----
    # These three mechanisms all push in the same direction, and stacking them
    # over-corrects: duplicating a few thousand positive windows until they match
    # hundreds of thousands of negatives makes the model memorise those windows, and
    # weighting the loss on top of an already balanced sampler weights it twice. Exactly
    # one rebalancing mechanism is active unless the configuration insists otherwise.
    use_balanced_batches = bool(det_cfg.get("balanced_batches", True))
    want_oversample = bool(det_cfg.get("oversample_positives", False))
    oversample = want_oversample and (
        not use_balanced_batches or bool(det_cfg.get("allow_stacked_rebalance", False))
    )
    strategy: dict[str, Any] = {
        "balanced_batches": use_balanced_batches,
        "oversample_requested": want_oversample,
        "oversample_applied": oversample,
    }

    if oversample:
        pos_idx = np.flatnonzero(y_fit == 1)
        neg_idx = np.flatnonzero(y_fit == 0)
        if len(pos_idx) > 0 and len(neg_idx) > len(pos_idx):
            target_ratio = float(det_cfg.get("oversample_target_ratio", 1.0))
            max_factor = float(det_cfg.get("max_oversample_factor", 5.0))
            n_needed = int(np.ceil(target_ratio * len(neg_idx)))
            # Cap the duplication factor: past this point each positive appears so many
            # times that the fit measures memorisation rather than generalisation.
            n_needed = min(n_needed, int(np.ceil(max_factor * len(pos_idx))))
            take = rng.choice(pos_idx, size=n_needed, replace=True)
            x_fit = np.concatenate([x_fit[neg_idx], x_fit[take]], axis=0)
            y_fit = np.concatenate([y_fit[neg_idx], y_fit[take]], axis=0)
            shuffle = rng.permutation(len(y_fit))
            x_fit, y_fit = x_fit[shuffle], y_fit[shuffle]
            strategy["oversample_duplication_factor"] = float(n_needed / len(pos_idx))

    # ---- augmentation, applied to the fitting split only ----
    n_synth = 0
    if synthetic is not None and len(synthetic) > 0:
        synth = np.asarray(synthetic, dtype=np.float32)
        x_fit = np.concatenate([x_fit, synth], axis=0)
        y_fit = np.concatenate([y_fit, np.ones(len(synth), dtype=np.int64)], axis=0)
        n_synth = len(synth)

    model = build_detector(
        cfg,
        n_channels=int(x_fit.shape[1]),
        sample_length=int(x_fit.shape[2]),
    ).to(device)
    epochs = int(det_cfg.get("epochs", cfg.get("classifier_epochs", 30)))
    batch_size = int(det_cfg.get("batch_size", 64))
    lr = float(det_cfg.get("lr", 1e-3))
    patience = int(det_cfg.get("patience", 8))
    stop_metric = str(det_cfg.get("early_stop_metric", "auprc")).lower()

    n_pos = int((y_fit == 1).sum())
    n_neg = int((y_fit == 0).sum())

    x_fit_t = torch.as_tensor(x_fit)
    y_fit_t = torch.as_tensor(y_fit, dtype=torch.float32)

    if use_balanced_batches and n_pos > 0 and n_neg > 0:
        w_pos = 0.5 / n_pos
        w_neg = 0.5 / n_neg
        sample_w = np.where(y_fit == 1, w_pos, w_neg).astype(np.float64)
        sample_w /= sample_w.sum()
    else:
        sample_w = None

    # The sampler already presents the classes in equal proportion, so a positive
    # weight on top of it would count the same correction twice.
    weighting_would_double = sample_w is not None or strategy.get("oversample_applied", False)
    if weighting_would_double and not bool(det_cfg.get("allow_stacked_rebalance", False)):
        pos_weight = None
    elif bool(det_cfg.get("use_pos_weight", True)):
        pos_weight = float(n_neg) / max(1, n_pos)
    else:
        pos_weight = None
    strategy["pos_weight"] = pos_weight
    strategy["focal_gamma"] = float(det_cfg.get("focal_gamma", 2.0))

    window_augment = bool(det_cfg.get("window_augment", False))
    strategy["window_augment"] = window_augment
    if window_augment:
        strategy["window_augment_settings"] = {
            key: det_cfg.get(key)
            for key in (
                "augment_time_shift",
                "augment_scale",
                "augment_channel_dropout",
                "augment_noise_std",
            )
        }

    criterion = FocalLoss(gamma=float(det_cfg.get("focal_gamma", 2.0)), pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=float(det_cfg.get("weight_decay", 1e-4))
    )

    steps_per_epoch = max(1, int(np.ceil(len(x_fit_t) / batch_size)))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(float(det_cfg.get("warmup_fraction", 0.05)) * total_steps)
    min_lr_factor = float(det_cfg.get("min_lr_factor", 0.05))
    use_schedule = str(det_cfg.get("lr_schedule", "cosine")).lower() == "cosine"

    def _lr_factor(step: int) -> float:
        if not use_schedule:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, _lr_factor)

    # An exponential moving average of the weights. Under balanced sampling the
    # gradient is dominated by a small set of repeatedly drawn positives, so the raw
    # iterate is noisy from epoch to epoch; the average is what gets scored, and the
    # early-stopping decision is made on the same weights that produce the final
    # probabilities.
    ema_decay = float(det_cfg.get("ema_decay", 0.999))
    ema = _WeightEMA(model, decay=ema_decay) if ema_decay > 0 else None

    best_score, best_state, stale, epochs_run = -np.inf, None, 0, 0
    history: list[dict] = []
    save_history = bool(det_cfg.get("save_history", True))
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        if sample_w is not None:
            batches = (
                torch.as_tensor(
                    rng.choice(len(x_fit_t), size=batch_size, replace=True, p=sample_w),
                    dtype=torch.long,
                )
                for _ in range(steps_per_epoch)
            )
        else:
            perm = torch.as_tensor(rng.permutation(len(x_fit_t)), dtype=torch.long)
            batches = (perm[start : start + batch_size] for start in range(0, len(perm), batch_size))

        for idx in batches:
            xb = x_fit_t[idx].to(device)
            yb = y_fit_t[idx].to(device)
            if window_augment:
                xb = _augment_windows(xb, det_cfg, rng)
            loss = criterion(model(xb), yb)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if float(det_cfg.get("grad_clip", 1.0)) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(det_cfg.get("grad_clip", 1.0)))
            optimiser.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            epoch_loss += float(loss.item())
            n_batches += 1

        backup = ema.swap_in(model) if ema is not None else None
        probs_val = _predict(model, x_val, device)
        # Threshold-free ranking metric for early stopping. AUPRC is preferred under
        # the natural TUH imbalance; AUROC remains available for ablations.
        if len(np.unique(y_val)) > 1:
            val_auroc = float(roc_auc_score(y_val, probs_val))
            if stop_metric in {"auroc", "auc", "roc_auc"}:
                score = val_auroc
            else:
                score = float(average_precision_score(y_val, probs_val))
            val_auprc = float(average_precision_score(y_val, probs_val))
        else:
            score = -float(np.mean((probs_val - y_val) ** 2))
            val_auroc = float("nan")
            val_auprc = float("nan")
        if save_history:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": epoch_loss / max(1, n_batches),
                    "val_auprc": val_auprc,
                    "val_auroc": val_auroc,
                    "stop_score": float(score),
                }
            )
        if score > best_score + 1e-6:
            best_score, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stop_now = False
        else:
            stale += 1
            stop_now = stale >= patience
        if backup is not None:
            ema.swap_out(model, backup)
        if stop_now:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    strategy["early_stop_score"] = float(best_score) if np.isfinite(best_score) else None
    strategy["ema_decay"] = ema_decay if ema is not None else None

    probs_val = _predict(model, x_val, device)
    threshold = choose_threshold(
        y_val,
        probs_val,
        metric=str(det_cfg.get("threshold_metric", "mcc")),
    )

    probs_test = _predict(model, x_test, device)
    y_test = np.asarray(y_test, dtype=np.int64)
    preds = (probs_test >= threshold).astype(int)

    both_classes = len(np.unique(y_test)) > 1
    tn, fp, fn, tp = (
        confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        if len(y_test)
        else (0, 0, 0, 0)
    )

    return DetectorResult(
        f1=float(f1_score(y_test, preds, zero_division=0)),
        auroc=float(roc_auc_score(y_test, probs_test)) if both_classes else float("nan"),
        auprc=float(average_precision_score(y_test, probs_test)) if both_classes else float("nan"),
        mcc=float(matthews_corrcoef(y_test, preds)) if both_classes else float("nan"),
        sensitivity=float(recall_score(y_test, preds, zero_division=0)),
        specificity=float(tn / max(1, tn + fp)),
        precision=float(precision_score(y_test, preds, zero_division=0)),
        threshold=float(threshold),
        brier=float(brier_score_loss(y_test, probs_test)) if both_classes else float("nan"),
        ece=expected_calibration_error(y_test, probs_test),
        n_train=int(len(x_fit)),
        n_val=int(len(x_val)),
        n_test=int(len(x_test)),
        n_synthetic_added=int(n_synth),
        epochs_run=epochs_run,
        wall_seconds=time.perf_counter() - t0,
        augmentation={
            **(augmentation_meta or {}),
            "n_real_positives_in_fit": int((y_train[tr_idx] == 1).sum()),
            "n_synthetic_added": int(n_synth),
            "positive_rate_before": float((y_train[tr_idx] == 1).mean()),
            "positive_rate_after": float((y_fit == 1).mean()),
            "validation_split_is_real_only": True,
            "imbalance_strategy": strategy,
            "detector_params": int(sum(p.numel() for p in model.parameters())),
        },
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        history=history,
        model=model,
    )
