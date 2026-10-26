"""Selecting synthetic windows for downstream augmentation.

Audit item E5. The old selection had a fallback that fired whenever no sample cleared
the fidelity threshold::

    keep = [int(i) for i in order if scores[i] >= min_efs][:top_k]
    if not keep:
        keep = order[: max(1, top_k // 4)].tolist()

Three things went wrong together. The threshold was 0.35 while the fidelity scores in
the same records ran from 0.03 to 0.08, so nothing ever cleared it and the fallback
fired on every fold of every gated run. The fallback then kept a quarter of the
requested count, silently turning a request for 200 windows into 50. Fifty synthetic
windows against roughly 3600 real seizure windows is an augmentation dose near one and
a half percent, which no detector could be expected to respond to. And because the
fallback ignores the threshold entirely, the windows that were injected were the ones
that had just failed the quality criterion.

So the reported finding that quality gating did not help was not a test of quality
gating. It was a test of adding fifty low-fidelity windows.

The rewrite makes the failure impossible to hide. The dose is stated in one place, the
threshold is validated against the observed score distribution before it is applied,
and when nothing clears it the configured policy decides: raise, fall back to the
observed median with that recorded, or disable the gate. Every path records what
happened and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


class AugmentationError(RuntimeError):
    """Raised when the requested augmentation cannot be delivered as specified."""


@dataclass
class DoseSpec:
    """How many synthetic windows to inject, and on what basis.

    Attributes:
        mode: ``balance`` injects enough to reach a target positive ratio;
            ``fixed`` injects a stated count; ``proportional`` injects a multiple of
            the available real positives.
        target_ratio: desired positive fraction, for ``balance``.
        count: absolute count, for ``fixed``.
        multiplier: multiple of real positives, for ``proportional``.
        max_add: hard ceiling regardless of mode.
    """

    mode: str = "balance"
    target_ratio: float = 1.0
    count: int = 200
    multiplier: float = 0.5
    max_add: int | None = None

    @classmethod
    def from_cfg(cls, cfg: dict) -> "DoseSpec":
        aug = cfg.get("augmentation", {}) or {}
        return cls(
            mode=str(aug.get("dose_mode", "balance")),
            target_ratio=float(aug.get("target_ratio", 1.0)),
            count=int(aug.get("dose_count", 200)),
            multiplier=float(aug.get("dose_multiplier", 0.5)),
            max_add=aug.get("max_add"),
        )

    def resolve(self, n_positive: int, n_negative: int, n_available: int) -> tuple[int, dict]:
        """Compute the dose and explain how it was derived."""
        if self.mode == "balance":
            # Windows needed so positives reach target_ratio of the negatives.
            desired = int(round(self.target_ratio * n_negative)) - n_positive
            requested = max(0, desired)
            basis = (
                f"balance to a positive ratio of {self.target_ratio} against "
                f"{n_negative} negatives, from {n_positive} real positives"
            )
        elif self.mode == "fixed":
            requested = int(self.count)
            basis = f"fixed count of {self.count}"
        elif self.mode == "proportional":
            requested = int(round(self.multiplier * n_positive))
            basis = f"{self.multiplier} times the {n_positive} real positives"
        else:
            raise AugmentationError(
                f"Unknown dose mode '{self.mode}'; expected balance, fixed or proportional"
            )

        capped = requested
        limits = []
        if self.max_add is not None and capped > int(self.max_add):
            capped = int(self.max_add)
            limits.append(f"max_add={self.max_add}")
        if capped > n_available:
            capped = int(n_available)
            limits.append(f"only {n_available} synthetic windows generated")

        return capped, {
            "dose_mode": self.mode,
            "requested": requested,
            "delivered": capped,
            "basis": basis,
            "limited_by": limits,
            "shortfall": requested - capped,
            "dose_fraction_of_real_positives": (
                float(capped) / n_positive if n_positive else float("nan")
            ),
        }


@dataclass
class SelectionResult:
    """The selected windows plus a full account of the selection."""

    selected: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.selected)


def select_for_augmentation(
    synthetic: np.ndarray,
    real_reference: np.ndarray,
    cfg: dict,
    n_positive: int,
    n_negative: int,
    score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    rng: np.random.Generator | None = None,
) -> SelectionResult:
    """Choose which synthetic windows to inject, and how many.

    Args:
        synthetic: candidate windows, shape ``(N, C, T)``.
        real_reference: real windows the fidelity score is measured against. Must come
            from the reporting split, never from the split used to select the
            generator checkpoint.
        cfg: resolved configuration.
        n_positive: real positive count in the training fold.
        n_negative: real negative count in the training fold.
        score_fn: per-sample fidelity score. Required when the gate is enabled.
        rng: randomness for the ungated path.

    Raises:
        AugmentationError: when the gate is enabled and nothing clears the threshold
            while the policy is ``error``, or when the delivered dose falls so far
            below the request that the result could not show an effect.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    aug_cfg = cfg.get("augmentation", {}) or {}
    dose = DoseSpec.from_cfg(cfg)

    if synthetic is None or len(synthetic) == 0:
        return SelectionResult(
            selected=np.empty((0,), dtype=np.float32),
            meta={"n_generated": 0, "n_selected": 0, "note": "no synthetic windows supplied"},
        )

    n_target, dose_meta = dose.resolve(n_positive, n_negative, len(synthetic))
    meta: dict = {"n_generated": int(len(synthetic)), **dose_meta}

    # Refuse silent under-delivery (ae_v5b: requested ~30k, delivered 1024 under
    # dose_mode: balance). A large shortfall is not a valid "balance" or "fixed" test.
    requested = int(dose_meta.get("requested", 0))
    shortfall = int(dose_meta.get("shortfall", 0))
    max_shortfall_fraction = float(aug_cfg.get("max_shortfall_fraction", 0.05))
    if requested > 0 and shortfall / requested > max_shortfall_fraction:
        limits = ", ".join(dose_meta.get("limited_by") or []) or "unknown limit"
        raise AugmentationError(
            f"Augmentation shortfall {shortfall}/{requested} "
            f"({shortfall / requested:.1%}) exceeds max_shortfall_fraction="
            f"{max_shortfall_fraction:.1%} (limited by: {limits}). "
            f"Raise n_synthetic / max_add, switch to dose_mode: fixed with a "
            f"deliverable dose_count, or raise max_shortfall_fraction deliberately. "
            "Reporting a capped dose as the configured protocol is not publishable."
        )

    min_dose_fraction = float(aug_cfg.get("min_dose_fraction", 0.0))
    observed_fraction = dose_meta["dose_fraction_of_real_positives"]
    if min_dose_fraction > 0 and np.isfinite(observed_fraction):
        if observed_fraction < min_dose_fraction:
            raise AugmentationError(
                f"Augmentation dose is {observed_fraction:.3%} of the {n_positive} real "
                f"positives, below the required minimum of {min_dose_fraction:.3%}. "
                f"Generate more synthetic windows or lower the requirement. A dose this "
                "small cannot produce a detectable change in the detector, so reporting "
                "it as a null result would be misleading."
            )

    if not bool(aug_cfg.get("quality_gate", False)):
        idx = rng.choice(len(synthetic), size=n_target, replace=False)
        meta.update({"mode": "ungated", "n_selected": int(n_target)})
        return SelectionResult(selected=synthetic[idx], meta=meta)

    if score_fn is None:
        raise AugmentationError("quality_gate is enabled but no score_fn was supplied")

    threshold = float(aug_cfg.get("min_score", 0.0))
    policy = str(aug_cfg.get("fallback", "error")).lower()

    scores = np.asarray(
        [score_fn(real_reference, synthetic[i : i + 1]) for i in range(len(synthetic))],
        dtype=np.float64,
    )
    meta["score_distribution"] = {
        "min": float(scores.min()),
        "median": float(np.median(scores)),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
    }
    meta["threshold_requested"] = threshold

    passing = int((scores >= threshold).sum())
    meta["n_passing_threshold"] = passing

    effective_threshold = threshold
    if passing < n_target:
        # The condition that used to trigger a silent quarter-dose.
        note = (
            f"only {passing} of {len(scores)} synthetic windows reach the fidelity "
            f"threshold {threshold:.4f}, but {n_target} are needed. Observed scores span "
            f"{scores.min():.4f} to {scores.max():.4f}."
        )
        if policy == "error":
            raise AugmentationError(
                note
                + " The threshold is unreachable for this generator, so gating cannot be "
                "evaluated as specified. Set augmentation.min_score to a value inside the "
                "observed range, or augmentation.fallback: median to rank without an "
                "absolute threshold and have that recorded."
            )
        if policy == "median":
            effective_threshold = float(np.median(scores))
            meta["fallback_applied"] = "median"
            meta["fallback_note"] = (
                note + " Ranking by score with the threshold relaxed to the observed "
                "median; this is a relative gate, not the absolute gate requested."
            )
        else:
            effective_threshold = float("-inf")
            meta["fallback_applied"] = "all"
            meta["fallback_note"] = note + " Threshold disabled; selection is by rank only."

    meta["threshold_effective"] = effective_threshold
    order = np.argsort(-scores)
    keep = [int(i) for i in order if scores[i] >= effective_threshold][:n_target]

    if len(keep) < n_target:
        # Top up by rank so the delivered dose still matches the request, and say so.
        extra = [int(i) for i in order if int(i) not in set(keep)][: n_target - len(keep)]
        keep.extend(extra)
        meta["topped_up_by_rank"] = len(extra)

    selected = synthetic[keep]
    meta.update(
        {
            "mode": "quality_gate",
            "n_selected": int(len(selected)),
            "mean_score_selected": float(scores[keep].mean()) if keep else float("nan"),
            "mean_score_all": float(scores.mean()),
        }
    )
    if len(selected) != n_target:
        raise AugmentationError(
            f"Delivered {len(selected)} windows against a target of {n_target}. "
            "The dose must match the request or the run must fail; a silent shortfall "
            "is what made the previous augmentation study uninterpretable."
        )
    return SelectionResult(selected=selected, meta=meta)


def subsample_codes(z: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Take a fraction of the codes, for the data-efficiency study.

    A floor keeps the sample large enough to fit a generator at all; when the floor
    binds it means the requested fraction was not delivered, so the caller should
    record the actual count rather than the nominal fraction.
    """
    frac = float(fraction)
    if frac >= 1.0:
        return z
    floor = max(int(z.shape[1]) * 4, 32)
    n_take = min(len(z), max(int(len(z) * frac), floor))
    idx = rng.choice(len(z), size=n_take, replace=False)
    return z[idx]
