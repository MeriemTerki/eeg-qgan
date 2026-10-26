"""Autoencoder acceptance gates.

Audit item E3. The previous implementation computed a reconstruction test, stored it
under a key named for that test, and then built the overall pass flag from the
saturation test alone::

    "gate_nmse_ok": nmse(recon_np, x_val) < 0.5,
    ...
    report["gates_pass"] = report["gate_saturation_ok"]

So the reconstruction test was dead. Every fold of the headline run logged a
normalised error between 1.07 and 1.13, meaning the reconstruction was no better than
emitting the global mean, and every one was recorded as passing. Two folds also failed
the saturation test at 0.068 and 0.061 against a threshold of 0.02, and the driver
printed a warning and continued; those are the same folds that carried the collapsed
latent coordinates and the worst quantum result.

Here the pass flag is the conjunction of every enabled gate, and the caller decides
between raising and recording an exclusion. Neither option is "print and continue".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def normalised_mse(recon: np.ndarray, x: np.ndarray) -> float:
    """Mean squared error divided by the variance of the target.

    A value of 1 means the reconstruction is no better than predicting the mean, so
    anything at or above 1 indicates the autoencoder has learned nothing useful.
    """
    mse = float(np.mean((recon - x) ** 2))
    var = float(np.var(x)) + 1e-8
    return mse / var


def variance_explained(recon: np.ndarray, x: np.ndarray) -> float:
    """1 - NMSE, the familiar R^2-style reading. Negative when worse than the mean."""
    return 1.0 - normalised_mse(recon, x)


def mean_autocorrelation(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Mean autocorrelation across channels and samples, vectorised over lags."""
    n, c, t = x.shape
    flat = x.reshape(n * c, t).astype(np.float64)
    flat = flat - flat.mean(axis=1, keepdims=True)
    denom = (flat**2).sum(axis=1) + 1e-8
    out = np.ones(max_lag + 1, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        num = (flat[:, :-lag] * flat[:, lag:]).sum(axis=1)
        out[lag] = float(np.mean(num / denom))
    return out


def autocorrelation_preservation(x: np.ndarray, recon: np.ndarray, max_lag: int = 50) -> float:
    a = mean_autocorrelation(x, max_lag)
    b = mean_autocorrelation(recon, max_lag)
    return float(1.0 - np.mean(np.abs(a - b)))


def latent_saturation(z: np.ndarray, threshold: float = 0.99) -> float:
    """Fraction of code entries pinned against the bounds of the tanh."""
    return float(np.mean(np.abs(z) > threshold))


@dataclass
class GateReport:
    """Outcome of every gate, with the pass flag derived from all of them."""

    nmse: float
    variance_explained: float
    acf_preservation: float
    latent_saturation_frac: float
    thresholds: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def failures(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok]

    def as_dict(self) -> dict:
        return {
            "nmse": self.nmse,
            "variance_explained": self.variance_explained,
            "acf_preservation": self.acf_preservation,
            "latent_saturation_frac": self.latent_saturation_frac,
            "thresholds": dict(self.thresholds),
            "checks": dict(self.checks),
            "passed": self.passed,
            "failures": self.failures(),
        }

    def explain(self) -> str:
        if self.passed:
            return "all autoencoder gates passed"
        parts = []
        for name in self.failures():
            if name == "reconstruction":
                parts.append(
                    f"reconstruction NMSE {self.nmse:.4f} exceeds "
                    f"{self.thresholds.get('max_nmse')} (a value at or above 1 means "
                    "the decoder is no better than emitting the mean)"
                )
            elif name == "saturation":
                parts.append(
                    f"latent saturation {self.latent_saturation_frac:.4f} exceeds "
                    f"{self.thresholds.get('max_saturation')}, so codes are pinned "
                    "against the tanh bounds and coordinates may have collapsed"
                )
            elif name == "autocorrelation":
                parts.append(
                    f"autocorrelation preservation {self.acf_preservation:.4f} is below "
                    f"{self.thresholds.get('min_acf_preservation')}"
                )
        return "; ".join(parts)


@torch.no_grad()
def evaluate_ae_gates(
    ae: torch.nn.Module,
    x_val: np.ndarray,
    device: str = "cpu",
    max_nmse: float = 0.5,
    max_saturation: float | None = None,
    min_acf_preservation: float | None = None,
    saturation_threshold: float = 0.99,
    batch_size: int = 256,
) -> GateReport:
    """Score an autoencoder against every acceptance gate.

    Args:
        ae: the trained autoencoder.
        x_val: held-out windows, shape ``(N, C, T)``.
        max_nmse: reconstruction ceiling. This gate is now enforced.
        max_saturation: ceiling on the fraction of saturated code entries.
        min_acf_preservation: optional floor on autocorrelation agreement.
        saturation_threshold: absolute code value counted as saturated.
        batch_size: evaluation batch size, so long validation sets fit in memory.
    """
    ae.eval()
    x_val = np.asarray(x_val, dtype=np.float32)

    recons: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for start in range(0, len(x_val), int(batch_size)):
        chunk = torch.as_tensor(x_val[start : start + int(batch_size)], device=device)
        recon, z = ae(chunk, denoise_decode=False)
        recons.append(recon.cpu().numpy())
        codes.append(z.cpu().numpy())

    recon_np = np.concatenate(recons, axis=0)
    z_np = np.concatenate(codes, axis=0)

    nmse_value = normalised_mse(recon_np, x_val)
    acf_value = autocorrelation_preservation(x_val, recon_np)
    sat_value = latent_saturation(z_np, saturation_threshold)

    thresholds = {
        "max_nmse": float(max_nmse),
        "max_saturation": None if max_saturation is None else float(max_saturation),
        "min_acf_preservation": min_acf_preservation,
    }
    checks = {
        "reconstruction": bool(nmse_value < float(max_nmse)),
    }
    # Saturation is a collapse diagnostic. A working tanh bottleneck often uses the
    # outer part of (-1, 1); failing the fold on that alone rejected decoders with
    # NMSE well below 0.5. Only enforce it when explicitly configured.
    if max_saturation is not None:
        checks["saturation"] = bool(sat_value < float(max_saturation))
    if min_acf_preservation is not None:
        checks["autocorrelation"] = bool(acf_value >= float(min_acf_preservation))

    return GateReport(
        nmse=nmse_value,
        variance_explained=1.0 - nmse_value,
        acf_preservation=acf_value,
        latent_saturation_frac=sat_value,
        thresholds=thresholds,
        checks=checks,
    )


class AEGateFailure(RuntimeError):
    """Raised when an autoencoder fails a gate and the policy is to stop."""

    def __init__(self, report: GateReport, fold_label: str = "") -> None:
        self.report = report
        where = f" for {fold_label}" if fold_label else ""
        super().__init__(
            f"Autoencoder gates failed{where}: {report.explain()}. "
            "Set autoencoder.on_gate_failure: exclude to record the fold as excluded "
            "and continue, or fix the architecture. Silently continuing is what "
            "allowed a decoder with NMSE above 1 into the headline result."
        )
