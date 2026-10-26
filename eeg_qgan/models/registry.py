"""Building generators by name, with the capacity target derived from the circuit.

The arm names are the vocabulary the configs and the result files speak, so they are
defined in one place. ``build_generator`` is the only way an arm gets constructed,
which is what guarantees that the capacity-matched twin is sized from the actual
circuit rather than from a number copied into a config and left to drift.

Audit F6 is enforced here: :func:`resolve_noise_dim` gives every arm in a comparison
the same noise width, so a difference in input dimension can never masquerade as a
difference between quantum and classical models.
"""
from __future__ import annotations

import torch.nn as nn

from eeg_qgan.models.classical import (
    CapacityMatchedGenerator,
    FourierGenerator,
    LargeMLPGenerator,
    LastStyleGenerator,
    MLPGenerator,
)
from eeg_qgan.models.quantum import StyleQuantumGenerator

QUANTUM_ARMS = ("quantum", "quantum_none", "quantum_linear", "quantum_ring", "quantum_full")
CLASSICAL_ARMS = (
    "classical_matched",
    "classical_equiv",
    "classical_large",
    "classical_last",
    "fourier",
)
NON_ADVERSARIAL_ARMS = ("gmm",)
RAW_ARMS = ("raw_wgan",)

ALL_ARMS = QUANTUM_ARMS + CLASSICAL_ARMS + NON_ADVERSARIAL_ARMS + RAW_ARMS


def is_quantum(arm: str) -> bool:
    return str(arm).startswith("quantum")


def entanglement_for(arm: str, cfg: dict) -> str:
    """Topology for a quantum arm: from the suffix when present, else the config."""
    quantum_cfg = cfg.get("quantum", {}) or {}
    if "_" in str(arm):
        suffix = str(arm).split("_", 1)[1]
        if suffix in ("none", "linear", "ring", "full"):
            return suffix
    return str(quantum_cfg.get("entanglement", cfg.get("quantum_entanglement", "ring")))


def quantum_parameter_target(cfg: dict) -> int:
    """Trainable parameter count of the circuit this config specifies.

    Built rather than computed from a formula so the number is the real one even when
    a classical expansion layer is present. That layer is exactly what made the
    nominal ``4 * layers * qubits`` figure wrong in the wider-latent runs.
    """
    return int(build_generator("quantum", cfg).count_params())


def resolve_noise_dim(cfg: dict) -> int:
    """The noise width shared by every arm in this comparison.

    Defaults to the latent width so all arms are full rank. Setting
    ``noise_dim: qubits`` reproduces the earlier behaviour in which the quantum arm
    received only as many noise coordinates as it had qubits.
    """
    latent_dim = int(cfg["latent_dim"])
    quantum_cfg = cfg.get("quantum", {}) or {}
    n_qubits = int(quantum_cfg.get("n_qubits", cfg.get("n_qubits", latent_dim)))
    requested = cfg.get("noise_dim", "latent")
    if requested in ("latent", None):
        return latent_dim
    if requested == "qubits":
        return n_qubits
    return int(requested)


def build_generator(arm: str, cfg: dict, output_scale: float = 1.0) -> nn.Module:
    """Construct one generator arm.

    Args:
        arm: an entry of :data:`ALL_ARMS`.
        cfg: the resolved configuration.
        output_scale: multiplier covering the target support, supplied by the fitted
            latent transform when the coordinate system is standardised.
    """
    latent_dim = int(cfg["latent_dim"])
    noise_dim = resolve_noise_dim(cfg)
    quantum_cfg = cfg.get("quantum", {}) or {}
    classical_cfg = cfg.get("classical", {}) or {}

    if is_quantum(arm):
        return StyleQuantumGenerator(
            latent_dim=latent_dim,
            n_qubits=int(quantum_cfg.get("n_qubits", cfg.get("n_qubits", latent_dim))),
            n_layers=int(quantum_cfg.get("n_layers", cfg.get("n_quantum_layers", 4))),
            entanglement=entanglement_for(arm, cfg),
            ansatz=str(quantum_cfg.get("ansatz", "style_qgan")),
            noise_dim=noise_dim,
            shots=quantum_cfg.get("shots"),
            cfg=cfg,
            output_scale=output_scale,
        )

    if arm == "classical_equiv":
        target = int(cfg.get("classical_equiv_target_params") or quantum_parameter_target(cfg))
        hidden = classical_cfg.get("equiv_hidden")
        return CapacityMatchedGenerator(
            latent_dim=latent_dim,
            target_params=target,
            noise_dim=noise_dim,
            hidden=tuple(int(h) for h in hidden) if hidden else None,
            output_scale=output_scale,
        )

    if arm == "classical_matched":
        return MLPGenerator(
            latent_dim=latent_dim,
            hidden=tuple(classical_cfg.get("matched_hidden", (10,))),
            noise_dim=noise_dim,
            output_scale=output_scale,
        )

    if arm == "classical_large":
        return LargeMLPGenerator(
            latent_dim=latent_dim,
            hidden=tuple(classical_cfg.get("large_hidden", (64, 64))),
            noise_dim=noise_dim,
            spectral_norm=bool(classical_cfg.get("large_spectral_norm", True)),
            output_scale=output_scale,
        )

    if arm == "classical_last":
        return LastStyleGenerator(
            latent_dim=latent_dim,
            hidden=tuple(classical_cfg.get("last_hidden", (50, 30))),
            noise_dim=noise_dim,
            output_scale=output_scale,
        )

    if arm == "fourier":
        return FourierGenerator(
            latent_dim=latent_dim,
            n_frequencies=int(classical_cfg.get("fourier_n_frequencies", 4)),
            noise_dim=noise_dim,
            output_scale=output_scale,
        )

    raise ValueError(
        f"Unknown generator arm '{arm}'. Adversarial and MMD arms: "
        f"{QUANTUM_ARMS + CLASSICAL_ARMS}. Non-adversarial: {NON_ADVERSARIAL_ARMS}. "
        f"Raw waveform: {RAW_ARMS}."
    )


def capacity_audit(arms: list[str], cfg: dict) -> dict:
    """Parameter counts for every arm, with the quantum arm as the reference.

    Written before training so that a run whose arms are not actually matched is
    identifiable from its own artefacts.
    """
    target = quantum_parameter_target(cfg)
    noise_dim = resolve_noise_dim(cfg)
    audit: dict = {
        "quantum_target_params": target,
        "shared_noise_dim": noise_dim,
        "latent_dim": int(cfg["latent_dim"]),
        "arms": {},
    }
    for arm in arms:
        if arm in NON_ADVERSARIAL_ARMS or arm in RAW_ARMS:
            audit["arms"][arm] = {"kind": arm, "n_params": None, "note": "no latent generator"}
            continue
        try:
            g = build_generator(arm, cfg)
            entry = g.describe()
            entry["ratio_vs_quantum"] = float(g.count_params()) / max(1, target)
            audit["arms"][arm] = entry
        except Exception as exc:
            audit["arms"][arm] = {"kind": arm, "error": str(exc)}

    widths = {
        a: e.get("noise_dim")
        for a, e in audit["arms"].items()
        if isinstance(e, dict) and e.get("noise_dim")
    }
    audit["noise_dim_consistent"] = len(set(widths.values())) <= 1
    if not audit["noise_dim_consistent"]:
        audit["noise_dim_warning"] = (
            f"Arms disagree on noise width {widths}, so their outputs are supported on "
            "manifolds of different dimension and the comparison is confounded."
        )
    return audit
