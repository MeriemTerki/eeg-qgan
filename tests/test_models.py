"""Generator construction, capacity matching and noise-width parity."""
from __future__ import annotations

import pytest
import torch

from eeg_qgan.models import (
    build_generator,
    capacity_audit,
    entangler_pairs,
    quantum_parameter_target,
)


def base_cfg(**over) -> dict:
    cfg = {
        "latent_dim": 8,
        "n_qubits": 8,
        "seed": 42,
        "quantum": {"n_layers": 4, "entanglement": "ring"},
        "classical": {},
    }
    cfg.update(over)
    return cfg


def test_circuit_parameter_count_is_four_per_qubit_per_layer():
    cfg = base_cfg()
    # 4 trainable scalars per qubit per layer: w_style, b_style, theta, phi.
    assert quantum_parameter_target(cfg) == 4 * 4 * 8


def test_all_entanglement_topologies_carry_identical_parameter_counts():
    """Only the controlled-Z wiring changes, so capacity is constant across topologies."""
    counts = {}
    for topo in ("none", "linear", "ring", "full"):
        cfg = base_cfg(quantum={"n_layers": 4, "entanglement": topo})
        counts[topo] = build_generator("quantum", cfg).count_params()
    assert len(set(counts.values())) == 1, counts

    # And the wiring really does differ.
    assert len(entangler_pairs(8, "none")) == 0
    assert len(entangler_pairs(8, "linear")) == 7
    assert len(entangler_pairs(8, "ring")) == 8
    assert len(entangler_pairs(8, "full")) == 28


def test_capacity_matched_twin_lands_within_one_percent_of_the_circuit():
    cfg = base_cfg()
    target = quantum_parameter_target(cfg)
    twin = build_generator("classical_equiv", cfg)
    actual = twin.count_params()
    assert abs(actual - target) / target < 0.02, f"target {target}, got {actual}"


def test_every_arm_receives_the_same_noise_width():
    """Audit F6: unequal noise width would confound the comparison."""
    cfg = base_cfg(latent_dim=32, n_qubits=8, quantum={"n_layers": 4, "entanglement": "ring"})
    arms = ["quantum", "classical_equiv", "classical_matched", "classical_large", "fourier"]
    audit = capacity_audit(arms, cfg)
    assert audit["noise_dim_consistent"], audit.get("noise_dim_warning")
    assert audit["shared_noise_dim"] == 32


def test_quantum_arm_reports_its_classical_fraction():
    """Audit F5: a wide latent adds a dense expansion layer that dominates the count."""
    cfg = base_cfg(latent_dim=32, n_qubits=8)
    g = build_generator("quantum", cfg)
    breakdown = g.parameter_breakdown()
    assert breakdown["quantum"] == 4 * 4 * 8
    assert breakdown["classical"] > breakdown["quantum"], "expansion layer should dominate"
    assert not breakdown["is_pure_quantum"]

    pure = build_generator("quantum", base_cfg(latent_dim=8, n_qubits=8))
    assert pure.parameter_breakdown()["is_pure_quantum"]


def test_n_qubits_above_latent_dim_raises_instead_of_clamping():
    with pytest.raises(ValueError, match="cannot exceed"):
        build_generator("quantum", base_cfg(latent_dim=4, n_qubits=8))


@pytest.mark.parametrize(
    "arm", ["quantum", "classical_equiv", "classical_matched", "classical_large", "fourier"]
)
def test_every_arm_emits_bounded_codes_of_the_right_shape(arm):
    cfg = base_cfg()
    g = build_generator(arm, cfg)
    out = g(g.sample_noise(6))
    assert out.shape == (6, 8)
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0 + 1e-5, "generators must stay inside the encoder's range"


def test_output_scale_widens_the_reachable_range():
    """The mechanism that makes a standardised target reachable (audit E2)."""
    cfg = base_cfg()
    plain = build_generator("classical_equiv", cfg, output_scale=1.0)
    scaled = build_generator("classical_equiv", cfg, output_scale=4.0)
    # Copy the weights but not the buffers, so only the scale differs.
    scaled.net.load_state_dict(plain.net.state_dict())

    noise = plain.sample_noise(64)
    with torch.no_grad():
        ratio = float(scaled(noise).abs().max() / plain(noise).abs().max())
    assert ratio == pytest.approx(4.0, rel=1e-4)


def test_output_scale_is_saved_with_the_checkpoint():
    """The scale must not be able to drift between training and evaluation."""
    cfg = base_cfg()
    trained = build_generator("classical_equiv", cfg, output_scale=3.5)
    restored = build_generator("classical_equiv", cfg, output_scale=1.0)
    restored.load_state_dict(trained.state_dict())
    assert float(restored.output_scale) == pytest.approx(3.5)


def test_product_topology_cannot_couple_coordinates():
    """Audit F7: without entanglers each readout depends on one noise coordinate."""
    cfg = base_cfg(quantum={"n_layers": 4, "entanglement": "none"})
    g = build_generator("quantum", cfg)
    coupling = g.coordinate_coupling(n_probes=6)
    assert coupling["is_separable"], coupling
    # Zero up to the simulator's floating-point residue, not exactly zero.
    assert coupling["coupling_ratio"] < 1e-5


def test_ring_topology_does_couple_coordinates():
    cfg = base_cfg(quantum={"n_layers": 4, "entanglement": "ring"})
    g = build_generator("quantum", cfg)
    coupling = g.coordinate_coupling(n_probes=6)
    assert not coupling["is_separable"]
    assert coupling["coupling_ratio"] > 1e-6
