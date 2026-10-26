"""Generators, critics and the autoencoder."""

from eeg_qgan.models.autoencoder import ConvDecoder, ConvEncoder, MultichannelAE
from eeg_qgan.models.base import LatentGenerator, choose_hidden_for_target, count_mlp_params
from eeg_qgan.models.classical import (
    CapacityMatchedGenerator,
    FourierGenerator,
    LargeMLPGenerator,
    LastStyleGenerator,
    MLPGenerator,
)
from eeg_qgan.models.critic import (
    SpectralDiscriminator,
    WassersteinCritic,
    build_critic,
    gradient_penalty,
)
from eeg_qgan.models.quantum import StyleQuantumGenerator, entangler_pairs
from eeg_qgan.models.registry import (
    ALL_ARMS,
    CLASSICAL_ARMS,
    QUANTUM_ARMS,
    build_generator,
    capacity_audit,
    entanglement_for,
    is_quantum,
    quantum_parameter_target,
    resolve_noise_dim,
)

__all__ = [
    "MultichannelAE",
    "ConvEncoder",
    "ConvDecoder",
    "LatentGenerator",
    "count_mlp_params",
    "choose_hidden_for_target",
    "MLPGenerator",
    "CapacityMatchedGenerator",
    "LargeMLPGenerator",
    "LastStyleGenerator",
    "FourierGenerator",
    "StyleQuantumGenerator",
    "entangler_pairs",
    "WassersteinCritic",
    "SpectralDiscriminator",
    "build_critic",
    "gradient_penalty",
    "build_generator",
    "capacity_audit",
    "quantum_parameter_target",
    "resolve_noise_dim",
    "is_quantum",
    "entanglement_for",
    "ALL_ARMS",
    "QUANTUM_ARMS",
    "CLASSICAL_ARMS",
]
