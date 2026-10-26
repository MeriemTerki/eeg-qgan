"""Capacity-matched latent generative models for multichannel seizure EEG.

Each experiment is a study file under ``configs/studies``. The evaluation rules
are described in ``docs/PROTOCOL.md``.
"""

__version__ = "1.0.0"

from eeg_qgan.config import ConfigError, RunSelection, load_config, list_studies
from eeg_qgan.seeding import resolve_device, set_seed, spawn_rng

__all__ = [
    "ConfigError",
    "RunSelection",
    "load_config",
    "list_studies",
    "set_seed",
    "spawn_rng",
    "resolve_device",
    "__version__",
]
