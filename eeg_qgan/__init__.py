"""Latent-space quantum and classical generative modelling for scarce seizure EEG.

This package consolidates eleven scattered experiment directories into one library.
See MIGRATION.md for the mapping from the old layout, and CHANGES_FROM_AUDIT.md for
the corrections applied to the pipeline.
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
