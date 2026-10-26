"""Capacity-matched latent generative models for multichannel seizure EEG.

One library replaces the earlier forked experiment directories. Each experiment is
a study file under ``configs/studies``. The defaults are the post-audit protocol
described in ``docs/PROTOCOL.md``. ``AUDIT_FIXES.md`` records the failure each
default closes, and ``MIGRATION.md`` maps the old entry points onto this package.
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
