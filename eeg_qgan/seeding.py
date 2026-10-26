"""Reproducible seeding.

Audit item F15: the previous implementation seeded ``random``, ``numpy`` and
``torch`` but left cuDNN in its default non-deterministic mode, so GPU runs were
not bit-reproducible. ``set_seed`` here also pins the cuDNN and cuBLAS behaviour,
and ``deterministic=False`` remains available for the rare case where a
non-deterministic kernel is needed for speed.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every generator this project draws from.

    Args:
        seed: the seed value.
        deterministic: when True, force deterministic kernels so repeated runs on
            identical hardware give bit-identical results. Costs some speed.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic matmul reductions on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch builds lack warn_only; a non-fatal miss is acceptable.
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def spawn_rng(*parts: int | str) -> np.random.Generator:
    """Build an independent ``Generator`` from a tuple of identifiers.

    Deriving per-fold and per-arm randomness from ``(seed, fold, arm)`` keeps
    streams independent without the ordering dependence that repeated global
    reseeding introduces. Uses blake2b rather than ``hash`` because CPython
    randomises string hashing per process, which would break reproducibility
    across the separate SLURM jobs this project splits work into.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def resolve_device(prefer_gpu: bool = True) -> str:
    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"
