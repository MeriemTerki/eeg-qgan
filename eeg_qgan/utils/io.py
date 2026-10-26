"""Filesystem and JSON helpers.

``save_json`` writes atomically and coerces numpy scalars, because a result file that
is half-written when the scheduler kills a job is worse than no file at all, and a
``TypeError`` on a numpy float at the end of a twelve-hour run loses the whole run.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NumpyEncoder(json.JSONEncoder):
    """Serialise the numpy and torch scalars that metric functions return."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            value = float(obj)
            # JSON has no literal for these, so record them as strings rather than
            # emitting invalid JSON that nothing downstream can parse.
            return value if np.isfinite(value) else str(value)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        if hasattr(obj, "as_dict"):
            return obj.as_dict()
        return super().default(obj)


def _sanitise(obj: Any) -> Any:
    """Replace non-finite floats so the output is valid JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    return obj


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write JSON atomically: to a temporary file, then replace the target."""
    path = Path(path)
    ensure_dir(path.parent)
    payload = json.dumps(_sanitise(data), indent=indent, cls=_NumpyEncoder)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
