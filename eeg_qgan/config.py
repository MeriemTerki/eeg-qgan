"""Configuration loading, layering and validation.

The old layout kept 28 YAML files that were mostly copies of one another,
differing only in a seed index or a fold subset so that long jobs could be split
across SLURM submissions. Here a study is one overlay on top of ``base.yaml``,
and job splitting is a command-line concern (``--seed-index``, ``--fold-range``)
rather than a reason to fork a config.

Validation is deliberately strict. Several audit findings were configuration keys
that silently did nothing (a top-k that was always overridden, a synthetic ratio
that was unreachable whenever the gate was on). ``validate`` refuses to run in
those states instead of quietly ignoring them.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = REPO_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` onto a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def load_config(
    study: str | Path,
    overrides: dict[str, Any] | None = None,
    base: str | Path | None = None,
) -> dict:
    """Load ``base.yaml`` then layer a study overlay and CLI overrides on top.

    ``study`` may be a bare study name (resolved under ``configs/studies``) or a
    path to a YAML file. An overlay may itself declare ``extends:`` to inherit
    from another overlay, which is how the seed and fold variants of a study are
    expressed without duplication.
    """
    base_path = Path(base) if base is not None else CONFIG_DIR / "base.yaml"
    cfg = _read_yaml(base_path)

    overlay_path = _resolve_study_path(study)
    chain: list[dict] = []
    seen: set[Path] = set()
    while overlay_path is not None:
        if overlay_path in seen:
            raise ConfigError(f"Circular 'extends' chain at {overlay_path}")
        seen.add(overlay_path)
        overlay = _read_yaml(overlay_path)
        chain.append(overlay)
        parent = overlay.pop("extends", None)
        overlay_path = _resolve_study_path(parent) if parent else None

    for overlay in reversed(chain):
        cfg = _deep_merge(cfg, overlay)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    cfg.setdefault("study", Path(_resolve_study_path(study)).stem)
    validate(cfg)
    return cfg


def _resolve_study_path(study: str | Path | None) -> Path | None:
    if study is None:
        return None
    p = Path(study)
    if p.suffix in {".yaml", ".yml"} and p.is_file():
        return p.resolve()
    for candidate in (
        CONFIG_DIR / "studies" / f"{study}.yaml",
        CONFIG_DIR / "studies" / f"{study}.yml",
        CONFIG_DIR / f"{study}.yaml",
    ):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Unknown study '{study}'. Available: "
        + ", ".join(sorted(p.stem for p in (CONFIG_DIR / 'studies').glob('*.yaml')))
    )


def list_studies() -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / "studies").glob("*.yaml"))


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

REQUIRED_KEYS = (
    "latent_dim",
    "n_channels",
    "sample_length",
    "n_folds",
    "n_seeds",
    "seed",
    "generator_arms",
)


def validate(cfg: dict) -> None:
    """Fail loudly on configurations that cannot mean what they appear to mean."""
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"Missing required config keys: {', '.join(missing)}")

    latent_dim = int(cfg["latent_dim"])
    n_qubits = int(cfg.get("n_qubits", latent_dim))
    if n_qubits > latent_dim:
        # Audit F/A22: this used to be clamped silently, so a typo changed the
        # circuit with no trace in the log.
        raise ConfigError(
            f"n_qubits ({n_qubits}) exceeds latent_dim ({latent_dim}). "
            "Set them equal for an all-quantum generator, or lower n_qubits. "
            "A silent clamp used to hide this."
        )

    if int(cfg["n_folds"]) < 2:
        raise ConfigError("n_folds must be at least 2")

    _validate_latent_space(cfg)
    _validate_quality_gate(cfg)
    _validate_splits(cfg)
    _validate_grouping(cfg)
    _validate_split_mode(cfg)


def _validate_latent_space(cfg: dict) -> None:
    """Audit E2: bounded generators cannot cover a unit-variance target."""
    mode = str(cfg.get("latent_transform", "identity")).lower()
    allowed = {"identity", "whiten", "whiten_scaled"}
    if mode not in allowed:
        raise ConfigError(f"latent_transform must be one of {sorted(allowed)}, got '{mode}'")

    if mode == "whiten" and not bool(cfg.get("generator_output_scale", False)):
        raise ConfigError(
            "latent_transform: whiten standardises the target to unit variance while "
            "every generator is bounded to (-1, 1), so roughly a third of the target "
            "mass is unreachable and the reported distance carries an irreducible "
            "floor. Use latent_transform: identity (the encoder is already bounded to "
            "the same range), or latent_transform: whiten_scaled, which gives the "
            "generator an output scale that covers the standardised support."
        )

    if "latent_whiten" in cfg:
        raise ConfigError(
            "latent_whiten is retired because it silently created the range mismatch "
            "described above. Use latent_transform: identity | whiten_scaled."
        )


def _validate_quality_gate(cfg: dict) -> None:
    """Audit E5: the gate must be reachable and the dose must be explicit."""
    aug = cfg.get("augmentation", {}) or {}
    # Prefer the nested keys used by the current package; keep legacy top-level
    # aliases so old overlays still fail loudly rather than silently no-op.
    gate_on = bool(aug.get("quality_gate", cfg.get("quality_gate_augmentation", False)))
    if gate_on:
        min_efs = float(aug.get("min_score", cfg.get("min_efs_for_augment", 0.0)))
        if not 0.0 <= min_efs <= 1.0:
            raise ConfigError(f"augmentation.min_score must lie in [0, 1], got {min_efs}")

        fallback = str(aug.get("fallback", cfg.get("quality_gate_fallback", "error"))).lower()
        if fallback not in {"error", "median", "all"}:
            raise ConfigError(
                "augmentation.fallback must be 'error', 'median' or 'all'. The old code "
                "silently kept the top quarter of the requested count while ignoring the "
                "threshold, which is how a 200-sample request became 50 samples that all "
                "failed the fidelity criterion."
            )

        if cfg.get("synthetic_ratio") is not None and cfg.get("quality_gate_top_k"):
            raise ConfigError(
                "synthetic_ratio and quality_gate_top_k are both set, but with the gate "
                "enabled only one can take effect. Set exactly one, or use "
                "augmentation.dose_count instead."
            )

    mode = str(aug.get("dose_mode", "fixed")).lower()
    n_synth = int(cfg.get("n_synthetic", 0))
    max_add = aug.get("max_add")
    clin = cfg.get("clinical") or {}
    detector_only = bool(clin.get("enabled")) and bool(clin.get("detector_only"))
    if mode == "fixed":
        dose_count = int(aug.get("dose_count", 0))
        if dose_count <= 0 and not detector_only:
            raise ConfigError("augmentation.dose_mode=fixed requires dose_count > 0")
        if detector_only and dose_count <= 0:
            # Real-only long-clip clinical ceiling: no synthetic dose.
            pass
        elif n_synth < dose_count:
            raise ConfigError(
                f"n_synthetic={n_synth} < augmentation.dose_count={dose_count}. "
                "Generators must produce at least as many windows as the detector dose."
            )
        elif max_add is not None and int(max_add) < dose_count:
            raise ConfigError(
                f"augmentation.max_add={max_add} < dose_count={dose_count}; "
                "the dose would be truncated before selection."
            )
    elif mode == "balance":
        # Soft guard: balance under TUH typically requests tens of thousands.
        # Runs with a tiny generation budget will raise at selection time via
        # max_shortfall_fraction; warn at config time when the budget is clearly
        # too small to be a genuine balance experiment.
        if n_synth < 16000:
            raise ConfigError(
                "dose_mode: balance with n_synthetic < 16000 is not publishable under "
                "TUH predefined splits (ae_v5b requested ~30k and delivered only the "
                "generation budget). Use dose_mode: fixed with an explicit dose_count, "
                "or raise n_synthetic / max_add to cover the true balance request."
            )


def _validate_splits(cfg: dict) -> None:
    """Audit E4: checkpoint selection and reporting must not share a split."""
    latent = cfg.get("latent", {}) or {}
    frac_select = float(latent.get("select_size", cfg.get("latent_select_size", 0.0)))
    frac_report = float(latent.get("report_size", cfg.get("latent_report_size", 0.0)))
    if frac_select <= 0.0 or frac_report <= 0.0:
        raise ConfigError(
            "Both latent_select_size and latent_report_size must be positive. "
            "Selecting the generator checkpoint on the same split used to report the "
            "distance makes every reported value best-of-N optimistic, and the bias "
            "differs per arm."
        )
    if frac_select + frac_report >= 0.8:
        raise ConfigError(
            f"latent_select_size + latent_report_size = {frac_select + frac_report:.2f} "
            "leaves too little data for generator training"
        )


def _validate_grouping(cfg: dict) -> None:
    """Audit E1: refuse to run unless the grouping key is stated explicitly."""
    key = str(cfg.get("group_key", "")).lower()
    allowed = {"patient", "recording", "file"}
    if key not in allowed:
        raise ConfigError(
            f"group_key must be one of {sorted(allowed)}, got '{key}'. This was "
            "previously an unstated fallback that grouped by window index, which "
            "splits one recording across folds and turns the reported score into "
            "within-recording interpolation."
        )


def _validate_split_mode(cfg: dict) -> None:
    """Outer protocol must be stated: TUH folders vs patient-id GroupKFold."""
    data = cfg.get("data") or {}
    mode = str(data.get("split_mode", cfg.get("split_mode", "predefined"))).lower()
    allowed = {"predefined", "grouped_cv"}
    if mode not in allowed:
        raise ConfigError(
            f"data.split_mode must be one of {sorted(allowed)}, got '{mode}'. "
            "predefined = train vs holdout folders (BinaryClassification_Data); "
            "grouped_cv = StratifiedGroupKFold on per-window patient sidecars."
        )
    if mode == "grouped_cv" and bool(
        data.get("require_patient_groups", cfg.get("require_patient_groups", True))
    ):
        if str(cfg.get("group_key", "")).lower() != "patient":
            raise ConfigError(
                "grouped_cv with require_patient_groups needs group_key: patient"
            )
    if mode == "predefined":
        holdouts = data.get("holdouts", cfg.get("holdouts", ("dev", "eval")))
        if isinstance(holdouts, str):
            holdouts = [h.strip() for h in holdouts.split(",") if h.strip()]
        if not holdouts:
            raise ConfigError(
                "split_mode predefined requires at least one holdout "
                "(data.holdouts, e.g. [dev, eval])"
            )


@dataclass
class RunSelection:
    """Which slice of a study this process should execute."""

    seed_index: int | None = None
    fold_start: int = 0
    fold_end: int | None = None
    arms: list[str] | None = None

    def seed_indices(self, n_seeds: int) -> list[int]:
        if self.seed_index is None:
            return list(range(n_seeds))
        if not 0 <= self.seed_index < n_seeds:
            raise ConfigError(
                f"seed_index {self.seed_index} out of range for n_seeds={n_seeds}"
            )
        return [self.seed_index]

    def fold_indices(self, n_folds: int) -> list[int]:
        end = n_folds if self.fold_end is None else min(self.fold_end, n_folds)
        if self.fold_start < 0 or self.fold_start >= end:
            raise ConfigError(
                f"Invalid fold range: [{self.fold_start}, {end}) with n_folds={n_folds}"
            )
        return list(range(self.fold_start, end))

    def tag(self) -> str:
        """A filename-safe tag so split jobs write to distinct files."""
        parts = []
        if self.seed_index is not None:
            parts.append(f"s{self.seed_index}")
        if self.fold_start or self.fold_end is not None:
            parts.append(f"f{self.fold_start}-{self.fold_end if self.fold_end is not None else 'end'}")
        return "_".join(parts) if parts else "all"
