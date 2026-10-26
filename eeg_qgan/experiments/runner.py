"""The single cross-validation driver.

One entry point replaces the ten copies of ``run_latent_qgan_cv.py`` that were spread
across the old directories. Which study runs is a configuration choice, and which slice
of it this process executes is a command-line choice, so splitting a long job across
SLURM submissions no longer requires forking a config.

The audit fixes that belong to the driver rather than to a component:

E6: the outer splitter is constructed inside the seed loop. Building it once meant every
seed saw identical folds, so ten seed-by-fold rows were five partitions counted twice.
The fold identifier is recorded per row so the clustering is available to the analysis
regardless.

E1: the grouping is validated before any training happens, and a grouping that looks
like fixed-size index chunks stops the run.

E4: three disjoint group-respecting splits per fold. The generator trains on one,
its checkpoint is chosen on the second, and the reported distance and the quality-gate
reference come from the third.

E3: the autoencoder gate decides. A fold whose decoder fails is either fatal or recorded
as excluded, never silently used.

F1: one latent coordinate system is fitted per fold and handed to every arm, so the
leaderboard compares distances measured the same way.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from eeg_qgan.config import RunSelection, list_studies, load_config
from eeg_qgan.data import (
    assert_grouping_is_plausible,
    group_summary,
    load_windows,
    make_outer_splitter,
    make_predefined_folds,
    nested_latent_splits,
)
from eeg_qgan.data.clips import (
    build_clips_from_indices,
    clinical_cfg,
    clinical_enabled,
    load_clinical_clips,
    pack_synthetic_clips,
)
from eeg_qgan.evaluation.augmentation import DoseSpec, select_for_augmentation
from eeg_qgan.evaluation.decoded import evaluate_decoded, sliced_wasserstein
from eeg_qgan.evaluation.mmd import mmd_noise_floor
from eeg_qgan.models.mixture import build_mixture
from eeg_qgan.models.registry import capacity_audit
from eeg_qgan.seeding import resolve_device, set_seed, spawn_rng
from eeg_qgan.training.adversarial import (
    decode_codes,
    report_distance,
    sample_latent_codes,
    train_adversarial,
)
from eeg_qgan.training.autoencoder import encode_windows, train_autoencoder
from eeg_qgan.training.detector import train_and_evaluate_detector
from eeg_qgan.training.latent_space import fit_latent_transform, support_diagnostics
from eeg_qgan.utils.artifacts import save_fold_artifacts, save_model_checkpoint
from eeg_qgan.utils.io import ensure_dir, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one latent generative study with grouped cross-validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("study", nargs="?", help=f"study name; available: {', '.join(list_studies())}")
    p.add_argument("--config", default=None, help="explicit path to a study YAML")
    p.add_argument("--seed-index", type=int, default=None, help="run only this seed index")
    p.add_argument("--fold-start", type=int, default=0, help="first fold index, inclusive")
    p.add_argument("--fold-end", type=int, default=None, help="last fold index, exclusive")
    p.add_argument("--arms", nargs="*", default=None, help="override the generator arms")
    p.add_argument("--output-root", default=None, help="where results are written")
    p.add_argument("--data-root", default=None, help="override data.root")
    p.add_argument("--skip-downstream", action="store_true", help="fidelity metrics only")
    p.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "override any config key, dotted for nested keys, value parsed as YAML: "
            "--set augmentation.dose_count=2048 latent.train_fraction=0.25"
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="validate config and data, then stop")
    p.add_argument("--list-studies", action="store_true")
    return p.parse_args(argv)


def _assign_dotted(target: dict, dotted_key: str, value: Any) -> None:
    """Write ``value`` at ``a.b.c`` inside a nested dict, creating levels as needed."""
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ValueError("empty override key")
    node = target
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.arms:
        overrides["generator_arms"] = list(args.arms)
    if args.data_root:
        overrides.setdefault("data", {})["root"] = args.data_root
    if args.output_root:
        overrides["output_root"] = args.output_root
    if args.skip_downstream:
        overrides["skip_downstream"] = True
    # Sweeps over dose and training fraction were documented as "vary it through an
    # overlay or a launcher loop" without any mechanism to do so, which is how a study
    # ends up reporting a single point as a curve.
    for item in args.overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        key, raw = item.split("=", 1)
        _assign_dotted(overrides, key.strip(), yaml.safe_load(raw))
    return overrides


def _run_detector_block(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    cfg: dict,
    device: str,
    seed: int,
    fold: int,
    synthetic_by_arm: dict[str, np.ndarray],
    n_pos: int,
    n_neg: int,
    real_reference: np.ndarray,
    label: str,
) -> dict:
    """Train baseline + augmented detectors; optional clinical clip packing."""
    clin = clinical_cfg(cfg)
    dose = DoseSpec.from_cfg(cfg)
    detector_records: dict = {}
    detector_models: dict[str, object] = {}

    baseline = train_and_evaluate_detector(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        cfg=cfg,
        device=device,
        synthetic=None,
        groups_train=groups_train,
        rng=spawn_rng(seed, fold, "detector_baseline"),
    )
    detector_records["baseline"] = baseline.as_dict()
    detector_models["baseline"] = baseline
    print(
        f"      detector baseline F1={baseline.f1:.4f} AUROC={baseline.auroc:.4f} "
        f"AUPRC={baseline.auprc:.4f} MCC={baseline.mcc:.4f}",
        flush=True,
    )

    n_clip = int(clin.get("clip_windows", 1)) if clinical_enabled(cfg) else 1
    for arm, decoded in synthetic_by_arm.items():
        # Under the clinical protocol the detector sees clips, but generators emit short
        # windows. dose_mode: balance / proportional are therefore resolved in *clip*
        # units (n_pos / n_neg are clip counts) and then multiplied by n_clip so the
        # packed dose matches the paper-style class-balance request. dose_mode: fixed
        # stays a short-window count (see study headers).
        sel_cfg = cfg
        if clinical_enabled(cfg) and n_clip > 1 and dose.mode in {"balance", "proportional"}:
            # Resolve the class-balance request in clip units first (n_pos/n_neg are
            # clip counts). Cap with max_add interpreted as max synthetic *clips*.
            n_avail_clips = max(0, len(decoded) // n_clip)
            desired_clips, _ = dose.resolve(
                n_positive=n_pos,
                n_negative=n_neg,
                n_available=n_avail_clips,
            )
            short_needed = int(desired_clips) * n_clip
            sel_cfg = copy.deepcopy(cfg)
            sel_cfg.setdefault("augmentation", {})
            sel_cfg["augmentation"]["dose_mode"] = "fixed"
            sel_cfg["augmentation"]["dose_count"] = short_needed
            # max_add was a clip ceiling above; do not re-apply it as a short-window cap.
            sel_cfg["augmentation"]["max_add"] = short_needed
        selection = select_for_augmentation(
            synthetic=decoded,
            real_reference=real_reference,
            cfg=sel_cfg,
            n_positive=n_pos,
            n_negative=n_neg,
            score_fn=None,
            rng=spawn_rng(seed, fold, f"select_{arm}"),
        )
        synth = selection.selected
        if clinical_enabled(cfg) and n_clip > 1 and synth is not None:
            packed = pack_synthetic_clips(
                synth, n_clip, crossfade=int(clin.get("synthetic_crossfade", 0))
            )
            if packed is None:
                print(
                    f"      detector +{arm}: skipped — fewer than {n_clip} synthetic "
                    "windows to pack into clinical clips",
                    flush=True,
                )
                continue
            synth = packed
        result = train_and_evaluate_detector(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            cfg=cfg,
            device=device,
            synthetic=synth,
            augmentation_meta=selection.meta,
            groups_train=groups_train,
            rng=spawn_rng(seed, fold, f"detector_{arm}"),
        )
        key = f"augmented_{arm}"
        detector_records[key] = result.as_dict()
        detector_models[key] = result
        print(
            f"      detector +{arm} F1={result.f1:.4f} AUROC={result.auroc:.4f} "
            f"(delta F1 {result.f1 - baseline.f1:+.4f}, "
            f"delta AUROC {result.auroc - baseline.auroc:+.4f}, "
            f"dose {result.n_synthetic_added})",
            flush=True,
        )

    return {
        "detector": detector_records,
        "detector_models": detector_models,
        "dose_spec": {
            "mode": dose.mode,
            "target_ratio": dose.target_ratio,
            "count": dose.count,
            "multiplier": dose.multiplier,
        },
    }


def run_fold(
    windows,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: dict,
    seed: int,
    seed_index: int,
    fold: int,
    device: str,
    out_dir: Path | None = None,
) -> dict:
    """Execute one fold: autoencoder, every arm, fidelity, then the detector."""
    label = f"seed{seed_index}_fold{fold}"
    fold_rng = spawn_rng(seed, fold, "fold")
    arms = list(cfg["generator_arms"])
    latent_cfg = cfg.get("latent", {}) or {}
    clin = clinical_cfg(cfg)

    y_train = windows.y[train_idx]
    groups_train = windows.groups[train_idx]

    record: dict = {
        "seed_index": seed_index,
        "seed": seed,
        "fold": fold,
        # The cluster identifier for the statistics: rows sharing a partition are not
        # independent, so the analysis must group by this rather than by row.
        "partition_id": f"fold{fold}",
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_positive": int((y_train == 1).sum()),
        "n_test_positive": int((windows.y[test_idx] == 1).sum()),
        "train_groups": sorted(set(str(g) for g in groups_train)),
        "test_groups": sorted(set(str(g) for g in windows.groups[test_idx])),
    }
    # A patient appearing on both sides would invalidate the whole protocol.
    leaked = set(record["train_groups"]) & set(record["test_groups"])
    if leaked:
        raise RuntimeError(
            f"{label}: {len(leaked)} groups appear in both train and test: "
            f"{sorted(leaked)[:5]}. Grouped cross-validation is broken."
        )

    # ---- clinical detector-only ceiling (long clips + STFT; no generative stage) ----
    if clinical_enabled(cfg) and bool(clin.get("detector_only", False)):
        print(
            f"    clinical detector-only: input_length={windows.x.shape[-1]} "
            f"(~{windows.x.shape[-1] / float(cfg.get('target_fs', 200)):.2f}s) "
            f"detector={((cfg.get('detector') or {}).get('kind'))}",
            flush=True,
        )
        block = _run_detector_block(
            x_train=windows.x[train_idx],
            y_train=y_train,
            x_test=windows.x[test_idx],
            y_test=windows.y[test_idx],
            groups_train=groups_train,
            cfg=cfg,
            device=device,
            seed=seed,
            fold=fold,
            synthetic_by_arm={},
            n_pos=int((y_train == 1).sum()),
            n_neg=int((y_train == 0).sum()),
            real_reference=windows.x[train_idx[: min(512, len(train_idx))]],
            label=label,
        )
        record["detector"] = block["detector"]
        record["dose_spec"] = block["dose_spec"]
        record["clinical"] = {
            "enabled": True,
            "detector_only": True,
            "clip_length": int(windows.x.shape[-1]),
            "duration_seconds": float(windows.x.shape[-1]) / float(cfg.get("target_fs", 200)),
            "provenance": (windows.provenance or {}).get("clinical"),
        }
        return record

    # ---- three disjoint splits over the seizure windows of this fold ----
    seizure_positions = np.where(y_train == 1)[0]
    if len(seizure_positions) < 12:
        raise RuntimeError(
            f"{label}: only {len(seizure_positions)} seizure windows in the training "
            "fold, too few to split three ways. Reduce n_folds."
        )
    splits = nested_latent_splits(
        indices=train_idx[seizure_positions],
        groups=groups_train[seizure_positions],
        select_size=float(latent_cfg.get("select_size", cfg.get("latent_select_size", 0.15))),
        report_size=float(latent_cfg.get("report_size", cfg.get("latent_report_size", 0.15))),
        seed=seed,
    )
    record["latent_splits"] = splits.sizes()
    print(
        f"    splits: generator train={len(splits.train)} "
        f"checkpoint select={len(splits.select)} report={len(splits.report)} "
        f"(disjoint by {windows.group_key})",
        flush=True,
    )

    # ---- autoencoder, on training-fold windows only ----
    ae_cfg = cfg.get("autoencoder", {}) or {}
    ae_train_idx = train_idx if bool(ae_cfg.get("train_all_classes", True)) else splits.train
    ae, ae_meta = train_autoencoder(
        x_train=windows.x[ae_train_idx],
        x_val=windows.x[splits.report],
        cfg=cfg,
        device=device,
        fold_label=label,
    )
    record["autoencoder"] = ae_meta
    if ae_meta["excluded"]:
        record["excluded"] = True
        record["exclusion_reason"] = ae_meta["exclusion_reason"]
        if out_dir is not None:
            record["artifacts"] = save_fold_artifacts(
                out_dir=out_dir,
                seed_index=seed_index,
                fold=fold,
                cfg=cfg,
                ae=ae,
                ae_meta=ae_meta,
                transform=None,
                z_report=None,
                arm_models={},
                fold_record=record,
            )
        return record

    # ---- encode, then fit the one coordinate system every arm shares ----
    z_train = encode_windows(ae, windows.x[splits.train], device)
    z_select = encode_windows(ae, windows.x[splits.select], device)
    z_report = encode_windows(ae, windows.x[splits.report], device)

    # Data-efficiency study: reduce generator training codes only. Select/report
    # stay full so the measurement precision is comparable across fractions.
    train_fraction = float(latent_cfg.get("train_fraction", 1.0))
    if train_fraction < 1.0:
        from eeg_qgan.evaluation.augmentation import subsample_codes

        n_before = int(len(z_train))
        z_train = subsample_codes(
            z_train, train_fraction, spawn_rng(seed, fold, "train_fraction")
        )
        record["latent_train_fraction"] = {
            "requested": train_fraction,
            "n_before": n_before,
            "n_after": int(len(z_train)),
            "delivered_fraction": float(len(z_train) / max(1, n_before)),
        }
        print(
            f"    latent.train_fraction={train_fraction}: "
            f"{n_before} → {len(z_train)} generator-train codes",
            flush=True,
        )

    transform = fit_latent_transform(
        z_train,
        mode=str(cfg.get("latent_transform", "identity")),
        std_floor=float(latent_cfg.get("std_floor", 1e-2)),
        robust=bool(latent_cfg.get("robust_scale", True)),
    )
    record["latent_transform"] = transform.as_dict()
    record["latent_support"] = support_diagnostics(z_report, transform)
    print(
        f"    latent transform={transform.mode} "
        f"reach={record['latent_support']['fraction_within_reach']:.3f} "
        f"worst coordinate={record['latent_support']['worst_coordinate_coverage']:.3f}",
        flush=True,
    )

    # The reference every arm is read against: real against real, same estimator, same n.
    record["mmd_noise_floor"] = mmd_noise_floor(
        np.asarray(transform.to_generator_space(z_report), dtype=np.float32),
        n_repeats=int(cfg.get("noise_floor_repeats", 50)),
        rng=spawn_rng(seed, fold, "floor"),
    )

    # ---- arms ----
    arm_records: dict = {}
    synthetic_by_arm: dict[str, np.ndarray] = {}
    arm_models: dict[str, dict] = {}
    n_synth = int(cfg.get("n_synthetic", 1024))

    for arm in arms:
        arm_rng = spawn_rng(seed, fold, arm)
        print(f"    arm {arm}", flush=True)
        t_arm = time.perf_counter()

        if arm == "gmm":
            mixture = build_mixture(cfg, seed=seed)
            fit_meta = mixture.fit(z_train, transform=transform)
            codes = mixture.sample(n_synth, transform=transform, rng=arm_rng)
            arm_report = {
                "arm": arm,
                "fit": fit_meta,
                "generator": mixture.describe(),
                "wall_seconds": time.perf_counter() - t_arm,
            }
            # GMM has no torch module; the fitted parameters are in fit_meta / describe.
            arm_models[arm] = {
                "generator": None,
                "critic": None,
                "codes": codes,
                "selection_step": None,
                "selection_distance": None,
                "gmm": mixture,
            }
        else:
            generator, critic, training_report = train_adversarial(
                z_train=z_train,
                z_select=z_select,
                arm=arm,
                cfg=cfg,
                device=device,
                transform=transform,
                rng=arm_rng,
            )
            codes = sample_latent_codes(generator, n_synth, device, transform=transform)
            arm_report = training_report.as_dict()
            arm_models[arm] = {
                "generator": generator,
                "critic": critic,
                "codes": codes,
                "selection_step": training_report.selection_step,
                "selection_distance": training_report.selection_distance,
            }

        # Reported distance: unbiased, matched n, on the split nothing else touched.
        arm_report["latent_distance"] = report_distance(
            z_report, codes, transform=transform, rng=arm_rng
        )
        # Second reading of the same comparison, sensitive to cross-coordinate
        # dependence where the RBF-MMD headline is weak (claim C5).
        arm_report["latent_sliced_wasserstein"] = sliced_wasserstein(
            np.asarray(transform.to_generator_space(z_report), dtype=np.float64),
            np.asarray(transform.to_generator_space(codes), dtype=np.float64),
            seed=int(seed) + int(fold),
        )
        decoded = decode_codes(ae, codes, device)
        synthetic_by_arm[arm] = decoded
        arm_models[arm]["decoded"] = decoded
        arm_report["decoded"] = evaluate_decoded(
            windows.x[splits.report], decoded, fs=float(cfg.get("target_fs", 200))
        )
        arm_report["n_synthetic"] = int(len(decoded))
        arm_records[arm] = arm_report

        print(
            f"      distance={arm_report['latent_distance']['mmd2']:+.5f} "
            f"floor={record['mmd_noise_floor']['mean']:+.5f} "
            f"acf={arm_report['decoded']['autocorrelation_distance']:.4f}",
            flush=True,
        )

    record["arms"] = arm_records

    # Persist weights and latents before the detector, so a crash in the detector
    # stage does not lose the generative artefacts.
    if out_dir is not None:
        save_cfg = cfg.get("save", {}) or {}
        figure_n = int(save_cfg.get("decoded_figure_windows", 0) or 0)
        real_figure = None
        if figure_n > 0:
            report_y = np.asarray(windows.y[splits.report], dtype=np.int64)
            pos_idx = np.flatnonzero(report_y == 1)
            if len(pos_idx):
                take = pos_idx[:figure_n]
                real_figure = np.asarray(windows.x[splits.report][take], dtype=np.float32)

        record["artifacts"] = save_fold_artifacts(
            out_dir=out_dir,
            seed_index=seed_index,
            fold=fold,
            cfg=cfg,
            ae=ae,
            ae_meta=ae_meta,
            transform=transform,
            z_report=z_report,
            arm_models=arm_models,
            fold_record=record,
            real_figure_windows=real_figure,
        )
        print(f"    artifacts -> {record['artifacts'].get('dir', 'disabled')}", flush=True)

        # GMM has no torch weights; keep its fitted parameters next to the other arms.
        if bool((cfg.get("save", {}) or {}).get("artifacts", True)):
            for arm, bundle in arm_models.items():
                mixture = bundle.get("gmm")
                if mixture is None:
                    continue
                gmm_dir = ensure_dir(
                    Path(out_dir) / "artifacts" / f"seed{seed_index}_fold{fold}" / "gmm"
                )
                gmm_path = gmm_dir / f"{arm}.json"
                payload = {
                    "arm": arm,
                    "description": mixture.describe(),
                    "fit": arm_records[arm].get("fit"),
                }
                gmm = getattr(mixture, "_gmm", None)
                if gmm is not None and getattr(gmm, "weights_", None) is not None:
                    payload["weights"] = np.asarray(gmm.weights_).tolist()
                    payload["means"] = np.asarray(gmm.means_).tolist()
                    payload["covariances"] = np.asarray(gmm.covariances_).tolist()
                save_json(payload, gmm_path)
                record.setdefault("artifacts", {}).setdefault("gmm", {})[arm] = (
                    gmm_path.relative_to(out_dir).as_posix()
                )

    # ---- downstream detector ----
    if bool(cfg.get("skip_downstream", False)):
        return record

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    if clinical_enabled(cfg):
        n_clip = int(clin.get("clip_windows", 8))
        stride = max(1, int(clin.get("clip_stride", max(1, n_clip // 2))))
        label_mode = str(clin.get("label_mode", "any"))
        require_contig = bool(clin.get("require_contiguous", True))
        train_clips = build_clips_from_indices(
            windows,
            train_idx,
            n_windows=n_clip,
            stride=stride,
            label_mode=label_mode,
            require_contiguous_file_order=require_contig,
        )
        test_clips = build_clips_from_indices(
            windows,
            test_idx,
            n_windows=n_clip,
            stride=stride,
            label_mode=label_mode,
            require_contiguous_file_order=require_contig,
        )
        print(
            f"    clinical clips: train={len(train_clips.y)} test={len(test_clips.y)} "
            f"length={train_clips.meta['clip_length']} "
            f"(~{train_clips.meta['clip_length'] / float(cfg.get('target_fs', 200)):.2f}s) "
            f"skipped_gaps train/test="
            f"{train_clips.meta['n_skipped_noncontiguous']}/"
            f"{test_clips.meta['n_skipped_noncontiguous']}",
            flush=True,
        )
        record["clinical"] = {
            "enabled": True,
            "detector_only": False,
            "train": train_clips.meta,
            "test": test_clips.meta,
            "duration_seconds": float(train_clips.meta["clip_length"])
            / float(cfg.get("target_fs", 200)),
        }
        det_x_train, det_y_train = train_clips.x, train_clips.y
        det_x_test, det_y_test = test_clips.x, test_clips.y
        det_groups = train_clips.groups
        # Reference for selection stays short-window report split (same as rematch).
        real_reference = windows.x[splits.report]
        # Dose accounting uses short-window class counts; clip packing happens inside.
        n_pos_det = int((det_y_train == 1).sum())
        n_neg_det = int((det_y_train == 0).sum())
    else:
        det_x_train = windows.x[train_idx]
        det_y_train = y_train
        det_x_test = windows.x[test_idx]
        det_y_test = windows.y[test_idx]
        det_groups = groups_train
        real_reference = windows.x[splits.report]
        n_pos_det, n_neg_det = n_pos, n_neg

    block = _run_detector_block(
        x_train=det_x_train,
        y_train=det_y_train,
        x_test=det_x_test,
        y_test=det_y_test,
        groups_train=det_groups,
        cfg=cfg,
        device=device,
        seed=seed,
        fold=fold,
        synthetic_by_arm=synthetic_by_arm,
        n_pos=n_pos_det,
        n_neg=n_neg_det,
        real_reference=real_reference,
        label=label,
    )
    detector_records = block["detector"]
    detector_models = block["detector_models"]
    record["detector"] = detector_records
    record["dose_spec"] = block["dose_spec"]

    # Detector weights are optional (off by default): they are useful for reuse but
    # multiply disk use by the number of arms.
    if (
        out_dir is not None
        and bool((cfg.get("save", {}) or {}).get("detectors", False))
        and bool((cfg.get("save", {}) or {}).get("artifacts", True))
    ):
        det_dir = ensure_dir(
            Path(out_dir) / "artifacts" / f"seed{seed_index}_fold{fold}" / "detectors"
        )
        det_paths: dict[str, str] = {}
        for name, result in detector_models.items():
            model = getattr(result, "model", None)
            if model is None:
                continue
            path = save_model_checkpoint(
                model,
                det_dir / f"{name}.pt",
                kind=f"detector:{name}",
                extra={"threshold": getattr(result, "threshold", None)},
            )
            det_paths[name] = Path(path).relative_to(out_dir).as_posix()
        record.setdefault("artifacts", {})["detectors"] = det_paths

    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_studies:
        print("Available studies:")
        for name in list_studies():
            print(f"  {name}")
        return 0
    if not args.study and not args.config:
        print("error: give a study name or --config", file=sys.stderr)
        return 2

    cfg = load_config(args.config or args.study, overrides=_build_overrides(args))
    selection = RunSelection(
        seed_index=args.seed_index,
        fold_start=args.fold_start,
        fold_end=args.fold_end,
        arms=args.arms,
    )

    device = resolve_device(bool(cfg.get("use_gpu", True)))
    out_dir = ensure_dir(Path(cfg.get("output_root", "results")) / str(cfg["study"]))
    n_seeds = int(cfg["n_seeds"])
    n_folds = int(cfg["n_folds"])

    print("=" * 78)
    print(f"study            {cfg['study']}")
    print(f"arms             {', '.join(cfg['generator_arms'])}")
    print(f"latent_dim       {cfg['latent_dim']}   transform {cfg.get('latent_transform')}")
    print(f"folds            {n_folds}   seeds {n_seeds}   device {device}")
    print(f"group_key        {cfg['group_key']}")
    print(f"output           {out_dir}")
    print("=" * 78, flush=True)

    windows = (
        load_clinical_clips(cfg)
        if clinical_enabled(cfg) and bool(clinical_cfg(cfg).get("detector_only", False))
        else load_windows(cfg)
    )
    if clinical_enabled(cfg):
        clin = clinical_cfg(cfg)
        print(
            f"clinical protocol  enabled clip_windows={clin.get('clip_windows', 8)} "
            f"detector_only={bool(clin.get('detector_only', False))} "
            f"input_length={windows.x.shape[-1]} "
            f"(~{windows.x.shape[-1] / float(cfg.get('target_fs', 200)):.2f}s)",
            flush=True,
        )
    split_mode = str((cfg.get("data") or {}).get("split_mode", cfg.get("split_mode", "predefined")))

    if split_mode == "predefined":
        # Outer protocol is train vs holdout folders. Grouping checks are for the
        # file-level groups used only inside nested generator splits.
        diagnostics = {
            "split_mode": "predefined",
            "group_key": "file",
            "protocol": (windows.provenance or {}).get("protocol"),
            "holdouts_present": (windows.provenance or {}).get("holdouts_present"),
            "n_groups": int(len(set(str(g) for g in windows.groups))),
            "warning": (
                "Outer evaluation is patient-disjoint by the TUH train/dev/eval folder "
                "structure. Inner train/select/report splits group by batch file because "
                "per-window patient ids are not present in BinaryClassification_Data."
            ),
        }
    else:
        diagnostics = assert_grouping_is_plausible(
            groups=windows.groups,
            y=windows.y,
            n_splits=n_folds,
            group_key=str(cfg["group_key"]),
            min_group_size=int(cfg.get("min_group_size", 25)),
        )
    print("data:", json.dumps(windows.summary(), indent=2), flush=True)
    print("grouping:", json.dumps(diagnostics, indent=2), flush=True)

    audit = capacity_audit(list(cfg["generator_arms"]), cfg)
    save_json(audit, out_dir / "capacity_audit.json")
    save_json(cfg, out_dir / "config_resolved.json")
    if not audit["noise_dim_consistent"]:
        raise SystemExit(audit["noise_dim_warning"])
    print(
        "capacity: quantum target "
        f"{audit['quantum_target_params']} params, shared noise width "
        f"{audit['shared_noise_dim']}",
        flush=True,
    )

    if args.dry_run:
        save_json(
            {
                "config": cfg,
                "data": windows.summary(),
                "grouping": diagnostics,
                "capacity": audit,
                "group_summary": group_summary(windows.groups, windows.y),
            },
            out_dir / "dry_run.json",
        )
        print(f"\nDry run complete. Wrote {out_dir / 'dry_run.json'}")
        return 0

    tag = selection.tag()
    results_path = out_dir / f"results_{tag}.json"
    rows: list[dict] = []

    for seed_index in selection.seed_indices(n_seeds):
        seed = int(cfg["seed"]) + seed_index
        set_seed(seed, deterministic=bool(cfg.get("deterministic", True)))

        if split_mode == "predefined":
            holdouts = (cfg.get("data") or {}).get("holdouts", cfg.get("holdouts", ("dev", "eval")))
            predefined = make_predefined_folds(windows, holdouts=holdouts)
            fold_indices = selection.fold_indices(len(predefined))
            fold_iter = [
                (fold, train_idx, test_idx, fold_name)
                for fold, (train_idx, test_idx, fold_name) in enumerate(predefined)
                if fold in fold_indices
            ]
        else:
            # Rebuilt per seed: this is the E6 fix. Constructing it once outside the loop
            # gave every seed identical folds and turned replicates into duplicates.
            splitter = make_outer_splitter(n_folds, seed=seed)
            fold_indices = selection.fold_indices(n_folds)
            fold_iter = [
                (fold, train_idx, test_idx, f"fold{fold}")
                for fold, (train_idx, test_idx) in enumerate(
                    splitter.split(windows.x, windows.y, windows.groups)
                )
                if fold in fold_indices
            ]

        for fold, train_idx, test_idx, fold_name in fold_iter:
            print(
                f"\n--- seed {seed_index} (value {seed}) fold {fold} ({fold_name}) ---",
                flush=True,
            )
            row = run_fold(
                windows=windows,
                train_idx=train_idx,
                test_idx=test_idx,
                cfg=cfg,
                seed=seed,
                seed_index=seed_index,
                fold=fold,
                device=device,
                out_dir=out_dir,
            )
            row["fold_name"] = fold_name
            rows.append(row)
            # Written after every fold so a job killed by the scheduler keeps its work.
            save_json(
                {
                    "study": cfg["study"],
                    "config": cfg,
                    "selection": {
                        "seed_index": selection.seed_index,
                        "fold_start": selection.fold_start,
                        "fold_end": selection.fold_end,
                    },
                    "data_summary": windows.summary(),
                    "grouping": diagnostics,
                    "capacity_audit": audit,
                    "rows": rows,
                    "completed": len(rows),
                },
                results_path,
            )
            print(f"    checkpoint written ({len(rows)} rows) -> {results_path}", flush=True)

    n_excluded = sum(1 for r in rows if r.get("excluded"))
    print(f"\nDone. {len(rows)} rows, {n_excluded} excluded. Results at {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
