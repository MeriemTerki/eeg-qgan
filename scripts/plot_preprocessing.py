#!/usr/bin/env python3
"""Before/after preprocessing figures for the thesis Implementation chapter.

Shows the same multichannel window at each stage of the loader:

  raw  ->  resampled (T=256)  ->  db6 wavelet denoised  ->  bounded normalisation

One seizure window and one non-seizure window are drawn side by side, with a few
channels stacked so the figure is readable in print. Colours follow the thesis pastel
palette (dark text, soft fills).

Examples::

    # On synthetic data (works offline, no cluster needed)
    python scripts/plot_preprocessing.py --data-root data/synthetic --out figures/preprocess

    # On the real TUH batches (run on Narval once sidecars exist)
    python scripts/plot_preprocessing.py \\
        --data-root "$DATA_ROOT" \\
        --out "$OUTPUT_ROOT/figures/preprocess" \\
        --channels 0,3,7,12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eeg_qgan.data.tuh import (  # noqa: E402
    LoaderConfig,
    NON_SEIZURE_DIR_NAMES,
    SEIZURE_DIR_NAMES,
    _iter_data_files,
    _load_array,
    _normalize_bounded,
    _resolve_class_dir,
    preprocess_stages,
    resolve_patient_ids,
    robust_range,
)

INK = "#000000"
GRID = "#D8D8D8"
AXIS = "#4A4A4A"
STAGE_COLOUR = {
    "raw": "#C9C9C9",
    "resampled": "#A9C8E8",
    "denoised": "#B7D8B1",
    "normalized": "#F3B9B5",
}
STAGE_EDGE = {
    "raw": "#3A3A3A",
    "resampled": "#2C5F8D",
    "denoised": "#3D6B39",
    "normalized": "#A6403B",
}
STAGE_TITLE = {
    "raw": "Raw loaded window",
    "resampled": "Resampled to T = 256",
    "denoised": "Daubechies-6 soft threshold",
    "normalized": "Bounded normalisation [-1, 1]",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": AXIS,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _recording_range(batch: np.ndarray, rng: np.random.Generator, cap: int = 400) -> tuple[float, float]:
    """One robust range per file, matching the loader's recording-scope sample."""
    n = int(batch.shape[0])
    take = min(int(cap), n)
    if take == n:
        return robust_range(batch)
    idx = rng.choice(n, size=take, replace=False)
    return robust_range(batch[idx])


def _window_rail_peak(window: np.ndarray, norm_range: tuple[float, float] | None) -> float:
    """Median-over-channels of max |normalised| amplitude. Used only to pick a readable window."""
    peaks = [
        float(np.max(np.abs(_normalize_bounded(window[c], norm_range))))
        for c in range(window.shape[0])
    ]
    return float(np.median(peaks))


def _select_readable_index(
    batch: np.ndarray,
    norm_range: tuple[float, float] | None,
    rng: np.random.Generator,
    target_q: float,
) -> int:
    """Pick a window near ``target_q`` of file-rail occupancy, dropping the top 2% artefacts."""
    n = int(batch.shape[0])
    cand = np.arange(n) if n <= 800 else rng.choice(n, size=800, replace=False)
    scores = np.asarray([_window_rail_peak(batch[i], norm_range) for i in cand], dtype=np.float64)
    order = np.argsort(scores)
    usable = order[: max(1, int(0.98 * len(order)))]
    k = int(np.clip(round(float(target_q) * (len(usable) - 1)), 0, len(usable) - 1))
    return int(cand[usable[k]])


def _pick_window(
    data_root: Path,
    label: int,
    sample_length: int,
    n_channels: int | None,
    seed: int,
    *,
    target_q: float,
    min_peak: float = 0.08,
) -> tuple[np.ndarray, str, tuple[float, float] | None]:
    """Load one raw window chosen so the recording-scope trace is visible on a slide."""
    cfg = LoaderConfig(
        data_root=str(data_root),
        sample_length=sample_length,
        n_channels=n_channels,
        denoise=False,
        normalize=False,
        require_patient_groups=False,
        max_batch_files=6,
        max_segments_per_batch=40,
        seed=seed,
    )
    root = Path(cfg.data_root)
    seizure_dir = _resolve_class_dir(root, cfg.seizure_dir, SEIZURE_DIR_NAMES)
    non_seizure_dir = _resolve_class_dir(root, cfg.non_seizure_dir, NON_SEIZURE_DIR_NAMES)
    directory = seizure_dir if label == 1 else non_seizure_dir
    files = list(_iter_data_files(directory))
    if not files:
        raise FileNotFoundError(f"No batch files under {directory}")

    rng = np.random.default_rng(seed + label)
    order = rng.permutation(len(files))
    best: tuple[float, np.ndarray, str, tuple[float, float]] | None = None

    for file_i in order[:6]:
        path = files[int(file_i)]
        batch = _load_array(path)
        if batch.ndim != 3:
            raise ValueError(f"Expected (N,C,T) in {path}, got {batch.shape}")
        batch = np.asarray(batch, dtype=np.float32)
        file_range = _recording_range(batch, rng)
        idx = _select_readable_index(batch, file_range, rng, target_q)
        peak = _window_rail_peak(batch[idx], file_range)
        try:
            ids, source = resolve_patient_ids(path, batch.shape[0], "patient")
        except Exception:
            ids = [f"unknown_{i}" for i in range(batch.shape[0])]
            source = "unavailable"
        meta = (
            f"{path.name}  window {idx}  patient {ids[idx]}  "
            f"(id source: {source}; rail peak {peak:.3f})"
        )
        window = np.asarray(batch[idx], dtype=np.float32)
        if best is None or peak > best[0]:
            best = (peak, window, meta, file_range)
        if peak >= min_peak:
            return window, meta, file_range

    assert best is not None
    return best[1], best[2], best[3]


def _time_axis(n: int, fs: float) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / float(fs)


def draw_multichannel(
    ax,
    window: np.ndarray,
    channels: list[int],
    colour: str,
    edge: str,
    fs: float,
    offset: float = 2.4,
) -> None:
    """Stack a few channels with a vertical offset so they do not overlap."""
    t = _time_axis(window.shape[1], fs)
    for k, ch in enumerate(channels):
        if ch >= window.shape[0]:
            continue
        sig = window[ch].astype(np.float64)
        # Per-channel display scale so a quiet channel is still visible next to a loud one.
        scale = np.percentile(np.abs(sig), 99) + 1e-8
        y = sig / scale + (len(channels) - 1 - k) * offset
        ax.plot(t, y, color=edge, lw=0.9, solid_capstyle="round")
        ax.fill_between(
            t,
            y - 0.15,
            y + 0.15,
            color=colour,
            alpha=0.25,
            linewidth=0,
        )
        ax.text(
            -0.02,
            (len(channels) - 1 - k) * offset,
            f"ch {ch}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0.0, t[-1] if len(t) else 1.0)
    ax.set_ylim(-offset * 0.6, (len(channels) - 0.4) * offset)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")


def make_panel_figure(
    seizure_stages: dict[str, np.ndarray],
    nonseizure_stages: dict[str, np.ndarray],
    channels: list[int],
    fs: float,
    seizure_meta: str,
    nonseizure_meta: str,
    out_dir: Path,
) -> None:
    apply_style()
    stage_order = ("raw", "resampled", "denoised", "normalized")
    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(10.5, 11.0),
        sharex="col",
        constrained_layout=True,
    )

    for row, stage in enumerate(stage_order):
        for col, (stages, title_prefix) in enumerate(
            (
                (seizure_stages, "Seizure"),
                (nonseizure_stages, "Non-seizure"),
            )
        ):
            ax = axes[row, col]
            draw_multichannel(
                ax,
                stages[stage],
                channels,
                colour=STAGE_COLOUR[stage],
                edge=STAGE_EDGE[stage],
                fs=fs if stage != "raw" else fs,  # raw may differ in length; still plot in seconds
                offset=2.6 if stage != "normalized" else 2.2,
            )
            if row == 0:
                ax.set_title(f"{title_prefix}", fontweight="bold", color=INK, pad=6)
            if col == 0:
                ax.set_ylabel(STAGE_TITLE[stage], color=INK)

    fig.suptitle(
        "EEG preprocessing pipeline (same windows through every stage)",
        fontsize=12,
        fontweight="bold",
        color=INK,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        f"Seizure: {seizure_meta}\nNon-seizure: {nonseizure_meta}\n"
        "Stages: raw loaded window → resample to T=256 → Daubechies-6 soft-threshold "
        "denoising → robust bounded normalisation to [-1, 1].",
        ha="center",
        va="top",
        fontsize=7.5,
        color=INK,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"fig_preprocess_before_after.{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


def make_overlay_figure(
    stages: dict[str, np.ndarray],
    channel: int,
    fs: float,
    label: str,
    out_dir: Path,
) -> None:
    """Resampled vs db6, then recording-scope rails, then a zoom of the same red trace."""
    apply_style()
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7.2, 6.6), sharex=True, constrained_layout=True)
    y = stages["normalized"][channel].astype(np.float64)
    peak = float(np.max(np.abs(y)))
    t = _time_axis(stages["resampled"].shape[1], fs)

    ax = axes[0]
    ax.plot(t, stages["resampled"][channel], color=STAGE_EDGE["resampled"], lw=1.0, label="Resampled")
    ax.plot(t, stages["denoised"][channel], color=STAGE_EDGE["denoised"], lw=1.0, label="Denoised (db6)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"{label}: channel {channel}", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ax.plot(t, y, color=STAGE_EDGE["normalized"], lw=1.0, label="Normalised (file rails)")
    ax.axhline(1.0, color=AXIS, ls="--", lw=0.7)
    ax.axhline(-1.0, color=AXIS, ls="--", lw=0.7)
    ax.set_ylabel("File rails")
    ax.set_ylim(-1.15, 1.15)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"Recording-scope $[-1,1]$, this window peak = {peak:.3f}", fontsize=10)

    ax = axes[2]
    ax.plot(t, y, color=STAGE_EDGE["normalized"], lw=1.1, label="Same trace, zoomed")
    zoom = max(1.25 * peak, 0.04)
    ax.set_ylim(-zoom, zoom)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Zoom")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Zoom under the rails (waveform the autoencoder actually sees)", fontsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "fig_preprocess_overlay_nonseizure"
        if label.lower().startswith("non")
        else "fig_preprocess_overlay_seizure"
    )
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="BinaryClassification_Data or synthetic root")
    p.add_argument("--out", default="figures/preprocess", help="output directory")
    p.add_argument("--sample-length", type=int, default=256)
    p.add_argument("--target-fs", type=float, default=200.0)
    p.add_argument("--n-channels", type=int, default=22)
    p.add_argument("--channels", default="0,3,7,12", help="comma-separated channel indices to plot")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--normalize-scope",
        choices=("recording", "window"),
        default="recording",
        help="recording = one robust range per batch file (clinical pin); "
        "window = one range per channel per window (legacy)",
    )
    args = p.parse_args()

    channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    data_root = Path(args.data_root)
    out_dir = Path(args.out)

    # Seizure: a louder window so the red trace occupies a visible fraction of the rails.
    # Non-seizure: a typical window so the two classes do not both fill [-1, 1].
    seizure_raw, seizure_meta, seiz_range = _pick_window(
        data_root,
        label=1,
        sample_length=args.sample_length,
        n_channels=args.n_channels,
        seed=args.seed,
        target_q=0.88,
    )
    ns_raw, ns_meta, ns_range = _pick_window(
        data_root,
        label=0,
        sample_length=args.sample_length,
        n_channels=args.n_channels,
        seed=args.seed,
        target_q=0.42,
    )
    if args.normalize_scope != "recording":
        seiz_range = None
        ns_range = None

    seizure_stages = preprocess_stages(
        seizure_raw, args.sample_length, denoise=True, normalize=True, norm_range=seiz_range
    )
    ns_stages = preprocess_stages(
        ns_raw, args.sample_length, denoise=True, normalize=True, norm_range=ns_range
    )

    # For the raw column, plot the original length; for later stages use target_fs.
    # If raw length differs from 256, its time axis still uses target_fs as an
    # approximation of the source rate (batches are already near 200 Hz).
    make_panel_figure(
        seizure_stages,
        ns_stages,
        channels=channels,
        fs=args.target_fs,
        seizure_meta=seizure_meta,
        nonseizure_meta=ns_meta,
        out_dir=out_dir,
    )
    make_overlay_figure(
        seizure_stages, channel=channels[0], fs=args.target_fs, label="Seizure", out_dir=out_dir
    )
    make_overlay_figure(
        ns_stages, channel=channels[0], fs=args.target_fs, label="Non-seizure", out_dir=out_dir
    )

    # Machine-readable note for the caption / Implementation chapter.
    note = out_dir / "preprocess_caption.txt"
    note.write_text(
        "Preprocessing pipeline applied identically to every arm.\n"
        "1. Load multichannel window from BinaryClassification_Data.\n"
        "2. Resample each channel to T = 256 samples (1.28 s at 200 Hz).\n"
        "3. Daubechies-6 wavelet soft-threshold denoising (universal threshold).\n"
        f"4. Robust bounded normalisation to [-1, 1] (scope={args.normalize_scope}, "
        "percentiles 0.5 and 99.5).\n"
        f"Seizure window: {seizure_meta}\n"
        f"Non-seizure window: {ns_meta}\n",
        encoding="utf-8",
    )
    print(f"wrote {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
