#!/usr/bin/env python3
"""Thesis figures: full training curves and synthetic waveform panels.

Reads fold JSON records (training histories) and optional artifact arrays
(decoded/<arm>_figure.npy, decoded/real_figure.npy).

Examples::

    python eeg_qgan/scripts/plot_thesis_figures.py \\
        --results-root eeg_qgan_results_thesis_figures/thesis_figure_capture

    python eeg_qgan/scripts/plot_thesis_figures.py \\
        --results-root eeg_qgan_results_paper_style_cpu/paper_style_cpu_deadline \\
        --pfe-copy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parents[1]
THESIS = REPO.parent
DEFAULT_PFE = THESIS / "Engineering_PFE" / "Figures" / "results_deadline"

INK = "#1a1a1a"
EDGE = "#333333"
COL = {
    "real": "#2C5F8D",
    "quantum": "#7B2D8E",
    "classical_equiv": "#C0392B",
    "gmm": "#27AE60",
    "baseline": "#555555",
    "augmented_quantum": "#7B2D8E",
    "augmented_classical_equiv": "#C0392B",
    "augmented_gmm": "#27AE60",
}
LABEL = {
    "quantum": "Hybrid Style-qGAN",
    "classical_equiv": "Equal-budget MLP",
    "gmm": "Gaussian mixture",
    "real": "Real seizure",
    "baseline": "Real only",
    "augmented_quantum": "Quantum augmented",
    "augmented_classical_equiv": "Twin augmented",
    "augmented_gmm": "GMM augmented",
}


PRINT_W = 7.0


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.labelcolor": INK,
            "axes.edgecolor": EDGE,
            "axes.linewidth": 1.05,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out / f"{name}{ext}", facecolor="white", dpi=600)
    plt.close(fig)
    print(f"wrote {out / name}.pdf")


def _load_rows(root: Path) -> list[dict]:
    rows = []
    paths = sorted(root.glob("results_*.json"))
    all_path = root / "results_all.json"
    if all_path.exists() and all_path not in paths:
        paths.append(all_path)
    for f in paths:
        blob = json.loads(f.read_text(encoding="utf-8"))
        rows.extend(blob.get("rows", []))
    return rows


def _mean_curve(points: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    if not points:
        return [], []
    by_x: dict[int, list[float]] = {}
    for x, y in points:
        by_x.setdefault(int(x), []).append(float(y))
    xs = sorted(by_x)
    return xs, [float(np.mean(by_x[x])) for x in xs]


def fig_autoencoder(rows: list[dict], out: Path) -> None:
    train_pts, val_pts, nmse_pts = [], [], []
    for r in rows:
        for pt in (r.get("autoencoder") or {}).get("history") or []:
            ep = int(pt["epoch"])
            train_pts.append((ep, float(pt["train_loss"])))
            val_pts.append((ep, float(pt["val_loss"])))
            nmse_pts.append((ep, float(pt["val_nmse"])))

    if not train_pts:
        print("skip AE curves: no history")
        return

    xe, tr = _mean_curve(train_pts)
    _, va = _mean_curve(val_pts)
    _, nm = _mean_curve(nmse_pts)

    fig, axes = plt.subplots(2, 1, figsize=(PRINT_W, 7.2), constrained_layout=True)
    axes[0].plot(xe, tr, color=COL["quantum"], lw=2.4, label="Train")
    axes[0].plot(xe, va, color=COL["classical_equiv"], lw=2.4, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction loss (MSE)")
    axes[0].set_title("Autoencoder")
    axes[0].legend(frameon=False, loc="best")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))

    axes[1].plot(xe, nm, color=INK, lw=2.4)
    axes[1].axhline(0.55, color=COL["classical_equiv"], ls="--", lw=1.6, label="Gate (0.55)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation NMSE")
    axes[1].set_title("Reconstruction gate")
    axes[1].legend(frameon=False, loc="best")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    lo = min(min(nm), 0.55) - 0.03
    hi = max(max(nm), 0.55) + 0.03
    axes[1].set_ylim(lo, hi)

    fig.suptitle("Autoencoder training — capture study, full 80 epochs", fontsize=13)
    _save(fig, out, "fig_training_ae_full")


def fig_wgan(rows: list[dict], out: Path) -> None:
    arms = ("quantum", "classical_equiv")
    fig, axes = plt.subplots(2, 1, figsize=(PRINT_W, 7.4), constrained_layout=True)
    any_data = False

    for arm in arms:
        g_pts, c_pts, mmd_pts = [], [], []
        for r in rows:
            arm_blob = (r.get("arms") or {}).get(arm) or {}
            for pt in arm_blob.get("history") or []:
                step = int(pt["step"])
                g_pts.append((step, float(pt["generator_loss"])))
                c_pts.append((step, -float(pt["critic_loss"])))
                if "selection_distance" in pt:
                    mmd_pts.append((step, float(pt["selection_distance"])))
        if not g_pts:
            continue
        any_data = True
        xs, gm = _mean_curve(g_pts)
        _, cm = _mean_curve(c_pts)
        lab = LABEL.get(arm, arm)
        axes[0].plot(xs, gm, lw=2.4, color=COL[arm], label=f"{lab} generator")
        axes[0].plot(xs, cm, lw=1.8, ls="--", color=COL[arm], alpha=0.9, label=f"{lab} |critic|")
        if mmd_pts:
            xm, mm = _mean_curve(mmd_pts)
            axes[1].plot(xm, mm, lw=2.4, marker="o", ms=6, color=COL[arm], label=lab)

    if not any_data:
        print("skip WGAN curves: no history")
        return

    axes[0].set_xlabel("Generator update")
    axes[0].set_ylabel("Adversarial loss")
    axes[0].set_title("WGAN-GP losses")
    axes[0].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    axes[1].set_xlabel("Generator update")
    axes[1].set_ylabel("Holdout selection MMD")
    axes[1].set_title("Checkpoint selection metric")
    axes[1].legend(frameon=False, loc="best")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    fig.suptitle("Latent WGAN-GP training — capture study", fontsize=13)
    _save(fig, out, "fig_training_wgan_full")


def fig_detector(rows: list[dict], out: Path) -> None:
    keys = [
        ("baseline", "baseline"),
        ("augmented_quantum", "augmented_quantum"),
        ("augmented_classical_equiv", "augmented_classical_equiv"),
        ("augmented_gmm", "augmented_gmm"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(PRINT_W, 10.2), constrained_layout=True)
    plotted = False

    for source, col_key in keys:
        loss_pts, auprc_pts, auroc_pts = [], [], []
        for r in rows:
            det = (r.get("detector") or {}).get(source) or {}
            for pt in det.get("history") or []:
                ep = int(pt["epoch"])
                loss_pts.append((ep, float(pt["train_loss"])))
                auprc_pts.append((ep, float(pt.get("val_auprc", np.nan))))
                auroc_pts.append((ep, float(pt.get("val_auroc", np.nan))))
        if not loss_pts:
            continue
        plotted = True
        lab = LABEL.get(source, source)
        xe, tl = _mean_curve(loss_pts)
        _, ap = _mean_curve([(a, b) for a, b in auprc_pts if np.isfinite(b)])
        _, ar = _mean_curve([(a, b) for a, b in auroc_pts if np.isfinite(b)])
        axes[0].plot(xe, tl, lw=2.4, color=COL[col_key], label=lab)
        if ap:
            axes[1].plot(xe[: len(ap)], ap, lw=2.4, color=COL[col_key], label=lab)
        if ar:
            axes[2].plot(xe[: len(ar)], ar, lw=2.4, color=COL[col_key], label=lab)

    if not plotted:
        print("skip detector curves: no history (rerun thesis_figure_capture)")
        return

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train focal loss")
    axes[0].set_title("Detector optimisation")
    axes[0].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation AUPRC")
    axes[1].set_title("Early-stopping metric")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Validation AUROC")
    axes[2].set_title("Validation ranking")
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    fig.suptitle("STFT–BiLSTM detector training — capture fold 0", fontsize=13)
    _save(fig, out, "fig_training_detector_full")


def _artifact_dir(root: Path, row: dict) -> Path | None:
    art = row.get("artifacts") or {}
    rel = art.get("dir")
    if not rel:
        return None
    return root / rel


def fig_ae_recon_panels(root: Path, rows: list[dict], out: Path, channels: list[int]) -> None:
    """Real window vs AE reconstruction from the capture checkpoint."""
    if not rows:
        return
    adir = _artifact_dir(root, rows[0])
    if adir is None or not adir.exists():
        print("skip AE recon: no artifacts")
        return
    real_path = adir / "decoded" / "real_figure.npy"
    ckpt_path = adir / "autoencoder.pt"
    if not real_path.exists() or not ckpt_path.exists():
        print("skip AE recon: missing real_figure.npy or autoencoder.pt")
        return

    try:
        import torch
        from eeg_qgan.models.autoencoder import MultichannelAE
    except Exception as exc:  # pragma: no cover
        print(f"skip AE recon: {exc}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    desc = ckpt.get("description") or {}
    arch = desc.get("architecture") or desc
    model = MultichannelAE(
        n_channels=int(arch.get("n_channels", 22)),
        sample_length=int(arch.get("sample_length", 256)),
        latent_dim=int(arch.get("latent_dim", 64)),
        bottleneck=str(arch.get("bottleneck", "temporal")),
        widths=tuple(arch.get("widths") or (128, 256, 256)),
        arch=str(arch.get("arch", "v2")),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    real = np.load(real_path)[0]
    with torch.no_grad():
        x = torch.from_numpy(real[None].astype(np.float32))
        recon_t, _z = model(x, denoise_decode=False)
        recon = recon_t[0].cpu().numpy()
        if recon.shape != real.shape:
            recon = model.decode(model.encode(x), denoise=False)[0].cpu().numpy()

    fs = 200.0
    t = np.arange(real.shape[-1]) / fs
    channels = channels[:2]
    fig, axes = plt.subplots(len(channels), 2, figsize=(PRINT_W, 2.4 * len(channels)), sharex=True, constrained_layout=True)
    if len(channels) == 1:
        axes = np.atleast_2d(axes)

    for i, ch in enumerate(channels):
        ymax = max(
            float(np.percentile(np.abs(real[ch]), 99.5)),
            float(np.percentile(np.abs(recon[ch]), 99.5)),
        ) * 1.25
        for j, (sig, title, color) in enumerate(
            (
                (real[ch], "Real", COL["real"]),
                (recon[ch], "Reconstruction", COL["quantum"]),
            )
        ):
            ax = axes[i, j]
            ax.plot(t, sig, color=color, lw=1.7)
            ax.set_ylim(-ymax, ymax)
            ax.set_title(f"{title} · ch {ch}", fontsize=12)
            if j == 0:
                ax.set_ylabel("Amplitude")
            if i == len(channels) - 1:
                ax.set_xlabel("Time (s)")
            ax.set_xlim(float(t[0]), float(t[-1]))

    fig.suptitle("Autoencoder reconstruction — capture fold 0 (gate evidence)", fontsize=13)
    _save(fig, out, "fig_ae_recon_panels")


def fig_synthetic_waveforms(root: Path, rows: list[dict], out: Path, channels: list[int]) -> None:
    if not rows:
        return
    row = rows[0]
    adir = _artifact_dir(root, row)
    if adir is None or not adir.exists():
        print("skip waveform panel: artifacts/ not found (sync from Narval or rerun capture study)")
        return

    real_path = adir / "decoded" / "real_figure.npy"
    if not real_path.exists():
        print(f"skip waveform panel: missing {real_path.name}")
        return

    real = np.load(real_path)
    fs = 200.0
    t = np.arange(real.shape[-1]) / fs

    arms = [
        ("real", real[0:1], COL["real"], "Real"),
        ("quantum", adir / "decoded" / "quantum_figure.npy", COL["quantum"], "Quantum"),
        ("classical_equiv", adir / "decoded" / "classical_equiv_figure.npy", COL["classical_equiv"], "Twin"),
        ("gmm", adir / "decoded" / "gmm_figure.npy", COL["gmm"], "GMM"),
    ]

    n_ch = min(2, len(channels))
    channels = channels[:n_ch]
    fig, axes = plt.subplots(
        len(arms),
        n_ch,
        figsize=(PRINT_W, 2.55 * len(arms)),
        sharex=True,
        constrained_layout=True,
    )
    if n_ch == 1:
        axes = np.atleast_2d(axes)
    if len(arms) == 1:
        axes = axes.reshape(1, -1)

    windows = []
    for key, src, _color, _title in arms:
        if key == "real":
            windows.append(real[0])
        else:
            p = Path(src)
            windows.append(np.load(p)[0] if p.exists() else None)

    col_lim = []
    for ch in channels:
        finite = [w[ch] for w in windows if w is not None]
        col_lim.append(max(float(np.percentile(np.abs(s), 99.5)) for s in finite) * 1.25)

    for i, (_key, _src, color, title) in enumerate(arms):
        window = windows[i]
        if window is None:
            for ax in np.atleast_1d(axes[i]):
                ax.axis("off")
            continue
        for j, ch in enumerate(channels):
            ax = axes[i, j]
            ax.plot(t, window[ch], color=color, lw=1.7)
            ymax = col_lim[j]
            ax.set_ylim(-ymax, ymax)
            ax.set_yticks([-ymax * 0.7, 0.0, ymax * 0.7])
            ax.tick_params(labelsize=10)
            if i == 0:
                ax.set_title(f"Channel {ch}", fontsize=13, pad=8)
            if j == 0:
                ax.set_ylabel(title, fontsize=12)
            else:
                ax.tick_params(labelleft=False)
            ax.set_xlim(float(t[0]), float(t[-1]))
            if i == len(arms) - 1:
                ax.set_xlabel("Time (s)")

    fig.suptitle("Decoded 1.28 s seizure windows — capture fold 0, shared AE", fontsize=13)
    _save(fig, out, "fig_synthetic_waveforms")


def fig_synthetic_spectrogram(root: Path, rows: list[dict], out: Path, channel: int = 0) -> None:
    if not rows:
        return
    adir = _artifact_dir(root, rows[0])
    if adir is None or not adir.exists():
        return
    real_path = adir / "decoded" / "real_figure.npy"
    q_path = adir / "decoded" / "quantum_figure.npy"
    if not real_path.exists() or not q_path.exists():
        print("skip spectrogram panel: missing figure arrays")
        return

    real = np.load(real_path)[0, channel]
    synth = np.load(q_path)[0, channel]
    fs = 200.0

    def spec(x: np.ndarray) -> np.ndarray:
        n_fft, hop = 128, 64
        frames = []
        for start in range(0, len(x) - n_fft + 1, hop):
            frame = x[start : start + n_fft] * np.hanning(n_fft)
            frames.append(np.abs(np.fft.rfft(frame)))
        return np.log1p(np.asarray(frames).T)

    s_real, s_syn = spec(real), spec(synth)
    vmax = max(s_real.max(), s_syn.max())

    fig, axes = plt.subplots(2, 1, figsize=(PRINT_W, 7.0), constrained_layout=True)
    for ax, data, title in (
        (axes[0], s_real, "Real seizure"),
        (axes[1], s_syn, LABEL["quantum"]),
    ):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=vmax,
            extent=[0, len(real) / fs, 0, fs / 2],
            interpolation="nearest",
            rasterized=True,
        )
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="log magnitude")
    fig.suptitle(f"Spectrogram comparison — capture fold 0, channel {channel}", fontsize=13)
    _save(fig, out, "fig_synthetic_spectrogram_quantum")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Study output dir containing results_*.json (and artifacts/ if synced)",
    )
    p.add_argument("--out", type=Path, default=REPO / "figures" / "thesis_figures")
    p.add_argument("--pfe-copy", action="store_true", help="Also copy PNG/PDF to Engineering_PFE/Figures/results_deadline")
    p.add_argument("--channels", default="0,7", help="Channel indices for waveform panel")
    args = p.parse_args()

    _style()
    root = args.results_root.resolve()
    rows = _load_rows(root)
    if not rows:
        print(f"No results JSON under {root}")
        return 1

    out = args.out.resolve()
    chs = [int(c.strip()) for c in str(args.channels).split(",") if c.strip()]

    fig_autoencoder(rows, out)
    fig_wgan(rows, out)
    fig_detector(rows, out)
    fig_ae_recon_panels(root, rows, out, chs)
    fig_synthetic_waveforms(root, rows, out, chs)
    fig_synthetic_spectrogram(root, rows, out, channel=chs[0])

    if args.pfe_copy:
        import shutil

        pfe = DEFAULT_PFE.resolve()
        pfe.mkdir(parents=True, exist_ok=True)
        for src in out.glob("fig_*.*"):
            if src.suffix.lower() in {".png", ".pdf"}:
                shutil.copy2(src, pfe / src.name)
        print(f"PFE copy -> {pfe}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
