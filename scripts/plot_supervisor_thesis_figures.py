#!/usr/bin/env python3
"""Supervisor figure notes F1–F5 for the Engineering PFE thesis.

    python eeg_qgan/scripts/plot_supervisor_thesis_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Patch

THESIS = Path(__file__).resolve().parents[2]
OUT = THESIS / "Engineering_PFE" / "Figures" / "results_deadline"

INK = "#1F2A33"
EDGE = "#333333"
NAVY = "#1F4E79"
SOFT = "#E8EEF4"
# Sized for A4 thesis at \linewidth without height scaling (see packages.tex result include).
FS_BASE = 13
FS_AXIS = 13
FS_TICK = 12
FS_TITLE = 14
FS_PANEL = 13
FS_SUP = 15
FS_BAR = 11
FS_LEGEND = 11
FS_F1_BOX = 10.5
FS_F1_BULLET = 10
FS_F1_HEAD = 13
COLS = {
    "hybrid": "#2C6E7A",
    "twin": "#C0392B",
    "gmm": "#27AE60",
    "fourier": "#A08968",
    "product": "#8FB4BC",
    "linear": "#5A8F99",
    "ring": "#2C6E7A",
    "full": "#1E4D56",
    "large": "#A67A74",
    "small": "#5E7A6A",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
            "font.size": FS_BASE,
            "axes.titlesize": FS_TITLE,
            "axes.labelsize": FS_AXIS,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "axes.labelcolor": INK,
            "axes.edgecolor": EDGE,
            "axes.linewidth": 1.0,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(OUT / f"{name}{ext}", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def _grouped_bar_labels(
    ax,
    groups: list[list[tuple]],
    lanes: tuple[float, float, float] = (0.002, 0.009, 0.016),
    inside_frac: float = 0.42,
    fontsize: int = 10,
) -> None:
    """Thesis-style grouped labels: fixed vertical lanes per arm + inside-bar when tall enough."""
    panel_max = max((e[1] for g in groups for e in g), default=0.01)
    for group in groups:
        gmax = max((e[1] for e in group), default=0.0)
        inside_thresh = max(inside_frac * max(gmax, panel_max), 0.006)
        for i, (bar, h, lab) in enumerate(group):
            if not np.isfinite(h):
                continue
            cx = bar.get_x() + bar.get_width() / 2
            if h >= inside_thresh:
                ax.text(
                    cx,
                    h * 0.52,
                    lab,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="white",
                    fontweight="bold",
                )
            else:
                y = max(h, 0.0) + lanes[min(i, len(lanes) - 1)]
                ax.text(
                    cx,
                    y,
                    lab,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="bold",
                )


def _bar_labels(ax, bars, labels, y_pad_frac=0.04, fontsize=FS_BAR, inside_min=0.08) -> None:
    """Place value labels above bars with automatic y-limit padding."""
    ymax = max((b.get_height() for b in bars), default=0.0)
    for bar, lab in zip(bars, labels):
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        if h >= inside_min * ymax and h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 0.55,
                lab,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if h > 0.15 * ymax else INK,
                fontweight="bold",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + y_pad_frac * ymax,
                lab,
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )
    if ymax > 0:
        ax.set_ylim(0, ymax * (1.0 + 3.5 * y_pad_frac))


def fig_f1_fairness_contract() -> None:
    """F1 — pipeline + fairness pin."""
    fig = plt.figure(figsize=(8.2, 7.4))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.15, 1.0], hspace=0.30)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, 2.2)
    ax_top.axis("off")

    stages = [
        (0.15, 0.35, 1.55, 1.35, "TUH-SZ\npatient-disjoint\nfolders"),
        (1.95, 0.35, 1.55, 1.35, "Shared AE\n$d{=}64$\ngate NMSE $\\leq 0.55$"),
        (3.75, 0.35, 1.85, 1.35, "Generators\nHybrid 704 / Twin 709\ncritic $[128,128]$"),
        (5.85, 0.35, 1.45, 1.35, "Pack $\\times 8$\n10.24 s clips"),
        (7.5, 0.35, 1.35, 1.35, "STFT–BiLSTM\nreal holdout only"),
    ]
    for x, y, w, h, txt in stages:
        ax_top.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor=SOFT,
                edgecolor=NAVY,
                linewidth=1.2,
            )
        )
        ax_top.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=FS_F1_BOX, color=INK)
    for x0, x1 in ((1.7, 1.95), (3.5, 3.75), (5.6, 5.85), (7.3, 7.5)):
        ax_top.annotate("", xy=(x1, 1.02), xytext=(x0, 1.02), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.3))
    ax_top.text(5.0, 1.95, "Experimental pipeline (one outer split)", ha="center", fontsize=FS_F1_HEAD, fontweight="bold", color=NAVY)

    ax_bot.set_xlim(0, 10)
    ax_bot.set_ylim(0, 3.2)
    ax_bot.axis("off")
    pins = [
        (0.15, 0.15, 3.05, 2.75, "Fairness pin", [
            "Shared AE, frozen after gate",
            "Matched total budget (82% classical readout)",
            "Shared critic, LR, batch, 1000 steps",
            "Checkpoint by holdout MMD (not WGAN loss)",
        ]),
        (3.45, 0.15, 3.05, 2.75, "Evaluation contract", [
            "Latent / decoded / detection never equated",
            "Synthetic only in detector training",
            "Real TUH dev/eval holdout only",
            "Dose logged (max_add=1024, ~910 clips)",
        ]),
        (6.75, 0.15, 3.05, 2.75, "Controls", [
            "Twin: capacity-matched MLP",
            "GMM: density reference",
            "Fourier: function-class control",
            "Large MLP: only if it converges",
        ]),
    ]
    for x, y, w, h, title, bullets in pins:
        ax_bot.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="white",
                edgecolor=EDGE,
                linewidth=1.0,
            )
        )
        ax_bot.text(x + 0.14, y + h - 0.22, title, ha="left", va="top", fontsize=FS_PANEL, fontweight="bold", color=NAVY)
        for i, line in enumerate(bullets):
            ax_bot.text(x + 0.18, y + h - 0.55 - i * 0.48, f"• {line}", ha="left", va="top", fontsize=FS_F1_BULLET, color=INK)

    fig.suptitle("Experimental contract: pipeline and fairness pin", fontsize=FS_SUP, y=0.98)
    _save(fig, "fig_f1_fairness_contract")


def fig_f2_rematch() -> None:
    """F2 — rematch bars; large MLP hatched failed 4/4."""
    import shutil

    arms = ["GMM", "Hybrid\nStyle-qGAN", "Larger\nMLP", "Equal-budget\ntwin", "Large MLP\nfailed 4/4"]
    means = [0.00085, 0.0158, 0.0214, 0.0883, 0.905]
    errs = [0.00049, 0.0025, 0.0033, 0.0096, 0.057]
    colors = [COLS["gmm"], COLS["hybrid"], COLS["small"], COLS["twin"], COLS["large"]]
    labels = [f"{m:.4f}".rstrip("0").rstrip(".") for m in means]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = np.arange(len(arms))
    bars = ax.bar(
        x,
        means,
        yerr=errs,
        color=colors,
        edgecolor=EDGE,
        linewidth=0.9,
        capsize=3,
        width=0.68,
        error_kw={"ecolor": EDGE, "linewidth": 0.8},
    )
    bars[-1].set_hatch("///")
    bars[-1].set_alpha(0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=FS_TICK)
    ax.set_ylabel(r"Holdout latent MMD$^2$ (lower is better)", fontsize=FS_AXIS)
    ax.set_title("Capacity rematch — matched hybrid win; large MLP is an optimisation failure", fontsize=FS_TITLE, pad=10)
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 2.5)
    for bar, lab, m, e in zip(bars, labels, means, errs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            (m + e) * 1.35,
            lab,
            ha="center",
            va="bottom",
            fontsize=FS_BAR,
        )
    ax.legend(
        handles=[
            Patch(facecolor=COLS["large"], edgecolor=EDGE, hatch="///", label="Failed to converge (4/4)"),
            Patch(facecolor=COLS["hybrid"], edgecolor=EDGE, label="Matched adversarial win"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=FS_LEGEND,
    )
    fig.subplots_adjust(bottom=0.20)
    _save(fig, "fig_latent_mmd_capacity")
    shutil.copy2(OUT / "fig_latent_mmd_capacity.pdf", OUT / "fig_f2_rematch_hatched.pdf")
    shutil.copy2(OUT / "fig_latent_mmd_capacity.png", OUT / "fig_f2_rematch_hatched.png")


def _panel_title(ax, text: str, pad: float = 18) -> None:
    ax.set_title(text, fontsize=FS_PANEL, loc="left", pad=pad, fontweight="bold")


def _hide_x_labels(ax) -> None:
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.tick_params(axis="x", length=0, pad=2)


def _show_x_labels(ax, labels, xlabel: str | None = None, labelpad: float = 12) -> None:
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=FS_TICK)
    ax.tick_params(axis="x", length=3, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_AXIS, labelpad=labelpad)


def fig_f3_decomposition() -> None:
    """F3 — entanglement + Fourier + critic reversal composite."""
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.8))
    fig.subplots_adjust(hspace=0.58, top=0.96, bottom=0.08, left=0.15, right=0.97)

    ax = axes[0]
    labs = ["Product", "Linear", "Ring", "Full", "Twin"]
    vals = [0.0107, 0.0139, 0.0159, 0.0181, 0.0932]
    cols = [COLS["product"], COLS["linear"], COLS["ring"], COLS["full"], COLS["twin"]]
    bars = ax.bar(np.arange(5), vals, color=cols, edgecolor=EDGE, width=0.62)
    _show_x_labels(ax, labs)
    ax.set_ylabel(r"MMD$^2$ $\downarrow$")
    _panel_title(ax, "(a) Entanglement costs")
    _bar_labels(ax, bars, [f"{v:.3f}" for v in vals], y_pad_frac=0.08)

    ax = axes[1]
    labs = ["Fourier", "Hybrid", "Twin"]
    vals = [0.0062, 0.0200, 0.079]
    cols = [COLS["fourier"], COLS["hybrid"], COLS["twin"]]
    bars = ax.bar(np.arange(3), vals, color=cols, edgecolor=EDGE, width=0.58)
    _show_x_labels(ax, labs)
    _panel_title(ax, "(b) Function class ($L{=}4$)")
    _bar_labels(ax, bars, [f"{v:.3f}" for v in vals], y_pad_frac=0.08)

    ax = axes[2]
    x = np.arange(2)
    w = 0.34
    h16 = [0.073, 0.058]
    h128 = [0.017, 0.088]
    b1 = ax.bar(x - w / 2, h16, width=w, color=[COLS["hybrid"], COLS["twin"]], edgecolor=EDGE, label="critic [16,16]")
    b2 = ax.bar(
        x + w / 2,
        h128,
        width=w,
        color=[COLS["hybrid"], COLS["twin"]],
        edgecolor=EDGE,
        alpha=0.45,
        label="critic [128,128]",
    )
    _show_x_labels(ax, ["Hybrid", "Twin"])
    _panel_title(ax, "(c) Critic reverses rematch")
    ax.set_ylim(0, 0.115)
    for bars, vals in ((b1, h16), (b2, h128)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=FS_BAR)
    ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    _save(fig, "fig_f3_decomposition")


def fig_f4_three_levels_gmm() -> None:
    """F4 — three levels stacked vertically; GMM wedge at every level."""
    import shutil

    fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.6))
    fig.subplots_adjust(hspace=0.55, top=0.96, bottom=0.09, left=0.17, right=0.97)
    arms = ["Hybrid", "Twin", "GMM"]
    colors = [COLS["hybrid"], COLS["twin"], COLS["gmm"]]
    x = np.arange(3)

    panels = [
        {
            "title": "Level 1 — Latent fidelity",
            "ylabel": r"Holdout MMD$^2$ $\downarrow$",
            "vals": [0.020, 0.086, 0.0008],
            "stars": [False, False, False],
            "log": True,
            "ylim": (5e-4, 0.22),
            "fmt": lambda v: f"{v:.4f}".rstrip("0").rstrip("."),
            "label_dy": lambda v: v * 1.45,
        },
        {
            "title": "Level 2 — Detection (primary endpoint)",
            "ylabel": r"$\Delta$AUROC vs real-only",
            "vals": [0.000, 0.005, 0.010],
            "stars": [False, True, True],
            "log": False,
            "ylim": (-0.003, 0.015),
            "fmt": lambda v: f"{v:+.3f}",
            "label_dy": lambda v: v + 0.0014,
        },
        {
            "title": "Level 3 — Association (secondary)",
            "ylabel": r"$\Delta$MCC vs real-only",
            "vals": [0.010, 0.029, 0.008],
            "stars": [True, True, True],
            "log": False,
            "ylim": (-0.002, 0.038),
            "fmt": lambda v: f"{v:+.3f}",
            "label_dy": lambda v: v + 0.0022,
        },
    ]

    for i, (ax, spec) in enumerate(zip(axes, panels)):
        bars = ax.bar(x, spec["vals"], color=colors, edgecolor=EDGE, width=0.56)
        if i < len(panels) - 1:
            _hide_x_labels(ax)
        else:
            _show_x_labels(ax, arms)
        ax.set_ylabel(spec["ylabel"], fontsize=FS_AXIS)
        _panel_title(ax, spec["title"], pad=14)
        if spec["log"]:
            ax.set_yscale("log")
        else:
            ax.axhline(0, color=EDGE, lw=0.9)
        ax.set_ylim(*spec["ylim"])
        ax.tick_params(axis="y", labelsize=FS_TICK)
        for bar, v, s in zip(bars, spec["vals"], spec["stars"]):
            lab = spec["fmt"](v) + ("*" if s else "")
            y = spec["label_dy"](v)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                lab,
                ha="center",
                va="bottom",
                fontsize=FS_BAR,
                fontweight="bold",
            )

    _save(fig, "fig_eval_levels_transfer")
    shutil.copy2(OUT / "fig_eval_levels_transfer.pdf", OUT / "fig_f4_three_levels_gmm.pdf")
    shutil.copy2(OUT / "fig_eval_levels_transfer.png", OUT / "fig_f4_three_levels_gmm.png")


def fig_f5_dose_fraction() -> None:
    """F5 — dose and fraction stacked vertically; exploratory 50% noted in caption."""
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.6))
    fig.subplots_adjust(hspace=0.50, top=0.91, bottom=0.12, left=0.13, right=0.98)
    arms = ["Hybrid", "Twin", "GMM"]
    colors = [COLS["hybrid"], COLS["twin"], COLS["gmm"]]
    w = 0.20
    bar_gap = 0.27
    legend_handles = [Patch(facecolor=c, edgecolor=EDGE, label=a) for a, c in zip(arms, colors)]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=FS_LEGEND,
        bbox_to_anchor=(0.53, 0.995),
        columnspacing=1.8,
        handletextpad=0.5,
    )

    def _fmt(v: float, sig: bool) -> str:
        return f"{v:+.2f}" + ("*" if sig else "")

    def _plot_panel(ax, x, data, sig, ylim, title, xlabels=None, xlabel=None):
        groups: list[list[tuple]] = [[] for _ in x]
        for i, arm in enumerate(arms):
            vals = data[arm]
            offsets = (i - 1) * bar_gap
            bars = ax.bar(x + offsets, vals, width=w, color=colors[i], edgecolor=EDGE)
            for j, (bar, v, s) in enumerate(zip(bars, vals, sig[arm])):
                groups[j].append((bar, v, _fmt(v, s)))
        ax.axhline(0, color=EDGE, lw=0.9)
        _grouped_bar_labels(ax, groups)
        ax.set_ylabel(r"$\Delta$AUROC", fontsize=FS_AXIS)
        _panel_title(ax, title, pad=10)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.55, len(x) - 0.45)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        if xlabels is None:
            _hide_x_labels(ax)
        else:
            _show_x_labels(ax, xlabels, xlabel=xlabel)
        return groups

    doses = [227, 455, 910]
    dose_data = {
        "Hybrid": [0.003, 0.008, 0.000],
        "Twin": [0.001, 0.013, 0.005],
        "GMM": [0.008, 0.002, 0.010],
    }
    dose_sig = {
        "Hybrid": [False, False, False],
        "Twin": [False, True, True],
        "GMM": [True, False, True],
    }
    _plot_panel(
        axes[0],
        np.arange(len(doses)),
        dose_data,
        dose_sig,
        ylim=(-0.002, 0.024),
        title="Dose sweep (delivered synthetic clips: 227 / 455 / 910)",
        xlabels=["227", "455", "910"],
        xlabel="Delivered synthetic clips",
    )

    fracs = [0.25, 0.5, 1.0]
    frac_data = {
        "Hybrid": [0.009, 0.019, 0.000],
        "Twin": [0.010, 0.011, 0.005],
        "GMM": [0.004, 0.010, 0.010],
    }
    frac_sig = {
        "Hybrid": [True, True, False],
        "Twin": [True, True, True],
        "GMM": [False, True, True],
    }
    _plot_panel(
        axes[1],
        np.arange(len(fracs)),
        frac_data,
        frac_sig,
        ylim=(-0.002, 0.038),
        title="Generator-data fraction",
        xlabels=["0.25", "0.5", "1.0"],
        xlabel="Generator training fraction",
    )
    axes[1].text(
        1.0,
        0.033,
        "exploratory",
        ha="center",
        va="bottom",
        fontsize=FS_LEGEND,
        color=NAVY,
        fontstyle="italic",
    )

    _save(fig, "fig_f5_dose_fraction")


def fig_f6_decoded_transfer() -> None:
    """F6 — decoded metrics: latent win does not imply decoded win on every axis."""
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4))
    fig.subplots_adjust(hspace=0.48, wspace=0.30, top=0.96, bottom=0.12, left=0.11, right=0.98)
    arms = ["GMM", "Hybrid", "Twin"]
    colors = [COLS["gmm"], COLS["hybrid"], COLS["twin"]]
    metrics = [
        ("ACF distance", [0.051, 0.078, 0.063]),
        ("Channel $L_1$", [0.177, 0.195, 0.190]),
        ("Spectral band", [0.050, 0.070, 0.049]),
        ("Decoded $W_1$", [0.033, 0.057, 0.074]),
    ]
    for i, (ax, (title, vals)) in enumerate(zip(axes.ravel(), metrics)):
        bars = ax.bar(np.arange(3), vals, color=colors, edgecolor=EDGE, width=0.58)
        if i < 2:
            _hide_x_labels(ax)
        pad = 16 if i < 2 else 10
        ax.set_title(title, fontsize=FS_PANEL, loc="left", pad=pad, fontweight="bold")
        ymax = max(vals) * 1.30
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.012 * ymax,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=FS_BAR,
                fontweight="bold",
            )
    _show_x_labels(axes[1, 0], arms)
    _show_x_labels(axes[1, 1], arms)
    _save(fig, "fig_f6_decoded_transfer")


def fig_f7_reproduction() -> None:
    """F7 — claim-1 reproduction checks: depth L=2 and pooled bottleneck."""
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.0))
    fig.subplots_adjust(hspace=0.52, top=0.96, bottom=0.13, left=0.15, right=0.97)

    ax = axes[0]
    labs = ["Hybrid", "Twin"]
    vals = [0.0163, 0.186]
    bars = ax.bar(np.arange(2), vals, color=[COLS["hybrid"], COLS["twin"]], edgecolor=EDGE, width=0.52)
    _hide_x_labels(ax)
    _panel_title(ax, "(a) Depth $L{=}2$ — hybrid remains below twin", pad=22)
    ax.set_ylabel(r"Holdout MMD$^2$ $\downarrow$")
    _bar_labels(ax, bars, [f"{v:.4f}" for v in vals], y_pad_frac=0.10, inside_min=0.20)

    ax = axes[1]
    vals = [0.0227, 0.0826]
    bars = ax.bar(np.arange(2), vals, color=[COLS["hybrid"], COLS["twin"]], edgecolor=EDGE, width=0.52)
    _show_x_labels(ax, labs)
    _panel_title(ax, "(b) Pooled bottleneck — ranking unchanged", pad=18)
    ax.set_ylabel(r"Holdout MMD$^2$ $\downarrow$")
    _bar_labels(ax, bars, [f"{v:.4f}" for v in vals], y_pad_frac=0.10, inside_min=0.20)

    _save(fig, "fig_f7_reproduction")


def main() -> int:
    _style()
    fig_f1_fairness_contract()
    fig_f2_rematch()
    fig_f3_decomposition()
    fig_f4_three_levels_gmm()
    fig_f5_dose_fraction()
    fig_f6_decoded_transfer()
    fig_f7_reproduction()
    print(f"done -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
