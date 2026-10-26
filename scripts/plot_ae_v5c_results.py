#!/usr/bin/env python3
"""Publication figures for ae_v5c (generative rematch + clinical detection suite).

Preferred input: a local mirror of Narval result roots (see RESULTS_AE_V5C.md).
Fallback: transcribed analyze tables under
  Engineering_PFE/Figures/results_ae_v5c/source_tables/

    python eeg_qgan/scripts/plot_ae_v5c_results.py
    python eeg_qgan/scripts/plot_ae_v5c_results.py --root path/to/eeg_qgan_results_ae_v5c
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

THESIS = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = THESIS / "eeg_qgan_results_ae_v5c"
DEFAULT_OUT = THESIS / "Engineering_PFE" / "Figures" / "results_ae_v5c"
SOURCE = DEFAULT_OUT / "source_tables"

FILL = {
    "quantum": "#2C6E7A",
    "quantum_none": "#8FB4BC",
    "quantum_linear": "#5A8F99",
    "quantum_ring": "#2C6E7A",
    "quantum_full": "#1E4D56",
    "classical_equiv": "#7A8490",
    "classical_matched": "#5E7A6A",
    "classical_large": "#A67A74",
    "gmm": "#7D8B5F",
    "baseline": "#9A9389",
    "fourier": "#A08968",
}
EDGE = "#3D4450"
INK = "#2C3038"
MUTED = "#5C6370"

ORDER = ["quantum", "classical_equiv", "classical_matched", "classical_large", "gmm"]
ENT_ORDER = [
    "quantum_none",
    "quantum_linear",
    "quantum_ring",
    "quantum_full",
    "classical_equiv",
]
# Short two-line ticks so labels never collide on a thesis page.
FULL_LABEL = {
    "quantum": "Quantum\n(8-qubit hybrid)",
    "quantum_none": "Product\n(no CZ)",
    "quantum_linear": "Linear\nentanglement",
    "quantum_ring": "Ring\nentanglement",
    "quantum_full": "Full\nentanglement",
    "classical_equiv": "Equal-budget\nMLP",
    "classical_matched": "Larger\nMLP",
    "classical_large": "Large MLP\n(not converged)",
    "gmm": "Gaussian\nmixture",
    "baseline": "Real only",
    "fourier": "Fourier\nanalogue",
}
SHORT = {
    "quantum": "Quantum (8-qubit hybrid)",
    "quantum_none": "Product (no CZ)",
    "quantum_linear": "Linear entanglement",
    "quantum_ring": "Ring entanglement",
    "quantum_full": "Full entanglement",
    "classical_equiv": "Equal-budget MLP",
    "classical_matched": "Larger MLP",
    "classical_large": "Large MLP (not converged)",
    "gmm": "Gaussian mixture",
    "baseline": "Real only",
    "fourier": "Fourier analogue",
}

CLS_METRICS = [
    ("auroc", "AUROC"),
    ("auprc", "AUPRC"),
    ("mcc", "MCC"),
    ("f1", "F1"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("precision", "Precision"),
    ("ece", "ECE ↓"),
    ("brier", "Brier ↓"),
    ("accuracy", "Accuracy"),
]


# Draw near A4 text width. Multi-panel figures are stacked, not squeezed sideways,
# so tick labels do not sit on top of each other.
PRINT_W = 7.0
PRINT_H_PANEL = 3.35

COMPACT_TICK = {
    "quantum": "Quantum",
    "quantum_none": "Product",
    "quantum_linear": "Linear",
    "quantum_ring": "Ring",
    "quantum_full": "Full",
    "classical_equiv": "Twin",
    "classical_matched": "Larger MLP",
    "classical_large": "Large MLP",
    "gmm": "GMM",
    "baseline": "Real",
    "fourier": "Fourier",
}


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
            "axes.titleweight": "medium",
            "axes.labelcolor": INK,
            "axes.edgecolor": EDGE,
            "axes.linewidth": 1.05,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 6,
        }
    )


def _tick(arm: str) -> str:
    return COMPACT_TICK.get(arm, str(arm).replace("_", " "))


def _pad_ylim(ax, lows, highs, zero_line: bool = False) -> None:
    lows = np.asarray(lows, float)
    highs = np.asarray(highs, float)
    lo = float(np.nanmin(lows))
    hi = float(np.nanmax(highs))
    span = max(hi - lo, abs(hi), abs(lo), 1e-6)
    pad = 0.22 * span
    if zero_line or lo < 0:
        ax.set_ylim(min(lo, 0.0) - pad, max(hi, 0.0) + pad)
    else:
        ax.set_ylim(0.0, hi + pad)


def _set_xticks(ax, labels, fontsize: int = 11) -> None:
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=fontsize, color=INK, linespacing=1.2)
    ax.tick_params(axis="x", pad=8, length=4)
    ax.tick_params(axis="y", length=4)


def _fmt_num(v: float) -> str:
    av = abs(float(v))
    if av == 0.0:
        return "0"
    if av < 0.01:
        return f"{v:.4f}"
    if av < 1.0:
        return f"{v:.3f}"
    return f"{v:.2f}"


def _label_bars(ax, fontsize: int = 8) -> None:
    """Print the bar height so a jury can read the value without guessing the axis."""
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-9)
    for rect in ax.patches:
        h = rect.get_height()
        if not np.isfinite(h) or rect.get_width() <= 0:
            continue
        x = rect.get_x() + rect.get_width() / 2.0
        if h >= 0:
            y = h + 0.018 * span
            va = "bottom"
        else:
            y = h - 0.018 * span
            va = "top"
        ax.text(x, y, _fmt_num(h), ha="center", va=va, fontsize=fontsize, color=INK, clip_on=False)


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(
            p,
            facecolor="white",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        print(f"wrote {p}")
    plt.close(fig)


def _err(means, lo, hi):
    m, lo, hi = np.asarray(means, float), np.asarray(lo, float), np.asarray(hi, float)
    return np.vstack([np.maximum(0.0, m - lo), np.maximum(0.0, hi - m)])


def _color(arm: str) -> str:
    return FILL.get(arm, "#CCCCCC")


def load_arms_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_arms_from_study(study_dir: Path) -> pd.DataFrame | None:
    p = study_dir / "tables" / "arms.csv"
    return load_arms_csv(p)


def collect_classification(study_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(study_dir.glob("results_*.json")):
        blob = json.loads(f.read_text(encoding="utf-8"))
        for r in blob.get("rows", []):
            det = r.get("detector") or {}
            for source, metrics in det.items():
                if not isinstance(metrics, dict):
                    continue
                if "auroc" not in metrics and "f1" not in metrics:
                    continue
                arm = source.replace("augmented_", "") if source.startswith("augmented_") else source
                row = {
                    "seed": r.get("seed_index", r.get("seed")),
                    "fold": r.get("fold"),
                    "fold_name": r.get("fold_name"),
                    "source": source,
                    "arm": arm,
                    "n_synthetic_added": int(metrics.get("n_synthetic_added") or 0),
                }
                for k, _ in CLS_METRICS:
                    if k in metrics and k != "accuracy":
                        row[k] = float(metrics[k])
                conf = metrics.get("confusion") or {}
                if conf:
                    tn = int(conf.get("tn", 0))
                    fp = int(conf.get("fp", 0))
                    fn = int(conf.get("fn", 0))
                    tp = int(conf.get("tp", 0))
                    denom = tn + fp + fn + tp
                    if denom > 0:
                        row["accuracy"] = float(tn + tp) / float(denom)
                rows.append(row)
    return pd.DataFrame(rows)


def load_detector_summary_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def fig_bar_mmd(
    arms: pd.DataFrame,
    out: Path,
    name: str,
    title: str,
    order: list[str] | None = None,
    note: str = "",
) -> None:
    m = arms[(arms.space == "latent") & (arms.metric == "mmd2")].copy()
    if m.empty:
        return
    order = order or [a for a in ORDER if a in set(m.arm)]
    present = [a for a in order if a in set(m.arm)]
    if not present:
        present = list(m.arm.unique())
    fig, ax = plt.subplots(figsize=(PRINT_W, 4.6))
    x = np.arange(len(present))
    means = [float(m.loc[m.arm == a, "mean"].iloc[0]) for a in present]
    lo = [float(m.loc[m.arm == a, "ci_low"].iloc[0]) for a in present]
    hi = [float(m.loc[m.arm == a, "ci_high"].iloc[0]) for a in present]
    bars = ax.bar(
        x,
        means,
        yerr=_err(means, lo, hi),
        color=[_color(a) for a in present],
        edgecolor=EDGE,
        linewidth=0.9,
        capsize=4,
        width=0.68,
        error_kw={"ecolor": EDGE, "elinewidth": 0.9},
    )
    for i, a in enumerate(present):
        if a == "classical_large":
            bars[i].set_hatch("///")
    _set_xticks(ax, [_tick(a) for a in present], fontsize=10)
    ax.set_ylabel(r"Holdout latent MMD$^2$  (lower is better)")
    ax.set_title(title, pad=8)
    _pad_ylim(ax, lo, hi, zero_line=False)
    _label_bars(ax)
    fig.tight_layout()
    _save(fig, out, name)


def fig_entanglement(arms: pd.DataFrame, out: Path) -> None:
    fig_bar_mmd(
        arms,
        out,
        "fig_entanglement_latent_mmd",
        "Entanglement ablation — latent fidelity at fixed parameter count",
        order=ENT_ORDER,
    )


def fig_paired_delta(arms: pd.DataFrame, out: Path, name: str, title: str, ref: str = "classical_equiv") -> None:
    m = arms[(arms.space == "latent") & (arms.metric == "mmd2")].set_index("arm")
    if ref not in m.index:
        return
    ref_m = float(m.loc[ref, "mean"])
    arms_plot = [a for a in m.index if a != ref]
    # Prefer a sensible display order
    preferred = [a for a in ENT_ORDER + ORDER if a in arms_plot]
    arms_plot = preferred + [a for a in arms_plot if a not in preferred]
    deltas = [float(m.loc[a, "mean"]) - ref_m for a in arms_plot]
    # Approximate CI half-width from arm CI + ref CI in quadrature (display only)
    half = []
    for a in arms_plot:
        a_hw = 0.5 * (float(m.loc[a, "ci_high"]) - float(m.loc[a, "ci_low"]))
        r_hw = 0.5 * (float(m.loc[ref, "ci_high"]) - float(m.loc[ref, "ci_low"]))
        half.append(float(np.sqrt(a_hw**2 + r_hw**2)))
    fig, ax = plt.subplots(figsize=(PRINT_W, 4.5))
    x = np.arange(len(arms_plot))
    ax.bar(
        x,
        deltas,
        yerr=half,
        color=[_color(a) for a in arms_plot],
        edgecolor=EDGE,
        linewidth=0.9,
        capsize=4,
        width=0.7,
        error_kw={"ecolor": EDGE, "elinewidth": 0.9},
    )
    ax.axhline(0.0, color=EDGE, linewidth=1.0)
    _set_xticks(ax, [_tick(a) for a in arms_plot], fontsize=10)
    ax.set_ylabel(r"$\Delta$ latent MMD$^2$ vs equal-budget MLP" "\n(negative favours the arm)")
    ax.set_title(title, pad=8)
    lo = [d - h for d, h in zip(deltas, half)]
    hi = [d + h for d, h in zip(deltas, half)]
    _pad_ylim(ax, lo, hi, zero_line=True)
    _label_bars(ax)
    fig.tight_layout()
    _save(fig, out, name)


def fig_dose_latent(dose_tables: dict[int, pd.DataFrame], out: Path) -> None:
    doses = sorted(dose_tables)
    arms = ["gmm", "quantum", "classical_equiv"]
    fig, ax = plt.subplots(figsize=(PRINT_W, 4.5))
    width = 0.25
    x = np.arange(len(doses))
    for i, arm in enumerate(arms):
        means, los, his = [], [], []
        for d in doses:
            m = dose_tables[d]
            sub = m[(m.space == "latent") & (m.metric == "mmd2") & (m.arm == arm)]
            means.append(float(sub["mean"].iloc[0]))
            los.append(float(sub["ci_low"].iloc[0]))
            his.append(float(sub["ci_high"].iloc[0]))
        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            yerr=_err(means, los, his),
            color=_color(arm),
            edgecolor=EDGE,
            linewidth=0.9,
            capsize=3,
            label=SHORT[arm],
            error_kw={"ecolor": EDGE, "elinewidth": 0.9},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"dose = {d}" for d in doses], fontsize=10)
    ax.set_ylabel(r"Holdout latent MMD$^2$  (lower is better)")
    ax.set_title("Dose-response study — generative fidelity across synthetic dose")
    ax.legend(frameon=False, fontsize=10)
    _save(fig, out, "fig_dose_latent_mmd")


def fig_dose_detection(det_rows: pd.DataFrame, out: Path) -> None:
    """ΔAUROC / ΔAUPRC / ΔMCC vs baseline across doses for quantum vs classical."""
    if det_rows.empty:
        return
    metrics = [("delta_auroc", r"$\Delta$AUROC"), ("delta_auprc", r"$\Delta$AUPRC"), ("delta_mcc", r"$\Delta$MCC")]
    arms = ["quantum", "classical_equiv", "gmm"]
    doses = sorted(det_rows.dose.unique())
    fig, axes = plt.subplots(3, 1, figsize=(PRINT_W, 9.6), constrained_layout=True, sharey=False)
    width = 0.25
    x = np.arange(len(doses))
    for ax, (key, title) in zip(axes, metrics):
        for i, arm in enumerate(arms):
            vals = []
            for d in doses:
                sub = det_rows[(det_rows.dose == d) & (det_rows.arm == arm)]
                vals.append(float(sub[key].iloc[0]) if len(sub) else np.nan)
            ax.bar(
                x + (i - 1) * width,
                vals,
                width=width,
                color=_color(arm),
                edgecolor=EDGE,
                linewidth=0.85,
                label=SHORT[arm] if ax is axes[0] else None,
            )
        ax.axhline(0.0, color=EDGE, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(d)) for d in doses], fontsize=10)
        ax.set_xlabel("Synthetic clips added")
        ax.set_title(title)
        vals_all = []
        for arm in arms:
            for d in doses:
                sub = det_rows[(det_rows.dose == d) & (det_rows.arm == arm)]
                if len(sub):
                    vals_all.append(float(sub[key].iloc[0]))
        if vals_all:
            _pad_ylim(ax, vals_all, vals_all, zero_line=True)
    axes[0].set_ylabel("Change vs real-only baseline")
    handles = [Patch(facecolor=_color(a), edgecolor=EDGE, label=SHORT[a]) for a in arms]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Detection under varying synthetic dose", fontsize=13)
    _save(fig, out, "fig_dose_detection_delta")


def fig_dataeff_latent(frac_tables: dict[str, pd.DataFrame], out: Path) -> None:
    fracs = sorted(frac_tables, key=lambda s: float(s))
    arms = ["gmm", "quantum", "classical_equiv"]
    fig, ax = plt.subplots(figsize=(PRINT_W, 4.5))
    width = 0.25
    x = np.arange(len(fracs))
    for i, arm in enumerate(arms):
        means, los, his = [], [], []
        for f in fracs:
            m = frac_tables[f]
            sub = m[(m.space == "latent") & (m.metric == "mmd2") & (m.arm == arm)]
            means.append(float(sub["mean"].iloc[0]))
            los.append(float(sub["ci_low"].iloc[0]))
            his.append(float(sub["ci_high"].iloc[0]))
        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            yerr=_err(means, los, his),
            color=_color(arm),
            edgecolor=EDGE,
            linewidth=0.85,
            capsize=3,
            label=SHORT[arm],
            error_kw={"ecolor": EDGE, "elinewidth": 0.8},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"train fraction = {f}" for f in fracs])
    ax.set_ylabel(r"Holdout latent MMD$^2$  (lower is better)")
    ax.set_title("Data-efficiency — latent fidelity under labelled scarcity")
    ax.legend(frameon=False, fontsize=10)
    _save(fig, out, "fig_dataeff_latent_mmd")


def fig_dataeff_detection(det_rows: pd.DataFrame, out: Path) -> None:
    if det_rows.empty:
        return
    metrics = [("auroc", "AUROC"), ("auprc", "AUPRC"), ("mcc", "MCC")]
    arms = ["baseline", "quantum", "classical_equiv", "gmm"]
    fracs = sorted(det_rows.frac.unique(), key=float)
    fig, axes = plt.subplots(3, 1, figsize=(PRINT_W, 9.6), constrained_layout=True)
    width = 0.2
    x = np.arange(len(fracs))
    for ax, (key, title) in zip(axes, metrics):
        for i, arm in enumerate(arms):
            vals = []
            for f in fracs:
                sub = det_rows[(det_rows.frac == f) & (det_rows.arm == arm)]
                vals.append(float(sub[key].iloc[0]) if len(sub) else np.nan)
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width=width,
                color=_color(arm),
                edgecolor=EDGE,
                linewidth=0.8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(f) for f in fracs])
        ax.set_xlabel("Train fraction")
        ax.set_title(title)
        ax.set_ylim(0, max(0.85, ax.get_ylim()[1]))
    handles = [Patch(facecolor=_color(a), edgecolor=EDGE, label=SHORT.get(a, a)) for a in arms]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Detection under labelled scarcity — absolute metrics", fontsize=13)
    _save(fig, out, "fig_dataeff_detection_metrics")


def fig_classification_grid(cls_df: pd.DataFrame, out: Path, study_tag: str, title: str) -> None:
    if cls_df.empty:
        return
    arms = ["baseline"] + [a for a in ORDER if a in set(cls_df.arm)]
    metrics = [m for m in CLS_METRICS if m[0] in cls_df.columns and m[0] in {"auroc", "auprc", "mcc", "f1"}]
    fig, axes = plt.subplots(2, 2, figsize=(PRINT_W, 7.0), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, lab) in zip(axes, metrics):
        means, stds = [], []
        for a in arms:
            sub = cls_df[cls_df.arm == a]
            means.append(float(sub[key].mean()))
            stds.append(float(sub[key].std(ddof=1)) if len(sub) > 1 else 0.0)
        x = np.arange(len(arms))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=[_color(a) for a in arms],
            edgecolor=EDGE,
            linewidth=0.9,
            capsize=4,
            width=0.68,
            error_kw={"ecolor": EDGE, "elinewidth": 1.0},
        )
        if "classical_large" in arms:
            bars[arms.index("classical_large")].set_hatch("///")
        _set_xticks(ax, [_tick(a) for a in arms], fontsize=11)
        ax.set_title(lab, fontsize=13, pad=8)
        ax.set_ylabel(lab)
        highs = [mu + sd for mu, sd in zip(means, stds)]
        _pad_ylim(ax, [0.0] * len(highs), highs, zero_line=False)
        top = ax.get_ylim()[1]
        ax.set_ylim(0.0, min(1.08, max(top, 0.2)))
        _label_bars(ax, fontsize=9)
    fig.suptitle(title, fontsize=13, y=1.02)
    _save(fig, out, f"fig_classification_metrics_{study_tag}")


def fig_classification_delta_primary(cls_df: pd.DataFrame, out: Path, study_tag: str, dose_note: str) -> None:
    if cls_df.empty or "baseline" not in set(cls_df.arm):
        return
    arms = [a for a in ORDER if a in set(cls_df.arm)]
    metrics_show = [
        ("auroc", r"$\Delta$AUROC"),
        ("auprc", r"$\Delta$AUPRC"),
        ("mcc", r"$\Delta$MCC"),
        ("f1", r"$\Delta$F1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(PRINT_W, 7.2), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, metrics_show):
        if key not in cls_df.columns:
            ax.axis("off")
            continue
        deltas, stds = [], []
        for a in arms:
            paired = []
            for _, g in cls_df.groupby(["seed", "fold"]):
                bv, av = g[g.arm == "baseline"], g[g.arm == a]
                if len(bv) and len(av) and key in bv and key in av:
                    paired.append(float(av[key].iloc[0]) - float(bv[key].iloc[0]))
            deltas.append(float(np.mean(paired)) if paired else np.nan)
            stds.append(float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0)
        x = np.arange(len(arms))
        bars = ax.bar(
            x,
            deltas,
            yerr=stds,
            color=[_color(a) for a in arms],
            edgecolor=EDGE,
            linewidth=0.85,
            capsize=3,
            width=0.72,
            error_kw={"ecolor": EDGE, "elinewidth": 0.85},
        )
        if "classical_large" in arms:
            bars[arms.index("classical_large")].set_hatch("///")
        ax.axhline(0.0, color=EDGE, linewidth=1.0)
        _set_xticks(ax, [_tick(a) for a in arms], fontsize=11)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_ylabel(title)
        lo = [d - s for d, s in zip(deltas, stds)]
        hi = [d + s for d, s in zip(deltas, stds)]
        _pad_ylim(ax, lo, hi, zero_line=True)
        _label_bars(ax, fontsize=9)
    fig.suptitle(
        "Change versus the real-only detector baseline",
        fontsize=13,
        y=1.03,
    )
    _save(fig, out, f"fig_classification_delta_{study_tag}")


def fig_classification_delta_extended(cls_df: pd.DataFrame, out: Path, study_tag: str) -> None:
    """Secondary detection deltas: calibration and operating-point metrics."""
    if cls_df.empty or "baseline" not in set(cls_df.arm):
        return
    arms = [a for a in ORDER if a in set(cls_df.arm)]
    metrics_show = [
        ("ece", r"$\Delta$ECE"),
        ("brier", r"$\Delta$Brier"),
        ("sensitivity", r"$\Delta$Sensitivity"),
        ("specificity", r"$\Delta$Specificity"),
        ("precision", r"$\Delta$Precision"),
        ("accuracy", r"$\Delta$Accuracy"),
    ]
    present = [(k, t) for k, t in metrics_show if k in cls_df.columns]
    if not present:
        return
    ncols = 2
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(PRINT_W, 3.4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, title) in zip(axes, present):
        deltas, stds = [], []
        for a in arms:
            paired = []
            for _, g in cls_df.groupby(["seed", "fold"]):
                bv, av = g[g.arm == "baseline"], g[g.arm == a]
                if len(bv) and len(av) and key in bv and key in av:
                    paired.append(float(av[key].iloc[0]) - float(bv[key].iloc[0]))
            deltas.append(float(np.mean(paired)) if paired else np.nan)
            stds.append(float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0)
        x = np.arange(len(arms))
        ax.bar(
            x,
            deltas,
            yerr=stds,
            color=[_color(a) for a in arms],
            edgecolor=EDGE,
            linewidth=0.85,
            capsize=3,
            width=0.72,
            error_kw={"ecolor": EDGE, "elinewidth": 0.85},
        )
        ax.axhline(0.0, color=EDGE, linewidth=1.0)
        _set_xticks(ax, [_tick(a) for a in arms], fontsize=11)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_ylabel(title)
        lo = [d - s for d, s in zip(deltas, stds)]
        hi = [d + s for d, s in zip(deltas, stds)]
        _pad_ylim(ax, lo, hi, zero_line=True)
        _label_bars(ax, fontsize=9)
    for ax in axes[len(present) :]:
        ax.axis("off")
    fig.suptitle(
        "Secondary metrics versus the real-only detector baseline",
        fontsize=13,
        y=1.03,
    )
    _save(fig, out, f"fig_classification_delta_extended_{study_tag}")


def collect_training_histories(study_dir: Path) -> dict[str, list[dict]]:
    """Aggregate AE and WGAN-GP histories across fold records."""
    ae_epochs: dict[int, list[float]] = {}
    ae_val: dict[int, list[float]] = {}
    ae_nmse: dict[int, list[float]] = {}
    gan_steps: dict[int, dict[str, list[float]]] = {}

    for f in sorted(study_dir.glob("results_*.json")):
        blob = json.loads(f.read_text(encoding="utf-8"))
        for r in blob.get("rows", []):
            ae = r.get("autoencoder") or {}
            for pt in ae.get("history") or []:
                ep = int(pt["epoch"])
                ae_epochs.setdefault(ep, []).append(float(pt["train_loss"]))
                ae_val.setdefault(ep, []).append(float(pt["val_loss"]))
                ae_nmse.setdefault(ep, []).append(float(pt["val_nmse"]))
            for arm_name, arm in (r.get("arms") or {}).items():
                if not isinstance(arm, dict) or arm_name == "gmm":
                    continue
                bucket = gan_steps.setdefault(arm_name, {})
                for pt in arm.get("history") or []:
                    step = int(pt["step"])
                    bucket.setdefault(step, {"critic": [], "generator": [], "mmd": []})
                    bucket[step]["critic"].append(float(pt["critic_loss"]))
                    bucket[step]["generator"].append(float(pt["generator_loss"]))
                    if "selection_distance" in pt:
                        bucket[step]["mmd"].append(float(pt["selection_distance"]))

    ae_curve = [
        {
            "epoch": ep,
            "train_loss": float(np.mean(ae_epochs[ep])),
            "val_loss": float(np.mean(ae_val[ep])),
            "val_nmse": float(np.mean(ae_nmse[ep])),
        }
        for ep in sorted(ae_epochs)
    ]
    gan_curves = {}
    for arm, by_step in gan_steps.items():
        gan_curves[arm] = [
            {
                "step": step,
                "critic_loss": float(np.mean(by_step[step]["critic"])),
                "generator_loss": float(np.mean(by_step[step]["generator"])),
                "selection_mmd": float(np.mean(by_step[step]["mmd"])),
            }
            for step in sorted(by_step)
        ]
    return {"ae": ae_curve, "gan": gan_curves}


def fig_training_curves(hist: dict, out: Path, tag: str, title_prefix: str = "") -> None:
    ae = hist.get("ae") or []
    gan = hist.get("gan") or {}
    if not ae and not gan:
        return

    fig, axes = plt.subplots(3, 1, figsize=(PRINT_W, 10.4), constrained_layout=True)
    if ae:
        epochs = [p["epoch"] for p in ae]
        axes[0].plot(epochs, [p["train_loss"] for p in ae], color=_color("quantum"), lw=2.4, label="Train")
        axes[0].plot(epochs, [p["val_loss"] for p in ae], color=_color("classical_equiv"), lw=2.4, label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Reconstruction loss (MSE)")
        if epochs and min(epochs) >= 50:
            axes[0].set_title("Autoencoder (logged tail of the 80-epoch run)")
            axes[0].set_xlim(60.5, 80.5)
            axes[0].set_xticks([61, 65, 70, 75, 80])
        else:
            axes[0].set_title("Autoencoder")
            axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        axes[0].legend(frameon=False, loc="best")
        axes[1].plot(epochs, [p["val_nmse"] for p in ae], color=INK, lw=2.4)
        axes[1].axhline(0.55, color="#c0392b", ls="--", lw=1.6, label="Gate (0.55)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation NMSE")
        axes[1].set_title("Reconstruction gate")
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        if epochs and min(epochs) >= 50:
            axes[1].set_xlim(60.5, 80.5)
            axes[1].set_xticks([61, 65, 70, 75, 80])
        nm = [p["val_nmse"] for p in ae]
        lo = min(min(nm), 0.55) - 0.03
        hi = max(max(nm), 0.55) + 0.03
        axes[1].set_ylim(lo, hi)
        axes[1].legend(frameon=False, loc="best")
    else:
        axes[0].axis("off")
        axes[1].axis("off")

    plotted = False
    for arm in ("quantum", "classical_equiv"):
        curve = gan.get(arm)
        if not curve:
            continue
        steps = [p["step"] for p in curve]
        axes[2].plot(
            steps,
            [p["generator_loss"] for p in curve],
            color=_color(arm),
            lw=2.4,
            marker="o",
            ms=7,
            label=f"{COMPACT_TICK.get(arm, arm)} generator",
        )
        axes[2].plot(
            steps,
            [-p["critic_loss"] for p in curve],
            color=_color(arm),
            lw=1.8,
            ls="--",
            marker="s",
            ms=6,
            alpha=0.9,
            label=f"{COMPACT_TICK.get(arm, arm)} |critic|",
        )
        plotted = True
    if plotted:
        axes[2].set_xlabel("Generator update")
        axes[2].set_ylabel("Adversarial loss")
        axes[2].set_title("WGAN-GP (quantum vs twin)")
        axes[2].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
        axes[2].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    else:
        axes[2].axis("off")

    prefix = f"{title_prefix} — " if title_prefix else ""
    fig.suptitle(f"{prefix}Training dynamics (mean over folds)", fontsize=13)
    _save(fig, out, f"fig_training_curves_{tag}")


def fig_eval_levels_auroc(cls_df: pd.DataFrame, arms: pd.DataFrame, out: Path) -> None:
    m = arms[(arms.space == "latent") & (arms.metric == "mmd2")].set_index("arm")
    if "quantum" not in m.index or "classical_equiv" not in m.index:
        return

    def mean_delta(arm: str, key: str) -> float:
        vals = []
        for _, g in cls_df.groupby(["seed", "fold"]):
            bv, av = g[g.arm == "baseline"], g[g.arm == arm]
            if len(bv) and len(av) and key in bv.columns:
                vals.append(float(av[key].iloc[0]) - float(bv[key].iloc[0]))
        return float(np.mean(vals)) if vals else np.nan

    fig, axes = plt.subplots(3, 1, figsize=(PRINT_W, 9.6), constrained_layout=True)
    panels = [
        (
            axes[0],
            "Level 1 — Latent fidelity",
            r"Holdout MMD$^2$ (lower is better)",
            [float(m.loc["quantum", "mean"]), float(m.loc["classical_equiv", "mean"])],
            True,
        ),
        (
            axes[1],
            "Level 2 — Detection ranking",
            r"$\Delta$AUROC vs real-only",
            [mean_delta("quantum", "auroc"), mean_delta("classical_equiv", "auroc")],
            False,
        ),
        (
            axes[2],
            "Level 3 — Detection association",
            r"$\Delta$MCC vs real-only",
            [mean_delta("quantum", "mcc"), mean_delta("classical_equiv", "mcc")],
            False,
        ),
    ]
    for ax, title, ylab, vals, lower_better in panels:
        ax.bar(
            [0, 1],
            vals,
            color=[_color("quantum"), _color("classical_equiv")],
            edgecolor=EDGE,
            linewidth=0.9,
            width=0.62,
        )
        _set_xticks(ax, ["Quantum", "Twin"], fontsize=12)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_ylabel(ylab)
        if not lower_better:
            ax.axhline(0.0, color=EDGE, linewidth=0.9)
            _pad_ylim(ax, vals, vals, zero_line=True)
        else:
            _pad_ylim(ax, vals, vals, zero_line=False)
        _label_bars(ax, fontsize=11)
    _save(fig, out, "fig_eval_levels_transfer")


def fig_scenario_contrast(arms: pd.DataFrame, out: Path) -> None:
    m = arms[(arms.space == "latent") & (arms.metric == "mmd2")].set_index("arm")
    if "quantum" not in m.index:
        return
    q = float(m.loc["quantum", "mean"])
    q_lo, q_hi = float(m.loc["quantum", "ci_low"]), float(m.loc["quantum", "ci_high"])
    panels = []
    if "classical_large" in m.index:
        panels.append(
            (
                "Scenario 1 — permissive control",
                "classical_large",
                "Large MLP (failed)",
                True,
            )
        )
    if "classical_equiv" in m.index:
        panels.append(
            (
                "Scenario 2 — capacity-controlled",
                "classical_equiv",
                "Twin",
                False,
            )
        )
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(PRINT_W, 4.4 * len(panels)), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, ctrl, ctrl_label, hatch) in zip(axes, panels):
        c = float(m.loc[ctrl, "mean"])
        c_lo, c_hi = float(m.loc[ctrl, "ci_low"]), float(m.loc[ctrl, "ci_high"])
        vals = [q, c]
        bars = ax.bar(
            [0, 1],
            vals,
            yerr=_err(vals, [q_lo, c_lo], [q_hi, c_hi]),
            color=[_color("quantum"), _color(ctrl)],
            edgecolor=EDGE,
            linewidth=0.9,
            capsize=4,
            width=0.62,
            error_kw={"ecolor": EDGE, "elinewidth": 0.9},
        )
        if hatch:
            bars[1].set_hatch("///")
        _set_xticks(ax, ["Quantum", ctrl_label.replace("\n", " ")], fontsize=12)
        ax.set_title(title, pad=8, fontsize=13)
        ax.set_ylabel(r"Holdout latent MMD$^2$  (lower is better)")
        _pad_ylim(ax, [q_lo, c_lo], [q_hi, c_hi], zero_line=False)
        _label_bars(ax, fontsize=11)
    _save(fig, out, "fig_scenario_contrast")


def write_cls_csvs(cls_df: pd.DataFrame, out: Path, study: str) -> None:
    if cls_df.empty:
        return
    keys = [k for k, _ in CLS_METRICS if k in cls_df.columns]
    g = cls_df.groupby("arm", sort=False)
    flat = pd.DataFrame({"arm": g.size().index})
    for k in keys:
        flat[f"{k}_mean"] = g[k].mean().values
        flat[f"{k}_std"] = g[k].std(ddof=1).values
    flat["n"] = g.size().values
    flat.to_csv(out / f"classification_summary_{study}.csv", index=False)
    cls_df.to_csv(out / f"classification_folds_{study}.csv", index=False)
    print(f"wrote classification CSVs for {study}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Main ae_v5c results root")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--source", type=Path, default=SOURCE, help="Fallback transcribed tables")
    args = p.parse_args()
    _style()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    src = args.source

    # ---- Generative: capacity rematch (full root or source) ----
    cap_arms = load_arms_from_study(args.root / "capacity_rematch")
    if cap_arms is None:
        cap_arms = load_arms_csv(src / "capacity_arms.csv")
    if cap_arms is not None:
        fig_bar_mmd(
            cap_arms,
            out,
            "fig_latent_mmd_capacity",
            "Capacity rematch — latent distribution fidelity (ae_v5c)",
            note="Primary generative claim: Style-qGAN vs capacity-matched classical twin.",
        )
        fig_scenario_contrast(cap_arms, out)
        fig_paired_delta(
            cap_arms,
            out,
            "fig_latent_delta_vs_classical",
            "Paired latent advantage vs capacity-matched classical twin (ae_v5c)",
        )

    # ---- Entanglement ----
    ent = load_arms_from_study(args.root / "entanglement_ablation")
    if ent is None:
        ent = load_arms_csv(src / "entanglement_arms.csv")
    if ent is not None:
        fig_entanglement(ent, out)
        fig_paired_delta(
            ent,
            out,
            "fig_entanglement_delta_vs_classical",
            "Entanglement topologies vs capacity-matched classical twin (ae_v5c)",
        )

    # ---- Critic width ----
    critic = load_arms_from_study(args.root / "critic_width")
    if critic is None:
        critic = load_arms_csv(src / "critic_arms.csv")
    if critic is not None:
        fig_bar_mmd(
            critic,
            out,
            "fig_critic_latent_mmd",
            "Critic-width control — latent rematch is not a critic-capacity artefact (ae_v5c)",
            order=["quantum", "classical_equiv"],
        )

    # ---- Dose response (separate roots or source) ----
    dose_tables: dict[int, pd.DataFrame] = {}
    det_dose_rows = []
    for d in (512, 2048, 8192):
        study = args.root.parent / f"eeg_qgan_results_ae_v5c_dose{d}" / "dose_response"
        arms = load_arms_from_study(study) if study.exists() else None
        if arms is None:
            arms = load_arms_csv(src / f"dose{d}_arms.csv")
        if arms is not None:
            dose_tables[d] = arms
        # detection from JSON if present
        if study.exists():
            cls = collect_classification(study)
            if not cls.empty:
                base = cls[cls.arm == "baseline"]
                for arm in ("quantum", "classical_equiv", "gmm"):
                    sub = cls[cls.arm == arm]
                    if sub.empty or base.empty:
                        continue
                    row = {"dose": d, "arm": arm}
                    for key, _ in CLS_METRICS:
                        if key not in cls.columns:
                            continue
                        row[key] = float(sub[key].mean())
                        row[f"delta_{key}"] = float(sub[key].mean() - base[key].mean())
                    det_dose_rows.append(row)
        else:
            det_csv = load_detector_summary_csv(src / f"dose{d}_detector.csv")
            if det_csv is not None:
                for _, r in det_csv.iterrows():
                    det_dose_rows.append(r.to_dict())
    if dose_tables:
        fig_dose_latent(dose_tables, out)
    det_dose = pd.DataFrame(det_dose_rows)
    if not det_dose.empty:
        fig_dose_detection(det_dose, out)
        det_dose.to_csv(out / "dose_detection_summary.csv", index=False)

    # ---- Data efficiency ----
    frac_tables: dict[str, pd.DataFrame] = {}
    det_frac_rows = []
    for f in ("0.25", "0.5", "1.0"):
        study = args.root.parent / f"eeg_qgan_results_ae_v5c_frac{f}" / "data_efficiency"
        arms = load_arms_from_study(study) if study.exists() else None
        if arms is None:
            arms = load_arms_csv(src / f"dataeff_f{f}_arms.csv")
        if arms is not None:
            frac_tables[f] = arms
        if study.exists():
            cls = collect_classification(study)
            if not cls.empty:
                for arm in ["baseline", "quantum", "classical_equiv", "gmm"]:
                    sub = cls[cls.arm == arm]
                    if sub.empty:
                        continue
                    row = {"frac": f, "arm": arm}
                    for key, _ in CLS_METRICS:
                        if key in cls.columns:
                            row[key] = float(sub[key].mean())
                    det_frac_rows.append(row)
        else:
            det_csv = load_detector_summary_csv(src / f"dataeff_f{f}_detector.csv")
            if det_csv is not None:
                for _, r in det_csv.iterrows():
                    row = r.to_dict()
                    row["frac"] = f
                    det_frac_rows.append(row)
    if frac_tables:
        fig_dataeff_latent(frac_tables, out)
    det_frac = pd.DataFrame(det_frac_rows)
    if not det_frac.empty:
        fig_dataeff_detection(det_frac, out)
        det_frac.to_csv(out / "dataeff_detection_summary.csv", index=False)

    # ---- Detection from capacity / decoded (full JSON) ----
    for study_name, dose_note in (
        ("capacity_rematch", "synthetic dose = 8 192"),
        ("decoded_equivalence", "decoded-equivalence arms"),
    ):
        study_dir = args.root / study_name
        if not study_dir.exists():
            continue
        cls = collect_classification(study_dir)
        if cls.empty:
            continue
        fig_classification_grid(
            cls,
            out,
            study_name.replace("_rematch", "").replace("_equivalence", ""),
            f"Seizure detector metrics by augmentation arm — {study_name.replace('_', ' ')} (ae_v5c, E1)",
        )
        fig_classification_delta_primary(cls, out, study_name.split("_")[0], dose_note)
        write_cls_csvs(cls, out, study_name.split("_")[0])
        if study_name == "capacity_rematch" and cap_arms is not None:
            fig_eval_levels_auroc(cls, cap_arms, out)

    # numbers markdown
    lines = [
        "# ae_v5c numbers for Evaluation tables",
        "",
        "Regenerate figures: `python eeg_qgan/scripts/plot_ae_v5c_results.py`",
        "",
        "## Generative (primary contribution)",
        "",
        "- Lead with latent MMD rematch + entanglement ablation.",
        "- Clinical transfer figures report AUROC / AUPRC / MCC / F1 / sensitivity / specificity / precision / ECE.",
        "",
        "## Detection (secondary; wait for detboost probe 564326 before rewriting absolute claims)",
        "",
        "- Current rematch baseline AUROC ≈ 0.70–0.71 under patient-disjoint E1 + 1.28 s windows.",
        "- Synthetic quantum dose is null / slightly negative on ΔAUROC in finished dose & scarcity studies.",
        "",
    ]
    (out / "numbers_for_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'numbers_for_tables.md'}")


if __name__ == "__main__":
    main()
