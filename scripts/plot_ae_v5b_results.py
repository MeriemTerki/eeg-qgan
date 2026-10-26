#!/usr/bin/env python3
"""Publication figures from ae_v5b (pastel, full labels, classification metrics).

    python eeg_qgan/scripts/plot_ae_v5b_results.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

THESIS = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = THESIS / "eeg_qgan_results_ae_v5b"
DEFAULT_OUT = THESIS / "Engineering_PFE" / "Figures" / "results_ae_v5b"

# Soft pastels + dark ink for print readability
FILL = {
    "quantum": "#A9C8E8",
    "classical_equiv": "#C5CED6",
    "classical_matched": "#B7D8B1",
    "classical_large": "#E6B8B0",
    "gmm": "#D9C9A8",
    "baseline": "#D8D4E0",
}
EDGE = "#2C2C2C"
INK = "#1F1F1F"
MUTED = "#555555"

ORDER = ["quantum", "classical_equiv", "classical_matched", "classical_large", "gmm"]
FULL_LABEL = {
    "quantum": "Quantum Style-qGAN\n(hybrid, 704 parameters)",
    "classical_equiv": "Classical capacity-equiv.\nMLP (709 parameters)",
    "classical_matched": "Classical matched\nMLP (1 354 parameters)",
    "classical_large": "Classical large MLP\n(12 480 parameters)*",
    "gmm": "Gaussian mixture\nmodel (GMM)",
    "baseline": "Real-only baseline\n(no synthetic data)",
}
SHORT_FULL = {
    "quantum": "Quantum Style-qGAN (704)",
    "classical_equiv": "Classical capacity-equiv. MLP (709)",
    "classical_matched": "Classical matched MLP (1 354)",
    "classical_large": "Classical large MLP (12 480)*",
    "gmm": "Gaussian mixture model",
    "baseline": "Real-only baseline",
}

CLS_METRICS = [
    ("f1", "F1 score"),
    ("auroc", "AUROC"),
    ("auprc", "Average precision (AUPRC)"),
    ("mcc", "Matthews correlation (MCC)"),
    ("sensitivity", "Sensitivity (recall)"),
    ("specificity", "Specificity"),
    ("precision", "Precision"),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": EDGE,
            "axes.linewidth": 0.9,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAF8",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, facecolor="white")
        print(f"wrote {p}")
    plt.close(fig)


def _err(means, lo, hi):
    m, lo, hi = np.asarray(means), np.asarray(lo), np.asarray(hi)
    return np.vstack([m - lo, hi - m])


def load_arms(study_dir: Path) -> pd.DataFrame:
    return pd.read_csv(study_dir / "tables" / "arms.csv")


def collect_classification(root: Path, study: str) -> pd.DataFrame:
    """Per-fold detector metrics including baseline and all augmented arms."""
    rows = []
    d = root / study
    for f in sorted(d.glob("results_*.json")):
        blob = json.loads(f.read_text(encoding="utf-8"))
        for r in blob.get("rows", []):
            det = r.get("detector") or {}
            for source, metrics in det.items():
                if not isinstance(metrics, dict) or "f1" not in metrics:
                    continue
                arm = source.replace("augmented_", "") if source.startswith("augmented_") else source
                rows.append(
                    {
                        "study": study,
                        "seed": r.get("seed_index", r.get("seed")),
                        "fold": r.get("fold"),
                        "fold_name": r.get("fold_name"),
                        "source": source,
                        "arm": arm,
                        **{k: float(metrics[k]) for k, _ in CLS_METRICS if k in metrics},
                        "n_synthetic_added": int(metrics.get("n_synthetic_added") or 0),
                    }
                )
    return pd.DataFrame(rows)


def summarise_cls(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = [k for k, _ in CLS_METRICS]
    g = df.groupby("arm", sort=False)
    out = g[keys].agg(["mean", "std", "count"]).reset_index()
    # flatten
    flat = pd.DataFrame({"arm": out["arm"]})
    for k, _ in CLS_METRICS:
        flat[f"{k}_mean"] = out[(k, "mean")]
        flat[f"{k}_std"] = out[(k, "std")]
        flat["n"] = out[(k, "count")]
    return flat


def ae_nmse_table(root: Path) -> pd.DataFrame:
    rows = []
    for study in ("capacity_rematch", "decoded_equivalence", "deadline_probe"):
        d = root / study
        if not d.exists():
            continue
        for f in sorted(d.glob("results_*.json")):
            blob = json.loads(f.read_text(encoding="utf-8"))
            for r in blob.get("rows", []):
                ae = r.get("autoencoder") or {}
                gates = ae.get("gates") or {}
                rows.append(
                    {
                        "study": study,
                        "seed": r.get("seed_index", r.get("seed")),
                        "fold": r.get("fold"),
                        "fold_name": r.get("fold_name"),
                        "nmse": float(gates.get("nmse", ae.get("best_val_nmse", np.nan))),
                        "passed": bool(gates.get("passed", not ae.get("excluded", True))),
                    }
                )
    return pd.DataFrame(rows)


def fig_latent_mmd(arms: pd.DataFrame, out: Path) -> None:
    m = arms[(arms.space == "latent") & (arms.metric == "mmd2")].copy()
    present = [a for a in ORDER if a in set(m.arm)]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(present))
    means = [float(m.loc[m.arm == a, "mean"].iloc[0]) for a in present]
    lo = [float(m.loc[m.arm == a, "ci_low"].iloc[0]) for a in present]
    hi = [float(m.loc[m.arm == a, "ci_high"].iloc[0]) for a in present]
    bars = ax.bar(
        x,
        means,
        yerr=_err(means, lo, hi),
        color=[FILL[a] for a in present],
        edgecolor=EDGE,
        linewidth=0.9,
        capsize=4,
        width=0.72,
        error_kw={"ecolor": EDGE, "elinewidth": 0.9},
    )
    for i, a in enumerate(present):
        if a == "classical_large":
            bars[i].set_hatch("///")
            bars[i].set_alpha(0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([FULL_LABEL[a] for a in present])
    ax.set_ylabel(r"Holdout latent MMD$^2$  (lower is better)")
    ax.set_title("Capacity rematch — latent distribution fidelity (ae_v5b, 6 folds)")
    ax.text(
        0.01,
        0.98,
        "* Classical large MLP failed to converge on all 6 folds\n"
        "  (optimisation failure — not a fair capacity comparison).",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED,
    )
    _save(fig, out, "fig_latent_mmd_capacity")


def fig_scenario_contrast(arms: pd.DataFrame, out: Path) -> None:
    m = arms[(arms.space == "latent") & (arms.metric == "mmd2")].set_index("arm")
    q = float(m.loc["quantum", "mean"])
    q_lo, q_hi = float(m.loc["quantum", "ci_low"]), float(m.loc["quantum", "ci_high"])
    panels = [
        (
            "Scenario 1 — permissive control",
            "classical_large",
            "Classical large MLP\n(12 480 parameters)",
            True,
        ),
        (
            "Scenario 2 — capacity-controlled",
            "classical_equiv",
            "Classical capacity-equiv.\nMLP (709 parameters)",
            False,
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for ax, (title, ctrl, ctrl_label, hatch) in zip(axes, panels):
        c = float(m.loc[ctrl, "mean"])
        c_lo, c_hi = float(m.loc[ctrl, "ci_low"]), float(m.loc[ctrl, "ci_high"])
        vals = [q, c]
        bars = ax.bar(
            [0, 1],
            vals,
            yerr=_err(vals, [q_lo, c_lo], [q_hi, c_hi]),
            color=[FILL["quantum"], FILL[ctrl]],
            edgecolor=EDGE,
            linewidth=0.9,
            capsize=4,
            width=0.62,
            error_kw={"ecolor": EDGE, "elinewidth": 0.9},
        )
        if hatch:
            bars[1].set_hatch("///")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Quantum Style-qGAN\n(hybrid, 704 parameters)", ctrl_label])
        ax.set_title(title, pad=8)
        if hatch:
            ax.text(
                0.5,
                0.94,
                "Large MLP: did not converge",
                transform=ax.transAxes,
                ha="center",
                fontsize=8.5,
                color="#8B3A3A",
            )
    axes[0].set_ylabel(r"Holdout latent MMD$^2$  (lower is better)")
    fig.suptitle(
        "Same quantum checkpoints; only the classical control changes (ae_v5b)",
        y=1.02,
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    _save(fig, out, "fig_scenario_contrast")


def fig_decoded_metrics(arms: pd.DataFrame, out: Path, study_tag: str, title_study: str) -> None:
    metrics = [
        ("autocorrelation_distance", "Autocorrelation\ndistance"),
        ("channel_abs_error", "Cross-channel\nabsolute error"),
        ("spectral_band_error", "Spectral band\nerror"),
        ("ks_statistic", "Kolmogorov–Smirnov\nstatistic"),
    ]
    focus = [a for a in ["quantum", "classical_equiv", "classical_matched", "gmm"] if a in set(arms.arm)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12.5, 4.2), sharey=False)
    for ax, (metric, title) in zip(axes, metrics):
        sub = arms[(arms.space == "decoded") & (arms.metric == metric)]
        xs = np.arange(len(focus))
        means = [float(sub.loc[sub.arm == a, "mean"].iloc[0]) for a in focus]
        lo = [float(sub.loc[sub.arm == a, "ci_low"].iloc[0]) for a in focus]
        hi = [float(sub.loc[sub.arm == a, "ci_high"].iloc[0]) for a in focus]
        ax.bar(
            xs,
            means,
            yerr=_err(means, lo, hi),
            color=[FILL[a] for a in focus],
            edgecolor=EDGE,
            linewidth=0.8,
            capsize=3,
            width=0.7,
            error_kw={"ecolor": EDGE, "elinewidth": 0.8},
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [
                "Quantum\nStyle-qGAN",
                "Classical\ncapacity-equiv.",
                "Classical\nmatched",
                "GMM",
            ][: len(focus)],
            fontsize=8.5,
        )
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("Decoded distance (lower is better)")
    legend = [
        Patch(facecolor=FILL[a], edgecolor=EDGE, label=SHORT_FULL[a]) for a in focus
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        f"Decoded-space fidelity — {title_study} (ae_v5b; reconstruction gate passed)",
        y=1.14,
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, out, f"fig_decoded_metrics_{study_tag}")


def fig_classification_panel(cls_df: pd.DataFrame, out: Path, study_tag: str) -> None:
    """Absolute F1 / AUROC / MCC / sensitivity with baseline."""
    arms = ["baseline"] + [a for a in ORDER if a in set(cls_df.arm)]
    metrics_show = [
        ("f1", "F1 score"),
        ("auroc", "AUROC"),
        ("mcc", "MCC"),
        ("sensitivity", "Sensitivity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, metrics_show):
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
            color=[FILL[a] for a in arms],
            edgecolor=EDGE,
            linewidth=0.85,
            capsize=3,
            width=0.72,
            error_kw={"ecolor": EDGE, "elinewidth": 0.85},
        )
        if "classical_large" in arms:
            bars[arms.index("classical_large")].set_hatch("///")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                "Real-only\nbaseline",
                "Quantum\nStyle-qGAN",
                "Classical\ncapacity-equiv.",
                "Classical\nmatched",
                "Classical\nlarge*",
                "GMM",
            ][: len(arms)],
            fontsize=8,
        )
        ax.set_title(title)
        ax.set_ylim(0, max(1.05 * max(means) if means else 1, 0.15))
    fig.suptitle(
        "Seizure detector classification metrics by augmentation arm\n"
        f"(ae_v5b {study_tag.replace('_', ' ')}; mean ± std across folds)",
        fontsize=12.5,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        "* Classical large MLP failed latent training on all folds; downstream bar is shown for completeness only.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout()
    _save(fig, out, f"fig_classification_metrics_{study_tag}")


def fig_classification_delta(cls_df: pd.DataFrame, out: Path, study_tag: str, dose_note: str) -> None:
    """Δ vs real-only baseline for F1, AUROC, MCC, sensitivity."""
    base = cls_df[cls_df.arm == "baseline"]
    if base.empty:
        return
    arms = [a for a in ORDER if a in set(cls_df.arm)]
    metrics_show = [
        ("f1", r"$\Delta$ F1"),
        ("auroc", r"$\Delta$ AUROC"),
        ("mcc", r"$\Delta$ MCC"),
        ("sensitivity", r"$\Delta$ Sensitivity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, metrics_show):
        bmean = float(base[key].mean())
        deltas, stds = [], []
        # paired delta per fold then mean/std
        for a in arms:
            merged = []
            for (seed, fold), g in cls_df.groupby(["seed", "fold"]):
                bv = g[g.arm == "baseline"]
                av = g[g.arm == a]
                if len(bv) and len(av):
                    merged.append(float(av[key].iloc[0]) - float(bv[key].iloc[0]))
            deltas.append(float(np.mean(merged)) if merged else np.nan)
            stds.append(float(np.std(merged, ddof=1)) if len(merged) > 1 else 0.0)
        x = np.arange(len(arms))
        bars = ax.bar(
            x,
            deltas,
            yerr=stds,
            color=[FILL[a] for a in arms],
            edgecolor=EDGE,
            linewidth=0.85,
            capsize=3,
            width=0.72,
            error_kw={"ecolor": EDGE, "elinewidth": 0.85},
        )
        if "classical_large" in arms:
            bars[arms.index("classical_large")].set_hatch("///")
        ax.axhline(0.0, color=EDGE, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                "Quantum\nStyle-qGAN",
                "Classical\ncapacity-equiv.",
                "Classical\nmatched",
                "Classical\nlarge*",
                "GMM",
            ][: len(arms)],
            fontsize=8,
        )
        ax.set_title(title)
    fig.suptitle(
        "Change vs real-only baseline (positive = better than baseline)\n"
        f"ae_v5b {study_tag.replace('_', ' ')} · {dose_note}",
        fontsize=12.5,
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, out, f"fig_classification_delta_{study_tag}")


def fig_downstream_f1_readable(cls_df: pd.DataFrame, out: Path, study_tag: str, dose_note: str) -> None:
    """Single clear ΔF1 figure with full names (replaces abbreviated version)."""
    arms = [a for a in ORDER if a in set(cls_df.arm)]
    deltas, lo, hi = [], [], []
    for a in arms:
        paired = []
        for _, g in cls_df.groupby(["seed", "fold"]):
            bv, av = g[g.arm == "baseline"], g[g.arm == a]
            if len(bv) and len(av):
                paired.append(float(av["f1"].iloc[0]) - float(bv["f1"].iloc[0]))
        paired = np.asarray(paired, dtype=float)
        deltas.append(float(paired.mean()))
        # simple mean±1.96*sem visual band if n>=2
        if len(paired) > 1:
            sem = paired.std(ddof=1) / np.sqrt(len(paired))
            lo.append(deltas[-1] - 1.96 * sem)
            hi.append(deltas[-1] + 1.96 * sem)
        else:
            lo.append(deltas[-1])
            hi.append(deltas[-1])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(arms))
    bars = ax.bar(
        x,
        deltas,
        yerr=_err(deltas, lo, hi),
        color=[FILL[a] for a in arms],
        edgecolor=EDGE,
        linewidth=0.9,
        capsize=4,
        width=0.7,
        error_kw={"ecolor": EDGE, "elinewidth": 0.9},
    )
    if "classical_large" in arms:
        bars[arms.index("classical_large")].set_hatch("///")
    ax.axhline(0.0, color=EDGE, linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([FULL_LABEL[a] for a in arms])
    ax.set_ylabel(r"$\Delta$F1 versus real-only baseline")
    ax.set_title(
        f"Downstream clinical utility — F1 change under synthetic augmentation\n"
        f"(ae_v5b, {dose_note})"
    )
    ax.text(
        0.01,
        0.02,
        "Error bars: approximate 95% interval from fold-wise paired differences.\n"
        "* Classical large MLP: latent training did not converge.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
        va="bottom",
    )
    _save(fig, out, f"fig_downstream_delta_{study_tag}")


def fig_ae_gate(ae: pd.DataFrame, out: Path) -> None:
    sub = ae[ae.study == "capacity_rematch"].sort_values(["seed", "fold"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(sub))
    colours = ["#B7D8B1" if p else "#E6B8B0" for p in sub.passed]
    ax.bar(x, sub.nmse, color=colours, edgecolor=EDGE, linewidth=0.85, width=0.75)
    ax.axhline(0.55, color="#8B3A3A", linestyle="--", linewidth=1.2, label="Gate threshold (max NMSE = 0.55)")
    ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.0, label="Previous aspirational gate (0.50)")
    labels = [f"Seed {int(s)}, {fn} fold" for s, fn in zip(sub.seed, sub.fold_name)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Autoencoder reconstruction NMSE")
    ax.set_title("Autoencoder quality gate — capacity rematch folds (all passed)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _save(fig, out, "fig_ae_gate_capacity")


def fig_eval_levels(cls_df: pd.DataFrame, arms: pd.DataFrame, out: Path) -> None:
    m = arms[(arms.space == "latent") & (arms.metric == "mmd2")].set_index("arm")
    d = arms[(arms.space == "decoded") & (arms.metric == "channel_abs_error")].set_index("arm")
    # paired delta F1
    def mean_delta(arm: str) -> float:
        vals = []
        for _, g in cls_df.groupby(["seed", "fold"]):
            bv, av = g[g.arm == "baseline"], g[g.arm == arm]
            if len(bv) and len(av):
                vals.append(float(av["f1"].iloc[0]) - float(bv["f1"].iloc[0]))
        return float(np.mean(vals)) if vals else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    pairs = [
        (
            axes[0],
            "Level 1 — Latent fidelity",
            r"Holdout MMD$^2$",
            [float(m.loc["quantum", "mean"]), float(m.loc["classical_equiv", "mean"])],
            True,
        ),
        (
            axes[1],
            "Level 2 — Decoded fidelity",
            "Cross-channel absolute error",
            [float(d.loc["quantum", "mean"]), float(d.loc["classical_equiv", "mean"])],
            True,
        ),
        (
            axes[2],
            "Level 3 — Clinical utility",
            r"$\Delta$F1 vs real-only",
            [mean_delta("quantum"), mean_delta("classical_equiv")],
            False,
        ),
    ]
    for ax, title, ylab, vals, lower_better in pairs:
        ax.bar(
            [0, 1],
            vals,
            color=[FILL["quantum"], FILL["classical_equiv"]],
            edgecolor=EDGE,
            linewidth=0.9,
            width=0.62,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["Quantum\nStyle-qGAN", "Classical\ncapacity-equiv."],
            fontsize=9,
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab)
        if not lower_better:
            ax.axhline(0.0, color=EDGE, linewidth=0.9)
    fig.text(0.355, 0.48, "does not\nimply", ha="center", va="center", fontsize=8, color=MUTED, style="italic")
    fig.text(0.645, 0.48, "does not\nimply", ha="center", va="center", fontsize=8, color=MUTED, style="italic")
    fig.suptitle(
        "Three evaluation levels — a latent improvement need not transfer downstream (ae_v5b)",
        fontsize=12,
        y=1.03,
    )
    fig.tight_layout()
    _save(fig, out, "fig_eval_levels_transfer")


def write_classification_csv(cls_df: pd.DataFrame, out: Path, study: str) -> None:
    summ = summarise_cls(cls_df)
    path = out / f"classification_summary_{study}.csv"
    summ.to_csv(path, index=False)
    print(f"wrote {path}")
    # also fold-level
    raw = out / f"classification_folds_{study}.csv"
    cls_df.to_csv(raw, index=False)
    print(f"wrote {raw}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    _style()

    cap = args.root / "capacity_rematch"
    dec = args.root / "decoded_equivalence"
    arms = load_arms(cap)
    ae = ae_nmse_table(args.root)
    cls_cap = collect_classification(args.root, "capacity_rematch")

    fig_latent_mmd(arms, args.out)
    fig_scenario_contrast(arms, args.out)
    fig_decoded_metrics(arms, args.out, "capacity", "capacity rematch")
    fig_classification_panel(cls_cap, args.out, "capacity")
    fig_classification_delta(cls_cap, args.out, "capacity", "synthetic dose ≤ 1 024 windows")
    fig_downstream_f1_readable(cls_cap, args.out, "capacity", "synthetic dose ≤ 1 024 windows")
    fig_ae_gate(ae, args.out)
    fig_eval_levels(cls_cap, arms, args.out)
    write_classification_csv(cls_cap, args.out, "capacity")

    if (dec / "tables" / "arms.csv").exists():
        darms = load_arms(dec)
        cls_dec = collect_classification(args.root, "decoded_equivalence")
        fig_decoded_metrics(darms, args.out, "decoded", "decoded equivalence")
        fig_classification_panel(cls_dec, args.out, "decoded")
        fig_classification_delta(cls_dec, args.out, "decoded", "synthetic dose ≤ 3 000 windows")
        fig_downstream_f1_readable(cls_dec, args.out, "decoded", "synthetic dose ≤ 3 000 windows")
        write_classification_csv(cls_dec, args.out, "decoded")

    # numbers markdown
    base_f1 = float(cls_cap[cls_cap.arm == "baseline"]["f1"].mean())
    q = arms[(arms.arm == "quantum") & (arms.metric == "mmd2")].iloc[0]
    lines = [
        "# ae_v5b numbers for Evaluation tables",
        "",
        f"- Real-only baseline F1 (mean over folds): {base_f1:.4f}",
        f"- Quantum latent MMD²: {q['mean']:.5f} [{q['ci_low']:.5f}, {q['ci_high']:.5f}] (704 parameters)",
        f"- Classification CSVs: classification_summary_capacity.csv / classification_folds_capacity.csv",
        "",
        "## Classification means (capacity rematch)",
        "",
    ]
    for arm in ["baseline"] + ORDER:
        sub = cls_cap[cls_cap.arm == arm]
        if sub.empty:
            continue
        lines.append(
            f"- {SHORT_FULL.get(arm, arm)}: "
            f"F1={sub.f1.mean():.4f}±{sub.f1.std():.4f}, "
            f"AUROC={sub.auroc.mean():.4f}±{sub.auroc.std():.4f}, "
            f"MCC={sub.mcc.mean():.4f}±{sub.mcc.std():.4f}, "
            f"Sens={sub.sensitivity.mean():.4f}, Spec={sub.specificity.mean():.4f}"
        )
    (args.out / "numbers_for_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'numbers_for_tables.md'}")


if __name__ == "__main__":
    main()
