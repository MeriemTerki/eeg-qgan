"""Results figures for the defence deck and for the PFE manuscript.

One script, two targets, one source of truth.

``--target slides``
    Wide canvases, large type, few marks: sized so that nothing is lost when a
    16:9 frame is projected. Written under the ``RES_P*`` names the deck uses.

``--target print``
    A4 column width, 9 pt type, vector PDF. Laid out for the page rather than
    the screen, which sometimes means a different panel arrangement for the
    same data (the decoded distances, for instance, go from one row of four to
    a two-by-two block).

Both targets read the campaign CSVs at run time and share every helper that
selects or aggregates a number, so a figure in the deck and the corresponding
figure in the manuscript cannot drift apart. Nothing here is transcribed by
hand; re-running the campaign changes both.

Outputs
-------
slides  ``eeg_qgan/figures/presentation/``   masters, PNG and PDF
        ``Presentation/figures/03_results/`` PNG copies, slide names
print   ``eeg_qgan/figures/manuscript/``     masters, PNG and PDF
        ``Engineering_PFE/Figures/results_deadline/`` PDF and PNG, chapter names

Usage
-----
    python eeg_qgan/scripts/plot_presentation_results.py               # both
    python eeg_qgan/scripts/plot_presentation_results.py --target print
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── locations ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "eeg_qgan_results_canonical"
PS8 = ROOT / "eeg_qgan_results_paper_style_cpu" / "paper_style_cpu_deadline"
D256 = ROOT / "eeg_qgan_results_dose_deadline_d256" / "dose_deadline"
D512 = ROOT / "eeg_qgan_results_dose_deadline_d512" / "dose_deadline"
F025 = ROOT / "eeg_qgan_results_dataeff_deadline_f0.25" / "dataeff_deadline"
F050 = ROOT / "eeg_qgan_results_dataeff_deadline_f0.5" / "dataeff_deadline"
FOLDS = ROOT / "eeg_qgan" / "figures" / "deadline_campaign" / "detection"

MASTER = {
    "slides": ROOT / "eeg_qgan" / "figures" / "presentation",
    "print": ROOT / "eeg_qgan" / "figures" / "manuscript",
}
DEST = {
    "slides": ROOT / "Presentation" / "figures" / "03_results",
    "print": ROOT / "Engineering_PFE" / "Figures" / "results_deadline",
}

# key -> (master stem, slide name, manuscript name)
NAMES = {
    "r1": ("PRES_R1_latent_rematch", "RES_P1_latent_rematch", "fig_res_rematch_mmd"),
    "r2": ("PRES_R2_mechanism", "RES_P2_mechanism", "fig_res_mechanism"),
    "r3": ("PRES_R3_clinical_transfer", "RES_P3_clinical_transfer", "fig_res_transfer_levels"),
    "r4": ("PRES_R4_decoded_metrics", "RES_P4_decoded_metrics", "fig_res_decoded"),
    "r5": ("PRES_R5_dose_scarcity", "RES_P5_dose_scarcity", "fig_res_dose_scarcity"),
    "r6": ("PRES_R6_robustness", "RES_P6_robustness", "fig_res_robustness"),
    "r7": ("PRES_R7_qubit_width", "RES_P7_qubit_width", "fig_res_qubit_width"),
}

# ── palette, taken from Presentation/preamble.tex ───────────────────────
HYBRID = "#1F4E79"   # ESIBlue
TWIN = "#C0392B"     # AlertRed
GMM = "#1E8449"      # OkGreen
FOURIER = "#A9762B"  # amber, distinct from all three above
SMALLMLP = "#6B7C93" # slate: an unmatched classical arm
LARGE = "#9AA0A6"    # grey: an arm that carries no admissible claim
BASE = "#4D4D4D"     # the real-only reference
ACCENT = "#2E75B6"   # ESIAccent
BAND = "#EAF1F9"     # ESILight, lightened: a background, never a foreground
ESIBLUE_SOFT = "#3A6B96"
INK = "#1A1A1A"
GRID = "#C9CFD6"

LABEL = {
    "quantum": "Hybrid Style-qGAN",
    "classical_equiv": "Capacity-matched twin",
    "classical_matched": "Slightly larger MLP",
    "gmm": "Gaussian mixture",
    "classical_large": "Larger MLP",
    "fourier": "Fourier analogue",
    "baseline": "Real-only",
}
COLOUR = {
    "quantum": HYBRID,
    "classical_equiv": TWIN,
    "classical_matched": SMALLMLP,
    "gmm": GMM,
    "classical_large": LARGE,
    "fourier": FOURIER,
    "baseline": BASE,
}

# ── per-target geometry and type ────────────────────────────────────────
# Print figures target an aspect above ~1.5 so that they fill the A4 column
# width under the 0.46--0.58 textheight cap imposed by \pferesultinclude.
FIGSIZE = {
    "r1": {"slides": (12.6, 4.9), "print": (6.5, 3.9)},
    "r2": {"slides": (13.4, 4.9), "print": (6.6, 4.1)},
    "r3": {"slides": (13.2, 4.6), "print": (6.5, 3.5)},
    "r4": {"slides": (13.6, 4.6), "print": (6.5, 4.4)},
    "r5": {"slides": (13.2, 4.6), "print": (6.5, 3.6)},
    "r6": {"slides": (13.2, 4.6), "print": (6.5, 3.6)},
    "r7": {"slides": (13.2, 4.6), "print": (6.5, 3.6)},
}

RC = {
    "slides": {
        "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "axes.linewidth": 1.1, "grid.linewidth": 0.8, "savefig.dpi": 200,
    },
    "print": {
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.7, "grid.linewidth": 0.5, "savefig.dpi": 400,
    },
}
# annotation type, bar-value type, error-bar weight, marker size
ANN = {"slides": 12.5, "print": 7.3}
VAL = {"slides": 13.0, "print": 7.2}
ELW = {"slides": 1.3, "print": 0.8}
CAP = {"slides": 4.5, "print": 2.2}
MRK = {"slides": 8.0, "print": 4.0}
LWD = {"slides": 2.4, "print": 1.4}

TARGET = "slides"   # set by main()


def t(table: dict):
    return table[TARGET]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 110,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            **RC[TARGET],
        }
    )


def _new(key: str, nrows: int = 1, ncols: int = 1, **kw):
    return plt.subplots(nrows, ncols, figsize=FIGSIZE[key][TARGET],
                        constrained_layout=True, **kw)


def _save(fig: plt.Figure, key: str) -> None:
    master_stem, slide_name, print_name = NAMES[key]
    out = MASTER[TARGET]
    out.mkdir(parents=True, exist_ok=True)
    dest = DEST[TARGET]
    dest.mkdir(parents=True, exist_ok=True)

    stem = master_stem if TARGET == "slides" else print_name
    png, pdf = out / f"{stem}.png", out / f"{stem}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)

    if TARGET == "slides":
        shutil.copyfile(png, dest / f"{slide_name}.png")
        print(f"  {slide_name}.png")
    else:
        # \methodfigure prefers the PDF; the PNG is kept for previews.
        shutil.copyfile(pdf, dest / f"{print_name}.pdf")
        shutil.copyfile(png, dest / f"{print_name}.png")
        print(f"  {print_name}.pdf")


# ── data access ─────────────────────────────────────────────────────────
def latent_mmd(study_dir: Path) -> pd.DataFrame:
    """Latent MMD^2 per arm, with the bootstrap interval already in the CSV."""
    df = pd.read_csv(study_dir / "tables" / "arms.csv")
    df = df[(df["metric"] == "mmd2") & (df["space"] == "latent")]
    return df.set_index("arm")


def decoded_metric(study_dir: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(study_dir / "tables" / "arms.csv")
    df = df[(df["metric"] == metric) & (df["space"] == "decoded")]
    return df.set_index("arm")


def detector(study_dir: Path, metric: str = "auroc") -> pd.DataFrame:
    df = pd.read_csv(study_dir / "tables" / "detector.csv")
    return df[df["metric"] == metric].set_index("source")


def _err(row: pd.Series) -> tuple[float, float]:
    """Asymmetric error bar from a mean and its interval."""
    return (
        max(row["mean"] - row["ci_low"], 0.0),
        max(row["ci_high"] - row["mean"], 0.0),
    )


def _ebar(ax, x, y, yerr=None, xerr=None) -> None:
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="none", ecolor=INK,
                elinewidth=t(ELW), capsize=t(CAP), zorder=4)


def _bar_labels(ax, bars, values, fmt="{:.3f}", dy=0.0, hi=None) -> None:
    """Print the value on each bar so the axis never has to be read precisely.

    ``hi`` is the upper error of each bar. Passing it lifts the label clear of
    the whisker; without it a wide interval draws a line through the text.
    """
    for i, (bar, value) in enumerate(zip(bars, values)):
        top = bar.get_height() + (0.0 if hi is None else hi[i]) + dy
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, top),
            ha="center", va="bottom", fontsize=t(VAL),
            fontweight="bold", color=INK,
        )


# ── R1: the capacity rematch ────────────────────────────────────────────
def fig_latent_rematch() -> None:
    """Latent MMD^2 for every arm of the rematch, on a log axis.

    Five arms spanning three orders of magnitude, so the axis has to be
    logarithmic; the arms that may legitimately be ranked against each other
    are the two sharing a parameter budget, and those rows are shaded.
    """
    d = latent_mmd(CANON / "rematch_deadline")
    order = ["gmm", "quantum", "classical_matched", "classical_equiv", "classical_large"]
    matched = {"quantum", "classical_equiv"}

    fig, ax = _new("r1")
    y = np.arange(len(order))[::-1]

    for yi, arm in zip(y, order):
        if arm in matched:
            ax.axhspan(yi - 0.46, yi + 0.46, color=BAND, zorder=0)

    for yi, arm in zip(y, order):
        row = d.loc[arm]
        lo, hi = _err(row)
        failed = arm == "classical_large"
        ax.barh(
            yi, row["mean"], height=0.60,
            color=COLOUR[arm], alpha=0.35 if failed else 1.0,
            hatch="//" if failed else None,
            edgecolor=COLOUR[arm], linewidth=1.2, zorder=3,
        )
        _ebar(ax, row["mean"], yi, xerr=[[lo], [hi]])
        ax.annotate(f"  {row['mean']:.4f}", (row["mean"] + hi, yi),
                    va="center", ha="left", fontsize=t(VAL),
                    fontweight="bold", color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{LABEL[a]}\n"
            + ("no budget" if a == "gmm"
               else f"{int(d.loc[a, 'parameters']):,} weights".replace(",", " "))
            for a in order
        ]
    )
    ax.set_xscale("log")
    ax.set_xlim(1.9e-4, 4.0)
    ax.set_xlabel("Latent MMD$^2$ against real seizure codes   "
                  "(log scale, lower is better)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    ax.annotate(
        "shaded: the capacity-matched pair,\nthe only two arms that may be ranked",
        xy=(0.30, 3.0), fontsize=t(ANN), style="italic", color=ESIBLUE_SOFT,
        va="center", ha="left", linespacing=1.35,
    )
    ax.annotate("did not converge in any of the 4 folds", xy=(0.020, -0.44),
                fontsize=t(ANN), style="italic", color=TWIN)
    _save(fig, "r1")


# ── R2: what explains the rematch ───────────────────────────────────────
def fig_mechanism() -> None:
    """Three candidate explanations of the rematch, isolated one per panel."""
    ent = latent_mmd(CANON / "entanglement_deadline")
    depth4 = latent_mmd(CANON / "depth_ablation")
    c128 = latent_mmd(CANON / "critic_width")
    c16 = latent_mmd(CANON / "critic_deadline")

    fig, axes = _new("r2", 1, 3)

    # Panel (a) and (b) hold four and three categories in a third of the
    # canvas. On a slide that is wide enough for stacked labels; on the page
    # the same labels collide, so print rotates them instead.
    rot = TARGET == "print"

    # (a) entanglement topology: the claimed quantum resource
    ax = axes[0]
    topo = ["quantum_none", "quantum_linear", "quantum_ring", "quantum_full"]
    names = (["Product (no CZ)", "Linear", "Ring (headline)", "All-to-all"] if rot
             else ["Product\n(no CZ)", "Linear", "Ring\n(headline)", "All-to-all"])
    vals = [ent.loc[k, "mean"] for k in topo]
    errs = np.array([_err(ent.loc[k]) for k in topo]).T
    bars = ax.bar(names, vals, color=[ACCENT, ACCENT, HYBRID, ACCENT],
                  width=0.66, zorder=3)
    _ebar(ax, names, vals, yerr=errs)
    _bar_labels(ax, bars, vals, "{:.3f}", dy=max(vals) * 0.05, hi=errs[1])
    ax.set_title("(a) Entanglement", fontweight="bold")
    ax.set_ylabel("Latent MMD$^2$")
    ax.set_ylim(0, max(vals) * 1.34)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    if rot:
        ax.tick_params(axis="x", rotation=28)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
    if rot:
        # The panel is monotone, so the text alone is unambiguous; an arrow at
        # this width crosses the bar-value labels it is meant to point past.
        ax.annotate("fewest CZ gates\nis best", xy=(-0.42, max(vals) * 1.14),
                    fontsize=t(ANN), style="italic", color=TWIN, va="top")
    else:
        ax.annotate("fewest CZ gates\nis best", xy=(0.06, vals[0] * 1.03),
                    xytext=(0.55, max(vals) * 1.13), fontsize=t(ANN),
                    style="italic", color=TWIN,
                    arrowprops=dict(arrowstyle="->", color=TWIN,
                                    linewidth=t(ELW)))

    # (b) function class: is the circuit's bias classically available
    ax = axes[1]
    fam = ["fourier", "quantum", "classical_equiv"]
    names = (["Fourier analogue", "Hybrid Style-qGAN", "Capacity-matched twin"] if rot
             else ["Fourier\nanalogue", "Hybrid\nStyle-qGAN", "Capacity-\nmatched twin"])
    vals = [depth4.loc[k, "mean"] for k in fam]
    errs = np.array([_err(depth4.loc[k]) for k in fam]).T
    bars = ax.bar(names, vals, color=[COLOUR[k] for k in fam], width=0.66, zorder=3)
    _ebar(ax, names, vals, yerr=errs)
    _bar_labels(ax, bars, vals, "{:.3f}", dy=max(vals) * 0.05, hi=errs[1])
    ax.set_title("(b) Function class", fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.30)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    if rot:
        ax.tick_params(axis="x", rotation=28)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
    ax.annotate("a classical Fourier model\nis closer still", xy=(0, vals[0]),
                xytext=(0.10, max(vals) * 1.11), fontsize=t(ANN),
                style="italic", color=TWIN)

    # (c) critic width: is the ranking a property of the judge
    ax = axes[2]
    x = np.arange(2)
    w = 0.34
    q = [c128.loc["quantum", "mean"], c16.loc["quantum", "mean"]]
    tw = [c128.loc["classical_equiv", "mean"], c16.loc["classical_equiv", "mean"]]
    qe = np.array([_err(c128.loc["quantum"]), _err(c16.loc["quantum"])]).T
    te = np.array([_err(c128.loc["classical_equiv"]),
                   _err(c16.loc["classical_equiv"])]).T
    b1 = ax.bar(x - w / 2, q, w, label="Hybrid", color=HYBRID, zorder=3)
    b2 = ax.bar(x + w / 2, tw, w, label="Twin", color=TWIN, zorder=3)
    _ebar(ax, x - w / 2, q, yerr=qe)
    _ebar(ax, x + w / 2, tw, yerr=te)
    top = max(q + tw)
    _bar_labels(ax, b1, q, "{:.3f}", dy=top * 0.04, hi=qe[1])
    _bar_labels(ax, b2, tw, "{:.3f}", dy=top * 0.04, hi=te[1])
    ax.set_xticks(x)
    ax.set_xticklabels(["Critic [128,128]\n(headline)", "Critic [16,16]\n(thin)"])
    ax.set_title("(c) Critic width", fontweight="bold")
    ax.set_ylim(0, top * 1.38)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8,
              handlelength=1.1)
    ax.annotate("ranking reverses", xy=(1, max(q[1], tw[1])),
                xytext=(0.56, top * 1.14), fontsize=t(ANN),
                style="italic", color=TWIN)

    _save(fig, "r2")


# ── R3: does the latent ranking reach the clinic ────────────────────────
def fig_clinical_transfer() -> None:
    """The same folds read at Level 1 and at Level 3."""
    det = detector(PS8, "auroc")
    base = det.loc["baseline", "mean"]
    arms = ["augmented_quantum", "augmented_classical_equiv", "augmented_gmm"]
    keys = ["quantum", "classical_equiv", "gmm"]
    names = ["Hybrid", "Twin", "GMM"]
    lat = latent_mmd(PS8)

    fig, axes = _new("r3", 1, 2, gridspec_kw={"width_ratios": [1.0, 1.15]})

    # left: the generative ordering
    ax = axes[0]
    vals = [lat.loc[k, "mean"] for k in keys]
    errs = np.array([_err(lat.loc[k]) for k in keys]).T
    bars = ax.bar(names, vals, color=[COLOUR[k] for k in keys], width=0.6, zorder=3)
    _ebar(ax, names, vals, yerr=errs)
    _bar_labels(ax, bars, vals, "{:.3f}", dy=max(vals) * 0.035, hi=errs[1])
    # The parenthetical fits a slide title; on the page the two panel titles
    # would run into each other, and the caption already states both facts.
    ax.set_title("Level 1 — latent MMD$^2$" if TARGET == "print"
                 else "Level 1 — latent MMD$^2$   (lower is better)",
                 fontweight="bold")
    ax.set_ylabel("Latent MMD$^2$")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # right: whether that ordering survives into detection
    ax = axes[1]
    y = np.arange(len(arms))[::-1]
    for yi, src, key in zip(y, arms, keys):
        row = det.loc[src]
        delta = row["delta_mean"]
        lo = max(delta - row["delta_ci_low"], 0.0)
        hi = max(row["delta_ci_high"] - delta, 0.0)
        ax.barh(yi, delta, height=0.55, color=COLOUR[key], zorder=3)
        _ebar(ax, delta, yi, xerr=[[lo], [hi]])
        ax.annotate(f"  {delta:+.4f}", (max(delta + hi, 0.0005), yi),
                    va="center", ha="left", fontsize=t(VAL),
                    fontweight="bold", color=INK)
    ax.axvline(0, color=BASE, linewidth=1.6, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(r"$\Delta$AUROC against the real-only detector")
    ax.set_xlim(-0.004, 0.026)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Level 3 — detection" if TARGET == "print"
                 else f"Level 3 — detection   (real-only AUROC = {base:.3f})",
                 fontweight="bold")
    _save(fig, "r3")


# ── R4: does a latent win survive decoding ──────────────────────────────
def fig_decoded_metrics() -> None:
    """Level 2. The four distances tabulated in the chapter, plus KS.

    One row of four reads well on a slide; on the page the same four panels
    are squeezed below legibility, so print uses a two-by-two block.
    """
    study = CANON / "rematch_deadline"
    metrics = [
        ("autocorrelation_distance", "Autocorrelation distance"),
        ("channel_abs_error", "Channel $L_1$ error"),
        ("spectral_band_error", "Spectral band error"),
        ("decoded_sliced_w1", "Decoded sliced $W_1$"),
    ]
    keys = ["gmm", "quantum", "classical_equiv"]
    names = ["GMM", "Hybrid", "Twin"]

    if TARGET == "print":
        fig, axgrid = _new("r4", 2, 2)
        axes = axgrid.ravel()
    else:
        fig, axes = _new("r4", 1, 4)

    for ax, (metric, title) in zip(axes, metrics):
        d = decoded_metric(study, metric)
        vals = [d.loc[k, "mean"] for k in keys]
        errs = np.array([_err(d.loc[k]) for k in keys]).T
        bars = ax.bar(names, vals, color=[COLOUR[k] for k in keys],
                      width=0.62, zorder=3)
        _ebar(ax, names, vals, yerr=errs)
        _bar_labels(ax, bars, vals, "{:.3f}", dy=max(vals) * 0.03, hi=errs[1])
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.30)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        # Call the hybrid-versus-twin comparison only where the bootstrap
        # intervals are disjoint; overlapping means are not a direction.
        hyb, twn = d.loc["quantum"], d.loc["classical_equiv"]
        if twn["ci_high"] < hyb["ci_low"]:
            verdict, colour = "twin closer", TWIN
        elif hyb["ci_high"] < twn["ci_low"]:
            verdict, colour = "hybrid closer", HYBRID
        else:
            verdict = None
        if verdict:
            ax.annotate(verdict, xy=(0.5, 0.93), xycoords="axes fraction",
                        ha="center", fontsize=t(ANN), style="italic", color=colour)

    if TARGET == "print":
        for ax in axgrid[:, 0]:
            ax.set_ylabel("Distance to real windows")
    else:
        axes[0].set_ylabel("Distance to real windows\n(lower is better)")
    _save(fig, "r4")


# ── R5: dose and scarcity ───────────────────────────────────────────────
def fig_dose_scarcity() -> None:
    """Synthetic dose on the left, generator-data fraction on the right."""
    keys = ["quantum", "classical_equiv", "gmm"]
    srcs = ["augmented_quantum", "augmented_classical_equiv", "augmented_gmm"]
    dose_dirs = [D256, D512, PS8]
    frac_dirs = [F025, F050, PS8]
    frac_x = [25, 50, 100]

    fig, axes = _new("r5", 1, 2)

    # (a) dose. Intervals are bars, not bands: three shaded regions this wide
    # merge into an unreadable wash.
    ax = axes[0]
    doses = [int(detector(d).loc["augmented_quantum", "mean_dose"]) for d in dose_dirs]
    for i, (key, src) in enumerate(zip(keys, srcs)):
        ys = np.array([detector(d).loc[src, "delta_mean"] for d in dose_dirs])
        los = np.array([detector(d).loc[src, "delta_ci_low"] for d in dose_dirs])
        his = np.array([detector(d).loc[src, "delta_ci_high"] for d in dose_dirs])
        xs = np.array(doses, dtype=float) * (1 + (i - 1) * 0.022)
        ax.errorbar(xs, ys, yerr=[ys - los, his - ys], fmt="-o", color=COLOUR[key],
                    linewidth=t(LWD), markersize=t(MRK), elinewidth=t(ELW),
                    capsize=t(CAP), label=LABEL[key], zorder=3 + i)
    ax.axhline(0, color=BASE, linewidth=1.5, zorder=2)
    ax.set_xticks(doses)
    ax.set_xlabel("Synthetic clips added to training")
    ax.set_ylabel(r"$\Delta$AUROC vs real-only")
    ax.set_title("(a) Synthetic dose", fontweight="bold")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # (b) generator-data fraction
    ax = axes[1]
    for i, (key, src) in enumerate(zip(keys, srcs)):
        ys = np.array([detector(d).loc[src, "delta_mean"] for d in frac_dirs])
        los = np.array([detector(d).loc[src, "delta_ci_low"] for d in frac_dirs])
        his = np.array([detector(d).loc[src, "delta_ci_high"] for d in frac_dirs])
        xs = np.array(frac_x, dtype=float) + (i - 1) * 1.6
        ax.errorbar(xs, ys, yerr=[ys - los, his - ys], fmt="-o", color=COLOUR[key],
                    linewidth=t(LWD), markersize=t(MRK), elinewidth=t(ELW),
                    capsize=t(CAP), label=LABEL[key], zorder=3 + i)
    ax.axhline(0, color=BASE, linewidth=1.5, zorder=2)
    ax.set_xticks(frac_x)
    ax.set_xticklabels([f"{v}%" for v in frac_x])
    # On the page this label is wider than its own axes, which pushes it past
    # the tight bounding box and clips the last word.
    ax.set_xlabel("Real seizure codes available\nto the generator" if TARGET == "print"
                  else "Real seizure codes available to the generator")
    ax.set_title("(b) Data scarcity", fontweight="bold")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    peak = detector(F050).loc["augmented_quantum", "delta_mean"]
    # Print has a caption to carry the caveat, so it only needs the value.
    note = (f"largest hybrid lift {peak:+.3f}\n(exploratory)" if TARGET == "print"
            else f"largest hybrid lift {peak:+.3f}\n"
                 "exploratory: twin and GMM lift too")
    ax.annotate(note, xy=(51.5, peak), xytext=(58, peak * 0.66),
                fontsize=t(ANN), style="italic", color=HYBRID,
                arrowprops=dict(arrowstyle="->", color=HYBRID, linewidth=t(ELW)))

    handles = [Line2D([], [], color=COLOUR[k], marker="o", linewidth=t(LWD),
                      markersize=t(MRK), label=LABEL[k]) for k in keys]
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               frameon=False, handlelength=1.8)
    _save(fig, "r5")


# ── R6: is the ranking an artefact of depth or of the host ──────────────
def fig_robustness() -> None:
    """Halve the circuit depth, then change the encoder geometry."""
    l2 = latent_mmd(CANON / "depth_deadline")
    l4 = latent_mmd(CANON / "depth_ablation")
    temp = latent_mmd(CANON / "bottleneck_comparison")
    pool = latent_mmd(CANON / "bottleneck_pooled")

    fig, axes = _new("r6", 1, 2)
    x = np.arange(2)
    # Print panels are half the width of the slide ones, so the full arm names
    # make the legend box wide enough to sit on top of the tallest bar label.
    short = {"quantum": "Hybrid", "classical_equiv": "Twin", "fourier": "Fourier"}
    leg = (lambda k: short[k]) if TARGET == "print" else (lambda k: LABEL[k])

    # (a) depth
    ax = axes[0]
    keys = ["quantum", "classical_equiv", "fourier"]
    w = 0.26
    top = l2.loc["classical_equiv", "mean"]
    for i, key in enumerate(keys):
        vals = [l2.loc[key, "mean"], l4.loc[key, "mean"]]
        errs = np.array([_err(l2.loc[key]), _err(l4.loc[key])]).T
        off = (i - 1) * w
        bars = ax.bar(x + off, vals, w, color=COLOUR[key], label=leg(key), zorder=3)
        _ebar(ax, x + off, vals, yerr=errs)
        _bar_labels(ax, bars, vals, "{:.3f}", dy=top * 0.02, hi=errs[1])
    ax.set_xticks(x)
    ax.set_xticklabels(["$L=2$ layers", "$L=4$ layers (headline)"])
    ax.set_ylabel("Latent MMD$^2$")
    ax.set_title("(a) Circuit depth", fontweight="bold")
    ax.set_ylim(0, top * 1.34)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")

    # (b) bottleneck
    ax = axes[1]
    keys = ["quantum", "classical_equiv"]
    w = 0.30
    top = temp.loc["classical_equiv", "mean"]
    for i, key in enumerate(keys):
        vals = [temp.loc[key, "mean"], pool.loc[key, "mean"]]
        errs = np.array([_err(temp.loc[key]), _err(pool.loc[key])]).T
        off = (i - 0.5) * w
        bars = ax.bar(x + off, vals, w, color=COLOUR[key], label=leg(key), zorder=3)
        _ebar(ax, x + off, vals, yerr=errs)
        _bar_labels(ax, bars, vals, "{:.3f}", dy=top * 0.02, hi=errs[1])
    ax.set_xticks(x)
    ax.set_xticklabels(["Temporal (headline)", "Pooled"])
    ax.set_title("(b) Autoencoder bottleneck", fontweight="bold")
    ax.set_ylim(0, top * 1.46)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center")
    _save(fig, "r6")


# ── R7: is the detection null an artefact of qubit count ────────────────
def fig_qubit_width() -> None:
    """Paired per-fold change in AUROC at eight and at four qubits.

    Pairing is within a fold: each arm is differenced against the real-only
    detector trained on the very same split, which is the only comparison the
    design supports.
    """
    fig, axes = _new("r7", 1, 2, sharey=True)
    keys = ["quantum", "classical_equiv", "gmm"]
    names = ["Hybrid", "Twin", "GMM"]

    for ax, tag, title in zip(
        axes, ["8q", "4q"],
        ["(a) Headline: 8 qubits", "(b) Width check: 4 qubits"],
    ):
        folds = pd.read_csv(FOLDS / f"classification_folds_paper_style_{tag}.csv")
        wide = folds.pivot_table(index=["seed", "fold"], columns="arm", values="auroc")
        means, sds, points = [], [], []
        for key in keys:
            paired = wide[key] - wide["baseline"]
            means.append(paired.mean())
            sds.append(paired.std(ddof=1))
            points.append(paired.values)

        ax.bar(names, means, width=0.58, color=[COLOUR[k] for k in keys], zorder=3)
        _ebar(ax, names, means, yerr=sds)
        for xi, vals in enumerate(points):      # every fold, shown honestly
            ax.scatter(np.full_like(vals, xi, dtype=float), vals,
                       s=26 if TARGET == "slides" else 11,
                       facecolor="white", edgecolor=INK,
                       linewidth=0.9, zorder=5)
        # The mean goes in the tick label: near a zero line, a floating
        # annotation lands on top of whichever bar happens to be positive.
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{n}\n{m:+.4f}" for n, m in zip(names, means)])
        ax.axhline(0, color=BASE, linewidth=1.6, zorder=6)
        ax.set_title(title, fontweight="bold")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"$\Delta$AUROC vs real-only" "\n" "(paired, per fold)")
    if TARGET == "slides":
        # A slide has no caption to carry this; the manuscript figure does.
        fig.text(0.5, -0.065,
                 "Open circles are the four individual folds; bars are their mean, "
                 "whiskers one standard deviation.",
                 ha="center", fontsize=t(ANN), style="italic", color=BASE)
    _save(fig, "r7")


BUILDERS = [
    fig_latent_rematch,
    fig_mechanism,
    fig_clinical_transfer,
    fig_decoded_metrics,
    fig_dose_scarcity,
    fig_robustness,
    fig_qubit_width,
]


def build(target: str) -> None:
    global TARGET
    TARGET = target
    _style()
    print(f"\n[{target}]")
    for builder in BUILDERS:
        builder()
    print(f"  masters: {MASTER[target].relative_to(ROOT)}")
    print(f"  copies:  {DEST[target].relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["slides", "print", "both"], default="both")
    args = ap.parse_args()
    targets = ["slides", "print"] if args.target == "both" else [args.target]
    for target in targets:
        build(target)


if __name__ == "__main__":
    main()


