#!/usr/bin/env python3
"""Publication figures for the current d=64 / 8q / recording-norm campaign.

Uses only local mirrors of Narval deadline studies. Does not mix ae_v5b/v5c
tables. Incomplete dose-512 / dataeff folders are listed, not plotted as 4-fold
jury results.

    python eeg_qgan/scripts/plot_deadline_campaign.py
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THESIS = Path(__file__).resolve().parents[2]
OUT = THESIS / "eeg_qgan" / "figures" / "deadline_campaign"
PFE_OUT = THESIS / "Engineering_PFE" / "Figures" / "results_deadline"
CANON = THESIS / "eeg_qgan_results_canonical"
PAPER8 = THESIS / "eeg_qgan_results_paper_style_cpu" / "paper_style_cpu_deadline"
PAPER4 = THESIS / "eeg_qgan_results_paper_style_q4" / "paper_style_q4"
DOSE256 = THESIS / "eeg_qgan_results_dose_deadline_d256" / "dose_deadline"
DOSE512 = THESIS / "eeg_qgan_results_dose_deadline_d512" / "dose_deadline"
FRAC25 = THESIS / "eeg_qgan_results_dataeff_deadline_f0.25" / "dataeff_deadline"
FRAC50 = THESIS / "eeg_qgan_results_dataeff_deadline_f0.5" / "dataeff_deadline"


def _load_plot():
    path = Path(__file__).resolve().parent / "plot_ae_v5c_results.py"
    spec = importlib.util.spec_from_file_location("plot_ae_v5c_results", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _n_json(study: Path) -> int:
    return len(list(study.glob("results_*.json"))) if study.exists() else 0


def fig_eval_levels(plot, cls_df: pd.DataFrame, arms: pd.DataFrame, out: Path) -> None:
    m = arms[(arms.space == "latent") & (arms.metric == "mmd2")].set_index("arm")
    if "quantum" not in m.index or "classical_equiv" not in m.index or cls_df.empty:
        return

    def mean_delta(arm: str, key: str) -> float:
        vals = []
        for _, g in cls_df.groupby(["seed", "fold"]):
            bv, av = g[g.arm == "baseline"], g[g.arm == arm]
            if len(bv) and len(av) and key in bv.columns:
                vals.append(float(av[key].iloc[0]) - float(bv[key].iloc[0]))
        return float(np.mean(vals)) if vals else np.nan

    fig, axes = plt.subplots(3, 1, figsize=(plot.PRINT_W, 9.6), constrained_layout=True)
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
            color=[plot._color("quantum"), plot._color("classical_equiv")],
            edgecolor=plot.EDGE,
            linewidth=0.9,
            width=0.55,
        )
        plot._set_xticks(ax, ["Quantum", "Twin"], fontsize=12)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_ylabel(ylab)
        if not lower_better:
            ax.axhline(0.0, color=plot.EDGE, linewidth=0.9)
            plot._pad_ylim(ax, vals, vals, zero_line=True)
        else:
            plot._pad_ylim(ax, vals, vals, zero_line=False)
        plot._label_bars(ax, fontsize=11)
    plot._save(fig, out, "fig_eval_levels_transfer")


def fig_qubit_mmd(plot, arms8: pd.DataFrame, arms4: pd.DataFrame, out: Path) -> None:
    rows = []
    for label, df in (("8 qubits", arms8), ("4 qubits", arms4)):
        if df is None or df.empty:
            continue
        m = df[(df.space == "latent") & (df.metric == "mmd2")]
        for arm in ("quantum", "classical_equiv", "gmm"):
            sub = m[m.arm == arm]
            if sub.empty:
                continue
            rows.append(
                {
                    "qubits": label,
                    "arm": arm,
                    "mean": float(sub["mean"].iloc[0]),
                    "lo": float(sub["ci_low"].iloc[0]),
                    "hi": float(sub["ci_high"].iloc[0]),
                }
            )
    if not rows:
        return
    tab = pd.DataFrame(rows)
    qubits = [q for q in ("8 qubits", "4 qubits") if q in set(tab.qubits)]
    arms = [a for a in ("gmm", "quantum", "classical_equiv") if a in set(tab.arm)]
    fig, ax = plt.subplots(figsize=(plot.PRINT_W, 4.4))
    x = np.arange(len(qubits))
    width = 0.25
    for i, arm in enumerate(arms):
        means, los, his = [], [], []
        for q in qubits:
            sub = tab[(tab.qubits == q) & (tab.arm == arm)]
            means.append(float(sub["mean"].iloc[0]))
            los.append(float(sub["lo"].iloc[0]))
            his.append(float(sub["hi"].iloc[0]))
        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            yerr=plot._err(means, los, his),
            color=plot._color(arm),
            edgecolor=plot.EDGE,
            linewidth=0.85,
            capsize=3,
            label=plot.SHORT[arm],
            error_kw={"ecolor": plot.EDGE, "elinewidth": 0.8},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(qubits, fontsize=11)
    ax.set_ylabel(r"Holdout latent MMD$^2$  (lower is better)")
    ax.set_title("8-qubit hybrid versus 4-qubit hybrid on the same latent host")
    ax.legend(frameon=False, fontsize=10)
    plot._pad_ylim(ax, tab["lo"], tab["hi"], zero_line=False)
    plot._label_bars(ax, fontsize=8)
    fig.tight_layout()
    plot._save(fig, out, "fig_qubit_latent_mmd")


def _plot_cls_study(plot, study: Path, det: Path, tag: str, title: str, dose_note: str, mmd_name: str) -> None:
    if _n_json(study) < 4:
        print(f"skip {tag}: {_n_json(study)}/4 JSON")
        return
    cls = plot.collect_classification(study)
    if cls.empty:
        print(f"skip {tag}: no detector rows")
        return
    plot.fig_classification_grid(cls, det, tag, title)
    plot.fig_classification_delta_primary(cls, det, tag, dose_note)
    plot.fig_classification_delta_extended(cls, det, tag)
    plot.write_cls_csvs(cls, det, tag)
    arms = plot.load_arms_from_study(study)
    if arms is not None:
        plot.fig_bar_mmd(
            arms,
            det,
            mmd_name,
            title.split("—")[0].strip() + " — latent MMD²",
            order=["quantum", "classical_equiv", "gmm"],
        )


def fig_dataeff_auroc(plot, complete: list[tuple[str, Path]], out: Path) -> None:
    records = []
    for frac, path in complete:
        det = path / "tables" / "detector.csv"
        if not det.exists():
            continue
        df = pd.read_csv(det)
        for source, arm in (
            ("augmented_quantum", "quantum"),
            ("augmented_classical_equiv", "classical_equiv"),
            ("augmented_gmm", "gmm"),
        ):
            sub = df[(df.source == source) & (df.metric == "auroc")]
            if sub.empty:
                continue
            records.append({"frac": frac, "arm": arm, "delta": float(sub["delta_mean"].iloc[0])})
    if not records:
        return
    tab = pd.DataFrame(records)
    fracs = list(dict.fromkeys(tab.frac))
    arms = ["quantum", "classical_equiv", "gmm"]
    fig, ax = plt.subplots(figsize=(plot.PRINT_W, 4.4))
    x = np.arange(len(fracs))
    width = 0.25
    for i, arm in enumerate(arms):
        vals = [float(tab[(tab.frac == f) & (tab.arm == arm)]["delta"].iloc[0]) if len(tab[(tab.frac == f) & (tab.arm == arm)]) else np.nan for f in fracs]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=plot._color(arm),
            edgecolor=plot.EDGE,
            linewidth=0.85,
            label=plot.SHORT[arm],
        )
    ax.axhline(0.0, color=plot.EDGE, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}" for f in fracs], fontsize=11)
    ax.set_xlabel("Generator training fraction (1.0 = full training fold)")
    ax.set_ylabel(r"$\Delta$AUROC vs real-only baseline")
    ax.set_title("Detection under generator-data scarcity")
    ax.legend(frameon=False, fontsize=10)
    lows = [float(v) for v in tab["delta"]]
    plot._pad_ylim(ax, lows, lows, zero_line=True)
    plot._label_bars(ax, fontsize=8)
    fig.tight_layout()
    plot._save(fig, out, "fig_dataeff_detection_delta")


def fig_dose_auroc(plot, complete: list[tuple[int, Path]], out: Path) -> None:
    """ΔAUROC from detector.csv for studies that are 4/4."""
    records = []
    for dose, path in complete:
        det = path / "tables" / "detector.csv"
        if not det.exists():
            continue
        df = pd.read_csv(det)
        for source, arm in (
            ("augmented_quantum", "quantum"),
            ("augmented_classical_equiv", "classical_equiv"),
            ("augmented_gmm", "gmm"),
        ):
            sub = df[(df.source == source) & (df.metric == "auroc")]
            if sub.empty:
                continue
            records.append(
                {
                    "dose": dose,
                    "arm": arm,
                    "delta": float(sub["delta_mean"].iloc[0]),
                    "sig": bool(sub["delta_significant"].iloc[0]),
                }
            )
    if not records:
        return
    tab = pd.DataFrame(records)
    doses = sorted(tab.dose.unique())
    arms = ["quantum", "classical_equiv", "gmm"]
    fig, ax = plt.subplots(figsize=(plot.PRINT_W, 4.4))
    x = np.arange(len(doses))
    width = 0.25
    for i, arm in enumerate(arms):
        vals = []
        for d in doses:
            sub = tab[(tab.dose == d) & (tab.arm == arm)]
            vals.append(float(sub["delta"].iloc[0]) if len(sub) else np.nan)
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=plot._color(arm),
            edgecolor=plot.EDGE,
            linewidth=0.85,
            label=plot.SHORT[arm],
        )
    ax.axhline(0.0, color=plot.EDGE, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in doses], fontsize=11)
    ax.set_xlabel("Synthetic clips added (operating dose delivers about 910)")
    ax.set_ylabel(r"$\Delta$AUROC vs real-only baseline")
    ax.set_title("Detection versus synthetic dose")
    ax.legend(frameon=False, fontsize=10)
    lows = [float(v) for v in tab["delta"]]
    plot._pad_ylim(ax, lows, lows, zero_line=True)
    plot._label_bars(ax, fontsize=8)
    fig.tight_layout()
    plot._save(fig, out, "fig_dose_detection_delta")


def _write_numbers(out: Path) -> None:
    text = """# Numbers and claims for the current architecture (jury)

Protocol: d=64, recording-norm, AE 80/20, 80k windows, 1000 GAN steps,
TUH E1 train vs dev/eval. Detection: 10.24 s clips, `stft_bilstm`.
Do not mix with ae_v5b / Study XI.

## What the pipeline change fixed

- AE reconstruction now passes the gate (val NMSE ~0.40 vs old ~0.67–0.85).
- Outer split is the TUH folder protocol (patient-disjoint by dataset design).
- Detection uses 10.24 s STFT-BiLSTM, not 1.28 s windows.

## Generative claim (supported)

Hybrid 8-qubit Style-qGAN **beats the equal-budget MLP** on latent MMD²
(~0.016 vs ~0.088 on rematch; ~0.020 vs ~0.086 on paper-style 8q).
GMM remains closest to real (~0.0008). `classical_large` failed to converge
on rematch (4/4) — do not treat it as a capacity result.

Entanglement: product (none) 0.011 < linear 0.014 < ring 0.016 < full 0.018,
all still << classical 0.093. Extra connectivity is not free under this budget.

Critic width: at **h16**, quantum MMD² 0.073 **loses** to the twin 0.058.
The rematch is critic-sensitive. Use rematch_deadline / paper-style (default
critic) for the headline generative claim; report h16 as a control.

## Detection claim (supported — do not overstate)

Paper-style 8q, dose ~910, 4/4 folds, baseline AUROC **0.821**:

- quantum ΔAUROC +0.000 n.s.
- classical_equiv ΔAUROC **+0.005\\***
- GMM ΔAUROC **+0.010\\***
- quantum does lift AUPRC/MCC; ECE gets worse

Dose 256 (4/4), ~227 clips, baseline AUROC **0.826**:

- quantum ΔAUROC +0.003 n.s. (AUPRC/MCC/F1 small \\*)
- GMM ΔAUROC **+0.008\\***

Dose 512 (4/4), ~455 clips, baseline AUROC **0.821**:

- quantum ΔAUROC +0.008 n.s. (AUPRC/MCC/F1 \\*)
- classical_equiv ΔAUROC **+0.013\\***
- GMM ΔAUROC +0.002 n.s.

Data-efficiency frac=0.25 (4/4), dose ~910, baseline AUROC **0.821**:

- quantum ΔAUROC **+0.009\\***
- classical_equiv ΔAUROC **+0.010\\***
- GMM ΔAUROC +0.004 n.s.

**Thesis sentence:** Quantum wins the capacity-controlled generative rematch.
It does **not** consistently win primary AUROC against a real-only STFT-BiLSTM
baseline. At full data, GMM or the equal-budget MLP is the detection winner.
At 25% generator data, both quantum and the MLP show a small significant
AUROC lift.
"""
    frac50 = FRAC50 / "tables" / "detector.csv"
    if _n_json(FRAC50) >= 4 and frac50.exists():
        df = pd.read_csv(frac50)
        def _row(source: str) -> pd.Series:
            return df[(df.source == source) & (df.metric == "auroc")].iloc[0]

        def _fmt(source: str) -> str:
            r = _row(source)
            star = r"\\*" if bool(r["delta_significant"]) else " n.s."
            return f"{float(r['delta_mean']):+.3f}{star}"

        base = float(_row("baseline")["mean"])
        text += f"""
Data-efficiency frac=0.5 (4/4), dose ~910, baseline AUROC **{base:.3f}**:

- quantum ΔAUROC **{_fmt("augmented_quantum")}**
- classical_equiv ΔAUROC **{_fmt("augmented_classical_equiv")}**
- GMM ΔAUROC **{_fmt("augmented_gmm")}**
"""
    else:
        text += """
## Still incomplete

- dataeff 0.5: waiting for `results_s1_f1-2.json` (4/4 + analyze)
"""
    text += "\n"
    (out / "numbers_for_tables.md").write_text(text, encoding="utf-8")


def _write_index(out: Path, pngs: list[str], status: list[str]) -> None:
    lines = [
        "# Deadline campaign figures",
        "",
        "Current architecture only. Regenerated by",
        "`python eeg_qgan/scripts/plot_deadline_campaign.py`.",
        "",
        "## Completeness",
        "",
        *status,
        "",
        "## Figures",
        "",
        *[f"- `{n}`" for n in pngs],
        "",
        "Copy PNG/PDF into `Engineering_PFE/Figures/` as needed.",
        "",
    ]
    (out / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    plot = _load_plot()
    plot._style()
    plot.FILL.setdefault("fourier", "#C4A574")
    plot.FULL_LABEL.setdefault("fourier", "Fourier surrogate\n(32k parameters)")
    plot.SHORT.setdefault("fourier", "Fourier surrogate")
    out = OUT
    mech = out / "mechanism"
    det = out / "detection"
    mech.mkdir(parents=True, exist_ok=True)
    det.mkdir(parents=True, exist_ok=True)

    specs = (
        (
            CANON / "rematch_deadline",
            "fig_latent_mmd_capacity",
            "Capacity rematch — latent distribution fidelity",
            ["quantum", "classical_equiv", "classical_matched", "classical_large", "gmm"],
        ),
        (
            CANON / "decoded_deadline",
            "fig_decoded_latent_mmd",
            "Decoded-space study — latent distribution fidelity",
            ["quantum", "classical_equiv", "classical_large", "gmm"],
        ),
        (
            CANON / "entanglement_deadline",
            "fig_entanglement_latent_mmd",
            "Entanglement topology — latent fidelity at fixed parameter count",
            ["quantum_none", "quantum_linear", "quantum_ring", "quantum_full", "classical_equiv"],
        ),
        (
            CANON / "depth_deadline",
            "fig_depth_L2_latent_mmd",
            "Circuit depth L = 2 — latent distribution fidelity",
            ["quantum", "classical_equiv", "fourier"],
        ),
        (
            CANON / "depth_ablation",
            "fig_depth_L4_latent_mmd",
            "Circuit depth L = 4 — latent distribution fidelity",
            ["quantum", "classical_equiv", "fourier"],
        ),
        (
            CANON / "critic_deadline",
            "fig_critic_h16_latent_mmd",
            "Thin critic (hidden width 16) — latent distribution fidelity",
            ["quantum", "classical_equiv"],
        ),
        (
            CANON / "critic_width",
            "fig_critic_h128_latent_mmd",
            "Default critic (hidden width 128) — latent distribution fidelity",
            ["quantum", "classical_equiv"],
        ),
        (
            CANON / "surrogate",
            "fig_surrogate_latent_mmd",
            "Function-class control — circuit versus Fourier analogue versus equal-budget MLP",
            ["quantum_none", "quantum", "classical_equiv", "fourier"],
        ),
        (
            CANON / "bottleneck_comparison",
            "fig_bottleneck_temporal_mmd",
            "Temporal bottleneck — latent distribution fidelity",
            ["quantum", "classical_equiv"],
        ),
        (
            CANON / "bottleneck_pooled",
            "fig_bottleneck_pooled_mmd",
            "Pooled bottleneck — latent distribution fidelity",
            ["quantum", "classical_equiv"],
        ),
    )
    for study, name, title, order in specs:
        arms = plot.load_arms_from_study(study)
        if arms is None:
            print(f"skip {study.name}: no tables/arms.csv")
            continue
        plot.fig_bar_mmd(arms, mech, name, title, order=order)
        plot.fig_paired_delta(
            arms,
            mech,
            name.replace("_mmd", "_delta").replace("_latent", ""),
            title.split("—")[0].strip() + " — Δ vs classical twin",
        )
        print(f"wrote mechanism figures for {study.name}")

    rematch_arms = plot.load_arms_from_study(CANON / "rematch_deadline")
    if rematch_arms is not None:
        plot.fig_scenario_contrast(rematch_arms, mech)

    paper_arms = plot.load_arms_from_study(PAPER8)
    paper_cls = plot.collect_classification(PAPER8) if PAPER8.exists() else pd.DataFrame()
    if paper_arms is not None:
        plot.fig_bar_mmd(
            paper_arms,
            det,
            "fig_paper_style_8q_latent_mmd",
            "Clinical detection folds — latent distribution fidelity",
            order=["quantum", "classical_equiv", "gmm"],
        )
    if not paper_cls.empty and _n_json(PAPER8) >= 4:
        plot.fig_classification_grid(
            paper_cls,
            det,
            "paper_style_8q",
            "Seizure detection — 8-qubit hybrid, 10.24 s spectrogram clips",
        )
        plot.fig_classification_delta_primary(
            paper_cls,
            det,
            "paper_style_8q",
            "real plus class-balanced synthetic training; real holdout; $\\approx$ 910 clips",
        )
        plot.fig_classification_delta_extended(paper_cls, det, "paper_style_8q")
        plot.write_cls_csvs(paper_cls, det, "paper_style_8q")
        if paper_arms is not None:
            fig_eval_levels(plot, paper_cls, paper_arms, det)

    if _n_json(PAPER8) >= 4:
        train_hist = plot.collect_training_histories(PAPER8)
        if train_hist.get("ae") or train_hist.get("gan"):
            plot.fig_training_curves(
                train_hist,
                det,
                "paper_style_8q",
                "Clinical detection campaign",
            )
    if _n_json(CANON / "rematch_deadline") >= 4:
        rematch_hist = plot.collect_training_histories(CANON / "rematch_deadline")
        if rematch_hist.get("ae") or rematch_hist.get("gan"):
            plot.fig_training_curves(
                rematch_hist,
                mech,
                "rematch",
                "Capacity-controlled rematch",
            )

    q4_cls = plot.collect_classification(PAPER4) if PAPER4.exists() else pd.DataFrame()
    q4_arms = plot.load_arms_from_study(PAPER4)
    if q4_arms is None and PAPER4.exists():
        # tables may be missing; MMD still lives in JSON via analyze later
        pass
    if not q4_cls.empty and _n_json(PAPER4) >= 4:
        plot.fig_classification_grid(
            q4_cls,
            det,
            "paper_style_4q",
            "Seizure detection — 4-qubit hybrid on the same 64-dimensional host",
        )
        plot.fig_classification_delta_primary(
            q4_cls,
            det,
            "paper_style_4q",
            "real plus class-balanced synthetic training; real holdout",
        )
        plot.fig_classification_delta_extended(q4_cls, det, "paper_style_4q")
        plot.write_cls_csvs(q4_cls, det, "paper_style_4q")
    if paper_arms is not None and q4_arms is not None:
        fig_qubit_mmd(plot, paper_arms, q4_arms, det)

    _plot_cls_study(
        plot,
        DOSE256,
        det,
        "dose_256",
        "Seizure detection — synthetic dose 256 clips",
        "approximately 227 clips admitted",
        "fig_dose256_latent_mmd",
    )
    _plot_cls_study(
        plot,
        DOSE512,
        det,
        "dose_512",
        "Seizure detection — synthetic dose 512 clips",
        "approximately 455 clips admitted",
        "fig_dose512_latent_mmd",
    )
    _plot_cls_study(
        plot,
        FRAC25,
        det,
        "dataeff_0.25",
        "Seizure detection — generator trained on 25% of the fold",
        "generator training fraction 0.25; operating dose",
        "fig_dataeff_0.25_latent_mmd",
    )
    _plot_cls_study(
        plot,
        FRAC50,
        det,
        "dataeff_0.5",
        "Seizure detection — generator trained on 50% of the fold",
        "generator training fraction 0.5; operating dose",
        "fig_dataeff_0.5_latent_mmd",
    )

    complete_dose = []
    if _n_json(DOSE256) >= 4 and (DOSE256 / "tables" / "detector.csv").exists():
        complete_dose.append((256, DOSE256))
    if _n_json(DOSE512) >= 4 and (DOSE512 / "tables" / "detector.csv").exists():
        complete_dose.append((512, DOSE512))
    if _n_json(PAPER8) >= 4 and (PAPER8 / "tables" / "detector.csv").exists():
        complete_dose.append((1024, PAPER8))
    fig_dose_auroc(plot, complete_dose, det)

    complete_frac = []
    if _n_json(FRAC25) >= 4 and (FRAC25 / "tables" / "detector.csv").exists():
        complete_frac.append(("0.25", FRAC25))
    if _n_json(FRAC50) >= 4 and (FRAC50 / "tables" / "detector.csv").exists():
        complete_frac.append(("0.5", FRAC50))
    if _n_json(PAPER8) >= 4 and (PAPER8 / "tables" / "detector.csv").exists():
        complete_frac.append(("1.0", PAPER8))
    fig_dataeff_auroc(plot, complete_frac, det)

    status = [
        f"- rematch / decoded / entanglement / depth L2+L4 / critic h16+h128 / surrogate: **4/4**",
        f"- paper_style 8q: **{_n_json(PAPER8)}/4** JSON",
        f"- paper_style 4q: **{_n_json(PAPER4)}/4** JSON",
        f"- dose 256: **{_n_json(DOSE256)}/4**",
        f"- dose 512: **{_n_json(DOSE512)}/4**",
        f"- dataeff 0.25: **{_n_json(FRAC25)}/4**",
        f"- dataeff 0.5: **{_n_json(FRAC50)}/4**",
    ]
    pngs = sorted(p.name for p in list(mech.glob("fig_*.png")) + list(det.glob("fig_*.png")))
    _write_numbers(out)
    _write_index(out, pngs, status)

    pfe = PFE_OUT
    pfe.mkdir(parents=True, exist_ok=True)
    for src in list(mech.glob("fig_*.*")) + list(det.glob("fig_*.*")):
        if src.suffix.lower() in {".png", ".pdf"}:
            shutil.copy2(src, pfe / src.name)
    for extra in ("INDEX.md", "numbers_for_tables.md"):
        src = out / extra
        if src.exists():
            shutil.copy2(src, pfe / extra)
    print(f"figures -> {out}")
    print(f"PFE copy -> {pfe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
