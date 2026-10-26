#!/usr/bin/env python3
"""Write every thesis figure pack into one folder: eeg_qgan/figures/.

Track A (generation / mechanism) is regenerated from ae_v5c tables or a
canonical JSON root. Track B (paper-style detection) is built from the
running paper_style_cpu_deadline study once results_*.json exist.

    python eeg_qgan/scripts/collect_all_figures.py
    python eeg_qgan/scripts/collect_all_figures.py --canonical-root path/to/eeg_qgan_results_canonical
    python eeg_qgan/scripts/collect_all_figures.py --paper-style-root path/to/eeg_qgan_results_paper_style_cpu
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

THESIS = Path(__file__).resolve().parents[2]
EEG = THESIS / "eeg_qgan"
FIG_ROOT = EEG / "figures"
AE_V5C_FIG = THESIS / "Engineering_PFE" / "Figures" / "results_ae_v5c"
DEFAULT_AE_ROOT = THESIS / "eeg_qgan_results_ae_v5c"
DEFAULT_CANONICAL = THESIS / "eeg_qgan_results_canonical"
DEFAULT_PAPER = THESIS / "eeg_qgan_results_paper_style_cpu"


def _load_plot_mod():
    path = Path(__file__).resolve().parent / "plot_ae_v5c_results.py"
    spec = importlib.util.spec_from_file_location("plot_ae_v5c_results", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p is not None and p.exists():
            return p
    return None


def _copy_existing_pack(src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.glob("fig_*.*"):
        if p.suffix.lower() not in {".png", ".pdf"}:
            continue
        shutil.copy2(p, dest / p.name)
        n += 1
    for name in ("dose_detection_summary.csv", "dataeff_detection_summary.csv", "numbers_for_tables.md"):
        src_f = src / name
        if src_f.exists():
            shutil.copy2(src_f, dest / name)
    return n


def _plot_canonical_studies(plot, root: Path, out: Path) -> None:
    """MMD bars for the deadline d=64 / recording-norm mechanism studies."""
    specs = (
        (
            "bottleneck_comparison",
            "fig_bottleneck_temporal_mmd",
            "Temporal bottleneck — latent rematch (d=64, recording-norm)",
            ["quantum", "classical_equiv"],
        ),
        (
            "bottleneck_pooled",
            "fig_bottleneck_pooled_mmd",
            "Pooled bottleneck — latent rematch (d=64, recording-norm)",
            ["quantum", "classical_equiv"],
        ),
        (
            "surrogate",
            "fig_surrogate_latent_mmd",
            "Surrogate study — latent rematch (d=64, recording-norm)",
            ["quantum_none", "quantum", "classical_equiv", "fourier"],
        ),
        (
            "depth_ablation",
            "fig_depth_L4_latent_mmd",
            "Depth L4 — latent rematch (d=64, recording-norm)",
            ["quantum", "classical_equiv", "fourier"],
        ),
        (
            "critic_width",
            "fig_critic_h128_latent_mmd",
            "Critic width 128 — latent rematch (d=64, recording-norm)",
            ["quantum", "classical_equiv"],
        ),
        (
            "rematch_deadline",
            "fig_latent_mmd_capacity",
            "Capacity rematch — latent (d=64, recording-norm, AE 80/20)",
            ["quantum", "classical_equiv", "classical_matched", "classical_large", "gmm"],
        ),
        (
            "decoded_deadline",
            "fig_decoded_latent_mmd",
            "Decoded rematch — latent (d=64, recording-norm, AE 80/20)",
            ["quantum", "classical_equiv", "classical_large", "gmm"],
        ),
        (
            "entanglement_deadline",
            "fig_entanglement_latent_mmd",
            "Entanglement — latent rematch (d=64, recording-norm)",
            ["quantum_none", "quantum_linear", "quantum_ring", "quantum_full", "classical_equiv"],
        ),
        (
            "depth_deadline",
            "fig_depth_L2_latent_mmd",
            "Depth L2 — latent rematch (d=64, recording-norm)",
            ["quantum", "classical_equiv", "fourier"],
        ),
        (
            "critic_deadline",
            "fig_critic_h16_latent_mmd",
            "Critic width 16 — latent rematch (d=64, recording-norm)",
            ["quantum", "classical_equiv"],
        ),
    )
    for study, name, title, order in specs:
        arms = plot.load_arms_from_study(root / study)
        if arms is None:
            print(f"skip {study}: no tables/arms.csv under {root / study}")
            continue
        plot.fig_bar_mmd(arms, out, name, title, order=order)
        plot.fig_paired_delta(
            arms,
            out,
            name.replace("_mmd", "_delta"),
            title.replace("latent rematch", "Δ vs classical twin"),
        )
        print(f"wrote canonical figures for {study}")


def _write_index(out: Path, *, track_a: Path, track_b: Path, paper_json: bool) -> None:
    a_figs = sorted(p.name for p in track_a.glob("fig_*.png"))
    b_figs = sorted(p.name for p in track_b.glob("fig_*.png"))
    lines = [
        "# Thesis figures — single pack (`eeg_qgan/figures`)",
        "",
        "All new figures go here. Do not mix Study XI (leaky split) or mix Track A",
        "latent MMD with Track B recording-norm MMD.",
        "",
        "## Config that every current job uses",
        "",
        "| Knob | Track A (generation) | Track B (paper-style, running now) |",
        "|------|----------------------|--------------------------------------|",
        "| Latent dim | **64** | **64** |",
        "| Qubits | **8** (hybrid Linear 8→64) | **8** |",
        "| Split | TUH predefined train vs dev/eval (E1) | same |",
        "| AE input norm | window | **recording** |",
        "| Detector | `cnn_bilstm_v2` on 1.28 s | `stft_bilstm` on 10.24 s clips |",
        "| Output | `eeg_qgan_results_canonical` or ae_v5c | `eeg_qgan_results_paper_style_cpu` |",
        "",
        "## Study → figure map",
        "",
        "| Study | Track | Figure(s) | Metrics |",
        "|-------|-------|-----------|---------|",
        "| `rematch_deadline` | A | `fig_latent_mmd_capacity` | latent MMD (matched/large arms) |",
        "| `entanglement_deadline` | A | `fig_entanglement_latent_mmd` | latent MMD (d=64, 4 topologies) |",
        "| `decoded_deadline` | A | `fig_decoded_latent_mmd` | decoded + latent MMD |",
        "| `dose_deadline` | B | dose 256/512 + paper_style 1024 | stft_bilstm Δ |",
        "| `dataeff_deadline` | B | frac 0.25/0.5 + paper_style 1.0 | stft_bilstm Δ |",
        "| `critic_width` / `critic_deadline` | A | h128 done; h16 pending | latent MMD |",
        "| `bottleneck_comparison` / `bottleneck_pooled` | A | done | AE NMSE, latent MMD |",
        "| `surrogate` | A | done | circuit vs Fourier vs MLP |",
        "| `depth_ablation` / `depth_deadline` | A | L4 done; L2 pending | MMD by `n_layers` |",
        "| `paper_style_cpu_deadline` | B | `fig_classification_metrics_paper_style` | AUROC AUPRC MCC accuracy |",
        "| `paper_style_q4` | B | same figures, 4q | only qubits change |",
        "",
        "## What you already have vs what is still running",
        "",
        f"- Track A PNG count in this folder: **{len(a_figs)}**",
        f"- Track B PNG count in this folder: **{len(b_figs)}**",
        f"- Track B JSON present locally: **{'yes' if paper_json else 'no — wait for Narval jobs 747356–359, then rsync'}**",
        "",
        "Do **not** put Study XI numbers in these folders.",
        "",
        "## Regenerate",
        "",
        "```",
        "python eeg_qgan/scripts/collect_all_figures.py",
        "```",
        "",
        "### Track A files",
        "",
    ]
    lines.extend(f"- `{n}`" for n in a_figs or ["*(none yet)*"])
    lines += ["", "### Track B files", ""]
    lines.extend(f"- `{n}`" for n in b_figs or ["*(none yet — paper_style jobs still running)*"])
    lines.append("")
    (out / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out / 'INDEX.md'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=FIG_ROOT)
    p.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    p.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    p.add_argument("--paper-style-root", type=Path, default=DEFAULT_PAPER)
    args = p.parse_args()

    out = args.out
    track_a = out / "track_a_generative"
    track_b = out / "track_b_detection"
    track_a.mkdir(parents=True, exist_ok=True)
    track_b.mkdir(parents=True, exist_ok=True)

    plot = _load_plot_mod()
    plot._style()
    plot.FILL.setdefault("fourier", "#C4A574")
    plot.FULL_LABEL.setdefault("fourier", "Fourier surrogate\n(32k parameters)")
    plot.SHORT.setdefault("fourier", "Fourier surrogate")

    canon = out / "canonical"
    canon.mkdir(parents=True, exist_ok=True)
    _plot_canonical_studies(plot, args.canonical_root, canon)

    if (args.canonical_root / "capacity_rematch").exists():
        rematch_parent = args.canonical_root
    elif args.ae_root.exists():
        rematch_parent = args.ae_root
    else:
        rematch_parent = args.ae_root

    print(f"Track A rematch parent: {rematch_parent}")
    old_argv = sys.argv
    try:
        sys.argv = [
            "plot_ae_v5c_results.py",
            "--root",
            str(rematch_parent),
            "--out",
            str(track_a),
            "--source",
            str(AE_V5C_FIG / "source_tables"),
        ]
        plot.main()
    finally:
        sys.argv = old_argv

    if AE_V5C_FIG.exists():
        n = _copy_existing_pack(AE_V5C_FIG, track_a)
        print(f"copied {n} existing ae_v5c figure files into {track_a}")

    paper_study = _first_existing(
        args.paper_style_root / "paper_style_cpu_deadline",
        args.canonical_root / "paper_style_cpu_deadline",
    )
    paper_json = False
    if paper_study is not None:
        n_json = len(list(paper_study.glob("results_*.json")))
        cls = plot.collect_classification(paper_study)
        paper_json = not cls.empty and n_json >= 4
        if not cls.empty and n_json < 4:
            print(f"Track B has {n_json}/4 fold JSON — wait for 747359 before detection figures")
        elif paper_json:
            plot.fig_classification_grid(
                cls,
                track_b,
                "paper_style",
                "Seizure detector — paper-style protocol (E1, recording-norm, 10.24 s, stft_bilstm)",
            )
            plot.fig_classification_delta_primary(
                cls,
                track_b,
                "paper_style",
                "real+balanced synthetic train / real holdout test",
            )
            plot.write_cls_csvs(cls, track_b, "paper_style")
            print(f"wrote Track B figures from {paper_study}")
        else:
            print(f"Track B study dir exists but has no detector rows yet: {paper_study}")
    else:
        print("Track B JSON not on this machine yet — rsync after jobs 747356–359 finish")

    _write_index(out, track_a=track_a, track_b=track_b, paper_json=paper_json)
    print(f"all figures → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
