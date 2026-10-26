#!/usr/bin/env python3
"""Generate and submit SLURM jobs for any study.

The old layout carried about thirty hand-written job scripts whose differences were a
config filename, a seed index and a fold range. They drifted apart, and several
referenced configs that no longer existed. Here the job script is generated from the
study's own resource profile, so splitting a long run across submissions is a flag
rather than a new file to maintain.

Typical use on a login node::

    # one job for the whole study
    python scripts/narval/submit.py capacity_rematch

    # one job per seed, which is how the long CPU runs were actually split
    python scripts/narval/submit.py capacity_rematch --split-by seed

    # one job per seed and fold, for maximum parallelism
    python scripts/narval/submit.py capacity_rematch --split-by seed-fold

    # see the scripts without submitting them
    python scripts/narval/submit.py capacity_rematch --split-by seed --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eeg_qgan.config import list_studies, load_config  # noqa: E402

# SLURM allocation (not the login user). Override with --account or CC_ACCOUNT.
# mrmterki is on def-khenni_cpu / def-khenni_gpu, not the unsplit def-khenni name.
DEFAULT_ACCOUNT = "def-khenni_cpu"


def _assign_dotted(target: dict, dotted_key: str, value: Any) -> None:
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


@dataclass
class Resources:
    """SLURM resource request for one job."""

    time: str = "24:00:00"
    cpus: int = 8
    mem: str = "48G"
    gpu: bool = False

    def directives(self, job_name: str, log_dir: str, account: str) -> list[str]:
        lines = [
            f"#SBATCH --account={account}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --cpus-per-task={self.cpus}",
            f"#SBATCH --mem={self.mem}",
            f"#SBATCH --output={log_dir}/%x-%j.out",
            f"#SBATCH --error={log_dir}/%x-%j.err",
            # Requeue on preemption: these runs are long and checkpoint after every fold,
            # so a requeued job resumes with its completed folds already written.
            "#SBATCH --requeue",
        ]
        if self.gpu:
            lines.insert(4, "#SBATCH --gres=gpu:1")
        return lines


def resources_for(cfg: dict, n_jobs: int) -> Resources:
    """Choose a resource request from the study's cost drivers.

    PennyLane simulation stays on CPU, but the autoencoder and detector dominate
    wall-clock on TUH-scale folds. Requesting a GPU for those stages is what makes
    a thesis-scale campaign finish; quantum circuit eval still uses the CPU cores.
    """
    arms = list(cfg.get("generator_arms", []))
    has_quantum = any(str(a).startswith("quantum") for a in arms)
    skip_downstream = bool(cfg.get("skip_downstream", False))
    steps = int((cfg.get("training", {}) or {}).get("generator_steps", 1500))
    shots = (cfg.get("quantum", {}) or {}).get("shots")

    # Work per job, used to size the walltime.
    folds = int(cfg["n_folds"]) * int(cfg["n_seeds"]) / max(1, n_jobs)
    cost = folds * len(arms) * steps / 1500.0
    if shots:
        # Parameter-shift gradients cost one circuit evaluation per parameter per step.
        cost *= 8.0

    # The generator budget alone underestimates cheap-generator studies, where the
    # autoencoder and the detector are what actually occupy the node. Measured on a
    # Narval GPU: about 45 minutes per fold per 300 autoencoder epochs at the capped
    # segment count, and a few minutes per detector arm.
    ae_epochs = int((cfg.get("autoencoder", {}) or {}).get("epochs", 300))
    ae_hours = folds * (ae_epochs / 300.0)
    det_hours = 0.0 if skip_downstream else folds * (len(arms) + 1) * 0.35
    clin = cfg.get("clinical") or {}
    if bool(clin.get("enabled")) and bool(clin.get("detector_only")):
        # Long-clip STFT ceiling: no AE / generator walltime, but the clip loader and
        # the detector both scale with how much of the corpus is read, and the caps
        # differ by an order of magnitude between the v1 and v2 studies. In this path
        # the volume is set by clips per file, not by the short-window segment cap.
        ae_hours = 0.0
        clips = clin.get("max_clips_per_file")
        if clips is None:
            segments = (cfg.get("data", {}) or {}).get("max_segments_per_batch") or 3200
            clips = float(segments) / max(1, int(clin.get("clip_windows", 8)))
        det_hours = folds * 1.5 * max(1.0, float(clips) / 64.0)
        has_quantum = False
        cost = folds

    if has_quantum:
        # GPU for AE/detector + enough CPU for the simulator.
        hours = max(12, min(168, int(8 * cost + ae_hours + det_hours)))
        if skip_downstream:
            hours = max(8, min(72, int(5 * cost + ae_hours)))
        if hours >= 72:
            return Resources(time="7-00:00:00", cpus=16, mem="64G", gpu=True)
        return Resources(time=f"{hours}:00:00", cpus=16, mem="64G", gpu=True)
    # Classical only.
    hours = max(6, min(48, int(4 * cost + ae_hours + det_hours) + 2))
    return Resources(time=f"{hours}:00:00", cpus=8, mem="48G", gpu=True)


def _hours(time: str) -> float:
    days, _, clock = time.rpartition("-")
    h, m, s = (int(v) for v in clock.split(":"))
    return (int(days) * 24 if days else 0) + h + m / 60.0 + s / 3600.0


def _format_time(hours: float) -> str:
    hours = max(1, min(168, int(round(hours))))
    return "7-00:00:00" if hours >= 168 else f"{hours}:00:00"


def to_cpu_only(resources: Resources) -> Resources:
    """Rewrite a GPU request as a CPU one with a walltime that can actually finish.

    Dropping ``--gres`` alone is what causes a CPU job to die at the wall: the estimate
    it inherited assumed a GPU for the autoencoder and detector. Those stages run
    roughly three times slower on Narval CPU nodes, and the queue for CPU is short
    enough that asking for the longer window costs little.
    """
    resources.gpu = False
    resources.cpus = max(resources.cpus, 16)
    resources.time = _format_time(3.0 * _hours(resources.time))
    return resources


def build_script(
    study: str,
    job_name: str,
    resources: Resources,
    runner_args: list[str],
    log_dir: str,
    account: str,
    data_root: str,
    output_root: str,
) -> str:
    directives = "\n".join(resources.directives(job_name, log_dir, account))
    args = " ".join(runner_args)
    export_data = f'export DATA_ROOT="{data_root}"\n' if data_root else ""
    export_out = f'export OUTPUT_ROOT="{output_root}"\n' if output_root else ""
    return f"""#!/bin/bash
{directives}

{export_data}{export_out}

# Generated by scripts/narval/submit.py. Edit the study config, not this file.

# Repair CRLF endings if the file was touched on Windows.
if grep -q $'\\r' "$0" 2>/dev/null; then
  sed -i 's/\\r$//' "$0"
  exec bash "$0" "$@"
fi

source "${{SLURM_SUBMIT_DIR:-$HOME/eeg_qgan}}/scripts/narval/_common.sh"

echo "study {study}"
echo "args  {args}"

python -m eeg_qgan.experiments.runner {study} \\
  --data-root "$DATA_ROOT" \\
  --output-root "$OUTPUT_ROOT" \\
  {args}

echo "finished $(date --iso-8601=seconds)"
"""


def plan_jobs(cfg: dict, split_by: str) -> list[tuple[str, list[str]]]:
    """Enumerate (suffix, runner arguments) pairs for the chosen split."""
    n_seeds = int(cfg["n_seeds"])
    n_folds = int(cfg["n_folds"])

    if split_by == "none":
        return [("all", [])]
    if split_by == "seed":
        return [(f"s{i}", [f"--seed-index {i}"]) for i in range(n_seeds)]
    if split_by == "seed-fold":
        return [
            (f"s{i}f{f}", [f"--seed-index {i}", f"--fold-start {f}", f"--fold-end {f + 1}"])
            for i in range(n_seeds)
            for f in range(n_folds)
        ]
    raise ValueError(f"Unknown --split-by '{split_by}'")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study", nargs="?", help="study name")
    p.add_argument(
        "--split-by",
        default="none",
        choices=("none", "seed", "seed-fold"),
        help="how to divide the study across jobs",
    )
    p.add_argument("--dry-run", action="store_true", help="write the scripts without submitting")
    p.add_argument("--time", default=None, help="override walltime, e.g. 3-00:00:00")
    p.add_argument("--cpus", type=int, default=None)
    p.add_argument("--mem", default=None)
    p.add_argument("--gpu", action="store_true", help="force a GPU request")
    p.add_argument("--no-gpu", action="store_true", help="force a CPU-only request")
    p.add_argument("--script-dir", default="scripts/narval/generated")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--list-studies", action="store_true")
    p.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="passed through to the runner, e.g. --set augmentation.dose_count=2048",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="suffix for the job and script names, so swept variants do not overwrite",
    )
    p.add_argument(
        "--account",
        default=os.environ.get("CC_ACCOUNT") or os.environ.get("SLURM_ACCOUNT") or DEFAULT_ACCOUNT,
        help="SLURM allocation account (not the login user). Default: def-khenni_cpu",
    )
    args = p.parse_args()

    if args.list_studies or not args.study:
        print("Available studies:")
        for name in list_studies():
            print(f"  {name}")
        return 0 if args.list_studies else 2

    # Apply --set before planning jobs. Otherwise n_folds=2 is ignored here and
    # predefined studies spawn empty f2/f3/f4 SLURM jobs from YAML n_folds: 5.
    cli_overrides: dict[str, Any] = {}
    for item in args.overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got '{item}'")
        key, raw = item.split("=", 1)
        _assign_dotted(cli_overrides, key.strip(), yaml.safe_load(raw))

    cfg = load_config(args.study, overrides=cli_overrides)

    jobs = plan_jobs(cfg, args.split_by)
    resources = resources_for(cfg, n_jobs=len(jobs))
    if args.mem:
        resources.mem = args.mem
    if args.gpu:
        resources.gpu = True
    overrides = list(args.overrides or [])
    if args.no_gpu:
        resources = to_cpu_only(resources)
        # Record the CPU run in the results JSON rather than leaving the config
        # claiming a GPU that the job never had.
        if not any(o.startswith("use_gpu=") for o in overrides):
            overrides.append("use_gpu=false")
    # An explicit --time or --cpus wins over the CPU rescaling above.
    if args.time:
        resources.time = args.time
    if args.cpus:
        resources.cpus = args.cpus

    script_dir = REPO_ROOT / args.script_dir
    script_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.log_dir).mkdir(parents=True, exist_ok=True)

    data_root = os.environ.get("DATA_ROOT", "")
    output_root = os.environ.get("OUTPUT_ROOT", "")
    print(f"study        {args.study}")
    print(f"user         {os.environ.get('USER', '?')}  (must be mrmterki, not linah03)")
    print(f"account      {args.account}")
    print(f"arms         {', '.join(cfg['generator_arms'])}")
    print(f"seeds/folds  {cfg['n_seeds']} x {cfg['n_folds']}")
    print(f"jobs         {len(jobs)} (split by {args.split_by})")
    print(
        f"resources    time={resources.time} cpus={resources.cpus} "
        f"mem={resources.mem} gpu={resources.gpu}"
    )
    print(f"data_root    {data_root or '(from _common.sh default)'}")
    print(f"output_root  {output_root or '(from _common.sh default)'}")
    print()

    extra = [f"--set {' '.join(overrides)}"] if overrides else []
    submitted = []
    for suffix, runner_args in jobs:
        job_name = f"{args.study}-{suffix}" + (f"-{args.tag}" if args.tag else "")
        script = build_script(
            args.study,
            job_name,
            resources,
            runner_args + extra,
            args.log_dir,
            account=args.account,
            data_root=data_root,
            output_root=output_root,
        )
        path = script_dir / f"{job_name}.sh"
        path.write_text(script, encoding="utf-8", newline="\n")
        path.chmod(0o755)

        if args.dry_run:
            print(f"  wrote {path.relative_to(REPO_ROOT)}")
            continue
        try:
            out = subprocess.run(
                ["sbatch", str(path)], capture_output=True, text=True, check=True
            )
            print(f"  {job_name}: {out.stdout.strip()}")
            submitted.append(job_name)
        except FileNotFoundError:
            print(
                "  sbatch not found: this must run on a cluster login node. "
                f"Script written to {path.relative_to(REPO_ROOT)}"
            )
        except subprocess.CalledProcessError as exc:
            print(f"  {job_name} FAILED: {exc.stderr.strip()}")

    print()
    if args.dry_run:
        print(f"Dry run. {len(jobs)} scripts in {script_dir.relative_to(REPO_ROOT)}")
    else:
        print(f"Submitted {len(submitted)} of {len(jobs)} jobs. Monitor with: squeue -u $USER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
