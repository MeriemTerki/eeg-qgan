# eeg-qgan

Capacity-matched latent generative models for multichannel seizure EEG.

A hybrid Style-qGAN and classical controls share one frozen autoencoder, one critic, and one parameter budget. Latent fidelity, decoded signal fidelity, and seizure detection are scored separately. A closer code is not reported as a clinical gain, and the code does not claim a quantum advantage.

![Evaluation pipeline: TUH-SZ windows, a frozen autoencoder, the hybrid generator beside a capacity-matched twin and a Gaussian mixture, then three separate scores](docs/architecture.svg)

## What this repository is

One Python package. Each experiment is a YAML study on top of `configs/base.yaml`, not a forked copy of the source. The defaults are the post-audit protocol:

- patient identity is read from sidecars and never invented from array indices
- each training fold is split three ways: train, checkpoint selection, and reporting
- reported MMD is the unbiased estimator, with matched sample sizes
- whitening of the latent codes is rejected if a config still asks for it
- a failed autoencoder gate or a collapsed augmentation dose stops the fold instead of being averaged in
- detection intervals resample whole folds, not duplicate rows

`AUDIT_FIXES.md` states what the previous code did and how to check each correction. `SUPERVISOR_R_PLAN.md` maps the remaining cluster runs.

## Install

Python 3.10 or newer.

```bash
pip install -e ".[dev]"
pytest tests -q
```

On the Narval cluster, once per account:

```bash
bash scripts/narval/setup_env.sh
```

## Run a study

```bash
python scripts/validate_studies.py
python -m eeg_qgan.experiments.runner smoke --data-root data/synthetic --dry-run
python -m eeg_qgan.experiments.runner capacity_rematch --data-root "$DATA_ROOT"
```

Cluster jobs are generated from the study:

```bash
python scripts/narval/submit.py capacity_rematch --split-by seed
python scripts/narval/submit.py entanglement_ablation_d8 --split-by seed --dry-run
```

The quantum arms use a CPU state-vector simulator. The autoencoder and the detector are the stages that use a GPU. Walltime and memory are taken from the study file.

## Studies that carry the paper

| Study | Question |
| --- | --- |
| `capacity_rematch` | Hybrid versus the capacity-matched twin, under the corrected protocol |
| `entanglement_ablation_d8` | Whether entanglement changes the result when whitening is off |
| `downstream_scoped` | Detection on real holdout clips; blocked until patient sidecars exist |
| `smoke` | Local wiring check on the synthetic dataset |

Before any patient-wise run:

```bash
python scripts/build_patient_sidecars.py --data-root "$DATA_ROOT" --dry-run
```

The command writes the sidecars or names exactly what is missing. `RUNBOOK.md` has the cluster sequence.

## Read the results

```bash
eeg-qgan-analyze results/capacity_rematch --reference classical_equiv --csv tables/
```

The summary drops folds the autoencoder gate excluded, bootstraps whole folds, and flags a gap smaller than the real-against-real noise floor, an arm that did not converge, and an augmentation dose that collapsed.

A dependence check that does not retrain:

```bash
python scripts/copula_shuffle_test.py results/capacity_rematch
```

## Layout

```
eeg_qgan/          package: data, models, training, evaluation, stats
configs/base.yaml  protocol defaults
configs/studies/   one YAML file per experiment
scripts/           local checks, plots, patient sidecars, SLURM submit
tests/             protocol checks (splits, MMD bias, gates)
demo/              local inference site; weights are not in this repository
docs/              figures for this page
```

TUH recordings, checkpoints, and result tables are not in the git history. Build a local smoke set with `scripts/make_synthetic_dataset.py`.

## Reproducibility

A fold's random stream comes from `(seed, fold, arm)` through a stable hash, so the same fold matches whether it runs alone or inside a larger job. Each fold is written as soon as it finishes. A requeued job continues from the folds already on disk.

## Citation

A paper is in preparation. Until it is public, cite the repository and the author, Meriem Terki.
