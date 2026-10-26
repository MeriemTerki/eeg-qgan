# Reproduction on Narval (Alliance Canada)

The tables in `results/published` were obtained from SLURM jobs on **Narval**, operated for the Digital Research Alliance of Canada (Alliance; formerly Compute Canada). They were not obtained from a local workstation.

Execute sections 1–4 on a Narval **login node**. Training itself is dispatched to compute nodes.

| Workload | Node | Resource request |
| --- | --- | --- |
| Environment creation, `sbatch` | Login | — |
| Autoencoder | Compute | 1 GPU |
| Generators (circuit and classical) | Compute | CPU |
| Detector | Compute | 1 GPU |
| Unit tests, `smoke --dry-run` | Workstation | CPU only; does not reproduce the tables |

## 1. Obtain the code

```bash
ssh <username>@narval.alliancecan.ca
cd "$HOME"
git clone https://github.com/MeriemTerki/eeg_qgan.git
cd eeg_qgan
```

The submission scripts assume the repository is at `$HOME/eeg_qgan`.

## 2. Install the environment

Run once on the login node. `scripts/narval/setup_env.sh` loads `python/3.11` and `scipy-stack`, creates `$SCRATCH/venvs/eeg_qgan` from the cluster wheelhouse, and installs this package.

```bash
bash scripts/narval/setup_env.sh
source "$SCRATCH/venvs/eeg_qgan/bin/activate"
python -c "import eeg_qgan, torch, pennylane, pywt; print(eeg_qgan.__version__)"
python scripts/validate_studies.py
```

## 3. Set data and output paths

Place the TUH-SZ BinaryClassification tree on scratch. The directory must contain `train`, `dev`, and `eval`.

```bash
export DATA_ROOT="$SCRATCH/path/to/BinaryClassification_Data"
export OUTPUT_ROOT="$SCRATCH/eeg_qgan_results"
export CC_ACCOUNT="<slurm-allocation>"
```

`OUTPUT_ROOT` defaults to `$SCRATCH/eeg_qgan_results` inside the job preamble (`scripts/narval/_common.sh`) when it is unset.

## 4. Submit the studies

`scripts/narval/submit.py` writes a SLURM script and calls `sbatch`. The reported capacity comparison was split by seed.

```bash
python scripts/narval/submit.py capacity_rematch --split-by seed --dry-run
python scripts/narval/submit.py capacity_rematch --split-by seed --account "$CC_ACCOUNT"
```

| Command | Study |
| --- | --- |
| `entanglement_ablation_d8` | Entanglement topologies |
| `downstream_scoped` | Detection with synthetic training clips |
| `--no-gpu` | CPU-only allocation for generator-only jobs |
| `--split-by seed-fold` | One job per seed and fold |

Monitor with `squeue -u "$USER"`. Standard output and error are written to `scripts/narval/logs`.

`downstream_scoped` requires per-window patient identifiers:

```bash
python scripts/build_patient_sidecars.py --data-root "$DATA_ROOT" --dry-run
```

## 5. Aggregate the outputs

On a login node, after the jobs have written their fold records:

```bash
source "$SCRATCH/venvs/eeg_qgan/bin/activate"
eeg-qgan-analyze "$OUTPUT_ROOT/capacity_rematch" \
  --reference classical_equiv \
  --csv "$OUTPUT_ROOT/tables/"
```

Confidence intervals resample entire folds. Under the official `dev` and `eval` partition, two seeds produce four rows and two independent partitions. The summary reports both counts.

## Workstation checks

The following commands validate the installation. They do not train on TUH-SZ and do not regenerate `results/published`.

```bash
pip install -e ".[dev]"
pytest tests -q
python scripts/make_synthetic_dataset.py --out data/synthetic
python -m eeg_qgan.experiments.runner smoke --data-root data/synthetic --dry-run
```
