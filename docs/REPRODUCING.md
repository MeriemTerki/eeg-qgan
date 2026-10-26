# Reproducing the runs

The published tables were not produced on a laptop. They were produced on **Narval**, a national high-performance cluster provided by the Digital Research Alliance of Canada (Alliance; formerly Compute Canada). Jobs are scheduled with SLURM. The corpus and the outputs reside on the scratch filesystem. The code and the Python environment reside on the login account.

A local machine is enough for the tests and for a dry run. It is not the machine that trained the autoencoder, the generators, or the detector.

| Stage | Where it ran | Hardware |
| --- | --- | --- |
| Environment and job submission | Narval login node | — |
| Autoencoder | Narval compute node | GPU |
| Hybrid circuit and classical generators | Narval compute node | CPU (state-vector simulation) |
| Detector | Narval compute node | GPU |
| Tests and `smoke --dry-run` | any machine | CPU |

## 1. Log in and place the code

Connect to a Narval login node and put this repository where the job scripts expect it, by default `$HOME/eeg_qgan`.

```bash
ssh <user>@narval.alliancecan.ca
cd $HOME
git clone https://github.com/MeriemTerki/eeg_qgan.git
cd eeg_qgan
```

## 2. Create the environment once

Run this on the login node, not inside a compute job. The script loads `python/3.11` and `scipy-stack`, builds a virtual environment under `$SCRATCH/venvs/eeg_qgan`, and installs the package. Alliance nodes prefer the cluster wheelhouse over a source build.

```bash
bash scripts/narval/setup_env.sh
source $SCRATCH/venvs/eeg_qgan/bin/activate
python -c "import eeg_qgan, torch, pennylane, pywt; print(eeg_qgan.__version__)"
python scripts/validate_studies.py
```

## 3. Point the jobs at the corpus

`DATA_ROOT` is the BinaryClassification directory of TUH-SZ on scratch. It must contain the official `train`, `dev`, and `eval` folders. `OUTPUT_ROOT` is where fold records are written. The job preamble uses `$SCRATCH/eeg_qgan_results` when `OUTPUT_ROOT` is unset.

```bash
export DATA_ROOT=$SCRATCH/path/to/BinaryClassification_Data
export OUTPUT_ROOT=$SCRATCH/eeg_qgan_results
```

## 4. Submit from the login node

`scripts/narval/submit.py` writes a SLURM script and calls `sbatch`. The allocation is `--account`, or the environment variable `CC_ACCOUNT`. Quantum-only work should be submitted with `--no-gpu`. Studies that train the autoencoder or the detector request one GPU.

```bash
# inspect the script without submitting
python scripts/narval/submit.py capacity_rematch --split-by seed --dry-run

# one job per seed, which is how the long runs were split
python scripts/narval/submit.py capacity_rematch --split-by seed

# one job per seed and fold
python scripts/narval/submit.py capacity_rematch --split-by seed-fold --account "$CC_ACCOUNT"
```

Other studies use the same command with a different name: `entanglement_ablation_d8`, `downstream_scoped`. After submission, `squeue -u $USER` lists the jobs. Logs are written under `scripts/narval/logs`.

`downstream_scoped` stops unless each window has a patient identifier.

```bash
python scripts/build_patient_sidecars.py --data-root "$DATA_ROOT" --dry-run
```

## 5. Read the finished jobs

Run the summary on a login node, against the directory the jobs actually wrote.

```bash
source $SCRATCH/venvs/eeg_qgan/bin/activate
eeg-qgan-analyze $OUTPUT_ROOT/capacity_rematch --reference classical_equiv --csv $OUTPUT_ROOT/tables/
```

Intervals are computed by resampling folds. The summary prints the number of distinct partitions next to the number of rows. With the official `dev` and `eval` folders, two seeds give four rows and two partitions.

## What can be checked without the cluster

These commands do not train on TUH-SZ and do not reproduce the published tables.

```bash
pip install -e ".[dev]"
pytest tests -q
python scripts/make_synthetic_dataset.py --out data/synthetic
python -m eeg_qgan.experiments.runner smoke --data-root data/synthetic --dry-run
```
