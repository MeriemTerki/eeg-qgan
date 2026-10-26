# Reproducing a run

## Local check

```bash
pip install -e ".[dev]"
pytest tests -q
python scripts/make_synthetic_dataset.py --out data/synthetic
python -m eeg_qgan.experiments.runner smoke --data-root data/synthetic --dry-run
python scripts/validate_studies.py
```

## Corpus run

Set `DATA_ROOT` to the BinaryClassification directory of TUH-SZ. The official `train`, `dev`, and `eval` folders must sit under that directory.

```bash
python -m eeg_qgan.experiments.runner capacity_rematch --data-root "$DATA_ROOT"
```

Useful overrides:

```bash
python -m eeg_qgan.experiments.runner capacity_rematch \
  --data-root "$DATA_ROOT" \
  --seed-index 0 \
  --fold-start 0 --fold-end 1
```

On a SLURM cluster, create the environment once with `bash scripts/narval/setup_env.sh`, then:

```bash
python scripts/narval/submit.py capacity_rematch --split-by seed
```

`--dry-run` prints the job scripts without submitting them. Quantum arms use a CPU state-vector simulator. The autoencoder and the detector are the stages that request a GPU.

## Patient-grouped folds

`downstream_scoped` does not run until each window has a patient identifier. Build those files, or read the reason they cannot be built, with:

```bash
python scripts/build_patient_sidecars.py --data-root "$DATA_ROOT" --dry-run
```

## Summaries

```bash
eeg-qgan-analyze results/capacity_rematch --reference classical_equiv --csv tables/
```

Intervals are obtained by resampling folds, not individual windows. The table states how many distinct partitions the rows come from.
