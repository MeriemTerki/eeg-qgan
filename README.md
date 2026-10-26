# eeg-qgan

Latent generative models for multichannel seizure EEG, compared at matched capacity.

A hybrid Style-qGAN and classical generators share one autoencoder, one critic, and one parameter budget. Latent fidelity, decoded signal structure, and seizure detection are evaluated separately. The package does not claim a quantum advantage.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/MeriemTerki/eeg_qgan.git
cd eeg_qgan
pip install -e ".[dev]"
pytest tests -q
```

## Data

The experiments use the TUH Seizure Corpus in its BinaryClassification layout: 22 bipolar channels, resampled to 200 Hz, cut into non-overlapping windows of 256 samples (1.28 s). The outer split is the official `train`/`dev` and `train`/`eval` partition. Pass the corpus root at launch. It is not stored in this repository.

```bash
python -m eeg_qgan.experiments.runner capacity_rematch --data-root "$DATA_ROOT"
```

`scripts/make_synthetic_dataset.py` writes a small stand-in under `data/synthetic` for a local check:

```bash
python scripts/make_synthetic_dataset.py --out data/synthetic
python -m eeg_qgan.experiments.runner smoke --data-root data/synthetic --dry-run
```

## Experiments

Each file in `configs/studies/` overrides `configs/base.yaml`.

| Study | Comparison |
| --- | --- |
| `capacity_rematch` | Hybrid generator against a classical network of the same size |
| `entanglement_ablation_d8` | Product, ring, and denser circuits on a matched latent width |
| `downstream_scoped` | Effect of synthetic windows on detection, scored on real holdout clips |
| `smoke` | Installation check |

```bash
python scripts/validate_studies.py
python scripts/narval/submit.py capacity_rematch --split-by seed --dry-run
```

Cluster setup and the commands that read a finished run are in [docs/REPRODUCING.md](docs/REPRODUCING.md). The evaluation rules are in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Layout

```
eeg_qgan/     models, data, training, evaluation
configs/      shared defaults and one file per study
scripts/      validation, synthetic data, cluster submission
tests/        splits, metrics, preprocessing, model shapes
docs/         protocol and reproduction notes
```

## Citation

```
Terki, M. eeg-qgan: capacity-matched latent generative models for multichannel seizure EEG.
https://github.com/MeriemTerki/eeg_qgan
```
