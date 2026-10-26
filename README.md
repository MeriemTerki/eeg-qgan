# eeg-qgan

Official implementation of a capacity-matched latent comparison for multichannel seizure EEG.

A hybrid Style-qGAN is trained in a latent space learned from the TUH Seizure Corpus and compared with classical generators under a shared autoencoder, a shared Wasserstein critic, and a matched parameter budget. Latent fidelity, decoded-signal fidelity, and downstream detection are reported separately.

**Environment.** The published runs were executed as SLURM jobs on Narval (Digital Research Alliance of Canada; formerly Compute Canada). Reproduction steps are in [docs/REPRODUCING.md](docs/REPRODUCING.md). The evaluation protocol is in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Experimental setup

| Item | Setting |
| --- | --- |
| Corpus | TUH-SZ, BinaryClassification release |
| Window | 22 bipolar channels × 256 samples, 200 Hz, 1.28 s |
| Outer split | Official `train` / `dev` and `train` / `eval` (patient-disjoint) |
| Representation | Frozen convolutional autoencoder, latent dimension 64, NMSE gate 0.55 |
| Hybrid generator | 8-qubit re-uploading circuit, analytic Pauli-Z readout, linear map 8 → 64 (704 parameters) |
| Matched control | Classical twin, 709 parameters |
| Density reference | Gaussian mixture in the same latent coordinates |
| Detector | STFT–BiLSTM on 8-window clips (10.24 s), scored on real holdout clips |
| Primary latent metric | Unbiased MMD² (lower is closer) |
| Primary detection metric | AUROC |

![Processing path from TUH-SZ windows through the shared autoencoder and the generators to the detector](docs/figures/pipeline.png)

## Results

Means ± standard deviation over four seed × fold records. The change in AUROC is paired against the real-only detector on the same folds. The 95% interval on the hybrid change contains zero.

![Latent MMD² and the paired change in AUROC](docs/figures/latent_and_detection.png)

| Model | Parameters | Latent MMD² ↓ | AUROC | ΔAUROC |
| --- | ---: | ---: | ---: | ---: |
| Real only | — | — | 0.821 ± 0.057 | — |
| Hybrid Style-qGAN | 704 | 0.0158 ± 0.0025 | 0.822 ± 0.052 | +0.0004 |
| Capacity-matched twin | 709 | 0.0883 ± 0.0096 | 0.827 ± 0.059 | +0.005 |
| Gaussian mixture | — | 0.00085 ± 0.00049 | 0.832 ± 0.044 | +0.010 |

Of the 704 hybrid parameters, 576 (82%) belong to the classical readout. Additional tables (decoded distances, entanglement topologies, critic width) are released in [`results/published`](results/published).

## Requirements

- Python ≥ 3.10
- PyTorch, PennyLane, NumPy, SciPy, scikit-learn, PyWavelets, PyYAML

```bash
git clone https://github.com/MeriemTerki/eeg_qgan.git
cd eeg_qgan
pip install -e ".[dev]"
pytest tests -q
```

## Repository structure

```
configs/             base configuration and one file per study
eeg_qgan/            data, models, training, evaluation, statistics
scripts/narval/      Alliance Canada environment and SLURM submission
results/published/   tables corresponding to the reported comparison
docs/PROTOCOL.md     evaluation protocol
docs/REPRODUCING.md  Narval reproduction procedure
docs/figures/        pipeline and result figures
tests/
```

| Study | Contents |
| --- | --- |
| `capacity_rematch` | Hybrid generator and capacity-matched twin |
| `entanglement_ablation_d8` | Product, ring, and denser entanglement |
| `downstream_scoped` | Synthetic augmentation; scores on real holdout clips |
| `smoke` | Configuration check on synthetic windows |
