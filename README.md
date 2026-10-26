# eeg-qgan

Capacity-matched latent generation for multichannel seizure EEG.

Ictal windows are scarce. This repository trains a hybrid Style-qGAN in a latent space learned from TUH-SZ, against classical generators that share its autoencoder, its critic, and its parameter budget. Latent fidelity, decoded structure, and seizure detection are reported as three separate outcomes.

## Contribution

The comparison isolates the generator. The autoencoder is frozen after a reconstruction gate, the critic and the update budget are shared, and the classical twin is sized to the full hybrid parameter count, including the classical readout.

On that contract the hybrid generator is closer to real seizure codes than the equal-budget twin. The margin is not explained by entanglement, and a classical model of the same function class is closer still. At the operating dose the latent margin does not move detection: the area under the ROC curve on real holdout clips is unchanged within its interval. A Gaussian mixture, which uses no critic, remains the closest model on the latent metric and is the detection gain at full data.

The result is a controlled evaluation, not a claim of quantum advantage. Expectation values are analytic.

## Reported results

Official patient-disjoint holdouts. Primary latent score is unbiased MMD² (lower is closer). Primary detection score is AUROC on real clips. The real-only detector sits at 0.821.

| Comparison | Result |
| --- | --- |
| Hybrid vs capacity-matched twin | MMD² 0.016 vs 0.088 (704 vs 709 parameters; 82% of the hybrid budget is the classical readout) |
| Entanglement | Product 0.011, ring 0.016, full 0.018 |
| Fourier reference | MMD² ≈ 0.006 |
| Gaussian mixture | MMD² ≈ 0.0008 |
| Detection at full dose | Hybrid ΔAUROC +0.0004, interval contains zero; twin +0.005; mixture +0.010 |
| Half the generator data | Hybrid ΔAUROC +0.019, not retained at full data |

## Pipeline

Windows of 1.28 s are encoded, generated, and decoded. Detection uses clips of 10.24 s. Synthetic clips may enter training. Scores are computed on real holdout clips.

![Pipeline from TUH-SZ through a shared autoencoder and the generators to the detector](docs/figures/pipeline.png)

The evaluation rules are written in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Run

Python 3.10 or newer. The corpus is not included.

```bash
git clone https://github.com/MeriemTerki/eeg_qgan.git
cd eeg_qgan
pip install -e ".[dev]"
pytest tests -q
python -m eeg_qgan.experiments.runner capacity_rematch --data-root "$DATA_ROOT"
```

| Study | Role |
| --- | --- |
| `capacity_rematch` | Hybrid against the matched twin |
| `entanglement_ablation_d8` | Circuit topology at fixed width |
| `downstream_scoped` | Detection on real holdout clips |
| `smoke` | Local check on synthetic windows |

Reproduction commands are in [docs/REPRODUCING.md](docs/REPRODUCING.md).

## Layout

```
eeg_qgan/   models, data, training, evaluation
configs/    defaults and one file per study
scripts/    checks, synthetic data, cluster submission
tests/      splits, metrics, preprocessing
docs/       protocol, reproduction notes, pipeline figure
```

## Citation

```
Terki, M. eeg-qgan: capacity-matched latent generative models for multichannel seizure EEG.
https://github.com/MeriemTerki/eeg_qgan
```
