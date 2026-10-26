# eeg-qgan

Latent generative models for multichannel seizure EEG, trained on the TUH Seizure Corpus and compared at matched capacity.

A hybrid Style-qGAN and a classical twin share one frozen autoencoder, one Wasserstein critic, and one parameter budget (704 against 709). Latent distance, decoded signal structure, and seizure detection are recorded as three outcomes.

## Finding

Under that match, the hybrid generator is closer to real seizure codes than the twin: holdout latent MMD² is 0.016 against 0.088. The margin does not come from entanglement. A product circuit is closer than the ring used in the headline run (0.011 against 0.016), and a full entangling pattern is worse (0.018). A Gaussian mixture with no critic is closer than both adversarial models (0.0008).

The latent order does not carry into detection. On real holdout clips the real-only AUROC is 0.821. Adding hybrid windows changes it by +0.0004, and the interval contains zero. The twin and the mixture are the arms that move the endpoint, by +0.005 and +0.010.

![Latent MMD² on the left and the paired change in AUROC on the right, for the hybrid, the twin, and the mixture](docs/figures/latent_and_detection.png)

| Arm | Parameters | Latent MMD² | AUROC | ΔAUROC |
| --- | ---: | ---: | ---: | ---: |
| Real only | — | — | 0.821 ± 0.057 | — |
| Hybrid Style-qGAN | 704 | 0.0158 ± 0.0025 | 0.822 ± 0.052 | +0.0004 |
| Capacity-matched twin | 709 | 0.0883 ± 0.0096 | 0.827 ± 0.059 | +0.005 |
| Gaussian mixture | — | 0.00085 ± 0.00049 | 0.832 ± 0.044 | +0.010 |

Values are means over four seed-by-fold records. Eighty-two percent of the hybrid parameters are the classical readout. The remaining tables, including decoded distances, entanglement, and critic width, are in [`results/published`](results/published).

## Method

Each window is 22 bipolar channels by 256 samples (1.28 s at 200 Hz). The autoencoder maps it to a 64-dimensional code and is frozen after a reconstruction check. Generators are trained in that space. Detection packs eight windows into a 10.24 s clip and is scored on real clips from the official `dev` and `eval` partitions. Synthetic clips may enter training only.

![TUH-SZ windows, the shared autoencoder, the generators, and the detector](docs/figures/pipeline.png)

The evaluation rules are in [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Reproduce

Python 3.10 or newer. The corpus is supplied at launch.

```bash
git clone https://github.com/MeriemTerki/eeg_qgan.git
cd eeg_qgan
pip install -e ".[dev]"
pytest tests -q
python -m eeg_qgan.experiments.runner capacity_rematch --data-root "$DATA_ROOT"
```

| Study | What it runs |
| --- | --- |
| `capacity_rematch` | Hybrid against the matched twin |
| `entanglement_ablation_d8` | Product, ring, and denser circuits |
| `downstream_scoped` | Detection on real holdout clips |
| `smoke` | A short local check |

Cluster submission and how to read a finished run are in [docs/REPRODUCING.md](docs/REPRODUCING.md).
