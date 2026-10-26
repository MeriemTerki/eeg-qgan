# Published results

Means and standard deviations over four seed-by-fold records on the official TUH-SZ holdouts (`train` against `dev`, and `train` against `eval`). Latent MMD² is lower when the generated codes are closer to real seizure codes. Detection is scored on real holdout clips. The hybrid ΔAUROC interval is [−0.0039, +0.0046].

| File | Content |
| --- | --- |
| `latent_rematch.csv` | Capacity comparison, including the unmatched controls |
| `entanglement.csv` | Product, linear, ring, and full circuits |
| `decoded.csv` | Distances after the shared decoder |
| `detection.csv` | AUROC, AUPRC, MCC, and the paired AUROC change |
| `critic.csv` | Default critic and the thin-critic control |

The corresponding figures are in `docs/figures/`.
