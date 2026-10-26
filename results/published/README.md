# Published tables

Statistics for the comparison reported in the repository README. Each mean and standard deviation is computed over four seed × fold records (`train`/`dev` and `train`/`eval`, two seeds). Latent MMD² is lower when generated codes are closer to real seizure codes. Detection metrics are computed on real holdout clips. The hybrid ΔAUROC interval is \([-0.0039, +0.0046]\).

The underlying jobs ran on Narval (Digital Research Alliance of Canada) as described in [docs/REPRODUCING.md](../../docs/REPRODUCING.md).

| File | Contents |
| --- | --- |
| `latent_rematch.csv` | Latent MMD² and parameter counts, including unmatched controls |
| `detection.csv` | AUROC, AUPRC, MCC, and paired ΔAUROC |
| `decoded.csv` | Distances after the shared decoder |
| `entanglement.csv` | Product, linear, ring, and full topologies |
| `critic.csv` | Default critic \([128,128]\) and thin critic \([16,16]\) |

Figures corresponding to the main comparison are `docs/figures/pipeline.png` and `docs/figures/latent_and_detection.png`.
