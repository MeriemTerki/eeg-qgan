# Evaluation protocol

This note specifies the protocol implemented by `configs/base.yaml` and used for the tables in `results/published`.

## 1. Computing environment

Reported training was performed on Narval, a national cluster of the Digital Research Alliance of Canada (Alliance; formerly Compute Canada), under SLURM.

| Component | Resource |
| --- | --- |
| Autoencoder | One GPU |
| Generators, including the hybrid circuit | CPU; PennyLane analytic state-vector simulation |
| Detector | One GPU |
| Corpus and job outputs | Project scratch (`$SCRATCH`) |

The circuit is not executed on quantum hardware. Job submission is described in [REPRODUCING.md](REPRODUCING.md).

## 2. Data

Inputs are multichannel scalp EEG windows

\[
x \in \mathbb{R}^{22 \times 256},
\]

sampled at 200 Hz (1.28 s), taken from the TUH Seizure Corpus in the BinaryClassification layout. Preprocessing is identical for every model: Daubechies-6 soft-threshold denoising, followed by a robust rescaling into \([-1, 1]\).

Detection operates on a longer clip. Eight contiguous windows are concatenated to 10.24 s. Synthetic clips may be added to the detector training set. They are excluded from the reported test set.

## 3. Partition

**Outer split.** Training uses the official `train` folder. Testing uses `dev` or `eval`. The folders are patient-disjoint by the corpus release.

**Inner split.** Within each training fold, generator examples are divided into three disjoint subsets:

1. parameter updates;
2. checkpoint selection;
3. reported latent distance.

The retained checkpoint minimises latent distance on subset 2. Published distances are computed on subset 3.

## 4. Models

**Autoencoder.** A convolutional autoencoder maps \(x\) to \(z \in (-1, 1)^{64}\) and reconstructs \(\hat{x}\). It is trained first and then frozen. A fold is retained only if validation normalised mean squared error is at most 0.55. All generators share this encoder and this decoder.

**Hybrid generator.** An eight-qubit data re-uploading circuit. The readout is the vector of analytic Pauli-\(Z\) expectations, followed by a linear map \(8 \to 64\). The parameter count is 704, of which 128 are circuit angles and 576 are classical readout weights.

**Capacity-matched twin.** A classical generator sized to the hybrid total (709 parameters).

**Gaussian mixture.** Fit in the same latent coordinates, with no adversarial critic. It is a density reference and is not capacity-matched.

Adversarial arms share the critic architecture, the learning rate, and the number of generator steps. The critic objective is the Wasserstein distance with gradient penalty.

## 5. Metrics

**Level 1 — latent fidelity.** Unbiased squared maximum mean discrepancy between generated and real holdout codes, with matched sample sizes. Lower is closer.

**Level 2 — decoded fidelity.** Distances on the reconstructed windows (autocorrelation, cross-channel error, spectral error, sliced Wasserstein, Kolmogorov–Smirnov). Lower is closer.

**Level 3 — detection.** Area under the ROC curve on real holdout clips is the primary endpoint. Area under the precision–recall curve and the Matthews correlation coefficient are reported alongside it. Accuracy is recorded and is not used for ranking.

A Level 1 difference is not reported as a Level 2 or Level 3 difference.
