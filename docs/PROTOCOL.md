# Evaluation protocol

This note is the contract implemented by `configs/base.yaml` and enforced in `eeg_qgan.config`. A study YAML may change a width, a dose, or an arm list. It may not silently restore a comparison the audit showed to be confounded.

The engineering thesis used an earlier campaign. This repository is the protocol a paper can cite. A number from the thesis is not a number from this protocol until the corresponding study has been rerun.

## Question

Does a hybrid Style-qGAN, placed in a latent space learned from multichannel scalp EEG, match the distribution of real seizure codes more closely than a classical generator that is allowed the same host, the same critic, and the same parameter budget? If it does, does that closeness survive decoding, and does it change seizure detection on real holdout clips?

Those are three questions. The code records three answers.

## Data

Windows come from the TUH Seizure Corpus, BinaryClassification release.

| Item | Value |
| --- | --- |
| Channels | 22, bipolar, fixed order |
| Sampling rate | 200 Hz |
| Window | 256 samples, 1.28 s, non-overlapping |
| Detector clip | 8 contiguous windows, 10.24 s |
| Outer split | Official `train` against `dev`, and `train` against `eval` |
| Denoising | Daubechies-6 soft threshold, identical for every arm |
| Scaling | Robust map to \([-1, 1]\) |

The outer folders are patient-disjoint by the corpus construction. That is the split used when `data.split_mode` is `predefined`. It does not depend on the random seed. Extra seeds then change initialisation, not the partition. A study that needs several distinct patient folds must set `split_mode: grouped_cv` and must supply per-window patient sidecars. `scripts/build_patient_sidecars.py` writes those files or stops and says what is missing. The loader does not invent a patient id from the array index.

Inside each training fold the generator sees three group-respecting pieces: one to train, one to select the checkpoint, one to report the distance. Selection and reporting are not the same split.

## Model

A convolutional autoencoder maps each window to a code in \((-1, 1)^{64}\) and back. The bottleneck keeps a temporal axis. There are no encoder–decoder skip connections, because the decoder later has to rebuild codes that a generator invented. A fold proceeds only if validation NMSE is at most 0.55. A failed gate raises or excludes the fold. It is not averaged in.

Generators are trained against one Wasserstein critic. The hybrid arm is an eight-qubit re-uploading circuit with a classical linear readout from 8 to 64. Most of its parameters sit in that readout. The capacity-matched twin is sized to the hybrid total, not to the circuit angles alone. A Gaussian mixture is fit in the same latent coordinates and is a density reference, not a twin. A large multilayer perceptron is a capacity audit. An arm that does not converge is reported as such.

Reported latent distance is the unbiased squared maximum mean discrepancy, with the two sample sizes matched. The biased estimator remains only as a training loss. Checkpoint choice uses holdout MMD on the selection split, not the adversarial loss.

Synthetic windows may enter detector training. They do not enter the test set. The detector is an STFT–BiLSTM. Its scores on the paper's clinical question are computed on real holdout clips. AUROC is the primary detection endpoint. Accuracy is recorded and is not used to rank arms.

## What a run is allowed to conclude

| Observation | Allowed statement |
| --- | --- |
| Lower latent MMD than the matched twin | The hybrid is the closer adversarial generator under this contract |
| Lower latent MMD than the mixture | Not claimed from a matched rematch. The mixture has a different budget and no critic |
| A decoded distance that ranks arms differently from MMD | Latent fidelity and decoded structure are different outcomes |
| A detection change whose interval contains zero | No detection benefit at that dose |
| A product circuit close to a classical Fourier model | The function class is available without claiming a quantum computer |

The package does not claim a quantum advantage. Expectation values are analytic. Hardware shot noise is a separate study.

## Where the rules live

| Rule | Code |
| --- | --- |
| Configurations that would reintroduce a known confound are rejected | `eeg_qgan/config.py` |
| Patient grouping and the three-way split | `eeg_qgan/data/splits.py` |
| Unbiased MMD | `eeg_qgan/evaluation/mmd.py` |
| One latent coordinate system per fold | `eeg_qgan/training/latent_space.py` |
| Seed-wise folds and the gate | `eeg_qgan/experiments/runner.py` |
| Fold-clustered intervals | `eeg_qgan/stats/inference.py` |

`AUDIT_FIXES.md` records the failure each rule closes.
