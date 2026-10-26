# Evaluation protocol

The numbers in `results/published` were produced on Narval, a national cluster of the Digital Research Alliance of Canada (Alliance; formerly Compute Canada). Training ran as SLURM jobs on compute nodes. A personal computer was used to prepare the code and to read the finished tables, not to train the models. How to repeat that launch is in [REPRODUCING.md](REPRODUCING.md).

## Signal

Each example is a window of scalp EEG, 22 channels by 256 samples, at 200 Hz (1.28 s). Every model sees the same preprocessing: a Daubechies-6 soft threshold, then a map of the amplitudes into \([-1, 1]\).

Detection does not use that window on its own. Eight contiguous windows form one clip of 10.24 s. Synthetic clips may be mixed into detector training. The detection numbers are computed on real holdout clips.

## Split

The outer split is the official corpus partition. Models train on `train` and are tested on `dev` or on `eval`. Those folders do not share patients.

Inside a training fold, the generator data are split three ways: fitting, checkpoint selection, and reporting. The checkpoint is the step with the lowest latent distance on the selection split. The distance in a table is measured on the reporting split.

## Representation

A convolutional autoencoder maps a window to a code in \((-1, 1)^{64}\) and back. It is trained first and then frozen. Every generator uses that same encoder and decoder. A fold is kept only when the normalised reconstruction error on held-out training windows is at most 0.55.

On Narval the autoencoder is the stage that requests a GPU.

## Generators

The hybrid generator is a data re-uploading circuit on eight qubits. Each qubit is read out by an analytic Pauli-\(Z\) expectation, and a linear map expands those eight values to 64 dimensions. The circuit is simulated. It does not run on a quantum processor, and it does not request a GPU.

The classical twin has the same total number of parameters as the hybrid model. A Gaussian mixture is fit to real codes in the same coordinates. It is a density reference. It is not sized to the hybrid budget.

All adversarial arms share one Wasserstein critic, one learning rate, and one step budget. Selection and reporting use an unbiased squared maximum mean discrepancy, with the two sample sizes matched.

## Detection

The detector is a spectrogram model with bidirectional recurrent layers. It is trained on a GPU node. Its primary score is the area under the receiver operating characteristic on real holdout clips. The area under the precision–recall curve and the Matthews correlation are reported with it. Accuracy is not the score used to rank models.

## How to read a row

A latent distance compares codes. A decoded distance compares waveforms after the frozen decoder. A detection score compares classifiers on real clips. A result at one of these stages is not rewritten as a result at another.
