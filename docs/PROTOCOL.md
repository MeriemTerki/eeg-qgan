# Evaluation protocol

## Signal

Each example is a window \(x \in \mathbb{R}^{22 \times 256}\) from scalp EEG at 200 Hz (1.28 s). Denoising is a Daubechies-6 soft threshold, applied in the same way to every model. Amplitudes are then mapped into \([-1, 1]\).

A detector does not see a single window. Eight contiguous windows form one clip of 10.24 s. Synthetic material may be added to detector training. Reported detection scores use real holdout clips only.

## Split

The default outer split is the corpus partition: training on `train`, testing on `dev` or on `eval`. These folders do not share patients.

Within a training fold, generator data are divided into three disjoint parts: fitting, checkpoint selection, and reporting. The checkpoint is the step with the lowest latent distance on the selection part. The number written in a table is the distance on the reporting part.

## Representation

A convolutional autoencoder maps a window to a code in \((-1, 1)^{64}\) and back. It is trained first and then frozen. Every generator uses that encoder and that decoder. A fold is kept only when the reconstruction error on held-out training windows is at most 0.55 in normalised mean squared error.

## Generators

The hybrid generator is a data re-uploading circuit on eight qubits, read out by analytic Pauli-\(Z\) expectations and expanded by a linear map to 64 dimensions. The classical twin is sized to the same total number of parameters. A Gaussian mixture is fit to real codes in the same coordinates. It is a density reference, not a capacity-matched control.

Training uses a Wasserstein critic with a gradient penalty. The critic, the learning rate, and the number of steps are shared. The latent distance used for selection and for reporting is an unbiased squared maximum mean discrepancy, with the two sample sizes matched.

## Detection

The detector is a spectrogram network with bidirectional recurrent layers. Its primary score is the area under the receiver operating characteristic on real holdout clips. Precision–recall area and the Matthews correlation are reported with it. Accuracy is not used to rank models.

## Reading a comparison

A difference in latent distance is a statement about codes. A difference after decoding is a statement about the reconstructed waveform. A difference in detection is a statement about the classifier. The three are recorded separately.
