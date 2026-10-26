# Audit corrections

Each item states what the old code did, why it distorted the result, what the new code
does, and how to check it. The ones under "Corrections that change conclusions" are
expected to move published numbers; those under "Corrections that change interpretation"
mostly change what the numbers are allowed to support.

## Corrections that change conclusions

### E1. Grouping did not group by patient

`binary_classification_loader.py` assigned a group key per window and, when no patient
information was available, fell back to `f"{batch_stem}_{i // max(1, n // 100)}"`. That
divides each batch file into about one hundred equal chunks of the array index regardless
of how many windows it holds, so a 2000-window batch produced 100 pseudo-groups of exactly
20 consecutive windows. Group-aware cross-validation then held out chunks of an array
rather than patients, so windows from a single recording, seconds apart and highly
correlated, appeared in both the training and test folds. Files read one segment at a time
took a second path and were grouped by filename, which is one group per batch file; that
is coarser but still not a patient, since each batch pools many patients.

Every patient-wise claim in the manuscript rests on this key. With the fallback active,
the reported score measures interpolation within recordings the model has already seen,
which is optimistically biased and not the quantity a clinical reader would assume.

Now `eeg_qgan/data/tuh.py` resolves identity from a sidecar file, an info CSV, or the
filename, and raises if none is available. There is deliberately no positional fallback.
`assert_grouping_is_plausible` additionally rejects a key that is too fine to be a patient
or whose groups are suspiciously uniform in size, since real recordings differ in length.

Verify: `pytest tests/test_splits_and_grouping.py -k grouping`. If the batch files lack
identity, `scripts/build_patient_sidecars.py` recovers it or explains what is missing.

**Expected effect:** all downstream detector metrics should fall. The gap between
augmented and baseline may narrow, widen or reverse; the previous gap was measured under
leakage and is not a reliable guide.

### E3. The autoencoder discarded time, and a failing gate did not stop the fold

`ConvEncoder` ended in `AdaptiveAvgPool1d(1)`, averaging each channel over the whole
window. A latent code carrying no temporal structure cannot represent the morphology that
distinguishes a seizure, so the generators were being asked to model a representation from
which the signal of interest had already been removed. The autocorrelation and spectral
distances then largely reflected the decoder's prior rather than the generator.

Separately, the quality gate computed reconstruction error and recorded a failure without
excluding the fold, so folds whose autoencoder had not learned anything usable still
contributed to the means.

Now the bottleneck keeps a temporal axis by default (`bottleneck: temporal`), and
`evaluate_ae_gates` raises `AEGateFailure`, which the runner records as an excluded fold.
The summariser reports how many folds were excluded and why, so a shrinking denominator
cannot pass unnoticed.

Verify: run `bottleneck_comparison` against `bottleneck_pooled`; the difference is the
size of this effect.

**Expected effect:** decoded-space fidelity should improve substantially for every arm.
Because it improves the shared decoder, it may compress the differences between arms.

### E4. Checkpoints were selected on the data used to report

The training loop chose the best generator step by the distance to a held-out latent set,
then reported the distance on that same set. Selecting on the reported quantity optimises
it directly, and the bias is largest for whichever arm is noisiest across steps, because
more steps means more chances for a favourable draw. That makes the comparison between
arms unfair in a way that is invisible in the output.

Now `nested_latent_splits` carves three patient-disjoint sets from each training fold:
one to train on, one to select the checkpoint, one to report. Selection uses the fast
biased estimator; reporting uses the unbiased one.

Verify: `pytest tests/test_splits_and_grouping.py -k ThreeWay`.

**Expected effect:** all latent distances should rise. Arms with noisier trajectories,
typically the quantum arm with parameter-shift gradients, should rise most, so the
previously reported ordering may not survive.

### E5. Augmentation dose could collapse to nothing

The quality gate filtered synthetic windows by a threshold that was frequently
unreachable. When nothing passed, the run continued and reported "augmented" metrics for a
detector that had received no synthetic data at all. The resulting null was reported as
evidence that augmentation does not help, when it was evidence that the gate rejected
everything.

Now `DoseSpec` states the dose explicitly, the delivered count is recorded per fold, and a
dose below the configured minimum raises rather than passing silently. The summariser
flags any source whose mean dose is under one window.

Verify: the smoke run prints the delivered dose per arm per fold, and
`results/*/summary.json` carries a warning when it collapses.

### E6. Extra seeds produced duplicate folds, not replicates

The cross-validation splitter was constructed once, outside the seed loop, so every seed
saw the same partition. Rows from different seeds were then treated as independent
observations when computing standard errors. Three seeds over five folds looked like
fifteen observations while containing five distinct partitions, which understates the
interval by roughly the square root of the replication factor and turns noise into
significance.

Now `make_outer_splitter` is called inside the seed loop, so seeds give genuinely
different partitions, and `eeg_qgan/stats/inference.py` bootstraps whole folds rather than
rows. The summariser reports the number of independent partitions next to the row count.

Verify: `pytest tests/test_splits_and_grouping.py -k Seeds`.

**Expected effect:** intervals widen. Some previously significant differences will stop
being significant, which is the correct outcome rather than a loss.

## Corrections that change interpretation

### F3. The MMD estimator was biased and sample-size dependent

The biased V-statistic includes the self-kernel diagonal, which adds a positive term that
shrinks as the sample grows. Comparing two arms evaluated on different numbers of samples
therefore compared their sample sizes as much as their distributions. Now
`mmd2_unbiased` drops the diagonal for all reporting, the sample size is matched across
arms, and the biased form is retained only as a differentiable training loss.

Verify: `pytest tests/test_mmd_bias.py`.

### The noise floor is now computed in every fold

A distance means nothing without knowing what two samples of real data score against each
other. That reference used to be a separate study, so most results were read without it.
It is now computed per fold and stored on every row, and the summariser prints it above
the leaderboard.

This exposes a specific trap the audit named. A paired test across folds can resolve a
difference far smaller than the floor, because pairing removes the fold-to-fold variance
the floor measures. Such a difference is statistically real and scientifically empty: it
is below the resolution of the distance itself. The summariser flags these explicitly
rather than leaving a significant result to be read as a quantum advantage. In the smoke
run this fires immediately, with a significant gap 0.078 times the floor width.

### F1, E2. One latent coordinate system

Whitening was applied inconsistently: the adversarial arms trained on whitened codes with
a bounded output activation, while the mixture baseline was fitted in the raw code space,
so the two were scored in different coordinates. The bounded activation also could not
reach the tails of the real distribution, so part of the reported distance was an
unreachable target rather than a modelling failure.

`LatentTransform` now defines one coordinate system used by every arm and every metric,
with an optional output scale, and `support_diagnostics` reports what fraction of real
codes lie inside the generator's reachable range. Below 90 percent the summariser says so.

### F6. Noise dimension parity

Arms were compared at matched parameter counts while receiving different noise widths, so
capacity was confounded with input width. `LatentGenerator` now requires an explicit
`noise_dim`, and `capacity_audit` refuses to run a study whose arms disagree.

### F12. The large classical generator had diverged

Its reported failure came from an unbounded activation and a learning rate that made
training unstable, so it measured an optimisation failure rather than a capacity limit,
yet appeared in tables as though it characterised large classical generators. It now uses
a bounded activation with optional spectral normalisation, and `_assess_convergence` marks
any arm that fails to converge. The summariser reports the count and states that those
folds do not carry information about capacity.

### F13. Correlation and autocorrelation errors cancelled

Cross-channel error was computed as a signed mean over channel pairs, so positive and
negative errors cancelled and a generator with no channel structure scored well. In the
smoke run the diagnostic makes this visible: real off-diagonal correlation 0.91 against
0.02 for an untrained arm, with a sign cancellation ratio near one. Errors are now
absolute, and autocorrelation compares full curves rather than a single lag.

### F5, F7. Quantum diagnostics

Parameter counts mixed classical projection weights with circuit parameters, so a
"parameter-matched" comparison was not matched on the quantum part. `parameter_breakdown`
now separates them. The entanglement diagnostic used an absolute threshold that simulator
floating-point noise could exceed; `coordinate_coupling` now uses a relative tolerance.

### Reproducibility

Per-fold randomness was derived with the built-in `hash`, which CPython randomises per
process. Splitting a study across SLURM jobs therefore gave different random streams for
the same nominal seed. `spawn_rng` now uses `blake2b`, so a fold is reproducible whether
it runs alone or with others. The Jacobian spectral norm, used for the Lipschitz argument,
was a random-probe approximation and is now computed exactly.

## Verification

```bash
pytest tests -q                                   # 34 tests
python scripts/validate_studies.py                # all 13 studies
python -m eeg_qgan.experiments.runner smoke --data-root data/synthetic
eeg-qgan-analyze results/smoke --reference classical_equiv
```

## Suggested order for re-running

1. `bottleneck_comparison` and `bottleneck_pooled`, to size the E3 effect cheaply before
   committing cluster time.
2. `capacity_rematch`, the headline result, under the corrected selection protocol.
3. `surrogate` and `entanglement_ablation`, which test whether any remaining difference
   has a mechanism behind it.
4. `dose_response` and `data_efficiency`, which need E1 to be settled first because they
   are downstream-metric studies.

A note on expectations. Several of these corrections work in the same direction: E1 removes
optimistic leakage from the detector results, and E4 removes optimistic bias from the
latent distances. It is a realistic outcome that the quantum advantage reported previously
does not survive them. If that happens, the finding is worth reporting on its own terms.
A negative result with a stated minimum detectable effect, a noise floor, and honest
patient-wise splits is publishable and considerably harder to dismiss than a positive
result resting on a grouping bug.
