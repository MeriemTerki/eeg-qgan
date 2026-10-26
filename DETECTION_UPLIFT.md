# Detection: what was actually limiting it, what changed, what to run

## Short answer

The 0.784 AUROC from `clinical_long_stft` was not a data ceiling. Three things in that
run were handicapping the detector, all of them in preprocessing and architecture rather
than in the split, and all three are now fixed. Nothing below weakens the patient-disjoint
protocol; the outer split is byte-for-byte the same.

The comparison that prompted this — Abou-Abbas et al. 2024, *Front. Artif. Intell.* 7:1437315,
91.73% accuracy — is not a target we should be trying to reach, for reasons set out in
[the last section](#on-the-9173-comparison). The honest rows of that same paper sit at
precision 0.61–0.63 and recall 0.71–0.73.

---

## What was limiting the score

### 1. Per-window normalisation erased the seizure amplitude cue

`_normalize_bounded` computed a 0.5–99.5 percentile range **per channel, per 1.28 s
window** and stretched that window to fill [-1, 1]. A 400 µV seizure burst and 20 µV of
background therefore arrived at the detector with the same amplitude. Amplitude increase
is the single most reliable electrographic seizure sign, and it was being divided out.
Per-channel scaling removed the second cue as well: seizure spread shows up as *relative*
amplitude across electrodes, and every channel was independently renormalised.

Fixed by `data.normalize_scope`:

| Value | Range computed over | Keeps |
|---|---|---|
| `window` (default, legacy) | each channel of each 1.28 s window | nothing about amplitude |
| `recording` | each batch file, shared across channels | amplitude contrast in time and across channels |

`recording` still removes what normalisation is *for* — between-subject and
between-amplifier gain differences. No statistic crosses a recording boundary, so this is
not leakage: the range for a test recording comes only from that recording, which is
information any offline reviewer has.

The default stays `window` so the saved ae_v5c generative results remain reproducible.
The detection studies set `recording` explicitly.

### 2. Long clips were assembled out of eight independently rescaled pieces

`load_clinical_clips` preprocessed each 1.28 s window and *then* concatenated eight of
them. So a 10.24 s clip was eight separately normalised segments glued end to end. That
destroys the slow amplitude envelope — the build-up that makes a long clip worth looking
at — and leaves a step discontinuity at each of the seven seams.

The loader now concatenates first and preprocesses the whole clip once.

### 3. The detector averaged the time axis away

`STFTCNNDetector` ends in `AdaptiveAvgPool2d((1, 1))`: the entire time-frequency plane
collapses to one vector. On a 1.28 s window that costs little. On a 10.24 s clip it
discards onset, evolution and propagation, which is the only reason to use a long clip at
all. It was also small (133 k parameters) relative to the task.

New `detector.kind: stft_bilstm` pools **frequency only**, so the recurrent layer sees one
feature vector per STFT frame, and attention pooling can concentrate on the frames where
the event sits. Roughly 1.2 M parameters at the study defaults.

### 4. A small slice of the corpus

`max_segments_per_batch: 400` with `max_clips_per_file: 64`. Raised to 2000 / 256 in
`clinical_ceiling_v2`. This is the cheapest change of the four and the least interesting,
but it is real.

---

## A confound the fixes introduce, and its mitigation

Once the detector keeps the time axis, packed synthetic clips become a problem. A real
clip is contiguous recording. A synthetic clip is eight independently sampled windows
placed end to end, so it has seams that no real clip has — and every synthetic clip is
labelled positive. A time-resolved detector can learn "seam means seizure", which would
make the augmentation contrast measure seam detection rather than sample quality, and it
would do so identically for every generator arm.

`clinical.synthetic_crossfade` overlap-adds adjacent windows through a raised-cosine ramp
(32 samples in `clinical_transfer_v2`). It does not make a synthetic clip genuinely
continuous — no window-level generator can — but it removes the trivially separable
artefact. Default is 0, which reproduces plain concatenation.

---

## What to run

Each rung isolates one change, so the paper can attribute the uplift instead of asserting
it. Run them in order and stop when the gain flattens.

```bash
cd ~/eeg_qgan && source $SCRATCH/venvs/eeg_qgan/bin/activate
OUT=$SCRATCH/eeg_qgan_results_clinical_v2

# R0 — baseline reproduction of the 0.784 result under the new code path.
python scripts/narval/submit.py clinical_ceiling_v2 --split-by seed --no-gpu --tag r0 \
  --set output_root=${OUT}_r0 data.normalize_scope=window detector.kind=stft_cnn \
        data.max_segments_per_batch=400 clinical.max_clips_per_file=64

# R1 — + recording-scope normalisation and clip-level preprocessing.
python scripts/narval/submit.py clinical_ceiling_v2 --split-by seed --no-gpu --tag r1 \
  --set output_root=${OUT}_r1 detector.kind=stft_cnn \
        data.max_segments_per_batch=400 clinical.max_clips_per_file=64

# R2 — + the time-resolved detector.
python scripts/narval/submit.py clinical_ceiling_v2 --split-by seed --no-gpu --tag r2 \
  --set output_root=${OUT}_r2 \
        data.max_segments_per_batch=400 clinical.max_clips_per_file=64

# R3 — + full data caps. This is the study default.
python scripts/narval/submit.py clinical_ceiling_v2 --split-by seed --no-gpu --tag r3 \
  --set output_root=${OUT}_r3
```

Then the question the paper actually asks:

```bash
# Quantum vs capacity-matched twin vs GMM, on top of the R3 ceiling.
python scripts/narval/submit.py clinical_transfer_v2 --split-by seed-fold --no-gpu \
  --set output_root=$OUT
```

Analysis is unchanged:

```bash
for r in r0 r1 r2 r3; do
  eeg-qgan-analyze ${OUT}_$r/clinical_ceiling_v2 --csv ${OUT}_$r/clinical_ceiling_v2/tables
done
eeg-qgan-analyze $OUT/clinical_transfer_v2 --csv $OUT/clinical_transfer_v2/tables
```

CPU is fine and is currently faster end to end than waiting in the GPU queue.
`--no-gpu` now triples the requested walltime and floors the request at 16 cores, which
is what the earlier CPU submissions were missing.

---

## What this does *not* change

The 1.28 s generative rematch (`capacity_rematch`, `dose_response`, `data_efficiency`,
the ablations) is untouched, because `normalize_scope` defaults to `window`. Those results
stand and the ae_v5c figures remain valid.

`clinical_transfer_v2` does use `recording`, which changes the autoencoder input, so its
**latent MMD numbers are not comparable to ae_v5c**. That is the intended division:
ae_v5c owns the generative leaderboard, the clinical studies own the downstream claim.
Do not put the two sets of MMD values in the same table.

---

## How to report quantum impact

Unchanged, and still the honest framing: **paired deltas against the shared real-only
baseline**, with fold-clustered intervals. ΔAUROC / ΔAUPRC / ΔMCC for `quantum` versus
`classical_equiv` versus `gmm`. The absolute AUROC belongs to the protocol; only the delta
belongs to the generator.

The supervisor's post-audit brief is explicit that downstream is secondary evidence and
that a drop under a real patient split is expected rather than alarming (escalation
trigger 6). A finding of "no transfer" is a result, not a failure, and it sits alongside
the generative rematch rather than competing with it.

---

## On the 91.73% comparison

Abou-Abbas et al. 2024 report 91.73% accuracy with WGAN-GP augmentation against 86% on
real data. Three things make it the wrong target:

**The augmentation happens before the split.** Section 4.1: *"It starts with data
augmentation (DA) using various Generative Adversarial Networks... Once the augmented
dataset is created, it is split into training and testing sets using a 10-fold
cross-validation approach."* The GAN is fit on the whole corpus, then the augmented pool
is divided. Synthetic samples derived from test-fold patients therefore appear in
training. That is the same class of defect as audit E1, and the 86% → 91.73% jump is its
expected signature. Their classifier CV is patient-wise; their generator is not.

**The metric is accuracy, not AUROC.** They never report AUROC or AUPRC. On a corpus
where roughly 6% of files contain seizure, accuracy is dominated by the negative class.
Table 3 reports real-only accuracy 0.86 with specificity 0.67 — arithmetically impossible
at 6% prevalence — so the evaluated set is already rebalanced, and the number does not
mean what a reader assumes.

**The input is different.** 836 hand-engineered features (44 per channel over 19 channels:
band powers, entropies, Hjorth parameters, wavelet energies) fed to an LSTM. Ours is a raw
22-channel waveform. Feature engineering of that kind is worth a lot on this task.

Their honest rows are the useful comparison: SMOTE, ADASYN and class weighting, under
patient-wise CV without a pre-split GAN, give precision 0.60–0.63, recall 0.71–0.73,
specificity 0.88. Against that, patient-disjoint AUROC near 0.78 with AUPRC 0.28 at ~5%
prevalence is in the right neighbourhood, and it is measured under a protocol theirs does
not meet.

Worth noting in the write-up: Khadidja Henni is an author on that paper. Framing our
result as "the same group's pipeline, re-measured under a split that closes the
augmentation leak" is a stronger and more collegial contribution than trying to match a
number that the leak produced.
