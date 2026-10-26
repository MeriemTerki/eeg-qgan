# Publishable campaign (ae_v5c): architecture, protocol, job order

What is correct and stays, what changed and why, and the order to run everything in.

## 1. Protocol: correct already, do not touch

| Layer | Status |
|-------|--------|
| Outer splits (E1): TUH `train` → `dev` / `eval`, patient-disjoint by folder | Keep |
| Detector protocol (F8 / A16): validation carved from real data before synthetics, threshold chosen on validation only | Keep |
| Autoencoder trained on training-fold windows only (F10), gate excludes or raises (E3) | Keep |
| Generator checkpoint selected on a split that is not the reported one (E4) | Keep |
| Capacity matching, noise-dimension parity, unbiased reported MMD, per-fold noise floor | Keep |

The old F1 near 0.74 came from a leaky grouping that audit E1 removed. It is not a
target and must not be restored.

## 2. What changed for ae_v5c

### Autoencoder (`autoencoder.arch: v2`)

The ae_v5c probe showed the reconstruction error was **still falling when the 300-epoch
budget ran out** (0.4759 at epoch 165 down to 0.4492 at epoch 300, `no_improve=3`). The
reported number was a budget limit, not a capacity limit.

- Residual blocks with group normalisation and channel attention at every scale.
- Wide stem kernel (7 samples) matched to low-frequency EEG morphology.
- Decoder upsamples then convolves instead of using transposed convolutions. A stride-2
  kernel-4 transpose distributes uneven numbers of contributions across neighbouring
  output samples, leaving a fixed ripple in every decoded window that is
  indistinguishable from generator error when decoded windows are scored.
- Budget 600 epochs, warmup then cosine decay, patience 60, STFT weight raised to 0.1.
- No encoder-to-decoder skip connections: the decoder must work on codes a generator
  invented.

`arch: v1` reproduces the ae_v5b stack exactly, which is what the architecture ablation
uses.

### Detector (`detector.kind: cnn_bilstm_v2`)

The probe reached AUROC 0.67 on the harder holdout and 0.74 on the easier one with a
two-layer stack that pooled its recurrent output by a plain mean.

- Residual convolutional stages with channel gating, three scales instead of two.
- Attention pooling over time, concatenated with mean and max summaries. A seizure
  occupies part of a window, so a plain mean dilutes the discriminative interval.
- A band-power branch: relative power per channel in delta, theta, alpha, beta and low
  gamma plus a log total, computed from the window inside the model. It is a fixed
  function of the input applied identically to real and synthetic windows, so it adds no
  information the raw window did not already carry.
- AdamW with weight decay, warmup then cosine decay, gradient clipping, and an
  exponential moving average of the weights that is what gets scored.

### One rebalancing mechanism, not three

ae_v5b stacked oversampling to a one-to-one ratio, class-balanced sampling, and a
positive class weight in the loss. Duplicating a few thousand seizure windows until they
match hundreds of thousands of negatives makes the fit memorise those windows, and
weighting the loss on top of an already balanced sampler applies the same correction
twice. Balanced sampling is now the single mechanism; duplication and loss weighting are
off unless `detector.allow_stacked_rebalance` is set deliberately. What was actually done
is recorded per fold under `augmentation.imbalance_strategy`.

### Dose that is delivered or the run fails

ae_v5b requested about 30 600 synthetic windows under `dose_mode: balance` and delivered
1 024, so its downstream arms were never the experiment the config described.

- Default is `dose_mode: fixed` with `dose_count: 8192` and a matching `n_synthetic`.
- A shortfall above `max_shortfall_fraction` (5 %) raises instead of reporting a capped
  dose as the configured protocol.
- Config validation refuses `dose_mode: balance` when the generation budget cannot cover
  the request.
- The summariser warns when the mean delivered dose falls short of the request.

### Sweeps that exist

`latent.train_fraction` was configured but never applied, and the dose sweep was a single
point. `--set KEY=VALUE` now overrides any config key from the command line, and
`subsample_codes` is wired into the runner, so the dose and scarcity curves are real
curves. `n_folds` in predefined studies is 2, matching the two holdouts.

## 3. Job order

```bash
export OUTPUT_ROOT=$SCRATCH/eeg_qgan_results_ae_v5c
cd ~/eeg_qgan && pip install -e . && python scripts/validate_studies.py
```

| # | Study | Purpose |
|---|-------|---------|
| 1 | `classification_probe` | New autoencoder and detector on the honest split; confirm dose 8192 with zero shortfall |
| 2 | `capacity_rematch` | Headline generation and detection table |
| 3 | `decoded_equivalence` | Whether a latent win survives decoding and reaches the detector |
| 4 | `bottleneck_comparison` + `bottleneck_pooled` | Size of the E3 temporal-bottleneck effect |
| 5 | `surrogate`, `entanglement_ablation`, `depth_ablation`, `critic_width`, `shot_noise` | Mechanism behind any residual gap |
| 6 | `dose_response` | Dose curve at stated fixed counts |
| 7 | `data_efficiency` | Scarcity curve with genuinely subsampled training codes |
| 8 | `qubit_ablation_4q` | Pure-quantum configuration, appendix |

Submit each as `python scripts/narval/submit.py <study> --split-by seed`.

### The two sweeps

```bash
# dose curve: one job per dose, each to its own results root
for d in 512 2048 8192; do
  python scripts/narval/submit.py dose_response --split-by seed --tag "d$d" \
    --set augmentation.dose_count=$d n_synthetic=$d augmentation.max_add=$d \
          output_root=$SCRATCH/eeg_qgan_results_ae_v5c_dose$d
done

# scarcity curve
for f in 0.25 0.5 1.0; do
  python scripts/narval/submit.py data_efficiency --split-by seed --tag "f$f" \
    --set latent.train_fraction=$f \
          output_root=$SCRATCH/eeg_qgan_results_ae_v5c_frac$f
done
```

### Architecture ablation (cheap, and worth a figure)

Shows a null augmentation result is not an artefact of a weak detector or a weak decoder.

```bash
python scripts/narval/submit.py classification_probe --split-by seed --tag detv1 \
  --set detector.kind=cnn_bilstm detector.conv_widths='[64,128]' \
        output_root=$SCRATCH/eeg_qgan_results_ae_v5c_detv1

python scripts/narval/submit.py classification_probe --split-by seed --tag aev1 \
  --set autoencoder.arch=v1 autoencoder.epochs=300 \
        output_root=$SCRATCH/eeg_qgan_results_ae_v5c_aev1
```

### Signal-jitter control (run it if there is time)

A gain from synthetic windows only means something if a circular time shift, a gain
change, a dropped channel and a little added noise do not deliver the same gain. This is
off by default so it cannot become a silent part of the headline; when on, it applies to
every arm including the real-only baseline.

```bash
python scripts/narval/submit.py classification_probe --split-by seed --tag jitter \
  --set detector.window_augment=true \
        output_root=$SCRATCH/eeg_qgan_results_ae_v5c_jitter
```

## 4. Per-fold checks before anything is written up

In each fold record under the study's `results_s*.json`:

- `autoencoder.gates.nmse` below `max_nmse`, and `excluded` false.
- `autoencoder.architecture.arch` is `v2`.
- `detector.augmentation.n_synthetic_added` equals `dose_count`, `shortfall` is 0 and
  `limited_by` is empty.
- `detector.augmentation.imbalance_strategy` shows one active mechanism.
- Baseline and augmented arms share the same detector configuration; only the synthetic
  windows differ.

The gate stays at 0.55 so the number is comparable with ae_v5b. Report the achieved NMSE,
which should be well below it, rather than the threshold.

## 5. Wording that keeps the claims defensible

- Outer evaluation is the predefined patient-disjoint TUH partition, not
  `StratifiedGroupKFold` over per-window patient identifiers.
- Inner generator train / select / report splits are **batch-file** disjoint, which is
  weaker than patient-disjoint. Say so.
- Style-qGAN at `d=64`, `q=8` is hybrid: roughly four fifths of its parameters are the
  classical expansion. `qubit_ablation_4q` is the pure-quantum configuration.
- Lead the clinical section with AUROC, AUPRC and MCC. F1 is secondary under a positive
  rate near five per cent, and the threshold that produces it is chosen on validation.
- The diverged large classical generator characterises an optimisation failure, not a
  capacity limit. Footnote or exclude it.
- ae_v5b generative results (latent MMD, capacity audit) remain usable. Its detection and
  augmentation numbers do not, because the dose was capped and the detector was the weak
  stack.
