# Clinical long-clip / STFT protocol (E1-preserving)

## Why this exists

The 1.28 s waveform detector under patient-disjoint TUH folders lands near
**AUROC ≈ 0.70**. That is the correct rematch clinical readout for the generative
study — but it is **not** comparable to papers that score **8–12 s** clips / STFT.

This protocol adds a **second** clinical evaluation:

| Protocol | Input | Detector | Role |
|----------|--------|----------|------|
| Rematch (existing) | 1.28 s waveform | `cnn_bilstm_v2` | Generative fidelity → clinical transfer |
| **Clinical STFT** | ~10.24 s contiguous clips | `stft_bilstm` | Absolute literature-style ceiling + quantum transfer |

Outer split stays **train vs dev/eval** (patient-disjoint). No leaky GroupKFold.

## Code

- `eeg_qgan/data/clips.py` — contiguous clip builder, `load_clinical_clips`, and
  `pack_synthetic_clips` with optional raised-cosine seam crossfade
- `eeg_qgan/models/detector.py` — `STFTCNNDetector` (`kind: stft_cnn`, v1) and
  `STFTBiLSTMDetector` (`kind: stft_bilstm`, keeps the time axis)
- `eeg_qgan/data/tuh.py` — `data.normalize_scope: window | recording`
- Runner hooks when `clinical.enabled: true`
- Studies:
  - `clinical_long_stft` / `clinical_long_stft_transfer` — v1, kept for the ablation
  - `clinical_ceiling_v2` — detector-only ceiling with all three fixes
  - `clinical_transfer_v2` — quantum / classical / GMM transfer on top of that ceiling

### v1 → v2

`clinical_long_stft` reached AUROC 0.784 / AUPRC 0.277 / MCC 0.294. Three limits in it
were fixed; see [`DETECTION_UPLIFT.md`](DETECTION_UPLIFT.md) for the full account and the
staged ablation that attributes the uplift.

| Fix | v1 | v2 |
|---|---|---|
| Amplitude | each 1.28 s window rescaled to fill [-1, 1] | one range per recording, shared across channels |
| Clip assembly | eight separately normalised pieces glued together | concatenate first, preprocess once |
| Detector | `stft_cnn`, averages the whole time-frequency plane | `stft_bilstm`, pools frequency and keeps time |
| Data | 400 segments/file, 64 clips/file | 2000 / 256 |
| Synthetic clips | plain concatenation, seams learnable as "positive" | 32-sample crossfade |

## What you do on Narval

Sync this repo, then:

```bash
cd ~/eeg_qgan
source $SCRATCH/venvs/eeg_qgan/bin/activate
pip install -e .   # if needed after pull

OUT=$SCRATCH/eeg_qgan_results_clinical_v2

# 1) Absolute ceiling, real clips only. CPU is currently faster end to end than the
#    GPU queue; --no-gpu triples the walltime request and floors it at 16 cores.
python scripts/narval/submit.py clinical_ceiling_v2 --split-by seed --no-gpu \
  --set output_root=$OUT

# 2) Only after (1) clears 0.784: the quantum transfer question on top of it.
python scripts/narval/submit.py clinical_transfer_v2 --split-by seed-fold --no-gpu \
  --set output_root=$OUT
```

The staged ablation (`r0`…`r3`) that attributes the uplift to each fix is in
[`DETECTION_UPLIFT.md`](DETECTION_UPLIFT.md).

Analyze:

```bash
eeg-qgan-analyze $OUT/clinical_ceiling_v2 --csv $OUT/clinical_ceiling_v2/tables

eeg-qgan-analyze $OUT/clinical_transfer_v2 \
  --reference classical_equiv --csv $OUT/clinical_transfer_v2/tables
```

## Thesis / paper rules

1. **Never mix** 1.28 s rematch AUROC and 10 s clip AUROC in one chart without labelling
   the protocol on each bar.
2. Lead generative claims with the latent MMD rematch (unchanged, ae_v5c).
3. `clinical_transfer_v2` uses `normalize_scope: recording`, which changes the autoencoder
   input. Its **latent MMD numbers do not belong in the ae_v5c leaderboard** — different
   representation, different coordinate system.
4. Report the clinical protocol as secondary: absolute scores plus quantum ΔAUROC under
   longer context, always as a paired delta against the shared real-only baseline.

## Expected outcome (honest)

Longer clips with amplitude preserved should raise absolute AUROC. None of that makes
quantum synthetics help. The two questions are independent, and a null transfer result on
this protocol is reportable in exactly the same way as on the 1.28 s one.
