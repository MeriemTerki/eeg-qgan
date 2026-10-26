# Why detection numbers look “low”, what is clinically fair, and what we changed

## What your rematch actually measured (ae_v5c, previous detector)

| Metric | Baseline mean | Role |
|--------|---------------|------|
| **AUROC** | ~0.71 | Primary ranking score |
| **AUPRC** | ~0.22 | Primary under imbalance (chance ≈ positive rate) |
| **MCC** | ~0.19 | Primary thresholded association |
| F1 | ~0.25 | Secondary — harsh at ~5% positives |
| Precision | ~0.18 | Bounded by prevalence |
| ECE / Brier | in fold JSON | Calibration (now exported by analyze) |

Synthetic dose **8192** was delivered; deltas were small / slightly negative. That is a
**null clinical transfer** of synthetics, not a failed generator study. Latent + decoded
fidelity for Style-qGAN vs `classical_equiv` remain the generative headline.

## Why literature AUROC ~0.90 is not an apples-to-apples bar

Many TUH papers reporting AUROC ≈ 0.85–0.92 use:

- **8–12 s** clips and/or **STFT / spectrogram** inputs  
- Different event-level post-processing and false-alarm/hour budgets  
- Sometimes patient-specific or less strict leakage controls  

This thesis uses **1.28 s** windows (`22×256` @ 200 Hz) and a **patient-disjoint**
`train→dev/eval` protocol (audit E1). Absolute F1 ~0.25 and AUROC ~0.70 under that
protocol are **not** the same claim as “AUROC 0.92 on 12 s STFT.” Do **not** restore
the old leaky split to recover F1 ~0.74.

## Metrics reviewers should see

**Generation:** latent MMD² (+ CI), paired Δ vs twin, decoded ACF / channel / spectral /
KS (only if AE gate passed), noise floor, convergence flags.

**Detection (lead with these):** AUROC, AUPRC, MCC, ECE (and Brier).  
**Also report:** sensitivity, specificity, precision, F1 at the validation-chosen
operating point (`threshold_metric: mcc`).

## Detector boost (ae_v5c-detboost) — already in configs

- Wider residual stack `[96,192,384]`, `blocks_per_stage: 2`, BiLSTM 192×2  
- Mild `window_augment` on **all** arms including baseline  
- Threshold chosen by **MCC** on real-only validation  
- Longer detector budget (100 epochs, patience 20)  
- Analyzer exports ECE / Brier / full clinical table  

## Job order under the deadline

1. Sync code; submit **`classification_probe`** (fast check).  
2. If mean eval-like AUROC ≥ ~0.78 (or clear lift vs 0.71), re-submit
   **`capacity_rematch`** and **`decoded_equivalence`** to a **new** output root
   (e.g. `eeg_qgan_results_ae_v5c_detboost`) so generative tables from the finished
   rematch stay usable.  
3. Leave mechanism jobs (`skip_downstream`) running — they do not need the detector.

## Honest thesis wording

Lead Results with generative rematch (latent + decoded). In Detection, report absolute
AUROC/AUPRC/MCC/ECE under E1, state the window-length protocol difference vs STFT
literature, and report the synthetic-dose effect as null/small-negative at 8192 unless
the boosted rematch shows otherwise.

## Secondary clinical protocol (long clip + STFT)

When absolute scores must be discussed next to STFT literature, use the **separate**
`clinical_long_stft` / `clinical_long_stft_transfer` studies (~10 s contiguous clips,
`stft_cnn`). Document both protocols in Methods; do not mix them in one unlabeled figure.
Details: `CLINICAL_STFT_PROTOCOL.md`.

## How to talk about “quantum impact on detection”

The clinical contribution of Style-qGAN is **not** automatic from a lower latent MMD.
Measure impact as paired deltas vs the real-only baseline:

- primary: ΔAUROC, ΔAUPRC, ΔMCC  
- also: sensitivity, specificity, precision, F1, ECE / Brier at the MCC-chosen threshold  

Compare `augmented_quantum` against `augmented_classical_equiv` and GMM under the **same**
dose and the **same** E1 split. A null or negative quantum Δ is publishable: it shows that
representational rematch and clinical transfer are separate claims.

Absolute AUROC under this protocol is expected near **0.70+** on 1.28 s windows; do not
treat literature STFT AUROC ≈ 0.9 as the success bar. Detboost (`classification_probe`, then
optional rematch into `*_detboost`) only tries to raise the **real-only** detector ceiling
while keeping E1.
