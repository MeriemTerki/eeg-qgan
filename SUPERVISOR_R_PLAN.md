# Post-audit work plan (R0–R3) mapped onto this repo

Status of every item in the supervisor's post-audit brief against the current code.
"Done" means implemented and tested, not that the job has been run on Narval.

---

## R0 — Gates

| Item | Status | Where |
|---|---|---|
| **R0.1** Cluster sidecar check for E1 | **Blocked on you** | `scripts/build_patient_sidecars.py`. Until this reports real per-window patient ids, `downstream_scoped` cannot run and the E1 answer stays "TUH predefined folders, patient-disjoint by the dataset authors' construction, batch-file groups inside train". |
| **R0.2** Whitening-off pilot | **Done and adopted** | `latent_whiten` is retired — `config.py` raises if it appears. `latent_transform: identity` is the default and every study uses it. |
| **R0.3** Frozen list applied to the draft | **Yours** | Not a code item. |

R0.1 is the one thing on this page that only you can do, and several decisions below
depend on it. Run it first.

---

## R1 — Metric and protocol infrastructure (no training)

| Item | Status | Where |
|---|---|---|
| **R1.1** Unbiased estimator, matched n | Done | `evaluation/mmd.py`: `mmd2_unbiased` removes the kernel diagonal; `mmd_report(match_n=True)` equalises both sides by default. |
| **R1.1** Floor at 300 partitions | Done | `noise_floor_repeats: 300` in `base.yaml` (was 50). |
| **R1.1** Shuffle test under the floor | **New** | `scripts/copula_shuffle_test.py`. Reads saved `z_report.npy`, no retraining. Places the shuffle against the floor distribution and prints an explicit verdict on claim C5, including the case where C5 has to be withdrawn. |
| **R1.2** Single coordinate system | Done | One `LatentTransform` per fold; GMM fits and samples through the same transform (`models/mixture.py`); reported distance and selection distance are separately labelled. |
| **R1.3** Sliced distance on 1-D projections of the joint | **New** | `evaluation/decoded.py::sliced_wasserstein`, reported per arm as `latent_sliced_w1` (latent) and `decoded_sliced_w1` (decoded). |
| **R1.3** Lagged cross-correlation, absolute value | **New** | `evaluation/decoded.py::lagged_cross_correlation_error`, reported as `lagged_xcorr_error`. Splits zero-lag from non-zero-lag so propagation error is not diluted by the instantaneous term. |
| **R1.4** Three-way split, selection in the reported space | Done | `data/splits.py::nested_latent_splits` → train / select / report, group-respecting. |
| **R1.5** Splitter inside the seed loop | Done | `make_outer_splitter` is called per seed in `runner.py`. |
| **R1.5** Fold-clustered bootstrap | Done | `stats/inference.py::cluster_bootstrap_mean` resamples whole folds. |
| **R1.6** Loud-failure hygiene | Done | AE gates raise or exclude by policy; augmentation shortfall raises; `latent_whiten` raises. |

**Run now, costs nothing:**

```bash
python scripts/copula_shuffle_test.py $SCRATCH/eeg_qgan_results_ae_v5c/capacity_rematch
```

---

## R2 — Reruns

| Item | Status | Study |
|---|---|---|
| **R2.1** Capacity-matched rematch, clean protocol | Partly — see the caveat below | `capacity_rematch` |
| **R2.2** Entanglement ablation at d=8, whitening off | **New, ready** | `entanglement_ablation_d8` |
| **R2.3** Rebuilt unified leaderboard | Arms not yet agreed | propose the final arm list to the supervisor before running |
| **R2.4** Surrogate fits redux | **Not implemented** | `surrogate` trains a Fourier generator independently (brief item E2). The *direct fit to the frozen PQC* (E-C / R2.4) does not exist yet. |
| **R2.5** Downstream, scoped | **New, gated on R0.1** | `downstream_scoped` |
| **R2.6** Pooling-free AE | Already the default | `autoencoder.bottleneck: temporal`; `bottleneck_pooled` reproduces the audited failure for contrast. |

### The R2.1 caveat you should raise with the supervisor

The brief asks for "2 seeds × 5 genuinely different folds". Under
`data.split_mode: predefined` there are exactly **two** outer folds — train vs `dev` and
train vs `eval` — and they do not depend on the seed. Extra seeds change initialisation
only, so they are pseudo-replicates at the fold level, which is the E6 problem the brief
is trying to close. `summarize.py` already reports this honestly ("4 rows, 2 independent
partitions"), and the cluster bootstrap treats folds as the unit.

Five genuinely different folds require `split_mode: grouped_cv`, which requires the R0.1
sidecars. So R2.1 and R2.5 are gated on the same thing. Worth flagging in your next
message to him rather than running 2×5 and reporting it as five replicates.

---

## R3 — Statistics, figures, atlas

Not started; largely depends on R2 outputs. `summarize.py` already produces MDE,
Holm-Bonferroni multiplicity correction and TOST-style equivalence intervals, so the
per-fold dot plots and the failure-atlas table are the main remaining pieces.

---

## Suggested order

1. **R0.1** sidecar check — gates R2.1 and R2.5.
2. **R1.1 shuffle test** — free, runs on saved artefacts, and decides whether claim C5
   survives. Do this while the sidecar check is running.
3. **`clinical_ceiling_v2`** — the detection work described in
   [`DETECTION_UPLIFT.md`](DETECTION_UPLIFT.md). Independent of everything above.
4. **`entanglement_ablation_d8`** (R2.2) — decides which branch of C4 gets written.
5. **`clinical_transfer_v2`** and, once sidecars exist, **`downstream_scoped`** (R2.5).
6. **R2.4** direct surrogate fit — needs new code; say the word and it gets built next.
