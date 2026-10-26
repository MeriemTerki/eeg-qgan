# Thesis figure capture (training curves + synthetic waveforms)

The canonical deadline JSON records only **truncated** autoencoder history (last 20
epochs) and **sparse** WGAN-GP checkpoints (every 250 steps). Detector per-epoch loss
was not logged at all. To populate the thesis with publication-quality training curves
and decoded waveform panels, rerun the dedicated capture study below.

## What gets captured

| Output | File(s) |
|--------|---------|
| Full AE train/val/NMSE curves | `results_*.json` → `autoencoder.history` |
| Dense WGAN-GP losses (every 25 steps) | `results_*.json` → `arms.<arm>.history` |
| Detector train loss + val AUPRC/AUROC | `results_*.json` → `detector.*.history` |
| Real vs synthetic 1.28 s windows | `artifacts/.../decoded/real_figure.npy`, `quantum_figure.npy`, … |
| Quantum vs real spectrogram | from the same figure arrays |

## If a capture job fails the AE gate (~0.558 vs 0.55)

That is **not** a new architecture failure. The 80k-window AE subsample was seeded
with Python `hash(fold_label)`, which changes between SLURM processes. Job 1158363
drew a subsample that reached NMSE ~0.39; job 1184337 drew another that stopped at
0.5581 and hit `AEGateFailure`. Sync `eeg_qgan/training/autoencoder.py` (it now uses
`spawn_rng`) and resubmit — do not raise `max_nmse`.

```bash
cd ~/eeg_qgan
# after rsync of the fixed autoencoder.py
export CC_ACCOUNT=def-khenni_cpu
export DATA_ROOT=/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/BinaryClassification_Data
export OUTPUT_ROOT=$SCRATCH/eeg_qgan_results_thesis_figures
python scripts/narval/submit.py thesis_figure_capture --no-gpu \
  --set output_root=$OUTPUT_ROOT
```

## Submit on Narval

```bash
ssh mrmterki@narval.computecanada.ca
cd ~/eeg_qgan
module load python/3.11 scipy-stack
source $SCRATCH/venvs/eeg_qgan/bin/activate
export CC_ACCOUNT=def-khenni_cpu
export DATA_ROOT=/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/BinaryClassification_Data
export OUTPUT_ROOT=$SCRATCH/eeg_qgan_results_thesis_figures

# One fold (~24–36 h) is enough for thesis panels:
bash scripts/narval/submit_canonical_fleet.sh --wave 5 --cpu
# submits seed0, both folds. For a single fold only:
python scripts/narval/submit.py thesis_figure_capture --no-gpu \
  --set output_root=$OUTPUT_ROOT selection.seed_index=0 selection.fold_start=0 selection.fold_end=1

# All four seed×fold records (change yaml to n_seeds=2 first):
python scripts/narval/submit.py thesis_figure_capture --split-by seed-fold --no-gpu \
  --set output_root=$OUTPUT_ROOT n_seeds=2
```

## Sync and plot locally

```bash
# From your workstation (adjust remote path):
scp -r mrmterki@narval.computecanada.ca:$SCRATCH/eeg_qgan_results_thesis_figures/thesis_figure_capture \
  ./eeg_qgan_results_thesis_figures/

python eeg_qgan/scripts/plot_thesis_figures.py \
  --results-root eeg_qgan_results_thesis_figures/thesis_figure_capture \
  --pfe-copy
```

Figures land in `eeg_qgan/figures/thesis_figures/` and are copied to
`Engineering_PFE/Figures/results_deadline/` when `--pfe-copy` is set.

## LaTeX includes (after plot)

- `fig_training_ae_full.png`
- `fig_training_wgan_full.png`
- `fig_training_detector_full.png`
- `fig_synthetic_waveforms.png`
- `fig_synthetic_spectrogram_quantum.png`
