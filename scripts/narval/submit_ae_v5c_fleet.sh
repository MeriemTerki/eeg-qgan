#!/bin/bash
# Submit the full ae_v5c publishable fleet on Narval.
#
# Prerequisites (once per login session):
#   cd ~/eeg_qgan
#   module load python/3.11 scipy-stack
#   source $SCRATCH/venvs/eeg_qgan/bin/activate
#   export DATA_ROOT=/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/BinaryClassification_Data
#   export OUTPUT_ROOT=$SCRATCH/eeg_qgan_results_ae_v5c
#
# Usage:
#   bash scripts/narval/submit_ae_v5c_fleet.sh
#
# Safe to run while classification_probe is still going: each study writes under
# its own subdirectory of OUTPUT_ROOT.

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

cd "${PROJECT:-$HOME/eeg_qgan}"
mkdir -p "$OUTPUT_ROOT" logs

echo "DATA_ROOT=$DATA_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo

# ---- Tier 1: headline detection + generation (do not skip) ----
python scripts/narval/submit.py capacity_rematch --split-by seed
python scripts/narval/submit.py decoded_equivalence --split-by seed
python scripts/narval/submit.py qubit_ablation_4q --split-by seed

# ---- Tier 2: AE bottleneck claim ----
python scripts/narval/submit.py bottleneck_comparison --split-by seed
python scripts/narval/submit.py bottleneck_pooled --split-by seed

# ---- Tier 3: mechanism / generative studies (no detector) ----
python scripts/narval/submit.py surrogate --split-by seed
python scripts/narval/submit.py entanglement_ablation --split-by seed
python scripts/narval/submit.py depth_ablation --split-by seed
python scripts/narval/submit.py critic_width --split-by seed
python scripts/narval/submit.py shot_noise --split-by seed

# ---- Tier 4: dose curve (separate roots so points do not overwrite) ----
for d in 512 2048 8192; do
  python scripts/narval/submit.py dose_response --split-by seed --tag "d${d}" \
    --set augmentation.dose_count="$d" n_synthetic="$d" augmentation.max_add="$d" \
          output_root="${SCRATCH}/eeg_qgan_results_ae_v5c_dose${d}"
done

# ---- Tier 5: scarcity curve ----
for f in 0.25 0.5 1.0; do
  python scripts/narval/submit.py data_efficiency --split-by seed --tag "f${f}" \
    --set latent.train_fraction="$f" \
          output_root="${SCRATCH}/eeg_qgan_results_ae_v5c_frac${f}"
done

echo
echo "Submitted. Monitor with: squeue -u \$USER"
squeue -u "$USER"
