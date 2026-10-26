#!/bin/bash
# Submit only studies that match the CURRENT architecture
# (paper_style_cpu_deadline / canonical_deadline).
#
# Do NOT submit these old names from this script:
#   capacity_rematch, decoded_equivalence, entanglement_ablation,
#   entanglement_ablation_d8, dose_response, data_efficiency,
#   qubit_ablation_4q, shot_noise, clinical_*
#
# PIN (baked into *_deadline YAML, not 600-epoch rematch):
#   latent_dim=64  n_qubits=8  recording-norm  E1 predefined
#   AE 80/20  80k windows  1000 GAN steps  n_folds=2  n_seeds=2
#
# Track B detection is already running:
#   paper_style_cpu_deadline 747356–747359
#   paper_style_q4           806555+
#
# Prerequisites (YOUR account: mrmterki@narval1 — not linah03):
#   whoami   # must print mrmterki
#   cd ~/eeg_qgan
#   module load python/3.11 scipy-stack
#   source $SCRATCH/venvs/eeg_qgan/bin/activate
#   export CC_ACCOUNT=def-khenni_cpu
#   export DATA_ROOT=/scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/BinaryClassification_Data
#   export OUTPUT_ROOT=$SCRATCH/eeg_qgan_results_canonical
#
# Usage:
#   bash scripts/narval/submit_canonical_fleet.sh --list
#   bash scripts/narval/submit_canonical_fleet.sh --wave 2 --cpu
#   bash scripts/narval/submit_canonical_fleet.sh --wave 4 --cpu
#   bash scripts/narval/submit_canonical_fleet.sh --wave 2 --dry-run

# Strip Windows CRLF before `set -o`, or pipefail\r is an invalid option.
if grep -q $'\r' "$0" 2>/dev/null; then
  sed -i 's/\r$//' "$0"
  exec bash "$0" "$@"
fi
set -euo pipefail

WAVE=""
DRY=()
DO_LIST=0
GPU_FLAG=(--no-gpu)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wave) WAVE="${2:-}"; shift 2 ;;
    --dry-run) DRY=(--dry-run); shift ;;
    --gpu) GPU_FLAG=(--gpu); shift ;;
    --cpu|--no-gpu) GPU_FLAG=(--no-gpu); shift ;;
    --list|-h|--help) DO_LIST=1; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

cat <<'MAP'

Current-architecture campaign
=============================

  Already running — do not resubmit
    paper_style_cpu_deadline   10.24 s stft_bilstm, 8q
    paper_style_q4             same pipeline, 4q, d=64

  Already finished on this stack — do not resubmit
    bottleneck_comparison, bottleneck_pooled, surrogate
    depth_ablation L4, critic_width h128

  wave 2  NEW mechanism studies (canonical_deadline YAML)
          rematch_deadline, decoded_deadline, entanglement_deadline
          depth_deadline (L2), critic_deadline (h16)
          skip_downstream, no clinical clips

  wave 4  NEW detection curves (extend paper_style_cpu_deadline)
          dose_deadline 256 / 512 clips  (1024 is the running paper_style job)
          dataeff_deadline 0.25 / 0.5    (1.0 is the running paper_style job)

  wave 5  Thesis figure capture (full training logs + decoded waveform panels)
          thesis_figure_capture — 1 seed x 1 fold by default; same architecture pin

  Never submit from here
    capacity_rematch, decoded_equivalence, entanglement_ablation,
    entanglement_ablation_d8, dose_response, data_efficiency,
    qubit_ablation_4q, shot_noise

MAP

if [[ "$DO_LIST" -eq 1 ]]; then
  exit 0
fi

if [[ -z "$WAVE" ]]; then
  echo "Pick a wave:  bash scripts/narval/submit_canonical_fleet.sh --wave 2"
  echo "Or list:      bash scripts/narval/submit_canonical_fleet.sh --list"
  exit 2
fi

: "${DATA_ROOT:?Set DATA_ROOT}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
export CC_ACCOUNT="${CC_ACCOUNT:-def-khenni_cpu}"

if [[ "${USER:-}" != "mrmterki" ]]; then
  echo "ERROR: logged in as '${USER:-unknown}', not mrmterki."
  echo "Log out of linah03 and ssh mrmterki@narval.computecanada.ca"
  exit 2
fi

cd "${PROJECT:-$HOME/eeg_qgan}"
mkdir -p "$OUTPUT_ROOT" logs

SPLIT=(--split-by seed-fold)

submit() {
  local study="$1"
  shift
  local extra=()
  local sets=()
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == --tag ]]; then
      extra+=(--tag "$2")
      shift 2
    else
      sets+=("$1")
      shift
    fi
  done
  echo
  echo ">>> $study ${extra[*]} --set ${sets[*]}"
  python scripts/narval/submit.py "$study" "${SPLIT[@]}" "${GPU_FLAG[@]}" "${DRY[@]}" \
    "${extra[@]}" --set "${sets[@]}"
}

echo "DATA_ROOT=$DATA_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "wave=$WAVE  device=${GPU_FLAG[*]}  dry=${DRY[*]:-off}"

case "$WAVE" in
  2)
    echo "Wave 2 — remaining mechanism studies (canonical_deadline architecture)."
    submit rematch_deadline      "output_root=${OUTPUT_ROOT}"
    submit decoded_deadline      "output_root=${OUTPUT_ROOT}"
    submit entanglement_deadline "output_root=${OUTPUT_ROOT}"
    submit depth_deadline        "output_root=${OUTPUT_ROOT}"
    submit critic_deadline       "output_root=${OUTPUT_ROOT}"
    ;;

  4)
    echo "Wave 4 — paper-style dose / scarcity (stft_bilstm, 10.24 s). Not the old 1.28 s rematch."
    for d in 256 512; do
      submit dose_deadline \
        "augmentation.max_add=${d}" \
        "output_root=${SCRATCH}/eeg_qgan_results_dose_deadline_d${d}" \
        --tag "d${d}"
    done
    for f in 0.25 0.5; do
      submit dataeff_deadline \
        "latent.train_fraction=${f}" \
        "output_root=${SCRATCH}/eeg_qgan_results_dataeff_deadline_f${f}" \
        --tag "f${f}"
    done
    ;;

  5)
    echo "Wave 5 — thesis figures (full AE/GAN/detector histories + decoded waveform subset)."
    submit thesis_figure_capture "output_root=${OUTPUT_ROOT:-$SCRATCH/eeg_qgan_results_thesis_figures}"
    ;;

  0|1|3)
    echo "Wave $WAVE is retired. Do not resubmit old studies."
    echo "  wave 0/1 already ran on this stack (paper_style + mechanism)."
    echo "  wave 3 was entanglement_ablation_d8 (old 8-d AE) — do not run."
    echo "Use --wave 2 (mechanism) or --wave 4 (paper-style dose/data-eff)."
    exit 2
    ;;

  *)
    echo "Unknown wave '$WAVE'. Use 2 or 4." >&2
    exit 2
    ;;
esac

echo
echo "Done. Monitor with: squeue -u \$USER"
echo "When a study finishes:"
echo "  eeg-qgan-analyze \$OUTPUT_ROOT/<study> --reference classical_equiv --csv \$OUTPUT_ROOT/<study>/tables"
squeue -u "$USER" || true
