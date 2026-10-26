#!/usr/bin/env bash
# Pull study artifact folds from Narval so extra models can run live in the demo.
#
# IMPORTANT
#   - Each study has its OWN autoencoder. Sync whole seed0_fold0 folders.
#   - Do not copy only *.pt generators into thesis_figure_capture.
#   - Narval now requires MFA: run this in an interactive terminal (Git Bash / WSL).
#
# Usage (Git Bash / WSL):
#   export NARVAL_HOST=mrmterki@narval.computecanada.ca
#   export REMOTE_SCRATCH=/scratch/linah03   # where results were written
#   bash demo/sync_from_narval.sh            # clinical fold only
#   SYNC_OPTIONAL=1 bash demo/sync_from_narval.sh
#
# Then restart:
#   cd eeg_qgan && python demo/server.py
# and pick the study in the website dropdown.

set -euo pipefail

NARVAL_HOST="${NARVAL_HOST:-mrmterki@narval.computecanada.ca}"
REMOTE_SCRATCH="${REMOTE_SCRATCH:-/scratch/linah03}"
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "Local thesis root: $LOCAL_ROOT"
echo "Narval host:       $NARVAL_HOST"
echo "Remote scratch:    $REMOTE_SCRATCH"
echo

MAIN=(
  "eeg_qgan_results_thesis_figures/thesis_figure_capture/artifacts/seed0_fold0"
)

OPTIONAL=(
  "eeg_qgan_results_canonical/rematch_deadline/artifacts/seed0_fold0"
  "eeg_qgan_results_canonical/entanglement_deadline/artifacts/seed0_fold0"
  "eeg_qgan_results_canonical/depth_deadline/artifacts/seed0_fold0"
  "eeg_qgan_results_canonical/surrogate/artifacts/seed0_fold0"
  "eeg_qgan_results_canonical/critic_deadline/artifacts/seed0_fold0"
)

sync_one() {
  local rel="$1"
  local remote="$REMOTE_SCRATCH/$rel"
  local local_dir="$LOCAL_ROOT/$rel"
  mkdir -p "$(dirname "$local_dir")"
  echo "==> $rel"
  if ssh "$NARVAL_HOST" "[ -d \"$remote\" ]"; then
    # Prefer rsync when available (resumable); fall back to scp.
    if command -v rsync >/dev/null 2>&1; then
      rsync -av --progress "$NARVAL_HOST:$remote/" "$local_dir/"
    else
      scp -r "$NARVAL_HOST:$remote" "$(dirname "$local_dir")/"
    fi
    echo "    ok -> $local_dir"
    if [[ -d "$local_dir/generators" ]]; then
      echo "    generators: $(ls "$local_dir/generators" | tr '\n' ' ')"
    fi
  else
    echo "    missing on Narval: $remote"
  fi
}

echo "=== Clinical fold (hybrid / twin / GMM + detectors) ==="
for rel in "${MAIN[@]}"; do
  sync_one "$rel"
done

if [[ "${SYNC_OPTIONAL:-0}" == "1" ]]; then
  echo
  echo "=== Optional mechanism / rematch folds ==="
  for rel in "${OPTIONAL[@]}"; do
    sync_one "$rel"
  done
else
  echo
  echo "Optional folds skipped. To pull rematch / entanglement / Fourier weights:"
  echo "  SYNC_OPTIONAL=1 bash demo/sync_from_narval.sh"
fi

echo
echo "Done. Restart the demo and choose the study in the UI."
echo "  cd eeg_qgan && python demo/server.py"
echo "  python demo/verify_studies.py"
