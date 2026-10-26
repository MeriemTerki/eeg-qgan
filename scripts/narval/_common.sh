#!/bin/bash
# Shared environment setup for every Narval job. Sourced, not executed.
#
# Keeps the module list, the virtualenv path and the thread limits in one place so a
# change does not have to be replicated across thirty job scripts, which is how the
# previous scripts drifted apart from one another.

set -eo pipefail

# Job scripts edited on Windows arrive with CRLF line endings, which bash rejects with a
# confusing error. Each launcher repairs itself before sourcing this file.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$HOME/eeg_qgan}"
cd "$PROJECT_DIR" || { echo "ERROR: cannot enter $PROJECT_DIR"; exit 1; }

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Thread limits matched to the allocation. Left unset, the numerical libraries spawn one
# thread per physical core on the node rather than per allocated core, which oversubscribes
# the CPU and slows the job down.
NTHREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export NUMEXPR_NUM_THREADS="$NTHREADS"

module load python/3.11
module load scipy-stack 2>/dev/null || true

VENV="${EEG_QGAN_VENV:-${SCRATCH:-$HOME}/venvs/eeg_qgan}"
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "ERROR: virtualenv not found at $VENV"
  echo "Create it once with: bash scripts/narval/setup_env.sh"
  exit 1
fi
source "$VENV/bin/activate"

# Fail early if the scientific stack is missing; sklearn imports scipy.
python -c "import scipy, sklearn" || {
  echo "ERROR: scipy/sklearn not importable in the venv."
  echo "On the login node run:"
  echo "  module load python/3.11 scipy-stack"
  echo "  source \$SCRATCH/venvs/eeg_qgan/bin/activate"
  echo "  pip install --no-index scipy scikit-learn numpy"
  exit 1
}

# Data root: set DATA_ROOT on the mrmterki login before submit.py.
# Do not rely on /scratch/linah03 unless that path is still group-readable.
if [[ -z "${DATA_ROOT:-}" ]]; then
  for cand in \
    "${SCRATCH}/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/BinaryClassification_Data" \
    "${SCRATCH}/BinaryClassification_Data" \
    /scratch/linah03/EpilepticSeizureProject/Dataset/THUSZ/edf/segment_5_eeg_all/BinaryClassification_Data
  do
    if [[ -d "$cand" ]]; then
      DATA_ROOT="$cand"
      break
    fi
  done
fi
if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "ERROR: DATA_ROOT is not set and no BinaryClassification_Data directory was found."
  echo "On mrmterki@narval1 export DATA_ROOT to your copy of the TUH data."
  exit 1
fi
export DATA_ROOT

# Results land on scratch, which is where large output belongs on this cluster.
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-$HOME}/eeg_qgan_results}"
export OUTPUT_ROOT
mkdir -p "$OUTPUT_ROOT"

echo "=============================================================="
echo "job          ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo "node         $(hostname)"
echo "project      $PROJECT_DIR"
echo "venv         $VENV"
echo "data root    $DATA_ROOT"
echo "output root  $OUTPUT_ROOT"
echo "cpus         $NTHREADS"
echo "started      $(date --iso-8601=seconds)"
echo "=============================================================="

python -c "import torch, sys; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Fail before burning an allocation if the package is not importable or the config is
# inconsistent, rather than after the queue wait.
python -c "import eeg_qgan; print('eeg_qgan', eeg_qgan.__version__)"
