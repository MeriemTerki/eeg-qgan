#!/bin/bash
# One-time virtualenv creation on Narval. Run on a login node.
#
#   bash scripts/narval/setup_env.sh
#
# Uses the cluster's prebuilt wheels rather than PyPI, which is what the Alliance
# documentation recommends and what avoids the long compile of scientific packages.

set -eo pipefail

module load python/3.11
module load scipy-stack 2>/dev/null || true

VENV="${EEG_QGAN_VENV:-${SCRATCH:-$HOME}/venvs/eeg_qgan}"
echo "Creating virtualenv at $VENV"
mkdir -p "$(dirname "$VENV")"
python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"

pip install --no-index --upgrade pip

# Install from the cluster wheelhouse where possible; fall back to PyPI for the rest.
echo "Installing from the cluster wheelhouse"
pip install --no-index torch numpy scipy scikit-learn pandas matplotlib pyyaml PyWavelets || true

echo "Installing remaining requirements"
pip install pennylane pytest PyWavelets || {
  echo "WARNING: PyPI unreachable from this node. Retry on a login node with network access."
  exit 1
}

PROJECT_DIR="${PROJECT_DIR:-$HOME/eeg_qgan}"
if [[ -f "$PROJECT_DIR/pyproject.toml" ]]; then
  pip install -e "$PROJECT_DIR"
fi

echo
echo "Environment ready. Verifying:"
python -c "import scipy, sklearn, numpy, torch, pennylane, yaml, pywt; print('scipy', scipy.__version__); print('sklearn', sklearn.__version__); print('torch', torch.__version__); print('pennylane', pennylane.__version__); print('pywt', pywt.__version__)"
python -c "import eeg_qgan; print('eeg_qgan', eeg_qgan.__version__)" 2>/dev/null || \
  echo "NOTE: run 'pip install -e $PROJECT_DIR' once the code is synced"

echo
echo "Next: submit a study with"
echo "  python scripts/narval/submit.py capacity_rematch --split-by seed"
