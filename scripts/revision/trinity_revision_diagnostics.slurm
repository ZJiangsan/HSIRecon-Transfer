#!/bin/bash
#SBATCH --job-name=hsi_revision_diag
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=hsi_revision_diag_%j.out
#SBATCH --error=hsi_revision_diag_%j.err
set -euo pipefail

ROOT="${HSIWHEAT40_ROOT:?Set HSIWHEAT40_ROOT to the existing HSIwheat_40 directory}"
SCRIPTS="${HSI_REVISION_SCRIPTS:-${SLURM_SUBMIT_DIR}/scripts/revision}"
PYTHON="${HSI_REVISION_PYTHON:-python}"
# On Trinity, load/activate a Python environment with numpy, pandas,
# matplotlib and torch before submission, or set HSI_REVISION_PYTHON.
"${PYTHON}" "${SCRIPTS}/revision_analyses.py" audit --root "${ROOT}"
"${PYTHON}" "${SCRIPTS}/revision_analyses.py" transfer-table --root "${ROOT}"
"${PYTHON}" "${SCRIPTS}/revision_analyses.py" bootstrap --root "${ROOT}" --nboot 5000
"${PYTHON}" "${SCRIPTS}/revision_analyses.py" spectra --root "${ROOT}" --device cuda --max-models 11
"${PYTHON}" "${SCRIPTS}/feature_deviation_diagnostics.py" --root "${ROOT}"
"${PYTHON}" "${SCRIPTS}/plot_real_spectra_only.py" --root "${ROOT}" --device cpu
