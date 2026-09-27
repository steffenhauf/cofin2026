#!/usr/bin/env bash
#SBATCH --job-name=portfolio-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=portfolio-merge-%j.out
#SBATCH --error=portfolio-merge-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/portfolio_sweep_slurm_stochastic_shocks}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cofin-mpl-merge-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

"$PYTHON_BIN" scripts/sweep_portfolio_rate_changes.py \
    --replot-only \
    --output-dir "$OUTPUT_DIR" --stochastic-shocks
