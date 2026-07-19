#!/usr/bin/env bash
#SBATCH --job-name=portfolio-profile
#SBATCH --array=0-5%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=portfolio-profile-%A_%a.out
#SBATCH --error=portfolio-profile-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/portfolio_sweep_slurm_stochastic_shocks}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS="${RUNS:-100}"
BACKEND="${BACKEND:-numba}"

profiles=(
    administration_heavy
    mixed
    manual_labor_heavy
    software_engineering_heavy
    knowledge_worker_heavy
    creative_worker_heavy
)

profile=${profiles[$SLURM_ARRAY_TASK_ID]}

cd "$PROJECT_DIR"

# Keep numerical libraries from creating threads inside each worker process.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/cofin-mpl-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

# If the cluster needs an environment activation, set PYTHON_BIN or activate it
# in the sbatch command before submission.
"$PYTHON_BIN" sweep_portfolio_rate_changes.py \
    --company-profiles "$profile" \
    --concurrency "$SLURM_CPUS_PER_TASK" \
    --backend "$BACKEND" \
    --runs "$RUNS" \
    --no-plots \
    --skip-combined-summary \
    --output-dir "$OUTPUT_DIR" --stochastic-shocks
