# Contributor Guide

## Repository layout

- `simulate_llm_efficiency.py` contains the core Monte Carlo model and its
  command-line entry point.
- `sweep_*.py` run budget, economics, and portfolio parameter sweeps.
- `plot_*.py`, `render_*.py`, and
  `package_period_frontier_return_rates.py` post-process existing results.
- `test_*.py` are the pytest regression tests.
- `simulation_config*.yaml` and `validation_config_*.yaml` contain model and
  validation inputs. Do not silently change assumptions in code when the
  corresponding value belongs in configuration.
- `slurm/` contains the tracked array and merge job templates.
- Generated CSV, HDF5, image, archive, cache, and Slurm log files are outputs,
  not source. Do not edit or commit them unless a task explicitly requires it.

The detailed model description and user-facing commands live in `README.md`.
Hardware assumptions are documented in `HARDWARE_BENCHMARKS.md`, and the
simulation algorithm is summarized in `SIMULATION_PSEUDOCODE.md`.

## Local setup and checks

Create a virtual environment from the repository root and install both
runtime and contributor dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
```

On systems that provide the project environment, `conda activate cofin` can
replace virtual-environment creation. Install `requirements-dev.txt` after
activation because the base `cofin` environment may not include pytest or the
formatters.

Formatting is intentionally limited to Python files tracked by Git. Black
uses a 79-column target and isort uses multi-line mode 4. Run them in this
order because isort's requested mode is not Black's native import layout:

```bash
git ls-files -z '*.py' | xargs -0 -n 1 black
git ls-files -z '*.py' | xargs -0 isort
git diff --exit-code -- '*.py'
```

Run regression tests with:

```bash
pytest -q
```

Keep changes surgical. Add or update a regression test for behavior changes,
and avoid reformatting generated outputs or untracked experiment files.

## Slurm runs

Submit from the repository root. The tracked array script maps its six array
indices to the six company profiles and writes one profile per task. Set an
explicit output directory and Python interpreter so compute nodes use the
same environment:

```bash
mkdir -p portfolio_sweep_slurm
ARRAY_JOB=$(sbatch --parsable \
  --export=ALL,OUTPUT_DIR="$PWD/portfolio_sweep_slurm",PYTHON_BIN="$PWD/.venv/bin/python" \
  slurm/portfolio_profile_array.sbatch)
sbatch --dependency="afterok:${ARRAY_JOB}" \
  --export=ALL,OUTPUT_DIR="$PWD/portfolio_sweep_slurm",PYTHON_BIN="$PWD/.venv/bin/python" \
  slurm/portfolio_profile_merge.sbatch
```

The array template defaults to 72 CPUs, 100 Monte Carlo runs, the Numba
backend, and at most six simultaneous tasks. Override job resources with
`sbatch` options and model controls with exported variables, for example:

```bash
sbatch --cpus-per-task=32 \
  --export=ALL,RUNS=20,BACKEND=numpy,OUTPUT_DIR="$PWD/smoke_sweep",PYTHON_BIN="$PWD/.venv/bin/python" \
  slurm/portfolio_profile_array.sbatch
```

Use `squeue -u "$USER"` to monitor jobs, `sacct -j "$ARRAY_JOB"` to inspect
completed task states, and `scancel "$ARRAY_JOB"` to cancel an array. Slurm
stdout and stderr use the `portfolio-profile-*` and `portfolio-merge-*`
patterns. Do not launch large sweeps merely to validate a code change; use a
small local run or reduced `RUNS` smoke submission first.
