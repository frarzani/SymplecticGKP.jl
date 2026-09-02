#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# generate_slurm.sh — SLURM array job generating dimension-reduced GKP-LDLC
# instances, one array task per number of modes n = N_BASE + SLURM_ARRAY_TASK_ID.
#
# BEFORE SUBMITTING (once, on a login node): precompile the environment so the
# heavy Oscar/Nemo build (pulled in transitively via LatticeDecoder) is cached:
#
#   julia --project=/path/to/SymplecticGKP \
#         -e 'using Pkg; Pkg.instantiate(); using SymplecticGKP'
#
# Consider exporting a shared JULIA_DEPOT_PATH so all array tasks reuse the same
# precompiled depot instead of each rebuilding Oscar.
#
# The CVP (distance) computation is exponential and is NOT safely interruptible,
# so each task is wrapped in coreutils `timeout`. A task killed by the timeout
# still leaves valid, distance-less instances on disk (distances_status other
# than "ok"); finish them later with a --distances-only run (more time budget).
#
# Submit e.g.:  sbatch --array=0-40 generate_slurm.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=gkp_ldlc_dim_red
#SBATCH --partition=cpu_homogen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

# ---- configuration (override via environment when submitting) --------------
JULIA_BIN=${JULIA_BIN:-julia}
PROJECT_DIR=${PROJECT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
OUTDIR=${OUTDIR:-"$PROJECT_DIR/data/generated"}
N_BASE=${N_BASE:-4}                 # n = N_BASE + array task id
D=${D:-4}
ATTEMPTS=${ATTEMPTS:-400}
INSTANCES=${INSTANCES:-5}
REDUCTION=${REDUCTION:-kz}
CVP_TIMEOUT=${CVP_TIMEOUT:-40h}     # per-task wall-clock guard for the CVP phase

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
N=$(( N_BASE + TASK_ID ))

mkdir -p "$OUTDIR"
echo "host=$(hostname) task=$TASK_ID n=$N outdir=$OUTDIR timeout=$CVP_TIMEOUT"

# `timeout` returns 124 when it kills the job; treat that as a non-fatal outcome
# (instances were saved in Phase A and can be completed with --distances-only).
set +e
timeout "$CVP_TIMEOUT" "$JULIA_BIN" --project="$PROJECT_DIR" \
    "$PROJECT_DIR/scripts/generate_worker.jl" \
    --n "$N" --d "$D" --attempts "$ATTEMPTS" --instances "$INSTANCES" \
    --seed "$TASK_ID" --outdir "$OUTDIR" --reduction "$REDUCTION"
status=$?
set -e

if [ "$status" -eq 124 ]; then
    echo "task=$TASK_ID n=$N: CVP timed out; distance-less instances left for --distances-only"
    exit 0
fi
exit "$status"
