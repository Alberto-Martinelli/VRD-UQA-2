#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --partition=cpu_sapphire
#SBATCH --output=slurm-build_dataset-%j.out

# Usage: sbatch scripts/slurm/build_dataset.sh

set -euo pipefail

START_TIME=$SECONDS
echo "Job started at: $(date)"

module purge
module load miniconda3/3.13.25
module load nvhpc/25.1

source "$HOME/VRD-UQA/scripts/env.sh"
mkdir -p "$HF_HOME"

if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi

cd "$HOME/VRD-UQA"

uv --version
export UV_LINK_MODE=copy
export UV_CACHE_DIR=/tmp/uv_cache_$SLURM_JOB_ID
uv sync -qq

uv run python finetuning/build_dataset.py

mv "$HOME"/slurm* "$HOME/VRD-UQA/" 2>/dev/null || true

ELAPSED=$(( SECONDS - START_TIME ))
printf 'Job finished at: %s\nTotal execution time: %02d:%02d:%02d (%ds)\n' \
    "$(date)" $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
